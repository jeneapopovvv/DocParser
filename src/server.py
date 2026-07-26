from fastapi import FastAPI, File, UploadFile, HTTPException, Form, Request
from fastapi.responses import JSONResponse, Response
from fastapi.middleware.cors import CORSMiddleware
from typing import Any, Dict, List, Optional
import os
import uuid
from pathlib import Path
from PIL import Image, ImageOps
import pdf2image
import logging
import base64
import httpx
from datetime import datetime
from io import BytesIO
import time
from pydantic import BaseModel, Field
import json
import re
import yaml
import shutil
import asyncio
from contextlib import asynccontextmanager
from config import supported_documents, document_schemas




def load_prompts(config_path: str = "prompts.yaml"):
  resolved = Path(__file__).resolve().parent / config_path
  if not resolved.exists():
    resolved = Path(config_path)
  with open(resolved, 'r', encoding='utf-8') as file:
    config = yaml.safe_load(file)
  if not config or not isinstance(config, dict):
    raise ValueError("prompts.yaml is empty or not a dict")
  return config


prompt_config = load_prompts()


class _Stats:
    """Thread-safe stats counters using asyncio.Lock."""
    def __init__(self):
        self._lock = asyncio.Lock()
        self.error_count = 0
        self.request_count = 0
        self.valid_count = 0
        self.invalid_count = 0
        self._invalid_ids: dict = {}  # ordered set (insertion order preserved)
        self.process_time_history: List[float] = []

    async def inc_request(self):
        async with self._lock:
            self.request_count += 1

    async def inc_valid(self):
        async with self._lock:
            self.valid_count += 1

    async def inc_invalid(self, file_id: str):
        async with self._lock:
            if file_id not in self._invalid_ids:
                self._invalid_ids[file_id] = None
                self.invalid_count += 1
                if len(self._invalid_ids) > 5000:
                    self._invalid_ids = dict(list(self._invalid_ids.items())[-2500:])

    async def inc_error(self):
        async with self._lock:
            self.error_count += 1

    async def record_time(self, elapsed: float):
        async with self._lock:
            if len(self.process_time_history) > 1000:
                self.process_time_history.pop(0)
            self.process_time_history.append(elapsed)

    async def snapshot(self) -> dict:
        async with self._lock:
            total = len(self.process_time_history)
            avg = sum(self.process_time_history) / total if total else 0
            return {
                "requests": self.request_count,
                "valid": self.valid_count,
                "invalid": self.invalid_count,
                "errors": self.error_count,
                "avg_process_time": avg,
            }


stats = _Stats()

_log_level = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=getattr(logging, _log_level, logging.INFO))
logger = logging.getLogger(__name__)


# Shared httpx client with connection pooling (created on startup, closed on shutdown)
_http_client: Optional[httpx.AsyncClient] = None


@asynccontextmanager
async def lifespan(app):
    global _http_client
    _http_client = httpx.AsyncClient(
        timeout=httpx.Timeout(VLLM_TIMEOUT, connect=10.0),
        limits=httpx.Limits(
            max_connections=200,
            max_keepalive_connections=100,
            keepalive_expiry=300,
        ),
    )
    asyncio.create_task(_periodic_cleanup())
    yield
    if _http_client:
        await _http_client.aclose()
        _http_client = None



app = FastAPI(lifespan=lifespan)

app.add_middleware(
  CORSMiddleware,
  allow_origins=["*"],
  allow_credentials=True,
  allow_methods=["*"],
  allow_headers=["*"],
)

# Configuration
def _safe_float_env(key: str, default: float) -> float:
    val = os.getenv(key)
    if val is None:
        return default
    try:
        return float(val)
    except (ValueError, TypeError):
        logger.warning(f"Invalid {key}={val!r}, using default {default}")
        return default


def _safe_int_env(key: str, default: int) -> int:
    val = os.getenv(key)
    if val is None:
        return default
    try:
        return int(val)
    except (ValueError, TypeError):
        logger.warning(f"Invalid {key}={val!r}, using default {default}")
        return default


UPLOAD_DIR = Path("uploads")
INVALID_DIR = UPLOAD_DIR / "invalid"
ERROR_DIR = UPLOAD_DIR / "error"


# Create directories
for directory in [INVALID_DIR, ERROR_DIR]:
  directory.mkdir(parents=True, exist_ok=True)



CLASSIFIER_URL: str = os.getenv("CLASSIFIER_URL", "")
EXTRACTOR_URL: str = os.getenv("EXTRACTOR_URL", "")
VLLM_TIMEOUT: float = _safe_float_env("VLLM_TIMEOUT", 30.0)
VLLM_MAX_RETRIES: int = _safe_int_env("VLLM_MAX_RETRIES", 0)
CLASSIFICATION_CONCURRENCY: int = _safe_int_env("CLASSIFICATION_CONCURRENCY", 5)
MIN_DISK_SPACE_GB: int = _safe_int_env("MIN_DISK_SPACE_GB", 20)
SAVE_INVALID_FILES: bool = os.getenv("SAVE_INVALID_FILES", "false") == "true"
MAX_IMAGES: int = _safe_int_env("MAX_IMAGES", 3)


ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".bmp", ".png", ".pdf"}
MAX_FILE_SIZE = 10 * 1024 * 1024
MIN_FILE_SIZE = 1024



async def _vllm_request(url: str, payload: dict, retries: int = VLLM_MAX_RETRIES) -> Optional[dict]:
    if _http_client is None:
        logger.error("_vllm_request called before startup")
        return None
    last_exc = None
    for attempt in range(retries + 1):
        try:
            response = await _http_client.post(url, json=payload)
            if response.status_code == 200:
                return response.json()
            if response.status_code in {429, 502, 503, 504} and attempt < retries:
                wait = 1.5 ** (attempt + 1)
                logger.warning(f"vLLM returned {response.status_code}, retry {attempt + 1}/{retries} in {wait:.1f}s")
                await asyncio.sleep(wait)
                continue
            logger.error(f"vLLM request failed: {response.status_code} - {response.text[:300]}")
            return None
        except (httpx.TimeoutException, httpx.RequestError) as e:
            last_exc = e
            if attempt < retries:
                wait = 1.5 ** (attempt + 1)
                logger.warning(f"vLLM request error (attempt {attempt + 1}/{retries}): {e}, retry in {wait:.1f}s")
                await asyncio.sleep(wait)
                continue
            logger.error(f"vLLM request failed after {retries + 1} attempts: {last_exc}")
            return None
    return None


class DocParserException(Exception):
    def __init__(self, status_code: int, error: str, task_id: str):
        self.status_code = status_code
        self.error = error
        self.taskId = task_id



class DocumentOverride(BaseModel):
    """User-supplied document override for classification and extraction."""
    id: str = Field(..., description="Document identifier")
    description: str = Field(..., description="Human-readable document description")
    schema: Dict[str, Any] = Field(..., description="Extraction schema for this document")


class Base64FileRequest(BaseModel):
    """Request model for base64 encoded file"""
    filename: str = Field(..., description="Original filename with extension")
    content: str = Field(..., description="Base64 encoded file content")
    documents: Optional[List[DocumentOverride]] = Field(default=None, description="Optional list of documents to override the default supported docs and schemas")


def get_error_details(error: str, task_id: str):
    return {
        "error": error,
        "taskId": task_id
    }


def validate_file_size(file) -> bool:
    """Validate file size"""
    file.file.seek(0, 2)
    file_size = file.file.tell()
    file.file.seek(0)
    return file_size >= MIN_FILE_SIZE and file_size <= MAX_FILE_SIZE



def get_file_extension(filename: str) -> str:
    """Get file extension in lowercase"""
    return Path(filename).suffix.lower()



def save_uploaded_file(file: UploadFile, filename: str):
    """Save uploaded file with unique name. Returns True on success."""
    try:
        disk_usage = shutil.disk_usage(INVALID_DIR)
        available_space_gb = disk_usage.free / (1024 ** 3)

        if available_space_gb < MIN_DISK_SPACE_GB:
            error_msg = f"Insufficient storage space: {available_space_gb:.2f}GB available (minimum {MIN_DISK_SPACE_GB}GB required)"
            logger.error(f"Failed to save file {filename}: {error_msg}")
            return False

        file.file.seek(0)
        file_path = INVALID_DIR / filename
        with open(file_path, "wb") as buffer:
            buffer.write(file.file.read())

        return True

    except Exception as e:
        logger.error(f"Failed to save file {filename}: {str(e)}", exc_info=True)
        return False


def _cleanup_old_files(directory: Path, days: int = 30):
    try:
        cutoff_time = time.time() - (days * 24 * 60 * 60)
        for filename in os.listdir(directory):
            filepath = os.path.join(directory, filename)
            if os.path.isfile(filepath):
                file_mod_time = os.path.getmtime(filepath)
                if file_mod_time < cutoff_time:
                    os.remove(filepath)
                    logger.info(f"Removed old file: {filepath}")
    except Exception as e:
        logger.error(f"Failed to clean old files: {str(e)}", exc_info=True)


async def _periodic_cleanup():
    """Background task that cleans up old files every hour."""
    while True:
        await asyncio.sleep(3600)
        try:
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, _cleanup_old_files, INVALID_DIR)
            await loop.run_in_executor(None, _cleanup_old_files, ERROR_DIR)
        except Exception as e:
            logger.error(f"Periodic cleanup failed: {e}")


def preprocess_document_image(image, min_short_side=1200, max_long_side=1600):

    # Auto-contrast for better text clarity
    img = image
    if img.mode != 'RGB':
        img = img.convert('RGB')

    img = ImageOps.autocontrast(img, cutoff=2)
    # Resize while preserving aspect ratio
    w, h = img.size
    scale = min(
        max_long_side / max(w, h),
        min_short_side / min(w, h),
        1.0  # don't upscale
    )
    new_w, new_h = int(w * scale), int(h * scale)
    img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
    
    return img


def image_to_base64(file: UploadFile, format: str) -> str:
    try:
        image_data = file.file.read()
        image = Image.open(BytesIO(image_data))
        image = preprocess_document_image(image)
        buffer = BytesIO()
        image.save(buffer, format=format)
        return base64.b64encode(buffer.getvalue()).decode('utf-8')

    except Exception as e:
        logger.error(f"Image encoding failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Image encoding failed")


def pil_image_to_base64(image: Image) -> str:
    try:
        image = preprocess_document_image(image)

        width, height = image.size
        logger.info(f"Image size: {width} x {height}")

        buffer = BytesIO()
        image.save(buffer, format="JPEG")

        img_bytes = buffer.getvalue()
        return base64.b64encode(img_bytes).decode('utf-8')
    except Exception as e:
        logger.error(f"Image encoding failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Image encoding failed")


def pdf_to_images(file: UploadFile, max_pages: int = None) -> List[str]:
    
    try:
        file_content = file.file.read()
        images = pdf2image.convert_from_bytes(
            file_content,
            dpi=200,
            fmt='jpeg'
        )
        
        # Limit to max_pages if specified
        if max_pages:
            images = images[:max_pages]

        if len(images) == 0:
            raise HTTPException(status_code=400, detail="Unable to process PDF file")

        result = []
        for img in images:
            base64 = pil_image_to_base64(img)
            result.append(f"data:image/jpeg;base64,{base64}")

        return result
        
    except Exception as e:
        logger.error(f"PDF conversion failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="PDF conversion failed")



def _normalize_string(value) -> Optional[str]:
    """Strip stray quotes/whitespace and reject non-meaningful values."""
    if isinstance(value, str):
        cleaned = value.strip().strip('"').strip("'").strip()
        if cleaned.lower() in {"", "null", "none", "n/a", "na"}:
            return None
        return cleaned
    if value is None or value is False:
        return None
    return str(value).strip()


def _normalize_date(value) -> Optional[str]:
    """Validate/repair a date string into strict DD/MM/YYYY format."""
    raw = _normalize_string(value)
    if raw is None:
        return None

    cleaned = re.sub(r'[^0-9a-zA-Z/.\- ]', '', raw)
    cleaned = cleaned.replace(" ", ".").replace("-", ".").replace("/", ".")

    # Handle the common YYYY.MM.DD / YYYY-MM-DD inversion from the LLM
    if re.match(r'^\d{4}\.\d{1,2}\.\d{1,2}$', cleaned):
        y, m, d = cleaned.split(".")
        cleaned = f"{d}.{m}.{y}"

    match = re.match(r'^(\d{1,2})\.(\d{1,2})\.(\d{2,4})$', cleaned)
    if not match:
        return None

    day, month, year = match.groups()
    day, month = int(day), int(month)
    if len(year) == 2:
        year = 2000 + int(year) if int(year) < 70 else 1900 + int(year)
    else:
        year = int(year)

    if not (1 <= day <= 31 and 1 <= month <= 12):
        return None

    try:
        datetime(year=year, month=month, day=day)
    except ValueError:
        return None

    return f"{day:02d}/{month:02d}/{year}"


def _normalize_gender(value) -> Optional[str]:
    raw = _normalize_string(value)
    if raw is None:
        return None
    lowered = raw.lower()
    if lowered.startswith("f") or "fem" in lowered:
        return "F"
    if lowered.startswith("m") or "male" in lowered:
        return "M"
    return None

def _normalize_iban(value) -> Optional[str]:
    if value is None:
        return None
    iban = "".join(ch for ch in str(value) if ch.isalnum())
    return iban if len(iban) >= 20 else None

def _normalize_number(value) -> Optional[str]:
    raw = _normalize_string(value)
    if raw is None:
        return None
    try:
        if "." in raw:
            float(raw)
        else:
            int(raw)
        return raw
    except (ValueError, TypeError):
        return None


def _normalize_field(field: str, value, field_type: str = None):
    if field_type == "date":
        return _normalize_date(value)
    if field == "gender":
        return _normalize_gender(value)
    if field == "iban":
        return _normalize_iban(value)
    if field_type == "number":
        return _normalize_number(value)
    return _normalize_string(value)


def _clean_section(section, schema):
    if not isinstance(section, dict):
        return None

    cleaned = {}
    for field, value in section.items():
        if field not in schema:
            continue

        field_schema = schema[field]

        if isinstance(field_schema, list):
            if not isinstance(value, list):
                continue
            if not field_schema:
                continue
            item_schema = field_schema[0]
            if not isinstance(item_schema, dict):
                continue
            cleaned_items = []
            for item in value:
                cleaned_item = _clean_section(item, item_schema)
                if cleaned_item is not None:
                    cleaned_items.append(cleaned_item)
            if cleaned_items:
                cleaned[field] = cleaned_items

        elif isinstance(field_schema, dict):
            if not isinstance(value, dict):
                continue
            cleaned_obj = _clean_section(value, field_schema)
            if cleaned_obj is not None:
                cleaned[field] = cleaned_obj

        else:
            field_type = field_schema if isinstance(field_schema, str) else None
            normalized = _normalize_field(field, value, field_type)
            if normalized is not None:
                cleaned[field] = normalized

    return cleaned if cleaned else None


def extract_json_content(content: str) -> str:
    """Strip markdown fences and recover a JSON object/array substring.

    LLM outputs frequently include ```json fences, leading/trailing prose,
    or multiple blocks. This extracts the first balanced JSON object/array.
    """
    if not isinstance(content, str):
        return ""

    cleaned = content.strip()

    # Remove markdown code fences (``` or ```json)
    fence = re.search(r"```(?:json)?\s*(.*?)\s*```", cleaned, re.DOTALL | re.IGNORECASE)
    if fence:
        cleaned = fence.group(1).strip()

    # Bounded substring search for the first balanced { ... } or [ ... ]
    for open_ch, close_ch in (("{", "}"), ("[", "]")):
        start = cleaned.find(open_ch)
        if start == -1:
            continue
        depth = 0
        in_str = False
        esc = False
        for i in range(start, len(cleaned)):
            ch = cleaned[i]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
                continue
            if ch == '"':
                in_str = True
            elif ch == open_ch:
                depth += 1
            elif ch == close_ch:
                depth -= 1
                if depth == 0:
                    return cleaned[start:i + 1]

    return cleaned


def parse_data(response: dict, schema: dict) -> dict:
 
    if response is None or not isinstance(response, dict):
        logger.warning("parse_data: response is not a dict, returning None")
        return None

    # The LLM may return a sentinel string indicating a rejected document
    if isinstance(response, dict) and set(response.keys()) <= {"other"}:
        return None
 
    return _clean_section(response, schema)



def resolve_document_config(request: Base64FileRequest) -> tuple[Dict[str, str], Dict[str, Dict[str, Any]]]:
    """Return the active supported document list and schemas for a request."""
    if not request.documents:
        return dict(supported_documents), dict(document_schemas)

    supported_docs: Dict[str, str] = {}
    schemas: Dict[str, Dict[str, Any]] = {}
    for document_override in request.documents:
        doc_id = str(document_override.id or "").strip()
        if not doc_id:
            continue
        supported_docs[doc_id] = str(document_override.description or doc_id)
        schemas[doc_id] = dict(document_override.schema or {})

    return supported_docs, schemas


async def classify_images(images: List[str], supported_documents_override: Optional[Dict[str, str]] = None) -> List[dict]:

    active_supported_documents = supported_documents_override or supported_documents
    doc_types_list = "\n".join(
        f"| {doc_id} | {description} |"
        for doc_id, description in active_supported_documents.items()
    )
    user_prompt = prompt_config.get('classification', "").replace("{{listPlaceholder}}", doc_types_list)
    sys_prompt = prompt_config.get('classification_system', "")

    semaphore = asyncio.Semaphore(CLASSIFICATION_CONCURRENCY)

    async def _classify_one(index: int, img: str) -> dict:
        async with semaphore:
            messages = [
                {"role": "system", "content": [{"type": "text", "text": sys_prompt}]},
                {"role": "user", "content": [
                    {"type": "text", "text": user_prompt},
                    {"type": "image_url", "image_url": {"url": img}}
                ]}
            ]

            payload = {
                "messages": messages, 
                "temperature": 0.0
            }

            content = None
            try:
                result = await _vllm_request(CLASSIFIER_URL, payload)
                if result is None:
                    logger.error(f"Classification failed page {index}")
                    return {"pageIndex": index, "documentType": "unknown", "confidence": 0.0}

                content = result["choices"][0]["message"]["content"]
                parsed = json.loads(content)

                document_type = parsed.get("documentType", "unknown") or "unknown"
                return {
                    "pageIndex": index,
                    "documentType": document_type,
                    "confidence": parsed.get("confidence", 0.0),
                    "reasoning": parsed.get("reasoning", "")
                }
            except Exception as e:
                logger.error(f"Classification error page {index}: {str(e)}\nContent: {content}", exc_info=True)
                return {"pageIndex": index, "documentType": "unknown", "confidence": 0.0}

    tasks = [_classify_one(i, img) for i, img in enumerate(images)]
    results = await asyncio.gather(*tasks)

    logger.debug(f"Classifying {len(images)} images with concurrency={CLASSIFICATION_CONCURRENCY}")
    logger.debug(f"Classification user prompt: {user_prompt}")
    logger.debug(f"Classification system prompt: {sys_prompt}")
    logger.debug(f"Classification results: {results}")

    return sorted(results, key=lambda x: x["pageIndex"])


def group_pages(
    classifications: List[dict],
    confidence_threshold: float = 0.0,
    max_noise_pages: int = 2,
    supported_documents_override: Optional[Dict[str, str]] = None,
) -> List[dict]:

    if not classifications:
        return []

    active_supported_documents = supported_documents_override or supported_documents
    supported_set = set(active_supported_documents.keys())
    groups: List[dict] = []
    current_type: Optional[str] = None
    current_indices: List[int] = []
    pending_noise_indices: List[int] = []
    noise_run = 0
    seen_supported = False

    def flush_current_group() -> None:
        nonlocal current_type, current_indices, pending_noise_indices, noise_run
        if current_type is not None:
            groups.append({"documentType": current_type, "pageIndices": current_indices[:]})
        current_type = None
        current_indices = []
        pending_noise_indices = []
        noise_run = 0

    for index, classification in enumerate(classifications):
        image_type = classification.get("documentType", "unknown") or "unknown"
        confidence = float(classification.get("confidence", 0.0) or 0.0)
        is_supported = image_type in supported_set and confidence >= confidence_threshold

        if is_supported:
            if current_type is None:
                current_type = image_type
                current_indices = pending_noise_indices + [index]
                pending_noise_indices = []
                seen_supported = True
            elif current_type == image_type:
                current_indices.append(index)
            else:
                flush_current_group()
                current_type = image_type
                current_indices = [index]
                seen_supported = True
            noise_run = 0
            continue

        if current_type is None:
            pending_noise_indices.append(index)
            continue

        if noise_run < max_noise_pages:
            current_indices.append(index)
            noise_run += 1
        else:
            flush_current_group()

    if current_type is not None:
        groups.append({"documentType": current_type, "pageIndices": current_indices[:]})

    return groups if seen_supported else []


async def extract_content(
    images: List[str],
    doc_type: str,
    document_schemas_override: Optional[Dict[str, Dict[str, Any]]] = None,
) -> dict:

    active_document_schemas = document_schemas_override or document_schemas
    schema = active_document_schemas.get(doc_type, None)
    if schema is None:
        logger.error(f"Unsupported document type for extraction: {doc_type}")
        return None
    
    sys_prompt = prompt_config.get('extraction_system', "")
    user_prompt = prompt_config.get('extraction', "").replace("{{jsonPlaceholder}}", json.dumps(schema, indent=2))

    messages = [
        {"role": "system", "content": [{"type": "text", "text": sys_prompt}]},
        {"role": "user", "content": [
            {"type": "text", "text": user_prompt},
            *[{"type": "image_url", "image_url": {"url": img}} for img in images]
        ]}
    ]

    payload = {
        "messages": messages, 
        "temperature": 0.0
    }

    result = await _vllm_request(EXTRACTOR_URL, payload)

    if result is None:
        logger.error(f"Extraction failed for {doc_type} after retries")
        return None

    content = result["choices"][0]["message"]["content"]

    logger.debug(f"Extraction system prompt for {doc_type}: {sys_prompt}")
    logger.debug(f"Extraction user prompt for {doc_type}: {user_prompt}")
    logger.debug(f"Extraction result for {doc_type}: {content}")

    try:
        json_data = json.loads(content)
        return parse_data(json_data, schema)
    except (json.JSONDecodeError, ValueError, TypeError):
        logger.error(f"Extraction JSON parse failed for {doc_type}: {content}", exc_info=True)
        return None


async def analyze_images(
    images: List[str],
    supported_documents_override: Optional[Dict[str, str]] = None,
    document_schemas_override: Optional[Dict[str, Dict[str, Any]]] = None,
) -> list:

    try:
        classifications = await classify_images(images, supported_documents_override=supported_documents_override) if CLASSIFIER_URL else []
        groups = group_pages(classifications, supported_documents_override=supported_documents_override) if CLASSIFIER_URL else []

        if not CLASSIFIER_URL or len(groups) == 0:
            logger.info(f"Classifier {'skipped' if not CLASSIFIER_URL else 'no groups formed'}, extracting all pages as single group")
            groups = [{"documentType": "unknown", "pageIndices": list(range(len(images)))}]

        async def _extract_one(group: dict) -> Optional[dict]:
            group_images = [images[i] for i in group["pageIndices"]]

            group_conf = [
                c["confidence"] for c in classifications
                if c["pageIndex"] in group["pageIndices"]
            ]
            confidence = max(group_conf) if group_conf else 0.0

            document_type = group.get("documentType", "unknown") or "unknown"
            data = await extract_content(
                group_images,
                document_type,
                document_schemas_override=document_schemas_override,
            )

            return {
                "documentType": document_type,
                "confidence": confidence,
                "data": data
            }

        extraction_tasks = [_extract_one(g) for g in groups]
        extraction_results = await asyncio.gather(*extraction_tasks, return_exceptions=True)
        results = [
            r for r in extraction_results
            if isinstance(r, dict) and r.get("data") is not None
        ]

        return results

    except httpx.TimeoutException:
        logger.error("Request timed out during analysis.")
        raise HTTPException(status_code=504, detail="Request timed out.")
    except httpx.RequestError as e:
        logger.error(f"API request failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="API request failed.")
    except DocParserException as e:
        raise e
    except Exception as e:
        logger.error(f"Error in analyze_images: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to process file.")


def make_file_id(file) -> str:
    return f"{file.filename}:{file.size}"


def contains_file(request: Base64FileRequest) -> bool:
    return request.content is not None and request.filename is not None and len(request.content) > 0 and len(request.filename) > 0


def decode_base64_file(request: Base64FileRequest) -> UploadFile:
    """Decode base64 string to UploadFile-like object"""
    try:
        file_content = base64.b64decode(request.content)
    except Exception as e:
        raise HTTPException(status_code=400, detail="Invalid base64 encoding")

    file_size = len(file_content)
    if file_size < MIN_FILE_SIZE or file_size > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file size. The file size must be between {MIN_FILE_SIZE / 1024} KB and {MAX_FILE_SIZE / (1024*1024)} MB."
        )

    file_ext = get_file_extension(request.filename)
    if file_ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type. Allowed: {', '.join(ALLOWED_EXTENSIONS)}"
        )

    # Create a BytesIO object that behaves like an UploadFile
    file_obj = BytesIO(file_content)
    file_obj.seek(0)

    class Base64UploadFile:
        def __init__(self, filename: str, file_obj: BytesIO, size: int):
            self.filename = filename
            self.file = file_obj
            self.size = size
            self.content_type = None

        def seek(self, offset: int, whence: int = 0):
            return self.file.seek(offset, whence)

        def tell(self):
            return self.file.tell()

        def read(self, size: int = -1):
            return self.file.read(size)

    return Base64UploadFile(request.filename, file_obj, file_size)


@app.middleware("http")
async def remove_headers(request: Request, call_next):
    response: Response = await call_next(request)
    if "content-length" in response.headers:
        del response.headers["content-length"]
    if "date" in response.headers:
        del response.headers["date"]
    return response


@app.exception_handler(DocParserException)
async def doc_parser_exception_handler(request: Request, exc: DocParserException):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.error,
            "taskId": exc.taskId
        }
    )


@app.post("/process")
async def process_file(
    request: Request,
    file: Optional[UploadFile] = File(None),
):

    task_uuid = str(uuid.uuid4())
    start_time = time.perf_counter()

    try:
        supported_documents_override: Optional[Dict[str, str]] = None
        document_schemas_override: Optional[Dict[str, Dict[str, Any]]] = None

        content_type = request.headers.get("content-type", "")
        if "application/json" in content_type:
            try:
                body = await request.json()
                req = Base64FileRequest(**body)
                if contains_file(req):
                    file = decode_base64_file(req)
                supported_documents_override, document_schemas_override = resolve_document_config(req)
            except HTTPException:
                raise
            except Exception:
                raise HTTPException(status_code=400, detail="Invalid base64 request body")


        if file is None:
            raise HTTPException(status_code=400, detail="Provide either file (multipart) or base64 request (JSON)")

        file_id = make_file_id(file)
        await stats.inc_request()
        if not validate_file_size(file):
            raise HTTPException(status_code=400, detail=f"Invalid file size. The file size must be between {MIN_FILE_SIZE / 1024} KB and {MAX_FILE_SIZE / (1024*1024)} MB.")
    
        file_ext = get_file_extension(file.filename)
        images_to_analyze: List[str] = []
        logger.info(f"[Task: {task_uuid}] Processing file: {file.filename}")

        if file_ext == ".jpg":
            base64 = image_to_base64(file, "JPEG")
            images_to_analyze.append(f"data:image/jpg;base64,{base64}")

        elif file_ext == ".jpeg":
            base64 = image_to_base64(file, "JPEG")
            images_to_analyze.append(f"data:image/jpeg;base64,{base64}")

        elif file_ext == ".bmp":
            base64 = image_to_base64(file, "BMP")
            images_to_analyze.append(f"data:image/bmp;base64,{base64}")

        elif file_ext == ".png":
            base64 = image_to_base64(file, "PNG")
            images_to_analyze.append(f"data:image/png;base64,{base64}")
            
        elif file_ext == ".pdf":
            images_to_analyze = pdf_to_images(file, max_pages=MAX_IMAGES)

        else:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type. Allowed: {', '.join(ALLOWED_EXTENSIONS)}"
            )


        vllm_result = await analyze_images(
            images_to_analyze,
            supported_documents_override=supported_documents_override,
            document_schemas_override=document_schemas_override,
        )
        if len(vllm_result) == 0:
            await stats.inc_invalid(file_id)
            if SAVE_INVALID_FILES:
                save_uploaded_file(file, f"{task_uuid}{file_ext}")
        else:
            await stats.inc_valid()


        end_time = time.perf_counter()
        processing_time = end_time - start_time
        await stats.record_time(processing_time)


        return JSONResponse(content={
            "success": len(vllm_result) > 0,
            "result": vllm_result
        })

    except HTTPException as er:
        await stats.inc_error()
        logger.error(f"[Task: {task_uuid}]. Error processing file: {str(er)}", exc_info=True)
        raise DocParserException(status_code=er.status_code, error=str(er.detail), task_id=task_uuid)
    
    except DocParserException as er:
        await stats.inc_error()
        logger.error(f"[Task: {task_uuid}]. Error processing file: {str(er)}", exc_info=True)
        raise er

    except Exception as e:
        await stats.inc_error()
        logger.error(f"[Task: {task_uuid}]. Error processing file: {str(e)}", exc_info=True)
        raise DocParserException(status_code=500, error="Failed to process file", task_id=task_uuid)



@app.get("/health")
async def health_check():
    """Health check endpoint with vLLM connectivity probe."""
    snapshot = await stats.snapshot()

    vllm_status = {}
    for name, url in [("classifier", CLASSIFIER_URL), ("extractor", EXTRACTOR_URL)]:
        if not url:
            vllm_status[name] = "not configured"
            continue
        try:
            base = url.rsplit("/v1", 1)[0] if "/v1" in url else url.rsplit("/", 1)[0]
            resp = await _http_client.get(f"{base}/health", timeout=5.0)
            vllm_status[name] = "healthy" if resp.status_code == 200 else f"unhealthy ({resp.status_code})"
        except Exception:
            vllm_status[name] = "unreachable"

    overall = "healthy"
    if any(v == "unreachable" for v in vllm_status.values()):
        overall = "degraded"

    return {
        "status": overall,
        "accepted": ALLOWED_EXTENSIONS,
        "requests": snapshot["requests"],
        "valid": snapshot["valid"],
        "invalid": snapshot["invalid"],
        "errors": snapshot["errors"],
        "max_file_size_mb": MAX_FILE_SIZE / (1024 * 1024),
        "avg_process_time": snapshot["avg_process_time"],
        "engine": vllm_status,
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=6060)

