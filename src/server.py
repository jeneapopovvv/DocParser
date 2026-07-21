from fastapi import FastAPI, File, UploadFile, HTTPException, Form, Request
from fastapi.responses import JSONResponse, Response
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional
import os
import uuid
from pathlib import Path
from PIL import Image, ImageOps
import pdf2image
import logging
import base64
import httpx
from datetime import datetime, timezone
from io import BytesIO
import time
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from pydantic import BaseModel
from enum import Enum
import json
from typing import Optional
import re
import yaml
import shutil

class DocParserException(Exception):
    def __init__(self, status_code: int, error: str, task_id: str):
        self.status_code = status_code
        self.error = error
        self.taskId = task_id


def load_prompts(config_path: str = "prompts.yaml"):
  with open(config_path, 'r', encoding='utf-8') as file:
    config = yaml.safe_load(file)
  return config


prompt_config = load_prompts()


error_count = 0
request_count = 0
valid_count = 0
invalid_count = 0
invalid_list: List[str] = []
process_time_history: List[float] = []

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


limiter = Limiter(key_func=get_remote_address)
app = FastAPI()
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)


# CORS middleware
app.add_middleware(
  CORSMiddleware,
  allow_origins=["*"],
  allow_credentials=True,
  allow_methods=["*"],
  allow_headers=["*"],
)

# Configuration
UPLOAD_DIR = Path("uploads")
INVALID_DIR = UPLOAD_DIR / "invalid"
ERROR_DIR = UPLOAD_DIR / "error"


# Create directories
for directory in [INVALID_DIR, ERROR_DIR]:
  directory.mkdir(parents=True, exist_ok=True)



VLLM_URL: str = os.getenv("VLLM_URL", "http://vllm:8000/v1/chat/completions")
VLLM_TIMEOUT: float = float(os.getenv("VLLM_TIMEOUT", "30")) #default 30 sec
SAVE_INVALID_FILES: bool = os.getenv("SAVE_INVALID_FILES", "false") == "true"
MAX_IMAGES: int = int(os.getenv("MAX_IMAGES", "3"))

# Allowed file types
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".bmp", ".png", ".pdf"}
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
MIN_FILE_SIZE = 1024  # 1KB



def get_error_details(error: str, task_id: str):
    return {
        "error": error,
        "taskId": task_id
    }


def validate_file_size(file: UploadFile) -> bool:
    """Validate file size"""
    file.file.seek(0, 2)
    file_size = file.file.tell()
    file.file.seek(0)
    return file_size >= MIN_FILE_SIZE and file_size <= MAX_FILE_SIZE



def get_file_extension(filename: str) -> str:
    """Get file extension in lowercase"""
    return Path(filename).suffix.lower()



def save_uploaded_file(file: UploadFile, filename: str):
    """Save uploaded file with unique name"""
    
    try:
        disk_usage = shutil.disk_usage(INVALID_DIR)
        available_space_gb = disk_usage.free / (1024 ** 3)  # Convert bytes to GB
        
        if available_space_gb < 20:
            error_msg = f"Insufficient storage space: {available_space_gb:.2f}GB available (minimum 5GB required)"
            logger.error(f"Failed to save file {filename}: {error_msg}", exc_info=True)
            return

        file.file.seek(0)
        file_path = INVALID_DIR / filename
        with open(file_path, "wb") as buffer:
          buffer.write(file.file.read())

    except Exception as e:
        logger.error(f"Failed to save file {filename}: {str(e)}", exc_info=True)

    remove_files_older_than(INVALID_DIR)


def remove_files_older_than(directory, days=30):
    
    try:
        cutoff_time = time.time() - (days * 24 * 60 * 60)

        for filename in os.listdir(directory):
            filepath = os.path.join(directory, filename)

            if os.path.isfile(filepath):
                file_mod_time = os.path.getmtime(filepath)

                if file_mod_time < cutoff_time:
                    os.remove(filepath)
                    logger.info(f"Removed: {filepath}")
                    
    except Exception as e:
        logger.error(f"Failed to clean invalid files: {str(e)}", exc_info=True)


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



VALID_BANKING_FIELDS = {
    "name", "accountType", "accountNumber", "iban", "branch", "date", "bic"
}
VALID_IDENTITY_FIELDS = {
    "personalNumber", "nationality", "name", "nameAr", "dateOfBirth", "expiryDate", "gender"
}
ACCEPTED_DOCUMENT_TYPES = [
    "passport", "national_id", "driver_license"
]

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


def _normalize_field(field: str, value):
    """Apply field-specific normalization/corrections."""
    if field in {"date", "dateOfBirth", "expiryDate"}:
        return _normalize_date(value)
    if field == "gender":
        return _normalize_gender(value)
    return _normalize_string(value)


def _clean_section(section, valid_fields):
    """Validate a section dict and keep only known, meaningful fields."""
    if not isinstance(section, dict):
        return None

    cleaned = {}
    for field, value in section.items():
        if field not in valid_fields:
            continue
        normalized = _normalize_field(field, value)
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


def parse_data(response: dict) -> dict:
    '''
    Validates that `response` conforms to the expected JSON schema and
    applies corrections where possible (field names, date formats, gender
    casing, stray markdown/quotes). Returns a cleaned dict, or None if the
    input is not a valid document structure.
    '''

    if response is None or not isinstance(response, dict):
        logger.warning("parse_data: response is not a dict, returning None")
        return None

    # The LLM may return a sentinel string indicating a rejected document
    if isinstance(response, dict) and set(response.keys()) <= {"other"}:
        return None

    docType = response.get('documentType')
    if docType not in ACCEPTED_DOCUMENT_TYPES:
        return None

    banking = _clean_section(response.get("bankingData"), VALID_BANKING_FIELDS)
    identity = _clean_section(response.get("identityData"), VALID_IDENTITY_FIELDS)

    if identity is not None and identity.get("documentType"):
        identity["documentType"] = docType

    # Required-field enforcement: a section is discarded (null) when its key
    # identifier is missing or empty after normalization.
    if identity is not None and not identity.get("personalNumber"):
        logger.info("parse_data: identityData dropped, missing personalNumber")
        identity = None

    if banking is not None and not banking.get("iban"):
        logger.info("parse_data: bankingData dropped, missing iban")
        banking = None

    if banking is None and identity is None:
        return None

    return {
        **({"bankingData": banking} if banking else {}),
        **({"identityData": identity} if identity else {}),
    }



async def analyze_images(images: List[str]) -> dict:

    try:
        image_contents = []
        for img in images:
            image_contents.append({
                "type": "image_url",
                "image_url": {
                    "url": img
                }
            })
        
        prompt = prompt_config['identification']

        messages = [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text", 
                        "text": prompt_config["system"]
                    }
                ]
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt
                    },
                    *image_contents
                ]
            }
        ]
        

        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        
        payload = {
            "messages": messages,
            "temperature": 0.0
        }
        
        async with httpx.AsyncClient(timeout=VLLM_TIMEOUT) as client:
            response = await client.post(
                VLLM_URL,
                headers=headers,
                json=payload
            )
            
            if response.status_code != 200:
                logger.error(f"Failed to process file. API error: {response.status_code} - {response.text}")
                raise HTTPException(status_code=500, detail="Failed to process file")
            
            result = response.json()
            
            content = result["choices"][0]["message"]["content"]
            jsonData = None
            try:
                jsonData = json.loads(extract_json_content(content))
            except (json.JSONDecodeError, ValueError, TypeError):
                logger.error("Failed to process file. LLM output is not valid JSON.")

            response = parse_data(jsonData)
            if response is None:
                logger.error(f"Failed to extract data. {content}")

            return response
            
    except httpx.TimeoutException:
        logger.error("Failed to process file. Request timed out.")
        raise HTTPException(status_code=504, detail="Failed to process file. Request timed out.")
    except httpx.RequestError as e:
        logger.error(f"Failed to process file.API request failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to process file.")
    except DocParserException as e:
        raise e
    except Exception as e:
        logger.error(f"Failed to process file.Error calling API: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to process file.")


def avg_processing_time():
    global process_time_history

    total = len(process_time_history)
    avg = 0
    if total == 0:
        return total, avg

    avg = sum(process_time_history) / total
    return total, avg


def make_file_id(file: UploadFile) -> str:
    return f"{file.filename}:{file.size}"



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
    file: UploadFile = File(...)
):
    global request_count, invalid_count, valid_count, process_time_history, error_count, invalid_list

    task_uuid = str(uuid.uuid4())
    start_time = time.perf_counter()

    try:
        file_id = make_file_id(file)
        request_count += 1
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


        vllm_result = await analyze_images(images_to_analyze)
        if vllm_result is None:
            if file_id not in invalid_list:
                invalid_list.append(file_id)
                invalid_count += 1

                if SAVE_INVALID_FILES:
                    save_uploaded_file(file, f"{task_uuid}{file_ext}")
        else:
            valid_count += 1


        end_time = time.perf_counter()
        processing_time = end_time - start_time
        if len(process_time_history) > 1000: #keep last 1000 requests
            process_time_history.pop(0)
        process_time_history.append(processing_time)


        return JSONResponse(content={
            "success": vllm_result is not None,
            "result": vllm_result
        })

    except HTTPException as er:
        error_count += 1
        logger.error(f"[Task: {task_uuid}]. Error processing file: {str(er)}", exc_info=True)
        raise DocParserException(status_code=er.status_code, error=str(er.detail), task_id=task_uuid)
    
    except DocParserException as er:
        error_count += 1
        logger.error(f"[Task: {task_uuid}]. Error processing file: {str(er)}", exc_info=True)
        raise er

    except Exception as e:
        error_count += 1
        logger.error(f"[Task: {task_uuid}]. Error processing file: {str(e)}", exc_info=True)
        raise DocParserException(status_code=500, error="Failed to process file", task_id=task_uuid)



@app.get("/health")
async def health_check():
    """Health check endpoint"""
    total, avg = avg_processing_time()
    return {
        "status": "healthy",
        "accepted": ".jpg, .jpeg, .png, .bmp, .pdf",
        "requests": request_count,
        "valid": valid_count,
        "invalid": invalid_count,
        "errors": error_count,
        "max_file_size_mb": MAX_FILE_SIZE / (1024 * 1024),
        "avg_process_time": avg
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=6060)

