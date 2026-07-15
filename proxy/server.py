from fastapi import FastAPI, File, UploadFile, HTTPException, Form, Request
from fastapi.responses import JSONResponse
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

ownership_doc_types = ['title_deed', 'lease_agreement', 'shares_certificate', 'allotment_letter', 'certificate_of_title', 'certificate_of_ownership', 'plot_certificate', 'certificate_of_lease']

collection = {
  "DT0002": {
    "doc_type": ["national_id"], 
    "prompt": prompt_config['identification'],
    "title": "National ID"
  },
  "DT0049": {
    "doc_type": ["passport"], 
    "prompt": prompt_config['identification'],
    "title": "Passport"
  },
  "DT0081": {
    "doc_type": ["military_id"], 
    "prompt": prompt_config['identification'],
    "title": "Military ID"
  },
  "DT0030": {
    "doc_type": ["certificate_of_registration"], 
    "prompt": prompt_config['certificate'],
    "title": "Certificate of Registration"
  },
  "DT0075": {
    "doc_type": ["certificate_of_incorporation"], 
    "prompt": prompt_config['certificate'],
    "title": "Certificate of Incorporation"
  },
  "DT0074": {
    "doc_type": ["taxpayer_registration_certificate", "pin_certificate"], 
    "prompt": prompt_config['kra_pin'],
    "title": "Business KRA Pin"
  },
  "DT0083": {
    "doc_type": ["taxpayer_registration_certificate", "pin_certificate"], 
    "prompt": prompt_config['kra_pin'],
    "title": "Individual KRA Pin"
  },
  "DT0076": {
    "doc_type": ownership_doc_types, 
    "prompt": prompt_config['ownership'],
    "title": "Title Deed"
  },
  "DT0077": {
    "doc_type": ownership_doc_types, 
    "prompt": prompt_config['ownership'],
    "title": "Lease Agreement"
  },
  "DT0078": {
    "doc_type": ownership_doc_types, 
    "prompt": prompt_config['ownership'],
    "title": "Shares Certificate"
  },
  "DT0079": {
    "doc_type": ownership_doc_types, 
    "prompt": prompt_config['ownership'],
    "title": "Allotment Letter"
  },
  "DT0094": {
    "doc_type": ["certificate_of_change_of_name"], 
    "prompt": prompt_config['certificate'],
    "title": "Certificate of Change of Name"
  },
}

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
MAX_IMAGES: int = int(os.getenv("MAX_IMAGES", "5"))

# Allowed file types
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".bmp", ".pdf"}
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
MIN_FILE_SIZE = 1024  # 1KB


def find_by_doc_type(collection: dict, doc_type: str) -> str | None:
    for entry in collection.values():
        if doc_type in entry["doc_type"]:
            return entry["title"]
    return None

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



def get_gender(arg):
    if isinstance(arg, str):
      gender = arg.strip().lower()
      if gender.startswith("f") or "fem" in gender:
        return "FEMALE"
      elif gender.startswith("m") or "male" in gender:
        return "MALE"

    return None


def get_score(arg):
    if arg is None:
        return 95
    try:
        return float(arg)
    except (ValueError, TypeError):
        return None


def get_doc_number(arg):
    if isinstance(arg, str):
        return re.sub(r'[^a-zA-Z0-9]', '', arg)
    return None


def get_certificate_number(arg):
    if isinstance(arg, str):
        return re.sub(r'[^a-zA-Z0-9/\-]', '', arg)
    return None


def get_id_number(arg):
    if isinstance(arg, str):
        return re.sub(r'[^0-9]', '', arg)
    return None


def get_date(arg):

    try:
        if isinstance(arg, str):
            date = re.sub(r'[^a-zA-Z0-9 \.\-/]', '', arg)
            if bool(re.match(r'^\d{4}-\d{2}-\d{2}$', date)):
                dt = datetime.strptime(date, "%Y-%m-%d")
                dt = dt.replace(tzinfo=timezone.utc)
                return int(dt.timestamp()) * 1000

            date = date.lower()
            alpha = re.sub(r'[^a-zA-Z]', '', date)
            if alpha:
                if alpha.startswith('jan') or 'jan' in alpha:
                    date = date.replace(alpha, "01")
                elif alpha.startswith('feb') or 'feb' in alpha:
                    date = date.replace(alpha, "02")
                elif alpha.startswith('mar') or 'mar' in alpha:
                    date = date.replace(alpha, "03")
                elif alpha.startswith('apr') or 'apr' in alpha:
                    date = date.replace(alpha, "04")
                elif alpha.startswith('may') or 'may' in alpha:
                    date = date.replace(alpha, "05")
                elif alpha.startswith('jun') or 'jun' in alpha:
                    date = date.replace(alpha, "06")
                elif alpha.startswith('jul') or 'jul' in alpha:
                    date = date.replace(alpha, "07")
                elif alpha.startswith('aug') or 'aug' in alpha:
                    date = date.replace(alpha, "08")
                elif alpha.startswith('sep') or 'sep' in alpha:
                    date = date.replace(alpha, "09")
                elif alpha.startswith('oct') or 'oct' in alpha:
                    date = date.replace(alpha, "10")
                elif alpha.startswith('nov') or 'nov' in alpha:
                    date = date.replace(alpha, "11")
                elif alpha.startswith('dec') or 'dec' in alpha:
                    date = date.replace(alpha, "12")

            date = date.replace(" ", ".").replace("-", ".").replace("/", ".")
            dt = datetime.strptime(date, "%d.%m.%Y")
            dt = dt.replace(tzinfo=timezone.utc)
            return int(dt.timestamp()) * 1000

    except Exception as e:
        logger.error(f"Failed to convert date {arg}: {str(e)}", exc_info=True)

    return None


def get_country(arg):
    if isinstance(arg, str):
        return re.sub(r'[^a-zA-Z ]', '', arg).upper()
    return None


def get_po_box(arg):
    if isinstance(arg, str):
        return re.sub(r'[^0-9 -]', '', arg)
    return None


def get_postal_code(arg):
    if isinstance(arg, str):
        return re.sub(r'[^0-9]', '', arg)[0:6]
    return None


def get_email(arg):
    if isinstance(arg, str) and '@' in arg:
        return arg.strip()
    return None


def get_phone_number(arg):
    if isinstance(arg, str):
        return re.sub(r'[^+0-9]', '', arg)
    return None

def get_name(name):
    first = ""
    middle = ""
    last = ""
    if isinstance(name, str):
        parts = name.split(" ")
        if len(parts) == 1:
          first = parts[0].strip()
        elif len(parts) == 2:
          first = parts[0].strip()
          last = parts[1].strip()
        elif len(parts) == 3:
          first = parts[0].strip()
          middle = parts[1].strip()
          last = parts[2].strip()
        elif len(parts) > 3:
          first = parts[0].strip()
          middle = parts[1].strip()
          last = " ".join(parts[2:])

    return first, middle, last




def get_passport_data(response: dict) -> dict:

    if response is None:
        return None
    
    first, middle, last = get_name(response.get("name"))
    gender = get_gender(response.get("gender"))
    country = get_country(response.get("country"))
    date_of_birth = get_date(response.get("date_of_birth"))
    passport_number = get_doc_number(response.get("passport_number"))
    confidence_score = get_score(response.get("confidence_score"))

    if passport_number is None or len(passport_number) < 7 or len(passport_number) > 12:
        return None

    return {
        "firstName": first,
        "middleName": middle,
        "lastName": last,
        "documentNumber": passport_number,
        "documentType": "30000CUSDO",
        "country": country,
        "gender": gender,
        "birthDate": date_of_birth,
        "confidenceScore": confidence_score
    }


def get_national_id_data(response: dict) -> dict:

    if response is None:
        return None
    
    first, middle, last = get_name(response.get("name"))
    gender = get_gender(response.get("gender"))
    date_of_birth = get_date(response.get("date_of_birth"))
    id = get_id_number(response.get("id_number"))
    serial = get_id_number(response.get("serial_number"))
    confidence_score = get_score(response.get("confidence_score"))
    country = response.get("country")
    if isinstance(country, str) and "ken" not in country.lower():
        return None

    #since VLM struggles with IDNO detection, additional validation is needed for id and serial
    id_number = None
    if isinstance(id, str) and len(id) >= 7 and len(id) <= 14:
        id_number = id

    if isinstance(serial, str) and len(serial) >= 7 and len(serial) <= 14:
        if id_number is None:
            id_number = serial
        elif len(id_number) > len(serial):
            id_number = serial

    
    if id_number is None or len(id_number) < 7 or len(id_number) > 14:
        return None

    return {
        "firstName": first,
        "middleName": middle,
        "lastName": last,
        "documentNumber": id_number,
        "documentType": "20000CUSDO",
        "country": "KENYA",
        "gender": gender,
        "birthDate": date_of_birth,
        "confidenceScore": confidence_score
    }


def get_military_id_data(response: dict) -> dict:

    if response is None:
        return None
    
    first, middle, last = get_name(response.get("name"))
    id_number = get_id_number(response.get("national_id"))
    confidence_score = get_score(response.get("confidence_score"))

    if id_number is None or len(id_number) < 7 or len(id_number) > 14:
        return None

    return {
        "firstName": first,
        "middleName": middle,
        "lastName": last,
        "documentNumber": id_number,
        "documentType": "40000CUSDO",
        "country": "KENYA",
        "gender": "MALE",
        "birthDate": None,
        "confidenceScore": confidence_score
    }


def get_reg_certificate_data(response: dict) -> dict:

    if response is None:
        return None
    
    doc_title = response.get("document_title", "")
    if isinstance(doc_title, str) and not bool(re.search(r'CERTIFICATE OF REGISTRATION', doc_title, re.IGNORECASE)):
        return None
    
    business_name = response.get("business_name")
    country = get_country(response.get("country"))
    bussines_number = get_certificate_number(response.get("bussines_number"))
    registration_number = get_certificate_number(response.get("registration_number"))
    confidence_score = get_score(response.get("confidence_score"))

    if bussines_number is None or len(bussines_number) == 0:
        if registration_number is None or len(registration_number) == 0:
            return None
        else:
            bussines_number = registration_number

    return {
        "businessName": business_name,
        "documentNumber": bussines_number,
        "documentType": "60000CUSDO",
        "country": country,
        "confidenceScore": confidence_score
    }


def get_inc_certificate_data(response: dict) -> dict:


    if response is None:
        return None
    
    doc_title = response.get("document_title", "")
    if isinstance(doc_title, str) and not bool(re.search(r'CERTIFICATE OF INCORPORATION', doc_title, re.IGNORECASE)):
        return None
    
    business_name = response.get("business_name")
    country = get_country(response.get("country"))
    bussines_number = get_certificate_number(response.get("bussines_number"))
    registration_number = get_certificate_number(response.get("registration_number"))
    confidence_score = get_score(response.get("confidence_score"))

    if registration_number is None or len(registration_number) == 0:
        if bussines_number is None or len(bussines_number) == 0:
            return None
        else:
            registration_number = bussines_number

    return {
        "businessName": business_name,
        "documentNumber": registration_number,
        "documentType": "61000CUSDO",
        "country": country,
        "confidenceScore": confidence_score
    }


def get_name_change_certificate_data(response: dict) -> dict:


    if response is None:
        return None
    
    doc_title = response.get("document_title", "")
    if isinstance(doc_title, str) and not bool(re.search(r'CERTIFICATE OF CHANGE OF NAME', doc_title, re.IGNORECASE)):
        return None
    
    business_name = response.get("business_name")
    country = get_country(response.get("country"))
    registration_number = get_certificate_number(response.get("registration_number"))
    confidence_score = get_score(response.get("confidence_score"))

    if registration_number is None or len(registration_number) == 0:
        return None

    return {
        "businessName": business_name,
        "documentNumber": registration_number,
        "documentType": "63000CUSDO",
        "country": country,
        "confidenceScore": confidence_score
    }

def get_kra_data(response: dict, doc_type: str) -> dict:

    if response is None:
        return None
    
    kra_pin = get_doc_number(response.get("pin", ""))
    doc_title = response.get("document_title", "")

    valid_pin = kra_pin is not None and bool(re.fullmatch(r'^[A-Z][0-9]{9}[A-Z]$', kra_pin))
    valid_doc = False

    if isinstance(doc_title, str):
        tax_reg_cert  = bool(re.search(r'Taxpayer Registration Certificate', doc_title, re.IGNORECASE))
        rev_auth      = bool(re.search(r'Kenya Revenue Authority', doc_title, re.IGNORECASE))
        tax_manag_sys = bool(re.search(r'Tax Management System', doc_title, re.IGNORECASE))
        pin_cert      = bool(re.search(r'PIN Certificate', doc_title, re.IGNORECASE))
        taxpayer_info = bool(re.search(r'Taxpayer Information', doc_title, re.IGNORECASE))
        taxpayer_data = bool(re.search(r'Data of the Taxpayer', doc_title, re.IGNORECASE))
        kra_eReturn   = bool(re.search(r'e-Return Acknowledgment Receipt', doc_title, re.IGNORECASE))
        tax_comp_cert = bool(re.search(r'Tax Compliance Certificate', doc_title, re.IGNORECASE))

        valid_doc = tax_reg_cert or rev_auth or tax_manag_sys or pin_cert or taxpayer_info or taxpayer_data or tax_comp_cert or kra_eReturn
        

    if not valid_pin and not valid_doc:
        return None
    
    confidence_score = get_score(response.get("confidence_score"))
    email = get_email(response.get("email"))
    phone = get_phone_number(response.get("phone"))
    po_box = get_po_box(response.get("po_box"))
    postal_code = get_postal_code(response.get("postal_code"))

    if doc_type == "taxpayer_registration_certificate" and isinstance(po_box, str):
        if '-' in po_box:
            parts = po_box.split(" - ")
            if len(parts) == 2:
                po_box = parts[0].strip()
                postal_code = parts[1].strip()

    if kra_pin is None or len(kra_pin) < 10 or len(kra_pin) > 14:
        return None

    return {
        "kraPin": kra_pin,
        "email": email,
        "phone": phone,
        "poBox": f"P.O. Box {po_box}" if po_box is not None else po_box,
        "postalCode": postal_code,
        "name": response.get("name", ""),
        "country": "KENYA",
        "documentType": "62000CUSDO",
        "county": response.get("county"),
        "district": response.get("district"),
        "city": response.get("city"),
        "street": response.get("street"),
        "building": response.get("building"),
        "confidenceScore": confidence_score
    }


def find_plot_number(arr: List[str]):
    max_slash = 0
    plot = None

    for item in arr:
        if isinstance(item, str) and '/' in item:
            parts = item.split("/")
            if max_slash < len(parts):
                max_slash = len(parts)
                plot = item

    return plot


def get_ownership_data(response: dict, vllm_doc_type: str) -> dict:

    if response is None:
        return None
    
    title_number = response.get("title_number", "")
    parcel_number = response.get("parcel_number", "")
    plot_number = response.get("plot_number", "")
    land_registry = response.get("land_registry", "")

    plot = None
    if vllm_doc_type == "certificate_of_title":
        plot = title_number

    elif vllm_doc_type == "plot_certificate":
        plot = parcel_number

    elif vllm_doc_type == "title_deed":
        plot = title_number

    elif vllm_doc_type == "certificate_of_ownership":
        data = []
        if land_registry is not None and len(land_registry) > 0:
            data.append(land_registry)
        
        if plot_number is not None and len(plot_number) > 0:
            data.append(plot_number)

        if len(data) > 0:
            plot = "/".join(data)

    else:
        plot = find_plot_number([title_number, parcel_number, plot_number, land_registry])

    if not isinstance(plot, str) or len(plot) == 0:
        return None

    return {
        "plotNumber": plot,
        "confidenceScore": get_score(response.get("confidence_score")),
        "nationalId": get_id_number(response.get("national_id"))
    }


async def analyze_images(images: List[str], doc_type: str, task_uuid: str) -> dict:

    try:
        image_contents = []
        for img in images:
            image_contents.append({
                "type": "image_url",
                "image_url": {
                    "url": img
                }
            })
        
        prompt = collection.get(doc_type).get("prompt")
        validate_doc_type = collection.get(doc_type).get("doc_type")

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
            if content.startswith("```json") and content.endswith("```"):
                content = content[8:-4]

            response = json.loads(content)

            vllm_doc_type = response.get("document_type")
            is_valid = vllm_doc_type in validate_doc_type
            extracted_data = None
            if is_valid:
                data = response.get("data")
                if vllm_doc_type == "passport":
                    extracted_data = get_passport_data(data)

                elif vllm_doc_type == "national_id":
                    extracted_data = get_national_id_data(data)

                elif vllm_doc_type == "military_id":
                    extracted_data = get_military_id_data(data)

                elif vllm_doc_type == "taxpayer_registration_certificate" or vllm_doc_type == "pin_certificate":
                    extracted_data = get_kra_data(data, vllm_doc_type)

                elif vllm_doc_type == "certificate_of_registration":
                    extracted_data = get_reg_certificate_data(data)

                elif vllm_doc_type == "certificate_of_incorporation":
                    extracted_data = get_inc_certificate_data(data)

                elif vllm_doc_type == "certificate_of_change_of_name":
                    extracted_data = get_name_change_certificate_data(data)

                elif vllm_doc_type in ownership_doc_types:
                    extracted_data = get_ownership_data(data, vllm_doc_type)

            if is_valid and extracted_data is None:
                logger.error(f"Unable to extract data from VLLM response for task: {task_uuid} \nResponse: {content}")
            elif not is_valid:
                logger.error(f"Invalid document type from VLLM response for task: {task_uuid} \nResponse: {content}")
                
                provided = find_by_doc_type(collection, vllm_doc_type)
                if provided is not None:
                  expected = collection.get(doc_type).get("title")
                  raise DocParserException(status_code=400, error=f"Invalid document type. Please provide '{expected}' instead of '{provided}'", task_id="")

            return extracted_data, is_valid
            
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
#@limiter.limit("1/minute")
async def process_file(
    request: Request,
    file: UploadFile = File(...),
    doc_type: str = Form(...)
):
    global request_count, invalid_count, valid_count, process_time_history, error_count, invalid_list

    task_uuid = str(uuid.uuid4())
    start_time = time.perf_counter()

    try:
        file_id = make_file_id(file)
        request_count += 1
        if not validate_file_size(file):
            raise HTTPException(status_code=400, detail=f"Invalid file size. The file size must be between {MIN_FILE_SIZE / 1024} KB and {MAX_FILE_SIZE / (1024*1024)} MB.")
    
        if doc_type not in collection:
            raise HTTPException(status_code=400, detail="Unsupported document type")
    
        file_ext = get_file_extension(file.filename)
        images_to_analyze: List[str] = []
        logger.info(f"[Task: {task_uuid}] Processing file: {file.filename} as {doc_type}")

        if file_ext == ".jpg":
            base64 = image_to_base64(file, "JPEG")
            images_to_analyze.append(f"data:image/jpg;base64,{base64}")

        elif file_ext == ".jpeg":
            base64 = image_to_base64(file, "JPEG")
            images_to_analyze.append(f"data:image/jpeg;base64,{base64}")

        elif file_ext == ".bmp":
            base64 = image_to_base64(file, "BMP")
            images_to_analyze.append(f"data:image/bmp;base64,{base64}")

        elif file_ext == ".pdf":
            images_to_analyze = pdf_to_images(file, max_pages=MAX_IMAGES)

        else:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type. Allowed: {', '.join(ALLOWED_EXTENSIONS)}"
            )

        vllm_result, is_valid = await analyze_images(images_to_analyze, doc_type, task_uuid)

        if vllm_result is None:
            if file_id not in invalid_list:
                invalid_list.append(file_id)
                invalid_count += 1

                if SAVE_INVALID_FILES:
                    save_uploaded_file(file, f"{doc_type}_{task_uuid}{file_ext}")
        else:
            valid_count += 1

        end_time = time.perf_counter()
        processing_time = end_time - start_time
        if len(process_time_history) > 1000: #keep last 1000 requests
            process_time_history.pop(0)
        process_time_history.append(processing_time)

        return JSONResponse(content={
            "success": is_valid,
            "processingTime": processing_time,
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
        "accepted": ".jpg, .jpeg, .bmp, .pdf",
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

