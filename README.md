
# Requirements
Nvidia driver 550+
16GB+ VRAM
40GB+ storage
16GB+ RAM

# Analysis of proxy/server.py

## 1) High-level summary
- FastAPI service that accepts uploaded document files (.jpg, .jpeg, .bmp, .pdf) and uses an external VLLM-style chat-completion HTTP API to extract structured document fields.
- Images / PDF pages are preprocessed, converted to base64 data URLs, packaged with a prompt (loaded from prompts.yaml) and sent to VLLM.
- VLLM is expected to return JSON describing document_type and data; the server validates and maps the VLLM output into normalized fields for a small set of document classes (passport, national_id, military_id, KRA/tax docs, registration/incorporation certificates, ownership docs).
- Tracks basic metrics (request_count, valid_count, invalid_count, avg processing time), supports saving invalid uploads, and exposes /health.

## 2) Configuration & environment
- VLLM_URL: env VLLM_URL or default "http://vllm:8000/v1/chat/completions"
- VLLM_TIMEOUT: default 30s
- SAVE_INVALID_FILES: env toggles saving invalid files to uploads/invalid
- MAX_IMAGES: max PDF pages to convert (default 5)
- File limits: allowed extensions {".jpg", ".jpeg", ".bmp", ".pdf"}; MAX_FILE_SIZE = 10 MB
- Upload directories: uploads/invalid and uploads/error created at start

## 3) Key file-level components
- load_prompts(): loads prompts.yaml and stores prompt_config (system-level prompt + per-document prompts)
- collection: mapping of external doc_type keys (e.g., "DT0002") to:
  - doc_type: list of vllm document_type strings that are considered valid for that collection entry
  - prompt: text prompt to send to VLLM for that doc class
- ownership_doc_types: list of vllm doc_type strings treated as ownership documents

## 4) Endpoints
- POST /process
  - Parameters: multipart UploadFile file and Form doc_type (must match a key in collection).
  - Generates a task_uuid for logging/errors.
  - Validates file size and extension.
  - Converts:
    - image files -> preprocess_image -> convert to JPEG/BMP and base64 data URL
    - PDF -> pdf2image.convert_from_bytes(...) -> per-page JPEG -> data URLs (limits pages to MAX_IMAGES)
  - Calls analyze_images(images_to_analyze, doc_type, task_uuid) to send data to VLLM and parse result.
  - If SAVE_INVALID_FILES is true and vllm_result is None it calls save_uploaded_file(...)
  - Updates valid/invalid counters and process_time_history
  - Returns JSON: { success: bool, processingTime: float (seconds), result: dict | null }
  - Exceptions are converted to DocParserException and returned with taskId.

- GET /health
  - Returns health object including request counts and avg processing time.

## 5) Image & PDF preprocessing
- preprocess_document_image(image, min_short_side=1200, max_long_side=1600):
  - Convert to RGB if needed
  - ImageOps.autocontrast(... cutoff=2) to improve readability
  - Resize preserving aspect ratio so that the shorter side >= min_short_side and the longer side <= max_long_side, but never upscales (scale ≤ 1.0).
  - Uses PIL.Image.Resampling.LANCZOS for high-quality resize.

- image_to_base64(file, format):
  - Reads file bytes, opens with PIL, preprocesses, saves into BytesIO with given format, returns base64 string.

- pil_image_to_base64(image):
  - Preprocesses PIL image, saves as JPEG and returns base64.

- pdf_to_images(file, max_pages=MAX_IMAGES):
  - Uses pdf2image.convert_from_bytes(... dpi=200, fmt='jpeg') to get per-page PIL Images
  - Converts each page with pil_image_to_base64 and returns list of data URLs
  - If pdf2image fails, raises HTTPException(500)

## 6) VLLM request & response handling (analyze_images)
- Builds messages list for VLLM:
  - system message: prompt_config["system"]
  - user message: prompt (from collection for the doc_type) plus the image contents: each image packaged as {"type":"image_url","image_url":{"url": img_data_url}}
- HTTP request:
  - POST to VLLM_URL with JSON payload: {"messages": messages, "temperature": 0.0}
  - Timeout controlled by VLLM_TIMEOUT, uses httpx.AsyncClient
- Interpreting VLLM response:
  - Expects JSON shape with result["choices"][0]["message"]["content"] (string)
  - If content is wrapped in a Markdown code block starting with "```json" and ending with "```", code strips that wrapper using content[8:-4]
  - Loads content via json.loads -> response dict with keys "document_type" and "data"
  - Checks that returned document_type is one of expected validate_doc_type entries configured for the posted doc_type
  - If valid, passes data to the appropriate extractor (get_passport_data, get_national_id_data, get_kra_data, etc.) which perform field-level normalization and validation
  - Returns tuple (extracted_data_or_None, is_valid_boolean)

## 7) Field extraction helpers (normalization & validation)
- Generic helpers:
  - get_gender, get_score (defaults to 95 if None), get_doc_number, get_certificate_number, get_id_number, get_date (parses multiple formats into epoch ms UTC), get_country, get_po_box, get_postal_code, get_email, get_phone_number
  - get_name splits a string into first, middle, last with reasonable heuristics

- Document-specific extractors:
  - get_passport_data: requires passport_number (7..12 chars after removing non-alnum), maps to documentType "30000CUSDO"
  - get_national_id_data: checks country mentions "ken" then picks best candidate between id_number and serial_number (7..14 digits), documentType "20000CUSDO", country "KENYA"
  - get_military_id_data: similar validation and returns documentType "40000CUSDO"
  - get_reg_certificate_data / get_inc_certificate_data: validate that document_title contains expected keywords, require registration_number, map to codes "60000CUSDO"/"61000CUSDO"
  - get_kra_data: pattern-matches document title for KRA related tokens, normalizes KRA pin length (10..14), extracts address pieces and other fields, maps to "62000CUSDO"
  - get_ownership_data: extracts plot number from various fields depending on vllm_doc_type; returns plotNumber + nationalId + confidence

## 8) Logging and metrics
- logging.basicConfig(level=logging.INFO)
- Logs:
  - Start of each task and VLLM errors or parsing failures
  - If VLLM returns an unexpected doc_type or extractor returns None, logs error with full VLLM content
- Metrics:
  - request_count, valid_count, invalid_count global counters (no locking)
  - process_time_history list stores last N (keeps last 1000) processing times and /health reports average

## 9) File saving & cleanup
- save_uploaded_file(file, filename) writes file contents to uploads/invalid/<filename> and calls remove_files_older_than(INVALID_DIR) to delete files older than 30 days.
- It also checks for available disk space and logs an error if <5GB (but does not abort).
- remove_files_older_than removes files older than cutoff and logs removals.

## 10) Error handling
- Custom DocParserException holds status_code, error string and taskId; exception handler returns those as JSON.
- analyze_images specifically handles httpx.TimeoutException (returns 504), httpx.RequestError (500), and general exceptions (500).
- process_file catches HTTPException and general Exception and rethrows as DocParserException which the handler converts to HTTP JSON response.

## 11) Startup
- If executed as __main__, runs uvicorn on host 0.0.0.0 port 6060


## 12) Expected output format and examples
- On success (valid extraction), /process returns:
  {
    "success": true,
    "processingTime": 1.23,
    "result": { ... normalized fields depending on doc type ... }
  }
- On invalid/unsupported or parse-failure:
  {
    "success": false,
    "processingTime": 1.23,
    "result": null
  }
- On error:
  {
    "error": "<message>",
    "taskId": "<uuid>"
  }


## 13) Diagram (https://www.mermaidchart.com/)

graph LR
  subgraph Client
    C1[User<br>Upload file]
    C2[User<br>Receive result]
  end

  subgraph API_Service
    A1[Endpoint<br>POST process]
    A2[Validate input<br>file size and type]
    A3[Preprocess file<br>image or pdf to images]
    A4[Encode images<br>base64 data URLs]
    A5[Build prompt<br>and request payload]
    A6[Parse VLLM JSON<br>validate document type]
    A7[Extract and normalize<br>fields or mark invalid]
    A8[Save invalid file<br>optional]
    A9[Update metrics<br>and logs]
    A10[Return JSON<br>response]
  end

  subgraph VLLM_Service
    V1[Receive request<br>with prompt and images]
    V2[Run model<br>vision plus language]
    V3[Return JSON<br>in message content]
  end

  %% Interactions
  C1 --> A1
  A1 --> A2
  A2 --> A3
  A3 --> A4
  A4 --> A5
  A5 --> V1
  V1 --> V2
  V2 --> V3
  V3 --> A6
  A6 --> A7
  A7 --> A8
  A7 --> A9
  A7 --> A10
  A10 --> C2