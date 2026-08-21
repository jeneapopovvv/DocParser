# DocParser

Document parsing server that classifies and extracts structured data from document images using vLLM models.

## Deploy/Redeploy
./redeploy.sh

## API Endpoints

### `POST /process`

Process a document image (JPG, PNG, BMP, PDF) and extract structured data.

**Request (multipart):**
- `file` - Document image or PDF

**Request (JSON):**
```json
{
  "filename": "doc.pdf",
  "content": "<base64>"
}
```

**Response:**
```json
{
  "success": true,
  "result": [{"documentType": "passport", "confidence": 0.95, "data": {...}}]
}
```

---

### `GET /documents`

List all supported document types.

**Response:**
```json
{
  "documents": [{"id": "passport", "description": "Passport", "schema": {...}}],
  "count": 6
}
```

---

### `GET /documents/{doc_id}`

Get a specific document type.

**Response:**
```json
{"id": "passport", "description": "Passport", "schema": {"passportNumber": "string", ...}}
```

---

### `POST /documents`

Create a new document type.

**Request:**
```json
{
  "id": "visa",
  "description": "Visa document",
  "schema": {"visaNumber": "string", "expiryDate": "date"}
}
```

**Response:** `201 Created` with document object.

---

### `PUT /documents/{doc_id}`

Update an existing document type.

**Request:**
```json
{
  "description": "Updated description",
  "schema": {"field1": "string"}
}
```

**Response:** Updated document object.

---

### `DELETE /documents/{doc_id}`

Delete a document type.

**Response:**
```json
{"message": "Document 'visa' deleted successfully"}
```

---

### `GET /health`

Health check with vLLM connectivity status.

**Response:**
```json
{
  "status": "healthy",
  "accepted": [".jpg", ".jpeg", ".bmp", ".png", ".pdf"],
  "requests": 100,
  "valid": 95,
  "invalid": 5,
  "errors": 0,
  "avg_process_time": 2.5,
  "engine": {"classifier": "healthy", "extractor": "healthy"}
}
```

---

## Configuration

| Env Variable | Default | Description |
|---|---|---|
| `CLASSIFIER_URL` | | vLLM classifier endpoint |
| `EXTRACTOR_URL` | | vLLM extractor endpoint |
| `VLLM_TIMEOUT` | 30.0 | Request timeout (seconds) |
| `VLLM_MAX_RETRIES` | 0 | Max retries on 429/502/503/504 |
| `SAVE_INVALID_FILES` | false | Save invalid uploads to disk |
| `MAX_IMAGES` | 3 | Max pages to process from PDF |
| `LOG_LEVEL` | INFO | Logging level |

## Document Storage

Document types are stored as JSON files in `src/documents/`. Default types are initialized on first run:

- `passport` - Passport
- `identity_card` - Government ID
- `driver_license` - Driver license
- `iban` - Financial account statements
- `lease_agreement` - Lease contract
- `cr_certificate` - Commercial Registration Certificate


## CRUD API endpoints:
- GET /documents - List all documents
- GET /documents/{doc_id} - Get a specific document
- POST /documents - Create a new document type
- PUT /documents/{doc_id} - Update an existing document
- DELETE /documents/{doc_id} - Delete a document
