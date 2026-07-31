"""Filesystem-based document storage for supported documents and schemas."""

import json
import os
import asyncio
from pathlib import Path
from typing import Any, Dict, List, Optional
import logging

logger = logging.getLogger(__name__)

DOCUMENTS_DIR = Path("documents")


def _ensure_documents_dir():
    """Create documents directory if it doesn't exist."""
    DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)


def _document_path(doc_id: str) -> Path:
    """Get the file path for a document."""
    return DOCUMENTS_DIR / f"{doc_id}.json"


def _validate_doc_id(doc_id: str) -> str:
    """Validate and sanitize document ID."""
    doc_id = doc_id.strip()
    if not doc_id:
        raise ValueError("Document ID cannot be empty")
    if not doc_id.replace("_", "").replace("-", "").isalnum():
        raise ValueError("Document ID can only contain alphanumeric characters, hyphens, and underscores")
    return doc_id


def load_documents_from_disk() -> tuple[Dict[str, str], Dict[str, Dict[str, Any]]]:
    """Load all documents from filesystem. Returns (supported_documents, document_schemas)."""
    _ensure_documents_dir()

    supported_documents = {}
    document_schemas = {}

    for file_path in DOCUMENTS_DIR.glob("*.json"):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            doc_id = data.get("id")
            if not doc_id:
                logger.warning(f"Skipping document file without id: {file_path}")
                continue

            supported_documents[doc_id] = data.get("description", doc_id)
            document_schemas[doc_id] = data.get("schema", {})
        except Exception as e:
            logger.error(f"Failed to load document from {file_path}: {e}")

    return supported_documents, document_schemas


def load_document(doc_id: str) -> Optional[Dict[str, Any]]:
    """Load a single document from filesystem."""
    _ensure_documents_dir()
    path = _document_path(doc_id)

    if not path.exists():
        return None

    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Failed to load document {doc_id}: {e}")
        return None


def save_document(doc_id: str, description: str, schema: Dict[str, Any]) -> Dict[str, Any]:
    """Save a document to filesystem. Creates or updates."""
    _ensure_documents_dir()
    doc_id = _validate_doc_id(doc_id)

    document = {
        "id": doc_id,
        "description": description,
        "schema": schema,
    }

    path = _document_path(doc_id)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(document, f, indent=2, ensure_ascii=False)

    logger.info(f"Saved document: {doc_id}")
    return document


def delete_document(doc_id: str) -> bool:
    """Delete a document from filesystem. Returns True if deleted."""
    _ensure_documents_dir()
    path = _document_path(doc_id)

    if not path.exists():
        return False

    path.unlink()
    logger.info(f"Deleted document: {doc_id}")
    return True


def list_documents() -> List[Dict[str, Any]]:
    """List all documents from filesystem."""
    _ensure_documents_dir()
    documents = []

    for file_path in sorted(DOCUMENTS_DIR.glob("*.json")):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            documents.append(data)
        except Exception as e:
            logger.error(f"Failed to read document from {file_path}: {e}")

    return documents


def document_exists(doc_id: str) -> bool:
    """Check if a document exists."""
    _ensure_documents_dir()
    return _document_path(doc_id).exists()


def initialize_default_documents():
    """Initialize default documents if no documents exist on disk."""
    _ensure_documents_dir()

    existing = list(DOCUMENTS_DIR.glob("*.json"))
    if existing:
        logger.info(f"Found {len(existing)} existing document files, skipping initialization")
        return

    logger.info("No documents found, initializing defaults")

    default_documents = {
        "passport": {
            "id": "passport",
            "description": "Passport",
            "schema": {
                "passportNumber": "string",
                "nationality": "string",
                "fullName": "string",
                "fullNameArabic": "string",
                "dateOfBirth": "date",
                "expiryDate": "date",
                "gender": "string"
            }
        },
        "identity_card": {
            "id": "identity_card",
            "description": "Government ID",
            "schema": {
                "personalNumber": "number",
                "nationality": "string",
                "fullName": "string",
                "fullNameArabic": "string",
                "dateOfBirth": "date",
                "expiryDate": "date",
                "gender": "string"
            }
        },
        "driver_license": {
            "id": "driver_license",
            "description": "Driver license",
            "schema": {
                "licenseNumber": "number",
                "nationality": "string",
                "fullName": "string",
                "fullNameArabic": "string",
                "dateOfBirth": "date",
                "gender": "string"
            }
        },
        "iban": {
            "id": "iban",
            "description": "Financial account statements from banking institutions",
            "schema": {
                "fullName": "string",
                "bankAccountNumber": "string",
                "iban": "string",
                "branch": "string",
                "date": "date",
                "bic/swift": "string"
            }
        },
        "lease_agreement": {
            "id": "lease_agreement",
            "description": "Lease agreement contract between landlord and tenant",
            "schema": {
                "landlords": [{
                    "name": "string",
                    "email": "string",
                    "phone": "string",
                    "cpr": "string"
                }],
                "tenants": [{
                    "name": "string",
                    "email": "string",
                    "phone": "string",
                    "cpr": "string"
                }],
                "propertyAddress": "string",
                "propertyType": "string",
                "leaseStartDate": "date",
                "leaseEndDate": "date"
            }
        },
        "cr_certificate": {
            "id": "cr_certificate",
            "description": "Commercial Registration Certificate issued by the government",
            "schema": {
                "registrationNumber": "string",
                "commercialName": "string",
                "commercialNameArabic": "string",
                "type": "string",
                "status": "string",
                "issueDate": "date",
                "expiryDate": "date"
            }
        }
    }

    for doc_id, doc_data in default_documents.items():
        save_document(doc_id, doc_data["description"], doc_data["schema"])

    logger.info(f"Initialized {len(default_documents)} default documents")


class DocumentStore:
    """Async wrapper around filesystem document storage."""

    def __init__(self):
        self._lock = asyncio.Lock()
        initialize_default_documents()
        self._supported_documents, self._document_schemas = load_documents_from_disk()

    @property
    def supported_documents(self) -> Dict[str, str]:
        return dict(self._supported_documents)

    @property
    def document_schemas(self) -> Dict[str, Dict[str, Any]]:
        return dict(self._document_schemas)

    async def list_documents(self) -> List[Dict[str, Any]]:
        """List all documents."""
        async with self._lock:
            return list_documents()

    async def get_document(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """Get a single document."""
        async with self._lock:
            doc_id = _validate_doc_id(doc_id)
            return load_document(doc_id)

    async def create_document(self, doc_id: str, description: str, schema: Dict[str, Any]) -> Dict[str, Any]:
        """Create a new document. Raises ValueError if document already exists."""
        async with self._lock:
            doc_id = _validate_doc_id(doc_id)
            if document_exists(doc_id):
                raise ValueError(f"Document '{doc_id}' already exists")

            document = save_document(doc_id, description, schema)
            self._supported_documents[doc_id] = description
            self._document_schemas[doc_id] = schema
            return document

    async def update_document(self, doc_id: str, description: str, schema: Dict[str, Any]) -> Dict[str, Any]:
        """Update an existing document. Raises ValueError if document doesn't exist."""
        async with self._lock:
            doc_id = _validate_doc_id(doc_id)
            if not document_exists(doc_id):
                raise ValueError(f"Document '{doc_id}' not found")

            document = save_document(doc_id, description, schema)
            self._supported_documents[doc_id] = description
            self._document_schemas[doc_id] = schema
            return document

    async def delete_document(self, doc_id: str) -> bool:
        """Delete a document. Returns True if deleted."""
        async with self._lock:
            doc_id = _validate_doc_id(doc_id)
            deleted = delete_document(doc_id)
            if deleted:
                self._supported_documents.pop(doc_id, None)
                self._document_schemas.pop(doc_id, None)
            return deleted

    async def refresh(self):
        """Reload all documents from disk."""
        async with self._lock:
            self._supported_documents, self._document_schemas = load_documents_from_disk()
