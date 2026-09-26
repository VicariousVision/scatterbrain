"""Documents router.

Exposes document upload, listing, retrieval, and deletion endpoints.

Endpoints
---------
POST /documents/upload
    Accept a multipart file upload, start background processing, return 202
    with the generated ``document_id``.

GET /documents
    Return a JSON array of all document records.

GET /documents/{document_id}
    Return a single document record by ID, or 404 if not found.

DELETE /documents/{document_id}
    Delete a document and all its chunks.
"""

from __future__ import annotations

import logging
from typing import List

from fastapi import APIRouter, File, HTTPException, UploadFile, status

from models.document import DocumentListItem, UploadResponse
from services.document_service import DocumentService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/documents")

# Module-level service singleton (registered by main.py at startup)
_document_service: DocumentService | None = None


def set_services(document_service: DocumentService) -> None:
    """Register the singleton service instance used by this router.
    
    Called once from ``backend/main.py`` during application startup.
    
    Parameters
    ----------
    document_service:
        The application-wide DocumentService.
    """
    global _document_service
    _document_service = document_service


def _require_document_service() -> DocumentService:
    if _document_service is None:
        raise RuntimeError(
            "DocumentService has not been initialised. "
            "Call documents_router.set_services() from main.py."
        )
    return _document_service


@router.post(
    "/upload",
    status_code=status.HTTP_202_ACCEPTED,
    response_model=UploadResponse,
)
async def upload_document(
    file: UploadFile = File(...),
) -> UploadResponse:
    """Accept a multipart file upload and start background processing.
    
    Returns 202 Accepted immediately with the generated ``document_id``.
    
    Supported file types: PDF, TXT
    """
    svc = _require_document_service()
    
    # Validate file type
    filename = file.filename or "unknown"
    if not (filename.lower().endswith(".pdf") or filename.lower().endswith(".txt")):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only PDF and TXT files are supported",
        )
    
    content = await file.read()
    record = await svc.upload(filename=filename, content=content)
    
    logger.info("Upload accepted: document_id=%s", record.document_id)
    return UploadResponse(document_id=record.document_id)


@router.get("/", response_model=List[DocumentListItem])
async def list_documents() -> List[DocumentListItem]:
    """Return all document records ordered by upload time (newest first)."""
    svc = _require_document_service()
    records = svc.list_documents()
    return [
        DocumentListItem(
            document_id=r.document_id,
            filename=r.filename,
            uploaded_at=r.uploaded_at,
            status=r.status,
            error=r.error,
        )
        for r in records
    ]


@router.get("/{document_id}", response_model=DocumentListItem)
async def get_document(document_id: str) -> DocumentListItem:
    """Return a single document record by ID."""
    svc = _require_document_service()
    record = svc.get_document(document_id)
    if record is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document '{document_id}' not found.",
        )
    return DocumentListItem(
        document_id=record.document_id,
        filename=record.filename,
        uploaded_at=record.uploaded_at,
        status=record.status,
        error=record.error,
    )


@router.delete("/{document_id}", status_code=status.HTTP_204_NO_CONTENT, response_model=None)
async def delete_document(document_id: str) -> None:
    """Delete a document and all its chunks from vector store and database."""
    svc = _require_document_service()
    record = svc.get_document(document_id)
    if record is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document '{document_id}' not found.",
        )
    
    await svc.delete_document(document_id)
    logger.info("Deleted document: %s", document_id)
