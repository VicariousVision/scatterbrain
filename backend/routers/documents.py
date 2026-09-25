"""Documents router.

Exposes the document upload, listing, retrieval, and graph-summary endpoints.

Endpoints
---------
POST /documents/upload
    Accept a multipart file upload, start background processing, return 202
    with the generated ``document_id``.

GET /documents
    Return a JSON array of all document records.

GET /documents/{document_id}
    Return a single document record by ID, or 404 if not found.

GET /documents/{document_id}/graph-summary
    Return the node and edge counts for a document's graph data.

Requirements: 1.2, 1.4, 7.2, 7.3, 7.4, 7.5
"""

from __future__ import annotations

import logging
from typing import List

from fastapi import APIRouter, File, Form, HTTPException, UploadFile, status

from backend.models.document import (
    DocumentListItem,
    GraphSummary,
    UploadResponse,
)
from backend.services.document_service import DocumentService
from backend.services.graph_service import GraphService, GraphServiceError

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/documents")

# ---------------------------------------------------------------------------
# Module-level service singletons (registered by main.py at startup)
# ---------------------------------------------------------------------------
# Using module-level singletons avoids the complexity of threading a Request
# object through every endpoint signature alongside UploadFile parameters.

_document_service: DocumentService | None = None
_graph_service: GraphService | None = None


def set_services(
    document_service: DocumentService,
    graph_service: GraphService,
) -> None:
    """Register the singleton service instances used by this router.

    Called once from ``backend/main.py`` during application startup.

    Parameters
    ----------
    document_service:
        The application-wide :class:`~backend.services.document_service.DocumentService`.
    graph_service:
        The application-wide :class:`~backend.services.graph_service.GraphService`.
    """
    global _document_service, _graph_service
    _document_service = document_service
    _graph_service = graph_service


def _require_document_service() -> DocumentService:
    if _document_service is None:
        raise RuntimeError(
            "DocumentService has not been initialised. "
            "Call documents_router.set_services() from main.py."
        )
    return _document_service


def _require_graph_service() -> GraphService:
    if _graph_service is None:
        raise RuntimeError(
            "GraphService has not been initialised. "
            "Call documents_router.set_services() from main.py."
        )
    return _graph_service


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.post(
    "/upload",
    status_code=status.HTTP_202_ACCEPTED,
    response_model=UploadResponse,
)
async def upload_document(
    file: UploadFile = File(...),
    index_type: str = Form("graphrag"),
) -> UploadResponse:
    """Accept a multipart file upload and start background processing.

    Returns 202 Accepted immediately with the generated ``document_id``.

    Requirements: 1.2, 1.4
    """
    svc = _require_document_service()
    content = await file.read()
    record = await svc.upload(
        filename=file.filename or "unknown",
        content=content,
        index_type=index_type,
    )
    logger.info("Upload accepted: document_id=%s", record.document_id)
    return UploadResponse(document_id=record.document_id)


@router.get("/", response_model=List[DocumentListItem])
async def list_documents() -> List[DocumentListItem]:
    """Return all document records.

    Requirements: 7.2, 7.3
    """
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
    """Return a single document record by ID.

    Requirements: 7.3
    """
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


@router.get("/{document_id}/graph-summary", response_model=GraphSummary)
async def get_graph_summary(document_id: str) -> GraphSummary:
    """Return the node and edge counts for a document's graph data.

    Requirements: 7.4, 7.5
    """
    doc_svc = _require_document_service()
    # Verify the document exists first.
    record = doc_svc.get_document(document_id)
    if record is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document '{document_id}' not found.",
        )

    graph_svc = _require_graph_service()
    try:
        summary = graph_svc.get_graph_summary(document_id)
    except GraphServiceError as exc:
        logger.error(
            "Graph summary failed for document_id=%s: %s", document_id, exc
        )
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Graph database is unavailable.",
        ) from exc

    return GraphSummary(
        node_count=summary["node_count"],
        edge_count=summary["edge_count"],
    )
