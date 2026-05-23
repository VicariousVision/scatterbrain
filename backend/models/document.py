"""Pydantic models for document upload, listing, and graph summary endpoints."""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel


class DocumentRecord(BaseModel):
    """Internal record tracking a document's processing state.

    Stored in the in-memory document store keyed by document_id.
    Status transitions: processing → completed | failed.
    """

    document_id: str
    filename: str
    uploaded_at: datetime
    status: Literal["processing", "completed", "failed"]
    error: str | None = None


class UploadResponse(BaseModel):
    """Response body for POST /documents/upload (202 Accepted).

    Requirements: 1.4
    """

    document_id: str


class DocumentListItem(BaseModel):
    """Single element in the GET /documents response array.

    Requirements: 7.3
    """

    document_id: str
    filename: str
    uploaded_at: datetime
    status: str  # "processing" | "completed" | "failed"


class GraphSummary(BaseModel):
    """Response body for GET /documents/{document_id}/graph-summary.

    Requirements: 7.5
    """

    node_count: int
    edge_count: int
