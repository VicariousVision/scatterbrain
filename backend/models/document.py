"""Pydantic models for document upload, listing, and search endpoints."""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel


class DocumentRecord(BaseModel):
    """Internal record tracking a document's processing state.

    Persisted in SQLite by ``DocumentDB``, keyed by document_id.
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
    error: str | None = None


class SearchResultItem(BaseModel):
    """Single match in the GET /search response array."""

    chunk_id: str
    document_id: str
    filename: str
    chunk_index: int
    text: str
    similarity: float
