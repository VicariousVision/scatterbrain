"""Pydantic models for document management and raw search APIs."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field


class DocumentRecord(BaseModel):
    """Internal persisted document processing/provenance record."""

    document_id: str
    filename: str
    uploaded_at: datetime
    status: Literal["processing", "completed", "failed"]
    error: str | None = None
    chunk_count: int = 0
    source_sha256: str | None = None
    document_title: str | None = None
    document_version: str | None = None
    extraction_method: str | None = None
    parser_version: str | None = None
    schema_version: int = 1
    needs_reingestion: bool = False


class UploadResponse(BaseModel):
    """Response body for ``POST /documents/upload`` (202)."""

    document_id: str


class DocumentListItem(BaseModel):
    """Document status returned by list/detail endpoints."""

    document_id: str
    filename: str
    uploaded_at: datetime
    status: str
    error: str | None = None
    chunk_count: int = 0
    source_sha256: str | None = None
    document_title: str | None = None
    document_version: str | None = None
    parser_version: str | None = None
    needs_reingestion: bool = False


class SearchResultItem(BaseModel):
    """Raw child search match with backward-compatible core fields."""

    chunk_id: str
    document_id: str
    filename: str
    chunk_index: int
    text: str
    similarity: float
    score: float | None = None
    child_id: str | None = None
    parent_id: str | None = None
    source_sha256: str | None = None
    document_version: str | None = None
    pdf_page_start: int | None = None
    pdf_page_end: int | None = None
    printed_page_start: str | None = None
    printed_page_end: str | None = None
    page_revisions: list[str] = Field(default_factory=list)
    section_path: list[str] = Field(default_factory=list)
    section_id: str | None = None
    clause_path: str | None = None
    heading: str | None = None
    breadcrumb: str | None = None
    content_type: str | None = None
    continues_from: str | None = None
    continues_to: str | None = None
    table_id: str | None = None
    table_header: list[str] = Field(default_factory=list)
    code: str | None = None
    defined_term: str | None = None
    cross_references: list[str] = Field(default_factory=list)
    extraction_method: str | None = None
    parser_version: str | None = None
    relation: str | None = None
    governing_context: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
