"""Raw hybrid child search endpoint with additive structured metadata."""

from __future__ import annotations

from typing import List, Optional

from fastapi import APIRouter, Query

from models.document import SearchResultItem
from services.retrieval_service import RetrievalService
from services.vector_store import VectorStore

router = APIRouter(prefix="/search")
_retrieval_service: RetrievalService | None = None


def set_services(
    retrieval_service: RetrievalService | None = None,
    vector_store: VectorStore | None = None,
) -> None:
    """Register the singleton retrieval service from ``main.py``.

    ``vector_store`` remains accepted as a compatibility adapter for older
    application/test wiring.
    """
    global _retrieval_service
    if retrieval_service is not None:
        _retrieval_service = retrieval_service
    elif vector_store is not None:
        _retrieval_service = RetrievalService(vector_store)
    else:
        raise ValueError("retrieval_service or vector_store is required")


def _require_retrieval_service() -> RetrievalService:
    if _retrieval_service is None:
        raise RuntimeError("RetrievalService has not been registered from main.py")
    return _retrieval_service


@router.get("/", response_model=List[SearchResultItem])
async def search(
    q: str = Query(..., min_length=1, description="Search query text"),
    top_k: int = Query(5, ge=1, le=50, description="Number of results to return"),
    document_id: Optional[str] = Query(
        None, description="Restrict search to a single document"
    ),
) -> List[SearchResultItem]:
    """Return diversified ranked children without an LLM call."""
    results = await _require_retrieval_service().search(
        query=q, top_k=top_k, document_id=document_id
    )
    items: list[SearchResultItem] = []
    for result in results:
        metadata = result.get("metadata") or {}
        items.append(
            SearchResultItem(
                chunk_id=str(result["id"]),
                document_id=str(metadata.get("document_id") or ""),
                filename=str(metadata.get("filename") or "unknown"),
                chunk_index=int(metadata.get("chunk_index") or 0),
                text=str(result.get("text") or ""),
                similarity=float(result.get("similarity", 0.0)),
                score=float(result.get("score", result.get("similarity", 0.0))),
                child_id=metadata.get("child_id") or result.get("id"),
                parent_id=metadata.get("parent_id"),
                source_sha256=metadata.get("source_sha256"),
                document_version=metadata.get("document_version"),
                pdf_page_start=metadata.get("pdf_page_start"),
                pdf_page_end=metadata.get("pdf_page_end"),
                printed_page_start=metadata.get("printed_page_start"),
                printed_page_end=metadata.get("printed_page_end"),
                page_revisions=metadata.get("page_revisions") or [],
                section_path=metadata.get("section_path") or [],
                section_id=metadata.get("section_id"),
                clause_path=metadata.get("clause_path"),
                heading=metadata.get("heading"),
                breadcrumb=metadata.get("breadcrumb"),
                content_type=metadata.get("content_type"),
                continues_from=metadata.get("continues_from"),
                continues_to=metadata.get("continues_to"),
                table_id=metadata.get("table_id"),
                table_header=metadata.get("table_header") or [],
                code=metadata.get("code"),
                defined_term=metadata.get("defined_term"),
                cross_references=metadata.get("cross_references") or [],
                extraction_method=metadata.get("extraction_method"),
                parser_version=metadata.get("parser_version"),
                relation=metadata.get("relation"),
                governing_context=metadata.get("governing_context"),
                metadata=metadata,
            )
        )
    return items
