"""Search router.

Exposes a raw semantic search endpoint over uploaded document chunks,
without going through the LLM. Useful for inspecting what the RAG
pipeline would retrieve for a given query, or for building a search UI.

Endpoints
---------
GET /search?q=...&top_k=5&document_id=...
    Return the top matching chunks ordered by similarity (descending).
"""

from __future__ import annotations

import logging
from typing import List, Optional

from fastapi import APIRouter, Query

from models.document import SearchResultItem
from services.vector_store import VectorStore

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/search")

# Module-level service singleton (registered by main.py at startup)
_vector_store: VectorStore | None = None


def set_services(vector_store: VectorStore) -> None:
    """Register the singleton VectorStore used by this router.

    Called once from ``backend/main.py`` during application startup.

    Parameters
    ----------
    vector_store:
        The application-wide VectorStore.
    """
    global _vector_store
    _vector_store = vector_store


def _require_vector_store() -> VectorStore:
    if _vector_store is None:
        raise RuntimeError(
            "VectorStore has not been registered. "
            "Call search_router.set_services() from main.py."
        )
    return _vector_store


@router.get("/", response_model=List[SearchResultItem])
async def search(
    q: str = Query(..., min_length=1, description="Search query text"),
    top_k: int = Query(5, ge=1, le=50, description="Number of results to return"),
    document_id: Optional[str] = Query(
        None, description="Restrict search to a single document"
    ),
) -> List[SearchResultItem]:
    """Semantic search over document chunks using cosine similarity.

    Returns the raw matched chunks (no LLM call), ordered by similarity
    descending.
    """
    vector_store = _require_vector_store()

    results = await vector_store.search(
        query=q,
        top_k=top_k,
        document_id=document_id,
    )

    return [
        SearchResultItem(
            chunk_id=r["id"],
            document_id=r["metadata"]["document_id"],
            filename=r["metadata"]["filename"],
            chunk_index=r["metadata"]["chunk_index"],
            text=r["text"],
            similarity=r["similarity"],
        )
        for r in results
    ]
