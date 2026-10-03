"""Chat router.

Exposes the chat query endpoint.

Endpoints
---------
POST /chat/query
    Accept a ChatRequest, run the RAG pipeline, and return a ChatResponse.
    
    Returns 503 Service Unavailable if the chat LLM is unavailable.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException, status

from config import settings
from models.chat import ChatRequest, ChatResponse
from models.content import ChatResult
from services.chat_service import ChatService
from services.llm_errors import LLMClientError

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/chat")

# Module-level service singleton (registered by main.py at startup)
_chat_service: ChatService | None = None


def set_services(chat_service: ChatService) -> None:
    """Register the singleton ChatService.
    
    Called once from ``backend/main.py`` during application startup.
    
    Parameters
    ----------
    chat_service:
        The application-wide ChatService.
    """
    global _chat_service
    _chat_service = chat_service


def _require_chat_service() -> ChatService:
    if _chat_service is None:
        raise RuntimeError(
            "ChatService has not been initialised. "
            "Call chat_router.set_services() from main.py."
        )
    return _chat_service


@router.post("/query", response_model=ChatResponse)
async def chat_query(request: ChatRequest) -> ChatResponse:
    """Process a chat query using the RAG pipeline.
    
    Retrieves relevant document chunks from the vector store and generates
    an answer using the LLM with grounded context.
    
    Returns 503 if the LLM service is unavailable.
    """
    svc = _require_chat_service()
    
    try:
        result = await svc.query(
            user_query=request.query,
            top_k=settings.retrieval_top_k,
        )
        if isinstance(result, ChatResult):
            response_text = result.answer
            retrieved_chunks = result.source_texts
            citations = result.citations
        else:  # Backward-compatible service doubles/older implementations.
            response_text, retrieved_chunks = result
            citations = []
    except LLMClientError as exc:
        logger.error("LLM unavailable during chat query: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="The LLM service is unavailable. Please try again later.",
        ) from exc
    except Exception as exc:
        logger.error("Chat query failed: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Chat query failed: {str(exc)}",
        ) from exc
    
    # Build updated history
    updated_history = request.history + [
        {"role": "user", "content": request.query},
        {"role": "assistant", "content": response_text},
    ]
    
    return ChatResponse(
        response=response_text,
        history=updated_history,
        retrieved_chunks=len(retrieved_chunks),
        citations=citations,
    )
