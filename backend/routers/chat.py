"""Chat router.

Exposes the chat query endpoint.

Endpoints
---------
POST /chat/query
    Accept a :class:`~backend.models.chat.ChatRequest`, run the graph-RAG
    pipeline via :class:`~backend.services.chat_service.ChatService`, and
    return a :class:`~backend.models.chat.ChatResponse`.

    Returns 503 Service Unavailable if Ollama is unreachable.

Requirements: 5.2, 5.5
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException, status

from backend.models.chat import ChatRequest, ChatResponse
from backend.services.chat_service import ChatService
from backend.services.ollama_client import OllamaClientError

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/chat")

# ---------------------------------------------------------------------------
# Module-level service singleton (registered by main.py at startup)
# ---------------------------------------------------------------------------

_chat_service: ChatService | None = None


def set_services(chat_service: ChatService) -> None:
    """Register the singleton :class:`~backend.services.chat_service.ChatService`.

    Called once from ``backend/main.py`` during application startup.

    Parameters
    ----------
    chat_service:
        The application-wide :class:`~backend.services.chat_service.ChatService`.
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


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.post("/query", response_model=ChatResponse)
async def chat_query(request: ChatRequest) -> ChatResponse:
    """Process a chat query using the graph-RAG pipeline.

    Accepts the user's query and current conversation history, retrieves
    relevant graph context, constructs a grounded prompt, and returns the
    LLM response together with the updated message history.

    Returns 503 if the Ollama LLM service is unavailable.

    Requirements: 5.2, 5.5
    """
    svc = _require_chat_service()
    try:
        response_text, updated_history = await svc.query(
            user_query=request.query,
            history=request.history,
        )
    except OllamaClientError as exc:
        logger.error("Ollama unavailable during chat query: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="The LLM service (Ollama) is currently unavailable. Please try again later.",
        ) from exc

    return ChatResponse(response=response_text, history=updated_history)
