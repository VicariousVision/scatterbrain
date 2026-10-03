"""Health check router.

Exposes GET /health which returns {"status": "ok"} with HTTP 200 when the
backend is running, plus the active chat LLM provider and model.

Requirements: 8.4
"""

from fastapi import APIRouter, Request

router = APIRouter()


@router.get("/health")
async def health_check(request: Request) -> dict:
    """Return a liveness response with the active chat LLM provider.

    Returns
    -------
    dict
        ``{"status": "ok", "llm_provider": ..., "llm_model": ...}``
    """
    state = request.app.state
    return {
        "status": "ok",
        "llm_provider": getattr(state, "llm_provider", "ollama"),
        "llm_model": getattr(state, "llm_model", None),
    }
