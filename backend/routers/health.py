"""Health check router.

Exposes GET /health which returns {"status": "ok"} with HTTP 200 when the
backend is running.

Requirements: 8.4
"""

from fastapi import APIRouter

router = APIRouter()


@router.get("/health")
async def health_check() -> dict:
    """Return a simple liveness response.

    Returns
    -------
    dict
        ``{"status": "ok"}``
    """
    return {"status": "ok"}
