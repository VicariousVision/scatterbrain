"""Pydantic models for the ``/chat/query`` endpoint."""

from __future__ import annotations

from pydantic import BaseModel, Field

from models.content import Citation


class ChatRequest(BaseModel):
    """Question and backward-compatible prior display history.

    Parameters
    ----------
    query:
        Natural-language question.
    history:
        Existing display messages; not treated as retrieved evidence.
    """

    query: str
    history: list[dict] = Field(default_factory=list)


class ChatResponse(BaseModel):
    """Grounded answer with additive structured source citations.

    Parameters
    ----------
    response:
        Generated answer text.
    history:
        Updated display history.
    citations:
        Exactly-used structured source locations; empty for legacy results.
    """

    response: str
    history: list[dict]
    retrieved_chunks: int = 0
    citations: list[Citation] = Field(default_factory=list)
