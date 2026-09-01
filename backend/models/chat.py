"""Pydantic models for the /chat/query endpoint.

Requirements: 5.2, 7.3
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel


class ChatRequest(BaseModel):
    """Request body for POST /chat/query.

    ``history`` is the current Chat Session message history as a list of
    ``{"role": "user"|"assistant", "content": str}`` dicts.  The Backend
    truncates this to the last 10 messages before building the LLM prompt.

    ``backend`` selects the LLM used for both Cypher generation and answer
    synthesis.  Accepted values:
      - ``"ollama"``     — local Mistral 7B (default)
      - ``"deepseek"``   — DeepSeek Chat API
      - ``"openrouter"`` — OpenRouter API (free tiers first)

    Requirements: 5.2
    """

    query: str
    history: list[dict]  # [{"role": "user"|"assistant", "content": str}]
    backend: Literal["ollama", "deepseek", "openrouter"] = "ollama"
    rag_mode: Literal["graphrag", "standard_rag"] = "graphrag"


class ChatResponse(BaseModel):
    """Response body for POST /chat/query (200 OK).

    ``response`` is the LLM-generated answer text.
    ``history`` is the updated message history with the user query and
    assistant response appended, ready to be stored in the Frontend's
    session state.
    ``backend`` echoes back the backend that was used, so the UI can display
    it alongside the response.
    ``generated_cypher`` is the Cypher query produced by Text2Cypher
    (or the keyword-fallback query), exposed for debugging and transparency.
    ``cypher_source`` indicates how the Cypher was obtained:
    ``"text2cypher"`` for LLM-generated, ``"keyword_fallback"`` for the
    direct keyword search, or ``None`` when no retrieval was attempted.

    Requirements: 5.2
    """

    response: str
    history: list[dict]
    backend: str = "ollama"
    generated_cypher: Optional[str] = None
    cypher_source: Optional[str] = None
