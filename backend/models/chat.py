"""Pydantic models for the /chat/query endpoint.

Requirements: 5.2, 7.3
"""

from pydantic import BaseModel


class ChatRequest(BaseModel):
    """Request body for POST /chat/query.

    ``history`` is the current Chat Session message history as a list of
    ``{"role": "user"|"assistant", "content": str}`` dicts.  The Backend
    truncates this to the last 10 messages before building the LLM prompt.

    Requirements: 5.2
    """

    query: str
    history: list[dict]  # [{"role": "user"|"assistant", "content": str}]


class ChatResponse(BaseModel):
    """Response body for POST /chat/query (200 OK).

    ``response`` is the LLM-generated answer text.
    ``history`` is the updated message history with the user query and
    assistant response appended, ready to be stored in the Frontend's
    session state.

    Requirements: 5.2
    """

    response: str
    history: list[dict]
