"""Pydantic models for the /chat/query endpoint."""

from __future__ import annotations

from pydantic import BaseModel


class ChatRequest(BaseModel):
    """Request body for POST /chat/query.
    
    ``history`` is the current chat session message history as a list of
    ``{"role": "user"|"assistant", "content": str}`` dicts.
    """
    
    query: str
    history: list[dict] = []


class ChatResponse(BaseModel):
    """Response body for POST /chat/query (200 OK).
    
    ``response`` is the LLM-generated answer text.
    ``history`` is the updated message history with the user query and
    assistant response appended.
    ``retrieved_chunks`` is the number of document chunks retrieved for context.
    """
    
    response: str
    history: list[dict]
    retrieved_chunks: int = 0
