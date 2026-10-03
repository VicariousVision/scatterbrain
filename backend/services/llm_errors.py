"""Shared error base for chat LLM clients (Ollama, Gemini)."""


class LLMClientError(Exception):
    """Raised when a chat LLM client fails to produce an answer."""
