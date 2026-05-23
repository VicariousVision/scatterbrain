"""
Ollama HTTP client for interacting with the locally-hosted LLM via the Ollama REST API.

Provides async methods for text generation and connectivity health checks.
"""

from __future__ import annotations

import httpx


class OllamaClientError(Exception):
    """Raised when the Ollama client encounters an error communicating with the Ollama API."""


class OllamaClient:
    """Thin async HTTP wrapper around the Ollama REST API.

    Args:
        base_url: Base URL of the Ollama server (e.g. ``http://localhost:11434``).
        model:    Name of the model to use for generation (e.g. ``llama3``).
    """

    def __init__(self, base_url: str, model: str) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model

    async def generate(self, prompt: str) -> str:
        """Send a prompt to the Ollama ``/api/generate`` endpoint and return the response text.

        Args:
            prompt: The full prompt string to send to the model.

        Returns:
            The model's response as a plain string.

        Raises:
            OllamaClientError: If the request fails or the response cannot be parsed.
        """
        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                response = await client.post(
                    f"{self.base_url}/api/generate",
                    json={"model": self.model, "prompt": prompt, "stream": False},
                )
                response.raise_for_status()
                return response.json()["response"]
        except httpx.HTTPStatusError as exc:
            raise OllamaClientError(
                f"Ollama API returned an error status {exc.response.status_code}: {exc.response.text}"
            ) from exc
        except httpx.RequestError as exc:
            raise OllamaClientError(
                f"Failed to connect to Ollama at {self.base_url}: {exc}"
            ) from exc
        except (KeyError, ValueError) as exc:
            raise OllamaClientError(
                f"Unexpected response format from Ollama: {exc}"
            ) from exc

    async def health_check(self) -> bool:
        """Check whether the Ollama server is reachable by hitting ``/api/tags``.

        Returns:
            ``True`` if the server responds with HTTP 200, ``False`` otherwise.
        """
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(f"{self.base_url}/api/tags")
                return response.status_code == 200
        except Exception:
            return False
