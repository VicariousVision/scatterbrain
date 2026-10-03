"""Async Gemini chat client (optional cloud LLM provider).

Mirrors ``OllamaClient.chat`` so ``ChatService`` can use either.

SDK facts verified against google-genai 2.28.0:
- ``errors.APIError`` exposes ``code``, ``status``, ``message`` and ``details``
  (the parsed JSON body; a 429 may carry a ``RetryInfo`` entry with
  ``retryDelay`` such as ``"30s"``).
- ``types.HttpOptions.timeout`` is in milliseconds.
- The async API is ``client.aio.models.generate_content`` / ``.get``.

Privacy: prompts and API keys are never logged or placed in error messages.
"""

from __future__ import annotations

import asyncio
import logging
import re
from typing import Any, Awaitable, Callable, Dict, List

from google import genai
from google.genai import errors, types

from services.llm_errors import LLMClientError

logger = logging.getLogger(__name__)

_RETRYABLE_CODES = {429, 503}
_MAX_ATTEMPTS = 3
_BACKOFF_BASE_SECONDS = 1.0
_MAX_DELAY_SECONDS = 30.0
_DELAY_RE = re.compile(r"^\s*([\d.]+)\s*s\s*$")


class GeminiClientError(LLMClientError):
    """Raised when the Gemini API call fails or returns no answer."""


def _retry_delay(exc: errors.APIError) -> float | None:
    """Extract the API-provided retry delay in seconds, if any."""
    details = exc.details
    if isinstance(details, dict):
        error = details.get("error", details)
        entries = error.get("details") if isinstance(error, dict) else None
        for entry in entries or []:
            if isinstance(entry, dict) and entry.get("retryDelay"):
                match = _DELAY_RE.match(str(entry["retryDelay"]))
                if match:
                    return float(match.group(1))
    headers = getattr(exc.response, "headers", None)
    if headers is not None:
        try:
            value = headers.get("retry-after")
            if value is not None:
                return float(value)
        except (TypeError, ValueError):
            pass
    return None


class GeminiClient:
    """Chat client for the Gemini Developer API.

    Parameters
    ----------
    api_key:
        Gemini API key.
    model:
        Model id, e.g. ``gemini-3.5-flash-lite``.
    timeout:
        Request timeout in seconds.
    temperature:
        Default sampling temperature.
    max_tokens:
        Default maximum output tokens.
    """

    provider_name = "gemini"

    def __init__(
        self,
        api_key: str,
        model: str,
        timeout: float = 60.0,
        temperature: float = 0.1,
        max_tokens: int = 1024,
    ) -> None:
        self.model = model
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._sleep: Callable[[float], Awaitable[None]] = asyncio.sleep
        self._client = genai.Client(
            api_key=api_key,
            http_options=types.HttpOptions(timeout=int(timeout * 1000)),
        )

    async def chat(
        self,
        messages: List[Dict[str, str]],
        *,
        json_schema: Dict[str, Any] | None = None,
        max_tokens: int | None = None,
        think: bool | None = None,  # accepted for interface parity; ignored
        temperature: float | None = None,
    ) -> str:
        """Send chat messages and return the answer text."""
        system = "\n\n".join(m["content"] for m in messages if m["role"] == "system")
        contents = [
            types.Content(
                role="model" if m["role"] == "assistant" else "user",
                parts=[types.Part(text=m["content"])],
            )
            for m in messages
            if m["role"] != "system"
        ]
        config = types.GenerateContentConfig(
            system_instruction=system or None,
            temperature=self._temperature if temperature is None else temperature,
            max_output_tokens=max_tokens or self._max_tokens,
        )
        if json_schema is not None:
            config.response_mime_type = "application/json"
            config.response_json_schema = json_schema

        response = None
        for attempt in range(1, _MAX_ATTEMPTS + 1):
            try:
                response = await self._client.aio.models.generate_content(
                    model=self.model, contents=contents, config=config
                )
                break
            except errors.APIError as exc:
                code = exc.code
                if code in _RETRYABLE_CODES and attempt < _MAX_ATTEMPTS:
                    delay = _retry_delay(exc)
                    if delay is None:
                        delay = _BACKOFF_BASE_SECONDS * 2 ** (attempt - 1)
                    delay = min(delay, _MAX_DELAY_SECONDS)
                    logger.warning(
                        "Gemini returned %s; retrying in %.1fs (attempt %d/%d)",
                        code, delay, attempt, _MAX_ATTEMPTS,
                    )
                    await self._sleep(delay)
                    continue
                if code == 429:
                    raise GeminiClientError(
                        "Gemini quota or rate limit exhausted (free tier limits); "
                        "wait and retry or check AI Studio quota"
                    ) from exc
                raise GeminiClientError(f"Gemini API error {code}") from exc
            except Exception as exc:  # transport errors, timeouts
                raise GeminiClientError(
                    f"Gemini request failed: {type(exc).__name__}"
                ) from exc

        text = getattr(response, "text", None)
        if not text:
            reason = None
            candidates = getattr(response, "candidates", None)
            if candidates:
                reason = getattr(candidates[0], "finish_reason", None)
            raise GeminiClientError(f"Empty Gemini response (finish reason: {reason})")
        return text

    async def health_check(self) -> bool:
        """Return True if the configured model is reachable."""
        try:
            await self._client.aio.models.get(model=self.model)
            return True
        except Exception:
            return False
