"""Ollama HTTP client for interacting with the locally-hosted LLM via the Ollama REST API.

Provides async methods for text generation, embeddings, and connectivity health checks.

Reliability notes (CPU-only, single Ollama process):
  - A module-level asyncio.Semaphore(1) serialises ALL outbound calls
    (embeddings and generation) so they never overlap.  Ollama on CPU cannot
    serve two model calls concurrently; the second request queues long enough
    to blow past the connect timeout or causes Ollama to restart.
  - Timeouts are split into components (connect/read/write/pool) because the
    failures seen in the traceback are ConnectTimeout, not ReadTimeout.  A
    flat timeout only on the read phase does not protect against a saturated
    Ollama process that won't accept new connections.
  - Connection errors (ConnectTimeout, ConnectError) are retried with
    exponential back-off (up to 3 attempts).  Application-level errors
    (bad status codes, malformed JSON) are NOT retried — they indicate a
    problem with the request itself, not transient server saturation.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time

import httpx

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Shared concurrency guard
# ---------------------------------------------------------------------------
# Semaphore(1): only one Ollama call (embedding OR generation) in-flight at a
# time.  Raise to 2 only if you have confirmed Ollama can handle 2 concurrent
# requests on your hardware without degrading (OLLAMA_NUM_PARALLEL >= 2 and
# sufficient RAM — ~8 GB free for two simultaneous mistral 7B inferences).
_OLLAMA_SEMAPHORE: asyncio.Semaphore | None = None


def _get_semaphore() -> asyncio.Semaphore:
    """Return the module-level semaphore, creating it lazily inside the running loop.

    asyncio.Semaphore must be created inside the event loop.  Creating it at
    module import time works in Python 3.10+ but raises a DeprecationWarning
    in 3.9 and fails if no loop is running.  Lazy init is the safe approach.
    """
    global _OLLAMA_SEMAPHORE
    if _OLLAMA_SEMAPHORE is None:
        _OLLAMA_SEMAPHORE = asyncio.Semaphore(1)
    return _OLLAMA_SEMAPHORE


# ---------------------------------------------------------------------------
# Retry configuration
# ---------------------------------------------------------------------------
_MAX_CONNECT_RETRIES = 3       # total attempts (1 initial + 2 retries)
_RETRY_BASE_DELAY = 2.0        # seconds; doubles on each retry (2 s, 4 s)

# Retryable exception types — transient connectivity failures only.
_RETRYABLE = (httpx.ConnectTimeout, httpx.ConnectError)

# ---------------------------------------------------------------------------
# Timeout configuration
# ---------------------------------------------------------------------------
# Split timeouts let us distinguish "Ollama won't accept the connection" from
# "Ollama accepted the connection but generation is slow".  The connect phase
# is where failures have been observed on CPU hardware.
_CALL_TIMEOUT = httpx.Timeout(
    connect=30.0,   # time to establish TCP connection
    read=120.0,     # time waiting for the response body (generation can be slow)
    write=30.0,     # time to send the request body
    pool=30.0,      # time waiting for a connection from the httpx pool
)
_HEALTH_TIMEOUT = httpx.Timeout(connect=5.0, read=5.0, write=5.0, pool=5.0)

# ---------------------------------------------------------------------------
# Windows log location hint
# ---------------------------------------------------------------------------
_WINDOWS_OLLAMA_LOG = os.path.join(
    os.environ.get("LOCALAPPDATA", r"C:\Users\<user>\AppData\Local"),
    "Ollama",
    "server.log",
)


def _connection_failure_hint(base_url: str, exc: Exception) -> str:
    """Return a detailed error string pointing to the Ollama server log."""
    return (
        f"Failed to connect to Ollama at {base_url}: {exc}. "
        f"Check whether Ollama is still running ('ollama serve' in a terminal). "
        f"On Windows the server log is usually at: {_WINDOWS_OLLAMA_LOG}"
    )


# ---------------------------------------------------------------------------
# Public exception
# ---------------------------------------------------------------------------


class OllamaClientError(Exception):
    """Raised when the Ollama client encounters an error communicating with the Ollama API."""


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------


class OllamaClient:
    """Thin async HTTP wrapper around the Ollama REST API.

    Args:
        base_url:        Base URL of the Ollama server (e.g. ``http://localhost:11434``).
        model:           Name of the model to use for generation (e.g. ``mistral``).
        embedding_model: Model name for embeddings; defaults to ``model`` if omitted.
        num_gpu:         GPU layers to offload.  ``0`` = CPU-only (default).
    """

    def __init__(
        self,
        base_url: str,
        model: str,
        embedding_model: str | None = None,
        num_gpu: int = 0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.embedding_model = embedding_model or model
        self.num_gpu = num_gpu

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _post_with_retry(
        self,
        path: str,
        payload: dict,
        *,
        timeout: httpx.Timeout = _CALL_TIMEOUT,
    ) -> httpx.Response:
        """POST ``payload`` to ``{base_url}{path}``, serialising through the
        module-level semaphore and retrying on transient connection failures.

        Only :data:`_RETRYABLE` exception types trigger a retry.  HTTP-level
        errors (4xx/5xx) and JSON-parse failures are NOT retried.

        Raises:
            OllamaClientError: After exhausting retries, or on a
                non-retryable request error.
        """
        semaphore = _get_semaphore()
        url = f"{self.base_url}{path}"
        last_exc: Exception | None = None

        for attempt in range(1, _MAX_CONNECT_RETRIES + 1):
            try:
                async with semaphore:
                    async with httpx.AsyncClient(timeout=timeout) as client:
                        response = await client.post(url, json=payload)
                return response          # success — exit retry loop
            except _RETRYABLE as exc:
                last_exc = exc
                if attempt < _MAX_CONNECT_RETRIES:
                    delay = _RETRY_BASE_DELAY * (2 ** (attempt - 1))
                    logger.warning(
                        "Ollama connect failure (attempt %d/%d) to %s: %s. "
                        "Retrying in %.1f s. %s",
                        attempt,
                        _MAX_CONNECT_RETRIES,
                        url,
                        exc,
                        delay,
                        _connection_failure_hint(self.base_url, exc),
                    )
                    await asyncio.sleep(delay)
                else:
                    logger.error(
                        "Ollama connect failure (attempt %d/%d) to %s: %s. "
                        "No more retries. %s",
                        attempt,
                        _MAX_CONNECT_RETRIES,
                        url,
                        exc,
                        _connection_failure_hint(self.base_url, exc),
                    )
            except httpx.RequestError as exc:
                # Non-retryable network error (e.g. invalid URL, DNS failure).
                raise OllamaClientError(
                    _connection_failure_hint(self.base_url, exc)
                ) from exc

        raise OllamaClientError(
            _connection_failure_hint(self.base_url, last_exc)
        ) from last_exc

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def generate_embedding(self, text: str) -> list[float]:
        """Send ``text`` to ``/api/embeddings`` and return the vector.

        Raises:
            OllamaClientError: On request failure or unexpected response format.
        """
        payload = {"model": self.embedding_model, "prompt": text}
        try:
            response = await self._post_with_retry("/api/embeddings", payload)
            response.raise_for_status()
            return response.json()["embedding"]
        except httpx.HTTPStatusError as exc:
            raise OllamaClientError(
                f"Ollama API returned status {exc.response.status_code}: {exc.response.text}"
            ) from exc
        except (KeyError, ValueError) as exc:
            raise OllamaClientError(
                f"Unexpected response format from Ollama /api/embeddings: {exc}"
            ) from exc

    async def generate(self, prompt: str, *, json_mode: bool = False, max_tokens: int | None = None) -> str:
        """Send ``prompt`` to ``/api/generate`` and return the response text.

        Args:
            prompt:     Full prompt string.
            json_mode:  If ``True``, sets ``format: "json"`` on the Ollama
                        request to constrain decoding to valid JSON output.
                        Use for extraction calls where the output schema is fixed.
            max_tokens: If set, passed as ``num_predict`` in the options dict
                        to cap output length.  Useful for extraction calls
                        where the output is always small (~150 tokens).

        Raises:
            OllamaClientError: On request failure or unexpected response format.
        """
        options: dict = {"num_gpu": self.num_gpu}
        if max_tokens is not None:
            options["num_predict"] = max_tokens

        payload: dict = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": options,
        }
        if json_mode:
            payload["format"] = "json"

        try:
            response = await self._post_with_retry("/api/generate", payload)
            response.raise_for_status()
            return response.json()["response"]
        except httpx.HTTPStatusError as exc:
            raise OllamaClientError(
                f"Ollama API returned status {exc.response.status_code}: {exc.response.text}"
            ) from exc
        except (KeyError, ValueError) as exc:
            raise OllamaClientError(
                f"Unexpected response format from Ollama /api/generate: {exc}"
            ) from exc

    async def health_check(self) -> bool:
        """Return ``True`` if Ollama responds with HTTP 200 on ``/api/tags``."""
        semaphore = _get_semaphore()
        try:
            async with semaphore:
                async with httpx.AsyncClient(timeout=_HEALTH_TIMEOUT) as client:
                    response = await client.get(f"{self.base_url}/api/tags")
                    return response.status_code == 200
        except Exception:
            return False
