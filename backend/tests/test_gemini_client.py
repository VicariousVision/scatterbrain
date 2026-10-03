"""Mock-only tests for GeminiClient (no network)."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from google.genai import errors

from services.gemini_client import GeminiClient, GeminiClientError
from services.llm_errors import LLMClientError


def _client(side_effect=None, text="ok") -> tuple[GeminiClient, AsyncMock]:
    client = GeminiClient(api_key="test-key", model="gemini-3.5-flash-lite")
    client._sleep = AsyncMock()
    mock = AsyncMock(
        side_effect=side_effect,
        return_value=SimpleNamespace(text=text, candidates=[]),
    )
    client._client.aio.models.generate_content = mock
    return client, mock


def _api_error(code: int, details: dict | None = None) -> errors.APIError:
    return errors.APIError(code, details or {"error": {"code": code, "message": "boom"}})


def test_error_is_llm_client_error() -> None:
    assert issubclass(GeminiClientError, LLMClientError)


def test_system_and_roles_mapped() -> None:
    client, mock = _client()
    messages = [
        {"role": "system", "content": "rules"},
        {"role": "user", "content": "q1"},
        {"role": "assistant", "content": "a1"},
        {"role": "user", "content": "q2"},
    ]
    assert asyncio_run(client.chat(messages, think=True)) == "ok"
    kwargs = mock.call_args.kwargs
    assert kwargs["model"] == "gemini-3.5-flash-lite"
    assert kwargs["config"].system_instruction == "rules"
    assert [c.role for c in kwargs["contents"]] == ["user", "model", "user"]


def test_json_schema_sets_mime_type() -> None:
    client, mock = _client()
    asyncio_run(client.chat([{"role": "user", "content": "q"}], json_schema={"type": "object"}))
    assert mock.call_args.kwargs["config"].response_mime_type == "application/json"


def test_api_error_becomes_client_error() -> None:
    client, _ = _client(side_effect=_api_error(400))
    with pytest.raises(GeminiClientError, match="400"):
        asyncio_run(client.chat([{"role": "user", "content": "q"}]))


def test_empty_text_raises() -> None:
    client, _ = _client(text="")
    with pytest.raises(GeminiClientError, match="Empty"):
        asyncio_run(client.chat([{"role": "user", "content": "q"}]))


def test_429_retried_then_succeeds_honoring_retry_delay() -> None:
    details = {"error": {"code": 429, "details": [{"retryDelay": "7s"}]}}
    ok = SimpleNamespace(text="fine", candidates=[])
    client, mock = _client(side_effect=[_api_error(429, details), ok])
    assert asyncio_run(client.chat([{"role": "user", "content": "q"}])) == "fine"
    assert mock.await_count == 2
    client._sleep.assert_awaited_once_with(7.0)


def test_503_uses_exponential_backoff() -> None:
    ok = SimpleNamespace(text="fine", candidates=[])
    client, _ = _client(side_effect=[_api_error(503), _api_error(503), ok])
    assert asyncio_run(client.chat([{"role": "user", "content": "q"}])) == "fine"
    assert [c.args[0] for c in client._sleep.await_args_list] == [1.0, 2.0]


def test_429_exhausted_gives_quota_message() -> None:
    client, mock = _client(side_effect=_api_error(429))
    with pytest.raises(GeminiClientError, match="quota"):
        asyncio_run(client.chat([{"role": "user", "content": "q"}]))
    assert mock.await_count == 3


def test_transport_error_wrapped() -> None:
    client, _ = _client(side_effect=TimeoutError("slow"))
    with pytest.raises(GeminiClientError, match="TimeoutError"):
        asyncio_run(client.chat([{"role": "user", "content": "q"}]))


def test_health_check_false_on_error() -> None:
    client, _ = _client()
    client._client.aio.models.get = AsyncMock(side_effect=_api_error(404))
    assert asyncio_run(client.health_check()) is False
    client._client.aio.models.get = AsyncMock(return_value=object())
    assert asyncio_run(client.health_check()) is True


def asyncio_run(coro):
    import asyncio

    return asyncio.run(coro)
