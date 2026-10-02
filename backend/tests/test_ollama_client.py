"""Unit tests for OllamaClient chat/generate payloads (no Ollama required)."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock

import httpx
import pytest

from services.ollama_client import OllamaClient, OllamaClientError


def _client() -> OllamaClient:
    return OllamaClient(base_url="http://test", model="m", num_gpu=0, num_ctx=4096)


def _response(path: str, status: int = 200, **kwargs) -> httpx.Response:
    return httpx.Response(status, request=httpx.Request("POST", f"http://test{path}"), **kwargs)


def test_chat_posts_messages_and_returns_content() -> None:
    client = _client()
    client._post_with_retry = AsyncMock(
        return_value=_response("/api/chat", json={"message": {"role": "assistant", "content": "hi"}})
    )
    messages = [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}]

    result = asyncio.run(client.chat(messages, think=False, max_tokens=64, temperature=0.1))

    assert result == "hi"
    path, payload = client._post_with_retry.await_args.args
    assert path == "/api/chat"
    assert payload == {
        "model": "m",
        "messages": messages,
        "stream": False,
        "options": {"num_gpu": 0, "num_ctx": 4096, "num_predict": 64, "temperature": 0.1},
        "think": False,
    }


def test_chat_omits_unset_optional_fields() -> None:
    client = _client()
    client._post_with_retry = AsyncMock(
        return_value=_response("/api/chat", json={"message": {"content": "ok"}})
    )

    asyncio.run(client.chat([{"role": "user", "content": "u"}]))

    payload = client._post_with_retry.await_args.args[1]
    assert "think" not in payload
    assert "format" not in payload
    assert payload["options"] == {"num_gpu": 0, "num_ctx": 4096}


def test_chat_raises_on_bad_body_and_http_error() -> None:
    client = _client()
    client._post_with_retry = AsyncMock(return_value=_response("/api/chat", json={"done": True}))
    with pytest.raises(OllamaClientError):
        asyncio.run(client.chat([{"role": "user", "content": "u"}]))

    client._post_with_retry = AsyncMock(return_value=_response("/api/chat", json={"message": None}))
    with pytest.raises(OllamaClientError):
        asyncio.run(client.chat([{"role": "user", "content": "u"}]))

    client._post_with_retry = AsyncMock(return_value=_response("/api/chat", 500, text="boom"))
    with pytest.raises(OllamaClientError):
        asyncio.run(client.chat([{"role": "user", "content": "u"}]))


def test_generate_sends_num_ctx_and_returns_response() -> None:
    client = _client()
    client._post_with_retry = AsyncMock(
        return_value=_response("/api/generate", json={"response": "text"})
    )

    assert asyncio.run(client.generate("p")) == "text"
    path, payload = client._post_with_retry.await_args.args
    assert path == "/api/generate"
    assert payload["options"]["num_ctx"] == 4096


def test_non_positive_num_ctx_rejected() -> None:
    with pytest.raises(ValueError):
        OllamaClient(base_url="http://test", model="m", num_ctx=0)
