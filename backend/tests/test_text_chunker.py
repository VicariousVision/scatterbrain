from __future__ import annotations

import pytest

from backend.services.text_chunker import chunk_text
from config import settings


@pytest.fixture
def chunk_settings(monkeypatch: pytest.MonkeyPatch):
    """Override the settings-driven chunk size/overlap for a single test."""

    def _apply(chunk_size: int, chunk_overlap: int) -> None:
        monkeypatch.setattr(settings, "chunk_size", chunk_size)
        monkeypatch.setattr(settings, "chunk_overlap", chunk_overlap)

    return _apply


def test_chunk_text_empty() -> None:
    assert chunk_text("") == []
    assert chunk_text(None) == []  # type: ignore[arg-type]


def test_chunk_text_small(chunk_settings) -> None:
    chunk_settings(20, 0)
    text = "Hello world!"
    assert chunk_text(text) == [text]


def test_chunk_text_large(chunk_settings) -> None:
    chunk_settings(25, 5)
    text = "Hello world! This is a long string that we want to split into smaller chunks."
    chunks = chunk_text(text)

    assert len(chunks) > 1
    for chunk in chunks:
        assert len(chunk) <= 25


def test_chunk_text_respects_spaces(chunk_settings) -> None:
    chunk_settings(12, 3)
    chunks = chunk_text("word1 word2 word3 word4")
    assert len(chunks) > 0
    for chunk in chunks:
        assert len(chunk) > 0
