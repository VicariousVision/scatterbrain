from __future__ import annotations

from backend.services.text_chunker import chunk_text


def test_chunk_text_empty() -> None:
    assert chunk_text("") == []
    assert chunk_text(None) == []  # type: ignore[arg-type]


def test_chunk_text_small() -> None:
    text = "Hello world!"
    assert chunk_text(text, chunk_size=20) == [text]


def test_chunk_text_large() -> None:
    text = "Hello world! This is a long string that we want to split into smaller chunks."
    chunks = chunk_text(text, chunk_size=25, chunk_overlap=5)
    
    assert len(chunks) > 1
    for chunk in chunks:
        assert len(chunk) <= 25


def test_chunk_text_respects_spaces() -> None:
    text = "word1 word2 word3 word4"
    chunks = chunk_text(text, chunk_size=12, chunk_overlap=3)
    assert len(chunks) > 0
    # Chunks should be non-empty strings
    for chunk in chunks:
        assert len(chunk) > 0
