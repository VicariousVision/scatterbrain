"""Text chunker service for splitting document text into semantic chunks."""

from __future__ import annotations


def chunk_text(
    text: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> list[str]:
    """Split input text into overlapping chunks of a specified maximum size.

    Attempts to split text at space or newline boundaries to avoid breaking words.

    Args:
        text: The raw input document text.
        chunk_size: Target maximum character length of each chunk.
        chunk_overlap: The target character overlap between adjacent chunks.

    Returns:
        A list of chunk strings.
    """
    if not text or chunk_size <= 0:
        return []

    if chunk_overlap >= chunk_size:
        chunk_overlap = chunk_size // 2

    chunks: list[str] = []
    start = 0
    text_len = len(text)

    while start < text_len:
        # Candidate end of window
        end = start + chunk_size
        if end >= text_len:
            chunk = text[start:].strip()
            if chunk:
                chunks.append(chunk)
            break

        # Search backward for a word/line boundary within a reasonable range (15% of chunk_size)
        search_min = max(start, end - int(chunk_size * 0.15))
        boundary = -1
        for i in range(end, search_min - 1, -1):
            if text[i] in ("\n", "\r", " ", "\t"):
                boundary = i
                break

        if boundary != -1:
            end = boundary + 1

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        # Move starting pointer for the next chunk, incorporating overlap
        next_start = end - chunk_overlap
        if next_start <= start:
            # Prevent infinite loops if progress is not made
            start = end
        else:
            start = next_start

    return chunks
