"""Generic recursive chunking retained as the non-legal/final fallback.

Legal manuals are routed through :mod:`services.legal_chunker`. Ordinary TXT
and unrelated PDFs still use LangChain's ``RecursiveCharacterTextSplitter``.
Markdown tables keep complete rows and repeat their real header in each group.
"""

from __future__ import annotations

import re
from typing import List, Sequence

from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import settings

_TABLE_ROW = re.compile(r"^\s*\|.*\|\s*$")
_TABLE_SEPARATOR = re.compile(r"^\s*\|(?:\s*:?-{3,}:?\s*\|)+\s*$")
_DEFAULT_SEPARATORS = ["\n\n", "\n", r"(?<=\.)\s+", " ", ""]


def chunk_text(text: str) -> List[str]:
    """Chunk generic text using the configured recursive fallback.

    ``CHUNK_SIZE`` and ``CHUNK_OVERLAP`` apply only here. Independent legal
    clauses never inherit this generic overlap.
    """
    if not text or not text.strip():
        return []
    chunks: list[str] = []
    for segment, is_table in _split_table_segments(text):
        if not segment.strip():
            continue
        if is_table:
            chunks.extend(_chunk_table(segment, settings.chunk_size))
        else:
            chunks.extend(
                _langchain_split(
                    segment,
                    chunk_size=settings.chunk_size,
                    chunk_overlap=settings.chunk_overlap,
                )
            )
    return [chunk for chunk in chunks if chunk.strip()]


def split_continuous_text(
    text: str,
    *,
    chunk_size: int,
    overlap: int,
) -> list[str]:
    """Force-split one continuous provision with an explicit continuity tail.

    This helper is intentionally not used between independent clauses. Legal
    parsing first tries sub-item, complete-row, and sentence boundaries, then
    invokes this final recursive fallback for a single overlong unit.
    """
    if not text.strip():
        return []
    return _langchain_split(text, chunk_size=chunk_size, chunk_overlap=overlap)


def _langchain_split(text: str, *, chunk_size: int, chunk_overlap: int) -> list[str]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=_DEFAULT_SEPARATORS,
        is_separator_regex=True,
        keep_separator=True,
        strip_whitespace=True,
    )
    return splitter.split_text(text)


def _is_table_block(lines: List[str]) -> bool:
    """Return whether lines begin with a Markdown header/separator pair."""
    return (
        len(lines) >= 2
        and _TABLE_ROW.match(lines[0]) is not None
        and _TABLE_SEPARATOR.match(lines[1]) is not None
    )


def _split_table_segments(text: str) -> List[tuple[str, bool]]:
    """Partition text into ordered prose and Markdown-table segments."""
    lines = text.split("\n")
    segments: list[tuple[str, bool]] = []
    prose: list[str] = []
    index = 0

    def flush_prose() -> None:
        if prose:
            segments.append(("\n".join(prose), False))
            prose.clear()

    while index < len(lines):
        if (
            _TABLE_ROW.match(lines[index])
            and index + 1 < len(lines)
            and _TABLE_SEPARATOR.match(lines[index + 1])
        ):
            flush_prose()
            end = index + 2
            while end < len(lines) and _TABLE_ROW.match(lines[end]):
                end += 1
            segments.append(("\n".join(lines[index:end]), True))
            index = end
        else:
            prose.append(lines[index])
            index += 1
    flush_prose()
    return segments


def _chunk_table(table: str, chunk_size: int) -> List[str]:
    """Pack complete rows, repeating the extracted header in every group."""
    lines = table.split("\n")
    if len(lines) < 2:
        return [table]
    prefix = f"{lines[0]}\n{lines[1]}"
    rows = lines[2:]
    if not rows:
        return [prefix]

    chunks: list[str] = []
    current: list[str] = []
    for row in rows:
        candidate = prefix + "\n" + "\n".join([*current, row])
        if current and len(candidate) > chunk_size:
            chunks.append(prefix + "\n" + "\n".join(current))
            current = [row]
        else:
            current.append(row)
    if current:
        chunks.append(prefix + "\n" + "\n".join(current))
    return chunks


def _recursive_split(
    text: str,
    separators: Sequence[str],
    chunk_size: int,
    chunk_overlap: int,
) -> List[str]:
    """Compatibility wrapper over LangChain's recursive splitter."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=list(separators),
        keep_separator=True,
        strip_whitespace=True,
    )
    return splitter.split_text(text)
