"""Text chunking service with structure-aware recursive splitting.

Provides semantic chunking with configurable size and overlap. Markdown tables
(as emitted by the PDF parser) are chunked specially so their structure
survives: a table is split on row boundaries rather than mid-row, and its
header row is repeated at the top of every chunk it spans, so each retrieved
chunk keeps its column names.
"""

from __future__ import annotations

import re
from typing import List

from config import settings


def chunk_text(text: str) -> List[str]:
    """Split text into semantically meaningful chunks, preserving tables.

    Prose is split on natural boundaries (double newlines, single newlines,
    sentences, words). Markdown tables are detected and chunked on row
    boundaries with their header repeated, so a table larger than
    ``chunk_size`` is never broken mid-row and body chunks never lose their
    column headers.

    Args:
        text: Input text to chunk.

    Returns:
        List of text chunks with configured size and overlap.
    """
    if not text or not text.strip():
        return []

    chunk_size = settings.chunk_size
    chunk_overlap = settings.chunk_overlap

    chunks: List[str] = []
    for segment, is_table in _split_table_segments(text):
        if not segment.strip():
            continue
        if is_table:
            chunks.extend(_chunk_table(segment, chunk_size))
        else:
            separators = ["\n\n", "\n", ". ", " ", ""]
            chunks.extend(
                _recursive_split(segment, separators, chunk_size, chunk_overlap)
            )
    return chunks


# A Markdown table row: starts and ends with a pipe (after stripping).
_TABLE_ROW = re.compile(r"^\s*\|.*\|\s*$")
# A Markdown alignment/separator row: | --- | --- | ...
_TABLE_SEPARATOR = re.compile(r"^\s*\|(?:\s*:?-{3,}:?\s*\|)+\s*$")


def _is_table_block(lines: List[str]) -> bool:
    """A run of >= 2 pipe rows with a separator as the second line is a table."""
    return (
        len(lines) >= 2
        and _TABLE_ROW.match(lines[0]) is not None
        and _TABLE_SEPARATOR.match(lines[1]) is not None
    )


def _split_table_segments(text: str) -> List[tuple[str, bool]]:
    """Partition text into ordered (segment, is_table) blocks.

    Contiguous runs of Markdown table rows (a header row followed by a
    separator row and any number of body rows) become table segments; every
    other run becomes a prose segment. Order is preserved.
    """
    lines = text.split("\n")
    segments: List[tuple[str, bool]] = []
    i = 0
    n = len(lines)
    prose_buffer: List[str] = []

    def _flush_prose() -> None:
        if prose_buffer:
            segments.append(("\n".join(prose_buffer), False))
            prose_buffer.clear()

    while i < n:
        # A table starts where a pipe row is immediately followed by a
        # separator row.
        if (
            _TABLE_ROW.match(lines[i])
            and i + 1 < n
            and _TABLE_SEPARATOR.match(lines[i + 1])
        ):
            _flush_prose()
            table_lines = [lines[i], lines[i + 1]]
            j = i + 2
            while j < n and _TABLE_ROW.match(lines[j]):
                table_lines.append(lines[j])
                j += 1
            segments.append(("\n".join(table_lines), True))
            i = j
        else:
            prose_buffer.append(lines[i])
            i += 1

    _flush_prose()
    return segments


def _chunk_table(table: str, chunk_size: int) -> List[str]:
    """Chunk a Markdown table on row boundaries, repeating the header.

    The header and separator rows are prepended to every chunk so each chunk
    is a self-contained, valid table with column names. Body rows are packed
    up to ``chunk_size``. A single row longer than ``chunk_size`` becomes its
    own chunk rather than being broken apart.
    """
    lines = table.split("\n")
    if len(lines) < 2:
        return [table]

    header = lines[0]
    separator = lines[1]
    body_rows = lines[2:]
    prefix = f"{header}\n{separator}"
    prefix_len = len(prefix) + 1  # +1 for the newline before the first body row

    if not body_rows:
        return [prefix]

    chunks: List[str] = []
    current: List[str] = []
    current_len = prefix_len

    for row in body_rows:
        row_len = len(row) + 1
        if current and current_len + row_len > chunk_size:
            chunks.append(prefix + "\n" + "\n".join(current))
            current = []
            current_len = prefix_len
        current.append(row)
        current_len += row_len

    if current:
        chunks.append(prefix + "\n" + "\n".join(current))
    return chunks


def _recursive_split(
    text: str,
    separators: List[str],
    chunk_size: int,
    chunk_overlap: int,
) -> List[str]:
    """Recursively split text on separators."""
    final_chunks = []
    
    # Get the current separator
    separator = separators[0] if separators else ""
    
    # Split on separator
    if separator:
        splits = text.split(separator)
    else:
        splits = list(text)
    
    # Merge splits into chunks
    current_chunk = []
    current_length = 0
    
    for split in splits:
        split_len = len(split)
        
        # If single split is too long and we have more separators, recurse
        if split_len > chunk_size and len(separators) > 1:
            # Save current chunk if it exists
            if current_chunk:
                final_chunks.append(separator.join(current_chunk))
                current_chunk = []
                current_length = 0
            
            # Recurse on the long split
            sub_chunks = _recursive_split(
                split,
                separators[1:],
                chunk_size,
                chunk_overlap,
            )
            final_chunks.extend(sub_chunks)
        else:
            # Add to current chunk if it fits
            if current_length + split_len + len(separator) <= chunk_size:
                current_chunk.append(split)
                current_length += split_len + len(separator)
            else:
                # Save current chunk and start new one
                if current_chunk:
                    final_chunks.append(separator.join(current_chunk))
                
                # Handle overlap
                if chunk_overlap > 0 and current_chunk:
                    # Keep last few items for overlap
                    overlap_text = separator.join(current_chunk)
                    if len(overlap_text) > chunk_overlap:
                        overlap_start = len(overlap_text) - chunk_overlap
                        overlap_text = overlap_text[overlap_start:]
                        current_chunk = [overlap_text, split]
                        current_length = len(overlap_text) + len(separator) + split_len
                    else:
                        current_chunk = [split]
                        current_length = split_len
                else:
                    current_chunk = [split]
                    current_length = split_len
    
    # Add final chunk
    if current_chunk:
        final_chunks.append(separator.join(current_chunk))
    
    return [chunk for chunk in final_chunks if chunk.strip()]
