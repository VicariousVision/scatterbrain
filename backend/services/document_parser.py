"""Document parser service for extracting raw text from uploaded files.

Supports PDF (via pdfplumber) and plain TXT files.

PDF extraction preserves document structure that plain ``extract_text`` would
flatten: tables are rendered as Markdown tables (pipe-delimited rows), and the
surrounding prose -- including indentation for nested lists and subsection
layout -- is kept in reading order. Downstream chunking and embedding then see
structured text rather than a space-collapsed blob.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import List, Optional, Tuple


class DocumentParsingError(Exception):
    """Raised when a document cannot be parsed due to corruption or an unreadable format."""


def parse_document(filename: str, content: bytes) -> str:
    """Extract text from PDF or TXT files.

    Args:
        filename: Original filename including extension.
        content:  Raw file bytes.

    Returns:
        Extracted text as a UTF-8 string.

    Raises:
        ValueError: If the file extension is not supported.
        DocumentParsingError: If the file is corrupted or unreadable.
    """
    ext = Path(filename).suffix.lower()
    if ext == ".pdf":
        return _parse_pdf(content)
    elif ext == ".txt":
        return _parse_txt(content)
    else:
        raise ValueError(f"Unsupported file type: {ext!r}. Only PDF and TXT are supported.")


def _parse_pdf(content: bytes) -> str:
    """Extract text from a PDF file page by page, joined with newlines.

    Args:
        content: Raw PDF bytes.

    Returns:
        Extracted text with pages separated by newline characters.

    Raises:
        DocumentParsingError: If the PDF cannot be opened or read.
    """
    try:
        import pdfplumber  # type: ignore
    except ImportError as exc:
        raise DocumentParsingError(
            "pdfplumber is not installed. Install it with: pip install pdfplumber"
        ) from exc

    try:
        pages: list[str] = []
        with pdfplumber.open(io.BytesIO(content)) as pdf:
            for page in pdf.pages:
                pages.append(_extract_page(page))
        # Separate pages with a blank line so page boundaries read as
        # paragraph breaks rather than concatenated lines.
        return "\n\n".join(p for p in pages if p.strip())
    except DocumentParsingError:
        raise
    except Exception as exc:
        raise DocumentParsingError(
            f"Failed to parse PDF: {exc}"
        ) from exc


def _extract_page(page) -> str:
    """Extract one page as text, rendering tables as Markdown in reading order.

    Tables are located via ``find_tables`` and rendered as Markdown. The text
    that falls inside each table's bounding box is removed from the prose
    extraction so table content is not duplicated. Tables and prose blocks are
    then ordered by their vertical (top) position on the page.

    Args:
        page: A ``pdfplumber`` page object.

    Returns:
        The page rendered as text, with any tables as Markdown tables.
    """
    try:
        tables = page.find_tables()
    except Exception:
        # If table detection fails for any reason, fall back to plain text so
        # the page is never lost.
        return page.extract_text() or ""

    if not tables:
        return page.extract_text() or ""

    # Blocks are (top_y, text) pairs sorted into reading order at the end.
    blocks: List[Tuple[float, str]] = []
    table_bboxes: List[Tuple[float, float, float, float]] = []

    for table in tables:
        markdown = _table_to_markdown(table.extract())
        if markdown:
            blocks.append((table.bbox[1], markdown))
            table_bboxes.append(table.bbox)

    # Extract prose from the regions outside every table bounding box so table
    # cell text is not repeated in the surrounding paragraphs.
    def _outside_tables(obj) -> bool:
        cx = (obj["x0"] + obj["x1"]) / 2
        cy = (obj["top"] + obj["bottom"]) / 2
        for x0, top, x1, bottom in table_bboxes:
            if x0 <= cx <= x1 and top <= cy <= bottom:
                return False
        return True

    non_table = page.filter(_outside_tables)
    prose = non_table.extract_text() or ""
    if prose.strip():
        # Anchor prose above the first table so, absent finer positioning, it
        # precedes the tables it introduces.
        top_anchor = min((b[1] for b in table_bboxes), default=0.0) - 1.0
        blocks.append((top_anchor, prose))

    blocks.sort(key=lambda b: b[0])
    return "\n\n".join(text for _, text in blocks)


def _table_to_markdown(rows: Optional[List[List[Optional[str]]]]) -> str:
    """Render an extracted table (list of rows of cells) as a Markdown table.

    The first row is treated as the header. ``None`` cells become empty
    strings, and newlines within a cell are flattened to spaces so each row
    stays on a single Markdown line. Returns an empty string for a table with
    no usable cells.

    Args:
        rows: Rows of cell strings as returned by ``Table.extract``.

    Returns:
        A Markdown table string, or ``""`` if there is nothing to render.
    """
    if not rows:
        return ""

    def _cell(value: Optional[str]) -> str:
        text = (value or "").replace("\n", " ").strip()
        # Escape pipes so cell content doesn't break the Markdown columns.
        return text.replace("|", "\\|")

    cleaned = [[_cell(c) for c in row] for row in rows]
    cleaned = [row for row in cleaned if any(cell for cell in row)]
    if not cleaned:
        return ""

    width = max(len(row) for row in cleaned)
    cleaned = [row + [""] * (width - len(row)) for row in cleaned]

    header = cleaned[0]
    body = cleaned[1:]

    lines = ["| " + " | ".join(header) + " |"]
    lines.append("| " + " | ".join(["---"] * width) + " |")
    for row in body:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def _parse_txt(content: bytes) -> str:
    """Decode plain-text file bytes as UTF-8.

    Args:
        content: Raw text file bytes.

    Returns:
        Decoded UTF-8 string.

    Raises:
        DocumentParsingError: If the bytes cannot be decoded as UTF-8.
    """
    try:
        return content.decode("utf-8")
    except (UnicodeDecodeError, ValueError) as exc:
        raise DocumentParsingError(
            f"Failed to decode TXT file as UTF-8: {exc}"
        ) from exc
