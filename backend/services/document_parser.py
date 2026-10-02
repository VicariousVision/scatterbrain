"""Page-aware PDF/TXT extraction for structured document ingestion.

PDF pages remain ordered objects containing coordinate-bearing prose and table
blocks. Running furniture is removed only after cross-page/coordinate analysis,
while printed page, revision, and section markers are retained as metadata.
The historical :func:`parse_document` text API remains a flattening wrapper.
"""

from __future__ import annotations

import hashlib
import io
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

from models.content import (
    PARSER_VERSION,
    ExtractedBlock,
    ExtractedPage,
    ParsedDocument,
    SourceSpan,
)

_MANUAL_TITLE = "Currency and Exchanges Manual for Authorised Dealers"
_SECTION_RE = re.compile(r"\b([A-K](?:\.\d+(?:\.\d+)*)?\.?)\b")
_PRINTED_PAGE_RE = re.compile(r"\b([A-Za-z]*\d+|[ivxlcdm]+)\s+of\s+\d+\b", re.I)
_REVISION_RE = re.compile(r"\b\d{1,3}/(?:19|20)\d{2}\b")
_DATE_RE = re.compile(r"\b(?:19|20)\d{2}-\d{2}-\d{2}\b")
_TOC_ENTRY_RE = re.compile(
    r"^\s*([A-K](?:\.\d+(?:\.\d+)*)?\.?)\s+(.+?)\.{2,}\s*(\d+)\s*$"
)
_NAV_MARKER_RE = re.compile(r"^\s*(\([A-Z]\)|\([ivxlcdm]+\))\s+(.+\S)\s*$")


class DocumentParsingError(Exception):
    """Raised when a document is corrupt or cannot be decoded."""


def parse_document(filename: str, content: bytes) -> str:
    """Extract a backward-compatible flattened text representation."""
    return parse_document_structured(filename, content).flatten()


def parse_document_structured(
    filename: str,
    content: bytes,
    pdf_pages: Sequence[Any] | None = None,
) -> ParsedDocument:
    """Extract a page-aware document.

    Parameters
    ----------
    filename:
        Original filename including extension.
    content:
        Exact uploaded bytes; used for parsing and stable SHA-256 identity.
    pdf_pages:
        Optional one-based PDF page numbers to extract. Already-open page
        objects or compact page dictionaries are also accepted by tests; each
        must expose a one-based ``page_number``/``pdf_page`` when it is not in
        natural sequence. Page numbers are never zero-based.

    Returns
    -------
    ParsedDocument
        Ordered pages/blocks plus detected document provenance.
    """
    ext = Path(filename).suffix.lower()
    source_sha256 = hashlib.sha256(content).hexdigest()
    if ext == ".txt":
        raw = _parse_txt(content)
        block = ExtractedBlock(
            block_type="prose",
            raw_text=raw,
            order=0,
            pdf_page=1,
            extraction_method="utf-8",
            source_spans=[SourceSpan(pdf_page=1, block_order=0)],
        )
        parsed = ParsedDocument(
            source_filename=filename,
            source_sha256=source_sha256,
            extraction_method="utf-8",
            pages=[ExtractedPage(pdf_page=1, blocks=[block])],
        )
        parsed.is_legal = _looks_like_legal_manual(parsed)
        return parsed
    if ext != ".pdf":
        raise ValueError(
            f"Unsupported file type: {ext!r}. Only PDF and TXT are supported."
        )

    try:
        selected_numbers = (
            [int(value) for value in pdf_pages]
            if pdf_pages is not None and all(isinstance(value, int) for value in pdf_pages)
            else None
        )
        if selected_numbers is not None:
            try:
                import pdfplumber  # type: ignore
            except ImportError as exc:
                raise DocumentParsingError(
                    "pdfplumber is not installed. Install it with: pip install pdfplumber"
                ) from exc
            with pdfplumber.open(io.BytesIO(content)) as pdf:
                if any(number < 1 or number > len(pdf.pages) for number in selected_numbers):
                    raise ValueError(
                        f"pdf_pages must contain one-based values from 1 to {len(pdf.pages)}"
                    )
                pages = [
                    _extract_page_structured(pdf.pages[number - 1], number)
                    for number in selected_numbers
                ]
        elif pdf_pages is not None:
            pages = []
            for fallback_number, page in enumerate(pdf_pages, start=1):
                if isinstance(page, dict):
                    number = int(page.get("pdf_page", fallback_number))
                else:
                    number = int(getattr(page, "page_number", fallback_number) or fallback_number)
                pages.append(_extract_page_structured(page, number))
        else:
            try:
                import pdfplumber  # type: ignore
            except ImportError as exc:
                raise DocumentParsingError(
                    "pdfplumber is not installed. Install it with: pip install pdfplumber"
                ) from exc
            with pdfplumber.open(io.BytesIO(content)) as pdf:
                pages = [
                    _extract_page_structured(page, number)
                    for number, page in enumerate(pdf.pages, start=1)
                ]
    except DocumentParsingError:
        raise
    except Exception as exc:
        raise DocumentParsingError(f"Failed to parse PDF: {exc}") from exc

    title, version = _document_provenance(pages)
    _classify_furniture_and_pages(pages)
    navigation = _navigation_entries(pages)
    parsed = ParsedDocument(
        source_filename=filename,
        source_sha256=source_sha256,
        document_title=title,
        document_version=version,
        extraction_method="pdfplumber-layout",
        parser_version=PARSER_VERSION,
        pages=pages,
        navigation=navigation,
        metadata={
            "pdf_page_count": len(pages),
            "front_matter_pages": [
                page.pdf_page for page in pages if page.content_type == "front_matter"
            ],
            "navigation_pages": [
                page.pdf_page for page in pages if page.content_type == "navigation"
            ],
        },
    )
    parsed.is_legal = _looks_like_legal_manual(parsed)
    return parsed


def _extract_page_structured(page: Any, pdf_page: int) -> ExtractedPage:
    """Extract one page into positionable blocks in visual reading order."""
    if isinstance(page, dict):
        return _extract_fixture_page(page, pdf_page)

    width = float(getattr(page, "width", 0.0) or 0.0)
    height = float(getattr(page, "height", 0.0) or 0.0)
    table_objects: list[Any]
    try:
        table_objects = list(page.find_tables())
    except Exception:
        table_objects = []

    table_bboxes: list[tuple[float, float, float, float]] = []
    pending: list[tuple[float, float, ExtractedBlock]] = []
    for table_index, table in enumerate(table_objects):
        try:
            rows = _clean_table_rows(table.extract())
            bbox = tuple(float(value) for value in table.bbox)
        except Exception:
            continue
        markdown = _table_to_markdown(rows)
        if not markdown:
            continue
        table_bboxes.append(bbox)  # type: ignore[arg-type]
        header = rows[0] if rows else []
        block = ExtractedBlock(
            block_type="table",
            raw_text=markdown,
            bbox=bbox,  # type: ignore[arg-type]
            order=0,
            pdf_page=pdf_page,
            table_id=f"p{pdf_page}-t{table_index + 1}",
            table_header=header,
            table_rows=rows[1:] if len(rows) > 1 else [],
            extraction_method="pdfplumber-table",
        )
        pending.append((bbox[1], bbox[0], block))

    words: list[dict[str, Any]] = []
    try:
        words = list(
            page.extract_words(
                x_tolerance=2,
                y_tolerance=3,
                keep_blank_chars=False,
                use_text_flow=False,
            )
            or []
        )
    except Exception:
        words = []

    if words:
        prose_words = [word for word in words if not _word_in_boxes(word, table_bboxes)]
        for raw_text, bbox in _words_to_prose_blocks(prose_words):
            block = ExtractedBlock(
                block_type="prose",
                raw_text=raw_text,
                bbox=bbox,
                order=0,
                pdf_page=pdf_page,
                extraction_method="pdfplumber-words",
            )
            pending.append((bbox[1], bbox[0], block))
    elif pending:
        # Table geometry succeeded but word extraction did not. Keep all
        # non-table prose via pdfplumber's filter as a conservative fallback;
        # coordinates are coarse in this exceptional path, but source is not
        # silently lost.
        try:
            def outside_tables(obj: dict[str, Any]) -> bool:
                return not _word_in_boxes(obj, table_bboxes)

            non_table = page.filter(outside_tables)
            raw = non_table.extract_text(layout=True) or non_table.extract_text() or ""
        except Exception:
            raw = ""
        if raw.strip():
            bbox = (0.0, 0.0, width, height)
            pending.append(
                (
                    0.0,
                    0.0,
                    ExtractedBlock(
                        block_type="prose",
                        raw_text=raw,
                        bbox=bbox,
                        order=0,
                        pdf_page=pdf_page,
                        extraction_method="pdfplumber-filter-fallback",
                    ),
                )
            )
    elif not pending:
        raw = page.extract_text(layout=True) or page.extract_text() or ""
        if raw.strip():
            bbox = (0.0, 0.0, width, height)
            pending.append(
                (
                    0.0,
                    0.0,
                    ExtractedBlock(
                        block_type="prose",
                        raw_text=raw,
                        bbox=bbox,
                        order=0,
                        pdf_page=pdf_page,
                        extraction_method="pdfplumber-text-fallback",
                    ),
                )
            )

    pending.sort(key=lambda item: (round(item[0], 2), round(item[1], 2)))
    blocks: list[ExtractedBlock] = []
    for order, (_, _, block) in enumerate(pending):
        block.order = order
        block.source_spans = [
            SourceSpan(
                pdf_page=pdf_page,
                bbox=block.bbox,
                block_order=order,
                char_start=0,
                char_end=len(block.raw_text),
            )
        ]
        blocks.append(block)
    return ExtractedPage(
        pdf_page=pdf_page,
        width=width or None,
        height=height or None,
        blocks=blocks,
    )


def _extract_fixture_page(data: dict[str, Any], pdf_page: int) -> ExtractedPage:
    """Build an extracted page from a compact deterministic test fixture."""
    blocks: list[ExtractedBlock] = []
    if data.get("blocks"):
        for order, raw_block in enumerate(data["blocks"]):
            block_data = dict(raw_block)
            block_data.setdefault("raw_text", block_data.pop("text", ""))
            block_data.setdefault("pdf_page", pdf_page)
            block_data.setdefault("order", order)
            block_data.setdefault("extraction_method", "fixture")
            block = ExtractedBlock.model_validate(block_data)
            if not block.source_spans:
                block.source_spans = [
                    SourceSpan(
                        pdf_page=pdf_page,
                        bbox=block.bbox,
                        block_order=order,
                        char_start=0,
                        char_end=len(block.raw_text),
                    )
                ]
            blocks.append(block)
    elif data.get("words") or data.get("tables"):
        for raw_text, bbox in _words_to_prose_blocks(data.get("words", [])):
            blocks.append(
                ExtractedBlock(
                    raw_text=raw_text,
                    bbox=bbox,
                    pdf_page=pdf_page,
                    extraction_method="fixture-words",
                )
            )
        for index, table in enumerate(data.get("tables", []), start=1):
            rows = _clean_table_rows(table.get("rows", []))
            bbox = tuple(table.get("bbox", (0.0, 0.0, 0.0, 0.0)))
            blocks.append(
                ExtractedBlock(
                    block_type="table",
                    raw_text=_table_to_markdown(rows),
                    bbox=bbox,  # type: ignore[arg-type]
                    pdf_page=pdf_page,
                    table_id=f"p{pdf_page}-t{index}",
                    table_header=rows[0] if rows else [],
                    table_rows=rows[1:] if len(rows) > 1 else [],
                    extraction_method="fixture-table",
                )
            )
        blocks.sort(key=lambda block: ((block.bbox or (0, 0, 0, 0))[1], (block.bbox or (0, 0, 0, 0))[0]))
    else:
        text = str(data.get("text", ""))
        if text:
            blocks.append(
                ExtractedBlock(
                    raw_text=text,
                    bbox=(0.0, 0.0, float(data.get("width", 0)), float(data.get("height", 0))),
                    pdf_page=pdf_page,
                    extraction_method="fixture-text",
                )
            )

    # Test dictionaries may intentionally be listed out of order to verify
    # that geometry, rather than input enumeration, controls reading order.
    blocks.sort(
        key=lambda block: (
            (block.bbox or (0.0, float(block.order), 0.0, 0.0))[1],
            (block.bbox or (float(block.order), 0.0, 0.0, 0.0))[0],
        )
    )
    for order, block in enumerate(blocks):
        block.order = order
        if not block.source_spans:
            block.source_spans = [
                SourceSpan(
                    pdf_page=pdf_page,
                    bbox=block.bbox,
                    block_order=order,
                    char_start=0,
                    char_end=len(block.raw_text),
                )
            ]
    return ExtractedPage(
        pdf_page=pdf_page,
        width=float(data.get("width", 595.0)),
        height=float(data.get("height", 842.0)),
        printed_page=data.get("printed_page"),
        page_revision=data.get("page_revision"),
        section_marker=data.get("section_marker"),
        content_type=data.get("content_type", "body"),
        blocks=blocks,
    )


def _word_in_boxes(
    word: dict[str, Any], boxes: Iterable[tuple[float, float, float, float]]
) -> bool:
    cx = (float(word.get("x0", 0)) + float(word.get("x1", 0))) / 2
    cy = (float(word.get("top", 0)) + float(word.get("bottom", 0))) / 2
    return any(x0 <= cx <= x1 and top <= cy <= bottom for x0, top, x1, bottom in boxes)


def _words_to_prose_blocks(
    words: Sequence[dict[str, Any]],
) -> list[tuple[str, tuple[float, float, float, float]]]:
    """Group positioned words into lines and nearby paragraph blocks."""
    if not words:
        return []
    ordered = sorted(words, key=lambda word: (float(word.get("top", 0)), float(word.get("x0", 0))))
    line_groups: list[list[dict[str, Any]]] = []
    for word in ordered:
        top = float(word.get("top", 0))
        if not line_groups:
            line_groups.append([word])
            continue
        current_top = sum(float(item.get("top", 0)) for item in line_groups[-1]) / len(line_groups[-1])
        if abs(top - current_top) <= 3.0:
            line_groups[-1].append(word)
        else:
            line_groups.append([word])

    rendered: list[tuple[str, tuple[float, float, float, float]]] = []
    for line in line_groups:
        line.sort(key=lambda word: float(word.get("x0", 0)))
        x0 = min(float(word.get("x0", 0)) for word in line)
        x1 = max(float(word.get("x1", word.get("x0", 0))) for word in line)
        top = min(float(word.get("top", 0)) for word in line)
        bottom = max(float(word.get("bottom", top)) for word in line)
        pieces: list[str] = []
        previous_x1: float | None = None
        for word in line:
            text = str(word.get("text", ""))
            wx0 = float(word.get("x0", 0))
            if pieces and previous_x1 is not None:
                gap = max(0.0, wx0 - previous_x1)
                pieces.append(" " * max(1, min(8, int(round(gap / 3.2)))))
            pieces.append(text)
            previous_x1 = float(word.get("x1", wx0))
        # Keep relative indentation as parser evidence without reproducing the
        # huge left page margin. Four PDF points is approximately one space.
        indent = " " * max(0, int(round((x0 - 36.0) / 4.0)))
        rendered.append((indent + "".join(pieces).rstrip(), (x0, top, x1, bottom)))

    blocks: list[tuple[str, tuple[float, float, float, float]]] = []
    current_lines: list[str] = []
    current_boxes: list[tuple[float, float, float, float]] = []
    previous_bottom: float | None = None
    previous_x0: float | None = None
    for text, bbox in rendered:
        gap = bbox[1] - previous_bottom if previous_bottom is not None else 0.0
        # A large vertical gap starts a new positionable prose block. A marked
        # indentation reset after a modest gap also separates headings/tables.
        new_block = bool(
            current_lines
            and (
                gap > 15.0
                or (gap > 8.0 and previous_x0 is not None and abs(bbox[0] - previous_x0) > 20)
            )
        )
        if new_block:
            blocks.append(("\n".join(current_lines), _union_boxes(current_boxes)))
            current_lines = []
            current_boxes = []
        current_lines.append(text)
        current_boxes.append(bbox)
        previous_bottom = bbox[3]
        previous_x0 = bbox[0]
    if current_lines:
        blocks.append(("\n".join(current_lines), _union_boxes(current_boxes)))
    return blocks


def _union_boxes(
    boxes: Sequence[tuple[float, float, float, float]],
) -> tuple[float, float, float, float]:
    return (
        min(box[0] for box in boxes),
        min(box[1] for box in boxes),
        max(box[2] for box in boxes),
        max(box[3] for box in boxes),
    )


def _classify_furniture_and_pages(pages: list[ExtractedPage]) -> None:
    """Remove repeated coordinate-band furniture and retain its metadata."""
    band_candidates: list[tuple[int, int, str]] = []
    for page_index, page in enumerate(pages):
        height = page.height or 842.0
        for block_index, block in enumerate(page.blocks):
            if not block.bbox:
                continue
            top, bottom = block.bbox[1], block.bbox[3]
            if top <= 72 or bottom >= height - 72:
                band_candidates.append((page_index, block_index, _furniture_key(block.raw_text)))
    frequencies = Counter(key for _, _, key in band_candidates if key)
    threshold = max(2, math.ceil(len(pages) * 0.35))

    for page_index, page in enumerate(pages):
        height = page.height or 842.0
        retained: list[ExtractedBlock] = []
        for block in page.blocks:
            text = " ".join(block.raw_text.split())
            lower = text.lower()
            bbox = block.bbox
            top_band = bool(bbox and bbox[1] <= 72)
            bottom_band = bool(bbox and bbox[3] >= height - 72)

            section_matches = _SECTION_RE.findall(text)
            if top_band and section_matches:
                page.section_marker = _canonical_section(section_matches[-1])

            page_match = _PRINTED_PAGE_RE.search(text) if bottom_band else None
            revision_match = _REVISION_RE.search(text) if bottom_band else None
            if page_match:
                page.printed_page = page_match.group(1)
            if revision_match:
                page.page_revision = revision_match.group(0)

            title_furniture = top_band and _MANUAL_TITLE.lower() in lower
            footer_furniture = bottom_band and bool(page_match)
            repeated_furniture = bool(
                (top_band or bottom_band)
                and frequencies.get(_furniture_key(text), 0) >= threshold
            )
            section_only = bool(re.fullmatch(r"\s*[A-K](?:\.\d+)*\.?\s*", text))
            if (title_furniture or footer_furniture or repeated_furniture) and not section_only:
                continue
            retained.append(block)

        page.blocks = retained
        for order, block in enumerate(page.blocks):
            block.order = order
            block.printed_page = page.printed_page
            block.page_revision = page.page_revision
            block.section_marker = page.section_marker
            block.source_spans = [
                SourceSpan(
                    pdf_page=page.pdf_page,
                    printed_page=page.printed_page,
                    page_revision=page.page_revision,
                    bbox=block.bbox,
                    block_order=order,
                    char_start=0,
                    char_end=len(block.raw_text),
                )
            ]

    # Manual-specific front matter signatures, with conservative generic
    # fallback. Version rows can carry a misleading running TOC header, so
    # their table columns decide the classification.
    toc_started = False
    section_index_active = False
    for page in pages:
        page_text = "\n".join(block.raw_text for block in page.blocks)
        lower = page_text.lower()
        starts_section_index = bool(re.search(r"(?mi)^\s*index\s*$", page_text))
        if starts_section_index:
            section_index_active = True
        if section_index_active:
            body_restart = bool(
                not starts_section_index
                and re.search(r"(?m)^\s*\(A\)\s+", page_text)
                and (
                    re.search(r"(?m)^\s*\([a-z]+\)\s+", page_text)
                    or re.search(r"\b(?:must|shall|regulations governing)\b", lower)
                )
            )
            if body_restart:
                section_index_active = False
            else:
                page.content_type = "navigation"
                continue
        version_rows = len(
            re.findall(r"(?m)^.*\b\d+\.\d+\b.*?\b(?:19|20)\d{2}-\d{2}-\d{2}\b", page_text)
        )
        if (
            ("version number" in lower and "issue date" in lower)
            or (page.pdf_page <= 4 and version_rows >= 3)
        ):
            page.content_type = "front_matter"
            continue
        if "version control sheet" in lower and page.pdf_page <= 4:
            page.content_type = "front_matter"
            continue
        if "table of contents" in lower:
            toc_started = True
        dotted_entries = len(re.findall(r"\.{4,}\s*\d+\s*$", page_text, re.M))
        if toc_started and (dotted_entries >= 2 or page.pdf_page <= 12):
            page.content_type = "navigation"
        elif toc_started:
            toc_started = False


def _furniture_key(text: str) -> str:
    normalized = " ".join(text.lower().split())
    normalized = _REVISION_RE.sub("<revision>", normalized)
    normalized = _PRINTED_PAGE_RE.sub("<page>", normalized)
    normalized = re.sub(r"\b[A-K](?:\.\d+)+\b", "<section>", normalized)
    return normalized


def _canonical_section(value: str) -> str:
    normalized = value.strip().rstrip(".")
    return normalized + "." if re.fullmatch(r"[A-K]", normalized) else normalized


def _document_provenance(pages: Sequence[ExtractedPage]) -> tuple[str | None, str | None]:
    all_text = "\n".join(block.raw_text for page in pages for block in page.blocks)
    title = _MANUAL_TITLE if _MANUAL_TITLE.lower() in all_text.lower() else None
    versions: list[tuple[tuple[int, int, int], str, str]] = []
    for match in re.finditer(
        r"\b(\d+\.\d+)\b[^\n]{0,60}?\b((?:19|20)\d{2}-\d{2}-\d{2})\b",
        all_text,
    ):
        date = tuple(int(part) for part in match.group(2).split("-"))
        versions.append((date, match.group(1), match.group(2)))
    if versions:
        _, number, issue = max(versions)
        return title, f"{number} ({issue})"
    dates = _DATE_RE.findall(all_text[:5000])
    return title, max(dates) if dates else None


def _navigation_entries(pages: Sequence[ExtractedPage]) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    current_section: str | None = None
    current_letter: str | None = None
    for page in pages:
        if page.content_type != "navigation":
            continue
        if page.section_marker:
            current_section = _canonical_section(page.section_marker)
        for block in page.blocks:
            for line in block.raw_text.splitlines():
                match = _TOC_ENTRY_RE.match(line)
                if match:
                    current_section = _canonical_section(match.group(1))
                    current_letter = None
                    entries.append(
                        {
                            "section_id": current_section,
                            "title": match.group(2).strip(),
                            "printed_page": match.group(3),
                            "pdf_page": page.pdf_page,
                        }
                    )
                    continue
                marker_match = _NAV_MARKER_RE.match(line)
                if not marker_match or not current_section:
                    continue
                marker = marker_match.group(1)
                raw_title = marker_match.group(2).strip()
                target_match = re.search(r"\.{2,}\s*(\d+)\s*$", raw_title)
                title = re.sub(r"\.{2,}\s*\d+\s*$", "", raw_title).strip()
                if marker[1].isupper():
                    current_letter = marker
                    clause_path = f"{current_section}{marker}"
                else:
                    clause_path = f"{current_section}{current_letter or ''}{marker}"
                entries.append(
                    {
                        "section_id": current_section,
                        "clause_path": clause_path,
                        "title": title,
                        "printed_page": target_match.group(1) if target_match else None,
                        "pdf_page": page.pdf_page,
                    }
                )
    return entries


def _looks_like_legal_manual(document: ParsedDocument) -> bool:
    body = document.flatten(include_navigation=True)
    score = 0
    if _MANUAL_TITLE.lower() in body.lower() or document.document_title == _MANUAL_TITLE:
        score += 3
    if len(re.findall(r"(?m)^\s*[A-K]\.\d+\s+", body)) >= 2:
        score += 2
    if len(re.findall(r"(?m)^\s*\([A-Z]\)\s+", body)) >= 2:
        score += 1
    if len(re.findall(r"(?m)^\s*\((?:[ivxlcdm]+|[a-z]+|\d+)\)\s+", body)) >= 3:
        score += 2
    if "table of contents" in body.lower() or "authorised dealer" in body.lower():
        score += 1
    return score >= 4


def _clean_table_rows(
    rows: Optional[Sequence[Sequence[Optional[str]]]],
) -> list[list[str]]:
    if not rows:
        return []
    cleaned: list[list[str]] = []
    width = max((len(row) for row in rows), default=0)
    for row in rows:
        values = [" ".join((cell or "").split()).replace("|", "\\|") for cell in row]
        values += [""] * (width - len(values))
        if any(values):
            cleaned.append(values)
    return cleaned


def _table_to_markdown(rows: Optional[Sequence[Sequence[Optional[str]]]]) -> str:
    """Render extracted table rows as Markdown without splitting cells."""
    cleaned = _clean_table_rows(rows)
    if not cleaned:
        return ""
    width = max(len(row) for row in cleaned)
    cleaned = [row + [""] * (width - len(row)) for row in cleaned]
    lines = ["| " + " | ".join(cleaned[0]) + " |"]
    lines.append("| " + " | ".join(["---"] * width) + " |")
    lines.extend("| " + " | ".join(row) + " |" for row in cleaned[1:])
    return "\n".join(lines)


def _extract_page(page: Any) -> str:
    """Backward-compatible one-page renderer used by older callers/tests."""
    number = int(getattr(page, "page_number", 1) or 1)
    extracted = _extract_page_structured(page, number)
    return "\n\n".join(block.raw_text for block in extracted.blocks)


def _parse_pdf(content: bytes) -> str:
    """Backward-compatible PDF flattening helper."""
    return parse_document_structured("document.pdf", content).flatten()


def _parse_txt(content: bytes) -> str:
    """Decode UTF-8 plain text or raise a parsing error."""
    try:
        return content.decode("utf-8")
    except (UnicodeDecodeError, ValueError) as exc:
        raise DocumentParsingError(f"Failed to decode TXT file as UTF-8: {exc}") from exc
