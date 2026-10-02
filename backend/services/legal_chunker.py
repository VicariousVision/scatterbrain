"""Hierarchy-aware legal parent/child chunking.

The parser recognizes the Authorised Dealers manual's A--K sections,
lettered subsections, Roman provisions, nested lower-case/numeric items,
definitions, visual tables, and BOP-style code lists. It carries hierarchy
across pages and emits unembedded parents plus searchable clause-first
children. Generic documents remain on the recursive fallback path.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import Iterable, Iterator, Sequence

from config import settings
from models.content import ChunkRecord, ExtractedBlock, ParsedDocument, SourceSpan
from services.text_chunker import chunk_text, split_continuous_text

_SECTION_LINE_RE = re.compile(
    r"^\s*((?:[A-K](?:\.\d+(?:\.\d+)*)+)|(?:[A-K]\.))\s+(.*\S)\s*$"
)
_COMPOSITE_RE = re.compile(
    r"^\s*([A-K](?:\.\d+)+)((?:\([A-Za-z0-9]+\))+)[.\s]*(.*)$"
)
_MARKER_RE = re.compile(r"^\s*\(([A-Za-z]+|\d+)\)\s*(.*)$")
_LEADING_MARKER_REFERENCE_RE = re.compile(
    r"^\s*(?:(?:\([A-Za-z0-9]+\))+\s+(?:above|below)\b|"
    r"\([A-Za-z0-9]+\)\s+(?:(?:and|or|to)\s+\([A-Za-z0-9]+\)\s+)(?:above|below)\b)",
    re.I,
)
_DEFINITION_RE = re.compile(
    r"^\s*([A-Z][A-Za-z0-9/&’'()\- ]{1,90}?)\s+means\b", re.I
)
_CODE_RE = re.compile(r"^\s*(\d{3}(?:\s+\d{2})?)\s+(.+\S)\s*$")
_CROSS_REF_RE = re.compile(
    r"\b([A-K](?:(?:\.\d+)+|\.)(?:\([A-Za-z0-9]+\))*)(?![A-Za-z0-9])"
)
_ROMAN_RE = re.compile(r"^(?=[ivxlcdm]+$)[ivxlcdm]+$")
_LIST_QUERY_WORDS = re.compile(
    r"\b(all|list|which|what are|codes?|categories|steps|requirements|following)\b",
    re.I,
)
_EXCEPTION_PREFIX = re.compile(
    r"^\s*(provided\s+that|except(?:ion)?\b|subject\s+to|unless\b|however\b)",
    re.I,
)


@dataclass
class _Line:
    clean: str
    raw: str
    indent: int
    pdf_page: int
    printed_page: str | None
    revision: str | None
    section_marker: str | None
    span: SourceSpan
    sequence: int


@dataclass
class _Node:
    level: int
    marker: str
    heading: str
    section_id: str | None
    parent: "_Node | None" = None
    items: list["_Node | _Line"] = field(default_factory=list)
    sequence: int = 0
    indent: int = 0
    defined_term: str | None = None

    @property
    def children(self) -> list["_Node"]:
        return [item for item in self.items if isinstance(item, _Node)]

    def lineage(self) -> list["_Node"]:
        nodes: list[_Node] = []
        current: _Node | None = self
        while current and current.level >= 0:
            nodes.append(current)
            current = current.parent
        return list(reversed(nodes))


@dataclass
class _TableGroup:
    blocks: list[ExtractedBlock]
    header: list[str]
    rows: list[list[str]]
    row_blocks: list[ExtractedBlock]


@dataclass
class _CodeGroup:
    lines: list[_Line]
    rows: list[tuple[str, str, list[_Line]]]
    section_id: str | None


def is_legal_document(document: ParsedDocument) -> bool:
    """Return whether structured legal parsing is appropriate."""
    if document.is_legal:
        return True
    text = document.flatten(include_navigation=True)
    score = 0
    if "authorised dealer" in text.lower() or "currency and exchanges manual" in text.lower():
        score += 3
    if len(re.findall(r"(?m)^\s*[A-K]\.\d+\s+", text)) >= 2:
        score += 2
    if len(re.findall(r"(?m)^\s*\([A-Z]\)\s+", text)) >= 2:
        score += 1
    if len(re.findall(r"(?m)^\s*\([ivxlcdm]+\)\s+", text)) >= 2:
        score += 2
    return score >= 4


def chunk_parsed_document(
    document: ParsedDocument,
    *,
    document_id: str | None = None,
) -> list[ChunkRecord]:
    """Create stable parent/child records from a structured document.

    Non-legal documents are delegated to :func:`chunk_text`, preserving the
    existing generic behavior. Legal records always obey the configured hard
    maximum; only forced splits of one continuous provision receive a
    continuity tail.
    """
    if not is_legal_document(document):
        return _generic_records(document, document_id=document_id)

    lines = _document_lines(document)
    code_groups, consumed = _extract_code_groups(lines)
    prose_lines = [line for line in lines if line.sequence not in consumed]
    roots = _parse_hierarchy(prose_lines)

    records: list[ChunkRecord] = []
    records.extend(_front_matter_records(document, document_id))
    records.extend(_hierarchy_records(document, roots, document_id))
    records.extend(_table_records(document, _stitch_tables(document), document_id))
    records.extend(_code_records(document, code_groups, document_id))

    parent_orders: dict[str, int] = {}
    for record in records:
        record.document_id = document_id
        if record.record_type == "child":
            parent_key = record.parent_id or ""
            record.child_order = parent_orders.get(parent_key, 0)
            parent_orders[parent_key] = record.child_order + 1
    return records


def _document_lines(document: ParsedDocument) -> list[_Line]:
    lines: list[_Line] = []
    sequence = 0
    for page in document.pages:
        if page.content_type != "body":
            continue
        for block in page.blocks:
            if block.block_type == "table":
                continue
            clean_lines = (block.clean_text or block.raw_text).splitlines()
            raw_lines = block.raw_text.splitlines()
            for index, clean in enumerate(clean_lines):
                if not clean.strip():
                    continue
                raw = raw_lines[index] if index < len(raw_lines) else clean
                leading = len(clean) - len(clean.lstrip(" "))
                source = block.source_spans[0] if block.source_spans else SourceSpan(
                    pdf_page=page.pdf_page,
                    printed_page=page.printed_page,
                    page_revision=page.page_revision,
                    bbox=block.bbox,
                    block_order=block.order,
                )
                lines.append(
                    _Line(
                        clean=clean.rstrip(),
                        raw=raw.rstrip(),
                        indent=leading,
                        pdf_page=page.pdf_page,
                        printed_page=page.printed_page,
                        revision=page.page_revision,
                        section_marker=page.section_marker,
                        span=source,
                        sequence=sequence,
                    )
                )
                sequence += 1
    return lines


def _parse_hierarchy(lines: Sequence[_Line]) -> list[_Node]:
    root = _Node(level=-1, marker="", heading="", section_id=None)
    stack: list[_Node] = [root]
    active_section: str | None = None

    for line in lines:
        stripped = line.clean.strip()
        composite = _COMPOSITE_RE.match(stripped)
        section_match = _SECTION_LINE_RE.match(stripped)

        if composite and line.indent <= 12:
            section = _canonical_section(composite.group(1))
            section_node = _start_node(
                stack,
                root,
                level=1,
                marker=section,
                heading="",
                section_id=section,
                sequence=line.sequence,
                indent=line.indent,
            )
            section_node.items.append(line)
            active_section = section
            marker_tokens = re.findall(r"\(([^)]+)\)", composite.group(2))
            for token_index, token in enumerate(marker_tokens):
                level = _marker_level(token, stack[-1], line.indent)
                heading = composite.group(3).strip() if token_index == len(marker_tokens) - 1 else ""
                node = _start_node(
                    stack,
                    root,
                    level=level,
                    marker=f"({token})",
                    heading=heading,
                    section_id=section,
                    sequence=line.sequence,
                indent=line.indent,
                )
                if token_index == len(marker_tokens) - 1:
                    node.items.append(line)
            continue

        if (
            section_match
            and line.indent <= 12
            and _is_section_identifier(section_match.group(1))
        ):
            section = _canonical_section(section_match.group(1))
            heading = section_match.group(2).strip()
            node = _start_node(
                stack,
                root,
                level=1,
                marker=section,
                heading=heading,
                section_id=section,
                sequence=line.sequence,
                indent=line.indent,
            )
            node.items.append(line)
            active_section = section
            continue

        page_section = (
            _canonical_section(line.section_marker)
            if line.section_marker
            else None
        )
        if page_section and page_section != active_section:
            active_section = page_section
            _start_node(
                stack,
                root,
                level=1,
                marker=active_section,
                heading="",
                section_id=active_section,
                sequence=line.sequence,
                indent=line.indent,
            )

        definition = (
            _DEFINITION_RE.match(stripped)
            if active_section and active_section.rstrip(".") == "A.1"
            else None
        )
        if definition:
            term = definition.group(1).strip()
            node = _start_node(
                stack,
                root,
                level=2,
                marker=f"definition:{_slug(term)}",
                heading=term,
                section_id=active_section,
                sequence=line.sequence,
                indent=line.indent,
            )
            node.defined_term = term
            node.items.append(line)
            continue

        if _LEADING_MARKER_REFERENCE_RE.match(stripped):
            stack[-1].items.append(line)
            continue

        marker_match = _MARKER_RE.match(stripped)
        if marker_match:
            token = marker_match.group(1)
            level = _marker_level(token, stack[-1], line.indent)
            heading = marker_match.group(2).strip()
            node = _start_node(
                stack,
                root,
                level=level,
                marker=f"({token})",
                heading=heading,
                section_id=active_section,
                sequence=line.sequence,
                indent=line.indent,
            )
            node.items.append(line)
            continue

        # Plain language, including "provided that" and exceptions, belongs
        # to the deepest active provision and therefore cannot drift into a
        # blind character window.
        stack[-1].items.append(line)

    return root.children


def _start_node(
    stack: list[_Node],
    root: _Node,
    *,
    level: int,
    marker: str,
    heading: str,
    section_id: str | None,
    sequence: int,
    indent: int,
) -> _Node:
    while len(stack) > 1 and stack[-1].level >= level:
        stack.pop()
    parent = stack[-1] if stack else root
    node = _Node(
        level=level,
        marker=marker,
        heading=heading[:240],
        section_id=section_id,
        parent=parent,
        sequence=sequence,
        indent=indent,
    )
    parent.items.append(node)
    stack.append(node)
    return node


def _marker_level(token: str, current: _Node, indent: int) -> int:
    if token.isdigit():
        return 5
    if token.isupper():
        return 2
    lower = token.lower()
    if _ROMAN_RE.fullmatch(lower):
        # A Roman marker is a peer when it appears at the active Roman
        # provision's indentation, even after nested lower/numeric items.
        # A more deeply indented Roman marker remains a nested item.
        if current.level <= 2:
            return 3
        roman_ancestor: _Node | None = current
        while roman_ancestor is not None and roman_ancestor.level != 3:
            roman_ancestor = roman_ancestor.parent
        if current.level == 3:
            return 3
        if roman_ancestor is not None and indent <= roman_ancestor.indent + 2:
            return 3
        return max(4, current.level + 1)
    if len(lower) == 1:
        return 4
    return 5


def _is_section_identifier(value: str) -> bool:
    value = value.rstrip(".")
    return bool(re.fullmatch(r"[A-K](?:\.\d+)*", value))


def _canonical_section(value: str) -> str:
    value = value.strip().rstrip(".")
    return value if "." in value else value + "."


def _hierarchy_records(
    document: ParsedDocument,
    roots: Sequence[_Node],
    document_id: str | None,
) -> list[ChunkRecord]:
    records: list[ChunkRecord] = []
    parent_order = 0
    for section in roots:
        if section.level != 1:
            continue
        candidates = [child for child in section.children if child.level == 2]
        if not candidates:
            candidates = [section]
        for candidate in candidates:
            parent_text = _node_text(candidate, raw=False)
            if not parent_text.strip():
                continue
            clause = _clause_path(candidate)
            content_type = _content_type(candidate, parent_text)
            logical_path = _logical_path(candidate)
            parent_id = _stable_id(
                document.source_sha256, "parent", logical_path, content_type, 0
            )
            parent_record = _record_from_node(
                document,
                candidate,
                record_type="parent",
                parent_id=parent_id,
                child_id=None,
                source_text=parent_text,
                raw_text=_node_text(candidate, raw=True),
                embedding_text="",
                content_type=content_type,
                parent_order=parent_order,
                document_id=document_id,
            )
            records.append(parent_record)
            parent_order += 1

            child_nodes = _primary_child_nodes(candidate)
            for node_index, child_node in enumerate(child_nodes):
                effective_parent_id = parent_id
                ancestor_parent_id: str | None = None
                child_text = _node_text(child_node, raw=False)
                if (
                    len(parent_text) > settings.legal_parent_max_chars
                    and child_node is not candidate
                ):
                    intermediate_id = _stable_id(
                        document.source_sha256,
                        "intermediate",
                        _logical_path(child_node),
                        content_type,
                        node_index,
                    )
                    records.append(
                        _record_from_node(
                            document,
                            child_node,
                            record_type="intermediate_parent",
                            parent_id=intermediate_id,
                            child_id=None,
                            source_text=child_text,
                            raw_text=_node_text(child_node, raw=True),
                            embedding_text="",
                            content_type=content_type,
                            parent_order=parent_order,
                            document_id=document_id,
                            ancestor_parent_id=parent_id,
                        )
                    )
                    parent_order += 1
                    effective_parent_id = intermediate_id
                    ancestor_parent_id = parent_id

                records.extend(
                    _child_records_for_node(
                        document,
                        child_node,
                        parent_id=effective_parent_id,
                        ancestor_parent_id=ancestor_parent_id,
                        ordinal=node_index,
                        content_type=content_type,
                        document_id=document_id,
                    )
                )
    return records


def _primary_child_nodes(parent: _Node) -> list[_Node]:
    if parent.defined_term:
        return [parent]
    romans = [child for child in parent.children if child.level == 3]
    if romans:
        return romans
    if parent.level == 1:
        lettered = [child for child in parent.children if child.level == 2]
        return lettered or [parent]
    return [parent]


def _child_records_for_node(
    document: ParsedDocument,
    node: _Node,
    *,
    parent_id: str,
    ancestor_parent_id: str | None,
    ordinal: int,
    content_type: str,
    document_id: str | None,
) -> list[ChunkRecord]:
    source = _node_text(node, raw=False).strip("\n")
    raw = _node_text(node, raw=True).strip("\n")
    if not source:
        return []
    pieces, forced = _split_legal_node(node)
    clause = _clause_path(node)
    breadcrumb = _breadcrumb(node)
    logical_path = _logical_path(node)
    ids = [
        _stable_id(
            document.source_sha256,
            "child",
            logical_path,
            content_type,
            ordinal * 1000 + index,
        )
        for index in range(len(pieces))
    ]
    records: list[ChunkRecord] = []
    for index, piece in enumerate(pieces):
        child_id = ids[index]
        record = _record_from_node(
            document,
            node,
            record_type="child",
            parent_id=parent_id,
            child_id=child_id,
            source_text=piece,
            raw_text=raw if len(pieces) == 1 else piece,
            embedding_text=_embedding_text(breadcrumb, clause, piece),
            content_type=content_type,
            child_order=ordinal * 1000 + index,
            document_id=document_id,
            ancestor_parent_id=ancestor_parent_id,
        )
        _apply_piece_provenance(
            record,
            _source_lines_for_piece(
                node,
                piece,
                overlap=settings.legal_forced_split_overlap_chars,
            ),
        )
        if len(pieces) > 1:
            record.continues_from = ids[index - 1] if index else None
            record.continues_to = ids[index + 1] if index + 1 < len(ids) else None
        records.append(record)
    # ``forced`` currently documents why links/overlap exist. Structural
    # overflows also need continuation links, but only forced pieces overlap.
    _ = forced
    return records


def _split_legal_node(node: _Node) -> tuple[list[str], bool]:
    hard = settings.legal_chunk_hard_max_chars
    target = settings.legal_chunk_target_chars
    overlap = settings.legal_forced_split_overlap_chars
    full = _node_text(node, raw=False).strip("\n")
    if len(full) <= hard:
        return [full], False

    # Preserve item order: consecutive direct lines form one unit and every
    # nested legal sub-item is its own safe boundary.
    units: list[str] = []
    line_buffer: list[str] = []

    def flush_lines() -> None:
        if line_buffer:
            value = "\n".join(line_buffer).strip("\n")
            if value:
                units.append(value)
            line_buffer.clear()

    for item in node.items:
        if isinstance(item, _Line):
            if item.clean.strip():
                line_buffer.append(item.clean.rstrip())
        else:
            flush_lines()
            value = _node_text(item, raw=False).strip("\n")
            if value:
                units.append(value)
    flush_lines()

    if len(units) > 1:
        packed: list[str] = []
        current = ""
        used_forced_fallback = False
        for unit in units:
            if len(unit) > hard:
                if current:
                    packed.append(current)
                    current = ""
                packed.extend(_forced_split(unit, hard=hard, overlap=overlap))
                used_forced_fallback = True
                continue
            candidate = f"{current}\n{unit}".strip("\n") if current else unit
            attach_exception = bool(_EXCEPTION_PREFIX.match(unit))
            if current and len(candidate) > hard:
                packed.append(current)
                current = unit
            elif current and len(current) >= target and not attach_exception:
                packed.append(current)
                current = unit
            else:
                current = candidate
        if current:
            packed.append(current)
        if packed and all(len(piece) <= hard for piece in packed):
            return packed, used_forced_fallback

    return _forced_split(full, hard=hard, overlap=overlap), True


def _forced_split(text: str, *, hard: int, overlap: int) -> list[str]:
    """Sentence/recursive split with an exact tail only inside this text."""
    base_size = hard - overlap if overlap else hard
    base = split_continuous_text(text, chunk_size=base_size, overlap=0)
    if len(base) <= 1:
        # Defensive character fallback for a splitter/provider regression.
        base = [text[index : index + base_size] for index in range(0, len(text), base_size)]
    pieces: list[str] = []
    for index, part in enumerate(base):
        if index == 0 or overlap == 0:
            piece = part
        else:
            tail = pieces[-1][-overlap:]
            piece = tail + part
        if len(piece) > hard:
            piece = piece[:hard]
        pieces.append(piece)
    return [piece for piece in pieces if piece.strip()]


def _source_lines_for_piece(
    node: _Node, piece: str, *, overlap: int
) -> list[_Line]:
    """Map a split quotation back to the source lines/pages it intersects."""
    lines = list(_iter_lines(node))
    if not lines:
        return []
    rendered = [line.clean.rstrip() for line in lines if line.clean.strip()]
    rendered_lines = [line for line in lines if line.clean.strip()]
    full = "\n".join(rendered)
    candidates = [piece.strip("\n")]
    if overlap and len(piece) > overlap:
        candidates.append(piece[overlap:].strip("\n"))
    start = -1
    matched = ""
    for candidate in candidates:
        if candidate:
            start = full.find(candidate)
            if start >= 0:
                matched = candidate
                break
    if start >= 0:
        end = start + len(matched)
        selected: list[_Line] = []
        offset = 0
        for source_line, text in zip(rendered_lines, rendered):
            line_end = offset + len(text)
            if line_end >= start and offset <= end:
                selected.append(source_line)
            offset = line_end + 1
        if selected:
            return selected

    # Recursive sentence splitting can normalize a boundary separator. Fall
    # back to deterministic token overlap rather than assigning every page of
    # a long parent to every child.
    piece_words = set(re.findall(r"[A-Za-z0-9]{3,}", piece.casefold()))
    selected = []
    for source_line in lines:
        line_words = set(re.findall(r"[A-Za-z0-9]{3,}", source_line.clean.casefold()))
        required = min(3, len(line_words))
        if required and len(piece_words & line_words) >= required:
            selected.append(source_line)
    return selected or lines


def _apply_piece_provenance(record: ChunkRecord, lines: Sequence[_Line]) -> None:
    if not lines:
        return
    ordered = sorted(lines, key=lambda line: line.sequence)
    record.pdf_page_start = min(line.pdf_page for line in ordered)
    record.pdf_page_end = max(line.pdf_page for line in ordered)
    printed = [line.printed_page for line in ordered if line.printed_page]
    record.printed_page_start = printed[0] if printed else None
    record.printed_page_end = printed[-1] if printed else None
    record.page_revisions = list(
        dict.fromkeys(line.revision for line in ordered if line.revision)
    )
    record.source_spans = _dedupe_spans(line.span for line in ordered)


def _record_from_node(
    document: ParsedDocument,
    node: _Node,
    *,
    record_type: str,
    parent_id: str,
    child_id: str | None,
    source_text: str,
    raw_text: str,
    embedding_text: str,
    content_type: str,
    document_id: str | None,
    child_order: int = 0,
    parent_order: int = 0,
    ancestor_parent_id: str | None = None,
) -> ChunkRecord:
    lines = list(_iter_lines(node))
    pages = [line.pdf_page for line in lines]
    printed = [line.printed_page for line in lines if line.printed_page]
    revisions = list(dict.fromkeys(line.revision for line in lines if line.revision))
    spans = _dedupe_spans(line.span for line in lines)
    clause = _clause_path(node)
    return ChunkRecord(
        record_type=record_type,  # type: ignore[arg-type]
        document_id=document_id,
        document_title=document.document_title,
        source_filename=document.source_filename,
        source_sha256=document.source_sha256,
        document_version=document.document_version,
        pdf_page_start=min(pages) if pages else None,
        pdf_page_end=max(pages) if pages else None,
        printed_page_start=printed[0] if printed else None,
        printed_page_end=printed[-1] if printed else None,
        page_revisions=revisions,
        section_path=_section_path(node),
        section_id=node.section_id,
        clause_path=clause,
        heading=node.heading or None,
        breadcrumb=_breadcrumb(node),
        parent_id=parent_id,
        ancestor_parent_id=ancestor_parent_id,
        child_id=child_id,
        child_order=child_order,
        parent_order=parent_order,
        content_type=content_type,  # type: ignore[arg-type]
        defined_term=node.defined_term,
        cross_references=_cross_references(source_text, clause),
        extraction_method=document.extraction_method,
        parser_version=document.parser_version,
        source_spans=spans,
        raw_text=raw_text,
        source_text=source_text,
        embedding_text=embedding_text,
    )


def _node_text(node: _Node, *, raw: bool) -> str:
    parts: list[str] = []
    for item in node.items:
        if isinstance(item, _Line):
            text = item.raw if raw else item.clean
            if text.strip():
                parts.append(text.rstrip())
        else:
            text = _node_text(item, raw=raw)
            if text.strip():
                parts.append(text.strip("\n"))
    return "\n".join(parts)


def _iter_lines(node: _Node) -> Iterator[_Line]:
    for item in node.items:
        if isinstance(item, _Line):
            yield item
        else:
            yield from _iter_lines(item)


def _section_path(node: _Node) -> list[str]:
    path: list[str] = []
    for current in node.lineage():
        if current.marker.startswith("definition:"):
            continue
        if current.marker:
            path.append(current.marker)
    return path


def _clause_path(node: _Node) -> str | None:
    tokens = _section_path(node)
    if not tokens:
        return node.section_id
    result = tokens[0]
    for token in tokens[1:]:
        result += token if token.startswith("(") else f".{token}"
    return result


def _logical_path(node: _Node) -> str:
    clause = _clause_path(node) or node.marker or "root"
    if node.defined_term:
        return f"{clause}:definition:{_slug(node.defined_term)}"
    return clause


def _breadcrumb(node: _Node) -> str:
    parts: list[str] = []
    for current in node.lineage():
        if current.level < 1:
            continue
        identifier = current.marker
        heading = " ".join(current.heading.split())[:160]
        if identifier.startswith("definition:"):
            identifier = "Definition"
        label = " ".join(part for part in (identifier, heading) if part).strip()
        if label and (not parts or label != parts[-1]):
            parts.append(label)
    return " > ".join(parts)


def _content_type(node: _Node, text: str) -> str:
    lower = f"{node.heading}\n{text[:300]}".lower()
    if node.defined_term or (node.section_id or "").rstrip(".") == "A.1":
        return "definition"
    if re.search(r"\b(schedule|annexure|appendix)\b", lower) and not re.search(
        r"\b(refer(?:red)? to|in terms of)\b.{0,30}\b(schedule|annexure|appendix)\b",
        lower,
    ):
        return "schedule"
    if re.search(r"\b(form|specimen)\b", lower) and "refer" not in lower[:100]:
        return "form"
    return "provision"


def _embedding_text(breadcrumb: str, clause: str | None, source: str) -> str:
    prefix = breadcrumb or clause or ""
    return f"Legal location: {prefix}\nSource text:\n{source}" if prefix else source


def _cross_references(text: str, own_clause: str | None = None) -> list[str]:
    refs: list[str] = []
    own_normalized = _normalize_clause(own_clause or "")
    for match in _CROSS_REF_RE.finditer(text):
        value = _normalize_clause(match.group(1))
        if value and value != own_normalized and value not in refs:
            refs.append(value)
    return refs


def _normalize_clause(value: str) -> str:
    return re.sub(r"\s+", "", value).rstrip(".")


def _stable_id(
    source_sha256: str,
    role: str,
    logical_path: str | None,
    content_type: str,
    ordinal: int,
) -> str:
    seed = "|".join(
        (source_sha256, role, _normalize_clause(logical_path or "root").lower(), content_type, str(ordinal))
    )
    digest = hashlib.sha256(seed.encode("utf-8")).hexdigest()[:24]
    prefix = "c" if role == "child" else "p"
    return f"{prefix}_{digest}"


def _dedupe_spans(spans: Iterable[SourceSpan]) -> list[SourceSpan]:
    seen: set[tuple[object, ...]] = set()
    result: list[SourceSpan] = []
    for span in spans:
        key = (span.pdf_page, span.block_order, span.bbox)
        if key not in seen:
            seen.add(key)
            result.append(span)
    return result


def _front_matter_records(
    document: ParsedDocument, document_id: str | None
) -> list[ChunkRecord]:
    records: list[ChunkRecord] = []
    order = 0
    for page in document.pages:
        if page.content_type == "body":
            continue
        text = "\n\n".join(
            (block.clean_text or block.raw_text).strip()
            for block in page.blocks
            if (block.clean_text or block.raw_text).strip()
        )
        if not text:
            continue
        parent_id = _stable_id(
            document.source_sha256,
            page.content_type,
            f"page-{page.pdf_page}",
            page.content_type,
            0,
        )
        records.append(
            ChunkRecord(
                record_type=page.content_type,
                document_id=document_id,
                document_title=document.document_title,
                source_filename=document.source_filename,
                source_sha256=document.source_sha256,
                document_version=document.document_version,
                pdf_page_start=page.pdf_page,
                pdf_page_end=page.pdf_page,
                printed_page_start=page.printed_page,
                printed_page_end=page.printed_page,
                page_revisions=[page.page_revision] if page.page_revision else [],
                section_id=page.section_marker,
                parent_id=parent_id,
                parent_order=order,
                content_type=page.content_type,
                extraction_method=document.extraction_method,
                parser_version=document.parser_version,
                source_spans=[span for block in page.blocks for span in block.source_spans],
                raw_text=text,
                source_text=text,
                embedding_text="",
            )
        )
        order += 1
    return records


def _stitch_tables(document: ParsedDocument) -> list[_TableGroup]:
    ordered: list[ExtractedBlock] = []
    for page in document.pages:
        if page.content_type == "body":
            ordered.extend(block for block in page.blocks if block.block_type == "table")
    groups: list[_TableGroup] = []
    for block in ordered:
        header = block.table_header or _markdown_header(block.clean_text or block.raw_text)
        rows = block.table_rows or _markdown_rows(block.clean_text or block.raw_text)
        if (
            groups
            and groups[-1].blocks[-1].pdf_page + 1 == block.pdf_page
            and _normalized_header(groups[-1].header) == _normalized_header(header)
            and header
        ):
            groups[-1].blocks.append(block)
            groups[-1].rows.extend(rows)
            groups[-1].row_blocks.extend([block] * len(rows))
        else:
            groups.append(
                _TableGroup(
                    blocks=[block],
                    header=header,
                    rows=list(rows),
                    row_blocks=[block] * len(rows),
                )
            )
    return groups


def _table_records(
    document: ParsedDocument,
    groups: Sequence[_TableGroup],
    document_id: str | None,
) -> list[ChunkRecord]:
    records: list[ChunkRecord] = []
    hard = settings.legal_chunk_hard_max_chars
    target = settings.legal_chunk_target_chars
    for group_index, group in enumerate(groups):
        first = group.blocks[0]
        last = group.blocks[-1]
        section = first.section_marker or last.section_marker
        table_id = _stable_id(
            document.source_sha256,
            "table",
            f"{section or 'table'}:{first.pdf_page}:{group_index}",
            "table",
            0,
        )
        parent_text = _render_table(group.header, group.rows)
        spans = _dedupe_spans(span for block in group.blocks for span in block.source_spans)
        common = dict(
            document_id=document_id,
            document_title=document.document_title,
            source_filename=document.source_filename,
            source_sha256=document.source_sha256,
            document_version=document.document_version,
            pdf_page_start=first.pdf_page,
            pdf_page_end=last.pdf_page,
            printed_page_start=first.printed_page,
            printed_page_end=last.printed_page,
            page_revisions=list(
                dict.fromkeys(
                    block.page_revision for block in group.blocks if block.page_revision
                )
            ),
            section_path=[section] if section else [],
            section_id=section,
            heading="Table",
            breadcrumb=f"{section} > Table" if section else "Table",
            parent_id=table_id,
            content_type="table",
            table_id=table_id,
            table_header=group.header,
            extraction_method=document.extraction_method,
            parser_version=document.parser_version,
            source_spans=spans,
        )
        records.append(
            ChunkRecord(
                record_type="parent",
                parent_order=group_index,
                raw_text=parent_text,
                source_text=parent_text,
                embedding_text="",
                **common,
            )
        )

        row_groups: list[list[list[str]]] = []
        current: list[list[str]] = []
        for row in group.rows:
            candidate = _render_table(group.header, [*current, row])
            if current and (len(candidate) > hard or len(_render_table(group.header, current)) >= target):
                row_groups.append(current)
                current = [row]
            else:
                current.append(row)
        if current or not group.rows:
            row_groups.append(current)

        child_sources: list[str] = []
        forced_runs: list[int | None] = []
        forced_run = 0
        for rows in row_groups:
            rendered = _render_table(group.header, rows)
            if len(rendered) <= hard:
                child_sources.append(rendered)
                forced_runs.append(None)
                continue
            # Preserve the complete row in the unembedded parent. Searchable
            # pieces repeat the true header and row key, then sentence-split
            # only the pathological description.
            row = rows[0] if rows else [rendered]
            key = row[0] if row else ""
            prefix = _render_table(group.header, [])
            available = max(100, hard - len(prefix) - len(key) - 8)
            description = " | ".join(row[1:]) if len(row) > 1 else row[0]
            pieces = _forced_split(
                description,
                hard=available,
                overlap=settings.legal_forced_split_overlap_chars,
            )
            forced_run += 1
            for piece in pieces:
                child_sources.append(f"{prefix}\n| {key} | {piece} |"[:hard])
                forced_runs.append(forced_run)

        ids = [
            _stable_id(
                document.source_sha256,
                "child",
                f"table:{table_id}",
                "table",
                index,
            )
            for index in range(len(child_sources))
        ]
        for index, source in enumerate(child_sources):
            record = ChunkRecord(
                record_type="child",
                child_id=ids[index],
                child_order=index,
                raw_text=source,
                source_text=source,
                embedding_text=_embedding_text(common["breadcrumb"], section, source),
                cross_references=_cross_references(source),
                **common,
            )
            _apply_table_piece_provenance(record, group, source)
            run_id = forced_runs[index]
            if run_id is not None:
                previous_same = index > 0 and forced_runs[index - 1] == run_id
                next_same = index + 1 < len(ids) and forced_runs[index + 1] == run_id
                record.continues_from = ids[index - 1] if previous_same else None
                record.continues_to = ids[index + 1] if next_same else None
            records.append(record)
    return records


def _apply_table_piece_provenance(
    record: ChunkRecord, group: _TableGroup, source: str
) -> None:
    selected: list[ExtractedBlock] = []
    normalized_source = " ".join(source.split()).casefold()
    for row, block in zip(group.rows, group.row_blocks):
        nonempty = [" ".join(cell.split()).casefold() for cell in row if cell.strip()]
        key = nonempty[0] if nonempty else ""
        if key and key in normalized_source:
            selected.append(block)
    if not selected:
        selected = group.blocks
    selected.sort(key=lambda block: (block.pdf_page, block.order))
    record.pdf_page_start = min(block.pdf_page for block in selected)
    record.pdf_page_end = max(block.pdf_page for block in selected)
    printed = [block.printed_page for block in selected if block.printed_page]
    record.printed_page_start = printed[0] if printed else None
    record.printed_page_end = printed[-1] if printed else None
    record.page_revisions = list(
        dict.fromkeys(block.page_revision for block in selected if block.page_revision)
    )
    record.source_spans = _dedupe_spans(
        span for block in selected for span in block.source_spans
    )


def _render_table(header: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    width = max([len(header), *(len(row) for row in rows)], default=1)
    normalized_header = list(header) + [""] * (width - len(header))
    lines = ["| " + " | ".join(normalized_header) + " |"]
    lines.append("| " + " | ".join(["---"] * width) + " |")
    for row in rows:
        values = list(row) + [""] * (width - len(row))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def _markdown_header(text: str) -> list[str]:
    rows = _parse_markdown(text)
    return rows[0] if rows else []


def _markdown_rows(text: str) -> list[list[str]]:
    rows = _parse_markdown(text)
    return rows[1:] if len(rows) > 1 else []


def _parse_markdown(text: str) -> list[list[str]]:
    result: list[list[str]] = []
    for index, line in enumerate(text.splitlines()):
        if not line.strip().startswith("|"):
            continue
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if index == 1 and all(re.fullmatch(r":?-{3,}:?", cell) for cell in cells):
            continue
        result.append(cells)
    return result


def _normalized_header(header: Sequence[str]) -> tuple[str, ...]:
    return tuple(re.sub(r"\W+", "", cell).lower() for cell in header)


def _extract_code_groups(lines: Sequence[_Line]) -> tuple[list[_CodeGroup], set[int]]:
    groups: list[_CodeGroup] = []
    consumed: set[int] = set()
    index = 0
    while index < len(lines):
        first = _CODE_RE.match(lines[index].clean)
        if not first:
            index += 1
            continue
        rows: list[tuple[str, str, list[_Line]]] = []
        group_lines: list[_Line] = []
        cursor = index
        while cursor < len(lines):
            match = _CODE_RE.match(lines[cursor].clean)
            if match:
                rows.append((match.group(1), match.group(2).strip(), [lines[cursor]]))
                group_lines.append(lines[cursor])
                cursor += 1
                continue
            if rows and cursor < len(lines):
                candidate = lines[cursor]
                # Wrapped descriptions are indented and don't begin another
                # legal heading. Stop after a clear section/clause boundary.
                if _SECTION_LINE_RE.match(candidate.clean.strip()) or _MARKER_RE.match(candidate.clean.strip()):
                    break
                if candidate.indent >= lines[index].indent and candidate.pdf_page <= group_lines[-1].pdf_page + 1:
                    code, description, row_lines = rows[-1]
                    rows[-1] = (code, f"{description} {candidate.clean.strip()}", [*row_lines, candidate])
                    group_lines.append(candidate)
                    cursor += 1
                    continue
            break
        if len(rows) >= 2:
            section = lines[index].section_marker
            groups.append(_CodeGroup(lines=group_lines, rows=rows, section_id=section))
            consumed.update(line.sequence for line in group_lines)
            index = cursor
        else:
            index += 1
    return groups, consumed


def _code_records(
    document: ParsedDocument,
    groups: Sequence[_CodeGroup],
    document_id: str | None,
) -> list[ChunkRecord]:
    records: list[ChunkRecord] = []
    hard = settings.legal_chunk_hard_max_chars
    target = settings.legal_chunk_target_chars
    for group_index, group in enumerate(groups):
        parent_id = _stable_id(
            document.source_sha256,
            "code-list",
            f"{group.section_id or 'codes'}:{group.lines[0].pdf_page}",
            "code_list",
            group_index,
        )
        parent_text = "\n".join(f"{code}  {description}" for code, description, _ in group.rows)
        pages = [line.pdf_page for line in group.lines]
        printed = [line.printed_page for line in group.lines if line.printed_page]
        revisions = list(dict.fromkeys(line.revision for line in group.lines if line.revision))
        spans = _dedupe_spans(line.span for line in group.lines)
        common = dict(
            document_id=document_id,
            document_title=document.document_title,
            source_filename=document.source_filename,
            source_sha256=document.source_sha256,
            document_version=document.document_version,
            pdf_page_start=min(pages),
            pdf_page_end=max(pages),
            printed_page_start=printed[0] if printed else None,
            printed_page_end=printed[-1] if printed else None,
            page_revisions=revisions,
            section_path=[group.section_id] if group.section_id else [],
            section_id=group.section_id,
            heading="Balance of payments codes",
            breadcrumb=f"{group.section_id or ''} > Balance of payments codes".strip(" >"),
            parent_id=parent_id,
            content_type="code_list",
            extraction_method=document.extraction_method,
            parser_version=document.parser_version,
            source_spans=spans,
        )
        records.append(
            ChunkRecord(
                record_type="parent",
                parent_order=group_index,
                raw_text=parent_text,
                source_text=parent_text,
                embedding_text="",
                **common,
            )
        )
        packed: list[list[tuple[str, str, list[_Line]]]] = []
        current: list[tuple[str, str, list[_Line]]] = []
        for row in group.rows:
            candidate = "\n".join(
                f"{code}  {description}" for code, description, _ in [*current, row]
            )
            if current and (len(candidate) > hard or len(candidate) >= target):
                packed.append(current)
                current = [row]
            else:
                current.append(row)
        if current:
            packed.append(current)
        child_specs: list[tuple[str, str | None, int | None]] = []
        forced_run = 0
        for rows in packed:
            source = "\n".join(f"{code}  {description}" for code, description, _ in rows)
            if len(source) <= hard:
                child_specs.append(
                    (source, "; ".join(row[0] for row in rows), None)
                )
                continue
            # One pathological code description is retained whole in the
            # unembedded parent and sentence-split into linked children. The
            # real code key is repeated, so code and description never drift.
            code, description, _ = rows[0]
            available = max(100, hard - len(code) - 2)
            forced_run += 1
            for part in _forced_split(
                description,
                hard=available,
                overlap=settings.legal_forced_split_overlap_chars,
            ):
                child_specs.append((f"{code}  {part}"[:hard], code, forced_run))

        ids = [
            _stable_id(
                document.source_sha256,
                "child",
                f"codes:{parent_id}",
                "code_list",
                child_index,
            )
            for child_index in range(len(child_specs))
        ]
        for child_index, (source, code, forced_run_id) in enumerate(child_specs):
            record = ChunkRecord(
                record_type="child",
                child_id=ids[child_index],
                child_order=child_index,
                code=code,
                raw_text=source,
                source_text=source,
                embedding_text=_embedding_text(common["breadcrumb"], group.section_id, source),
                cross_references=_cross_references(source),
                **common,
            )
            _apply_code_piece_provenance(record, group, source)
            if forced_run_id is not None:
                previous_same = (
                    child_index > 0 and child_specs[child_index - 1][2] == forced_run_id
                )
                next_same = (
                    child_index + 1 < len(child_specs)
                    and child_specs[child_index + 1][2] == forced_run_id
                )
                record.continues_from = ids[child_index - 1] if previous_same else None
                record.continues_to = ids[child_index + 1] if next_same else None
            records.append(record)
    return records


def _apply_code_piece_provenance(
    record: ChunkRecord, group: _CodeGroup, source: str
) -> None:
    selected = [
        line
        for code, _, row_lines in group.rows
        if re.search(rf"(?<!\d){re.escape(code)}(?!\d)", source)
        for line in row_lines
    ]
    if not selected:
        selected = group.lines
    _apply_piece_provenance(record, selected)


def _generic_records(
    document: ParsedDocument, *, document_id: str | None
) -> list[ChunkRecord]:
    text = document.flatten(include_navigation=True)
    chunks = chunk_text(text)
    if not chunks:
        return []
    pages = [page.pdf_page for page in document.pages]
    spans = [span for page in document.pages for block in page.blocks for span in block.source_spans]
    parent_id = _stable_id(document.source_sha256, "parent", "generic", "generic", 0)
    records: list[ChunkRecord] = [
        ChunkRecord(
            record_type="parent",
            document_id=document_id,
            document_title=document.document_title,
            source_filename=document.source_filename,
            source_sha256=document.source_sha256,
            document_version=document.document_version,
            pdf_page_start=min(pages) if pages else None,
            pdf_page_end=max(pages) if pages else None,
            parent_id=parent_id,
            content_type="generic",
            extraction_method=document.extraction_method,
            parser_version=document.parser_version,
            source_spans=_dedupe_spans(spans),
            raw_text=text,
            source_text=text,
            embedding_text="",
        )
    ]
    for index, source in enumerate(chunks):
        child_id = _stable_id(document.source_sha256, "child", "generic", "generic", index)
        records.append(
            ChunkRecord(
                record_type="child",
                document_id=document_id,
                document_title=document.document_title,
                source_filename=document.source_filename,
                source_sha256=document.source_sha256,
                document_version=document.document_version,
                pdf_page_start=min(pages) if pages else None,
                pdf_page_end=max(pages) if pages else None,
                parent_id=parent_id,
                child_id=child_id,
                child_order=index,
                content_type="generic",
                extraction_method=document.extraction_method,
                parser_version=document.parser_version,
                source_spans=_dedupe_spans(spans),
                raw_text=source,
                source_text=source,
                embedding_text=source,
            )
        )
    return records


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
