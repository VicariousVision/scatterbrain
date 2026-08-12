"""Provision-boundary chunker for the SARB Currency and Exchanges Manual.

Parses the manual's own numbering scheme instead of splitting on fixed
character counts.  Produces one chunk per lowest-level provision, preserving
the full path and heading trail needed for citation.

Manual numbering hierarchy (up to 5 levels):
  Level 0  B.4, A.1, C.2 ...               top-level section
  Level 1  (A), (B), (C) ...               subsection
  Level 2  (i), (ii), (iii) ...            clause
  Level 3  (a), (b), (c) ...              sub-clause
  Level 4  (aa), (bb), (cc) ...            sub-sub-clause
  Level 5  (1), (2), (3) ...              list item

Definitions in A.1 are parsed separately as flat term/definition pairs.

Output shapes
-------------
Provision chunk:
{
    "path": "B.4(B)(iv)(d)(bb)",
    "level": int,              # 0 = top section, 1 = subsection, etc.
    "heading_trail": [...],    # list of heading strings from root to this node
    "text": "..."              # this node's own text only (children excluded)
}

Definition chunk:
{
    "term": "CFC account",
    "path": "A.1",
    "text": "..."
}
"""

from __future__ import annotations

import re
import logging
from dataclasses import dataclass, field
from typing import Iterator

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Minimum text length to consider a chunk substantive.
# Chunks shorter than this (blank pages, TOC entries, headers) are dropped.
# ---------------------------------------------------------------------------
_MIN_CHUNK_CHARS = 40

# ---------------------------------------------------------------------------
# Regex patterns for each heading level
# ---------------------------------------------------------------------------
# Level 0: "B.4", "A.1", "C.12"  — letter dot number at start of line
_RE_L0 = re.compile(r"^([A-Z]\.\d+)\s+(.*)", re.MULTILINE)

# Level 1: "(A)", "(B)", ... "(ZZ)" — uppercase letter(s) in parens
_RE_L1 = re.compile(r"^\(([A-Z]{1,2})\)\s+(.*)", re.MULTILINE)

# Level 2: "(i)", "(ii)", "(iii)", "(iv)" ... — roman numerals in parens
_RE_L2 = re.compile(r"^\((i{1,3}|iv|vi{0,3}|ix|xi{0,3}|xiv|xv|xvi{0,3}|xix|xx{0,3})\)\s+(.*)", re.MULTILINE)

# Level 3: "(a)", "(b)", ... "(z)" — single lowercase letter in parens
_RE_L3 = re.compile(r"^\(([a-z])\)\s+(.*)", re.MULTILINE)

# Level 4: "(aa)", "(bb)", ... — double lowercase letter in parens
_RE_L4 = re.compile(r"^\(([a-z]{2})\)\s+(.*)", re.MULTILINE)

# Level 5: "(1)", "(2)", ... — digits in parens
_RE_L5 = re.compile(r"^\((\d+)\)\s+(.*)", re.MULTILINE)

# Ordered from most-specific to least to avoid ambiguity when trying each level.
# The tuple is (level_number, compiled_regex, marker_group_index, heading_group_index).
_LEVEL_PATTERNS: list[tuple[int, re.Pattern, int, int]] = [
    (4, _RE_L4, 1, 2),   # (aa) before (a) — longer match wins
    (3, _RE_L3, 1, 2),
    (2, _RE_L2, 1, 2),
    (1, _RE_L1, 1, 2),
    (5, _RE_L5, 1, 2),
    (0, _RE_L0, 1, 2),
]


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class ProvisionChunk:
    path: str
    level: int
    heading_trail: list[str]
    text: str

    def to_dict(self) -> dict:
        return {
            "path": self.path,
            "level": self.level,
            "heading_trail": self.heading_trail,
            "text": self.text,
        }


@dataclass
class DefinitionChunk:
    term: str
    path: str
    text: str

    def to_dict(self) -> dict:
        return {
            "term": self.term,
            "path": self.path,
            "text": self.text,
        }


# ---------------------------------------------------------------------------
# Non-substantive page detection
# ---------------------------------------------------------------------------

# Pages that are almost entirely whitespace/numbers (TOC, version control).
_RE_TOC_LINE = re.compile(
    r"^\s*(?:"
    r"\.{5,}"            # dotted leader line  "Section ........ 12"
    r"|Page \d+"         # "Page 12"
    r"|\d+\s*$"          # lone page number
    r"|Version control"
    r"|Table of [Cc]ontents"
    r"|CONTENTS"
    r")",
    re.MULTILINE,
)


def _is_non_substantive(text: str) -> bool:
    """Return True for pages that are TOC / version control boilerplate."""
    if len(text.strip()) < _MIN_CHUNK_CHARS:
        return True
    # If the majority of non-empty lines look like TOC entries, skip the block.
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if not lines:
        return True
    toc_hits = sum(1 for ln in lines if _RE_TOC_LINE.search(ln))
    return toc_hits / len(lines) > 0.5


# ---------------------------------------------------------------------------
# Token-level splitting
# ---------------------------------------------------------------------------

@dataclass
class _Token:
    """A heading token found in the raw text."""
    level: int
    marker: str        # e.g. "B.4", "(A)", "(iv)"
    heading_text: str  # text on the same line as the heading marker
    start: int         # character offset in raw text


def _tokenise(text: str) -> list[_Token]:
    """Walk the text and extract every heading token with its position."""
    tokens: list[_Token] = []

    for level, pattern, mg, hg in _LEVEL_PATTERNS:
        for m in pattern.finditer(text):
            tokens.append(_Token(
                level=level,
                marker=m.group(mg),
                heading_text=m.group(hg).strip(),
                start=m.start(),
            ))

    # Sort by position.  Where two patterns match the same position (should
    # not happen with well-formed text) prefer the higher level (lower number).
    tokens.sort(key=lambda t: (t.start, t.level))
    return tokens


def _build_path(trail: list[str], new_marker: str) -> str:
    """Concatenate heading markers into a path string like 'B.4(B)(iv)(d)(bb)'."""
    return "".join(trail) + new_marker


def _heading_label(marker: str, heading_text: str) -> str:
    """Human-readable heading label, e.g. '(B) Travel allowances'."""
    if heading_text:
        return f"({marker}) {heading_text}" if not marker[0].isalpha() or len(marker) == 1 and marker.isalpha() else f"{marker} {heading_text}"
    return marker


# ---------------------------------------------------------------------------
# Definition parser (section A.1)
# ---------------------------------------------------------------------------

# Matches a bold/ALL-CAPS term followed by a definition on the same or next line.
# Many PDF extractions produce definitions as: <TERM>\n<definition text>.
_RE_DEFINITION = re.compile(
    r"^([A-Z][A-Za-z /\-]{1,60})\s*[:\-–]\s*(.+?)(?=\n[A-Z][A-Za-z /\-]{1,60}\s*[:\-–]|\Z)",
    re.MULTILINE | re.DOTALL,
)


def _parse_definitions(section_text: str, section_path: str) -> list[DefinitionChunk]:
    """Extract term/definition pairs from a definitions section (e.g. A.1)."""
    chunks: list[DefinitionChunk] = []
    for m in _RE_DEFINITION.finditer(section_text):
        term = m.group(1).strip()
        definition = re.sub(r"\s+", " ", m.group(2)).strip()
        if len(definition) < _MIN_CHUNK_CHARS:
            continue
        chunks.append(DefinitionChunk(term=term, path=section_path, text=definition))
    return chunks


# ---------------------------------------------------------------------------
# Main parser
# ---------------------------------------------------------------------------

def parse_provisions(raw_text: str) -> tuple[list[ProvisionChunk], list[DefinitionChunk]]:
    """Parse the full manual text into provision and definition chunks.

    Args:
        raw_text: Full text of the manual as extracted by the document parser.

    Returns:
        (provisions, definitions) where:
          - provisions: list of ProvisionChunk objects, one per numbered node.
          - definitions: list of DefinitionChunk objects from A.1 glossary.
    """
    provisions: list[ProvisionChunk] = []
    definitions: list[DefinitionChunk] = []

    tokens = _tokenise(raw_text)
    if not tokens:
        logger.warning("provision_chunker: no heading tokens found in document text.")
        return provisions, definitions

    # Stack of (level, marker_string) for building paths.
    # The path stack tracks only the marker portion (e.g. "B.4", "(A)", "(iv)").
    path_stack: list[tuple[int, str]] = []
    heading_trail_stack: list[str] = []

    def _trim_stack_to_level(target_level: int) -> None:
        """Pop the stack down to the parent of target_level."""
        while path_stack and path_stack[-1][0] >= target_level:
            path_stack.pop()
            heading_trail_stack.pop()

    for idx, token in enumerate(tokens):
        # Determine the text owned by this token: from after the heading line
        # to just before the next token starts.
        token_end = tokens[idx + 1].start if idx + 1 < len(tokens) else len(raw_text)

        # The heading line itself ends at the first newline after token.start.
        heading_line_end = raw_text.find("\n", token.start)
        if heading_line_end == -1:
            heading_line_end = len(raw_text)

        # Own-text: from end-of-heading-line to start of next token,
        # minus any sub-headings (those will become their own tokens/chunks).
        own_text_raw = raw_text[heading_line_end:token_end].strip()
        # Strip child heading lines from the own-text block.  Any line that
        # looks like a heading at a deeper level is NOT this node's own text.
        own_text = _strip_child_headings(own_text_raw, token.level)

        # Build path and trail.
        _trim_stack_to_level(token.level)

        # For level-0 tokens the marker is like "B.4" (already dotted).
        # For sub-levels we wrap in parens: "(A)", "(i)", "(a)", etc.
        if token.level == 0:
            marker_str = token.marker          # "B.4" — already well-formed
        else:
            marker_str = f"({token.marker})"   # "(A)", "(iv)", "(a)", "(aa)", "(1)"

        parent_markers = [m for _, m in path_stack]
        path = "".join(parent_markers) + marker_str

        # Build heading trail.
        if token.heading_text:
            trail_label = f"{marker_str} {token.heading_text}"
        else:
            trail_label = marker_str
        current_trail = heading_trail_stack[:] + [trail_label]

        # Push this token onto the stack so children can reference it.
        path_stack.append((token.level, marker_str))
        heading_trail_stack.append(trail_label)

        # Skip non-substantive blocks.
        if _is_non_substantive(own_text) and len(own_text) < _MIN_CHUNK_CHARS:
            continue

        # Special-case: A.1 (definitions section) — parse as definitions.
        if path == "A.1" or path.startswith("A.1("):
            defs = _parse_definitions(own_text, path)
            if defs:
                definitions.extend(defs)
            # Also emit as a provision so the node exists in the graph.
            if len(own_text) >= _MIN_CHUNK_CHARS:
                provisions.append(ProvisionChunk(
                    path=path,
                    level=token.level,
                    heading_trail=current_trail,
                    text=own_text,
                ))
        else:
            if len(own_text) >= _MIN_CHUNK_CHARS:
                provisions.append(ProvisionChunk(
                    path=path,
                    level=token.level,
                    heading_trail=current_trail,
                    text=own_text,
                ))

    logger.info(
        "provision_chunker: parsed %d provision chunks, %d definition chunks.",
        len(provisions),
        len(definitions),
    )
    return provisions, definitions


def _strip_child_headings(text: str, parent_level: int) -> str:
    """Remove lines that are headings at a deeper level than parent_level.

    When we compute "own text" for a node, we don't want to include the
    heading lines of its children — those are captured as separate tokens.
    We only strip the heading marker line itself, not the child body text
    (which also gets its own token and is thus not re-included here because
    the slice boundary is the next token's start, not the parent's end).

    This is a lightweight guard; the primary boundary is the token-position
    slicing above.
    """
    if not text:
        return text

    child_patterns = [pat for lvl, pat, _, _ in _LEVEL_PATTERNS if lvl > parent_level]
    kept_lines: list[str] = []
    for line in text.splitlines():
        is_child_heading = any(pat.match(line.strip()) for pat in child_patterns)
        if not is_child_heading:
            kept_lines.append(line)
    return "\n".join(kept_lines).strip()
