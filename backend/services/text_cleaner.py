"""Text cleaning for parsed document content.

Normalizes raw extracted text (especially from PDFs) before chunking and
embedding. The goal is to remove formatting noise that hurts retrieval
quality -- control characters, zero-width junk, broken hyphenation, and
irregular whitespace -- while preserving the *structure* that carries meaning:
tables, nested lists, and subsection layout.

Design notes
------------
This cleaner is intentionally conservative on two fronts:

1. It does NOT lowercase, strip punctuation, or remove stopwords. Modern
   sentence-embedding models expect natural, cased text with punctuation, so
   those classic NLP steps would hurt retrieval rather than help it.

2. It preserves layout that encodes structure. Leading indentation (nested
   lists, code-like blocks) is kept, single newlines are kept (line
   structure), and Markdown table rows produced by the PDF parser pass through
   untouched. Only whitespace *within* a line's content is normalized, and
   only invisible/meaningless characters are removed.
"""

from __future__ import annotations

import re
import unicodedata

from models.content import ParsedDocument

# Control characters that carry no textual meaning. Tab and newline are kept
# (handled by the whitespace logic); everything else in the C0/C1 ranges plus
# the Unicode replacement char is removed.
_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f\ufffd]")

# Zero-width and bidi formatting characters that PDFs often embed.
_ZERO_WIDTH = re.compile(r"[\u200b-\u200f\u202a-\u202e\u2060\ufeff]")

# A word split across a line break with a hyphen, e.g. "inter-\nnational".
_HYPHEN_LINEBREAK = re.compile(r"([A-Za-z])-\n([a-z])")

# A run of horizontal whitespace (spaces, tabs, non-breaking/other spaces).
_HORIZONTAL_WS = re.compile(r"[ \t\u00a0\u2000-\u200a\u205f\u3000]+")

# Leading indentation captured separately so it can be preserved.
_LEADING_WS = re.compile(r"^([ \t\u00a0\u2000-\u200a\u205f\u3000]*)(.*)$")

# Three or more consecutive newlines collapse to a single paragraph break.
_EXCESS_BLANK_LINES = re.compile(r"\n{3,}")

# Tabs and non-breaking/exotic spaces normalized to plain spaces for indent
# measurement, so indentation depth is consistent regardless of source.
_INDENT_UNIFY = re.compile(r"[\t\u00a0\u2000-\u200a\u205f\u3000]")


def _looks_like_table_row(line: str) -> bool:
    """Return True for Markdown table rows, which must pass through unchanged.

    The PDF parser emits tables as pipe-delimited Markdown rows. Collapsing
    their internal spacing is fine, but they must keep their pipes and overall
    shape, so they are handled by the normal per-line path (which preserves
    pipes) rather than any special casing. This helper exists to document the
    intent and to guard the alignment/separator row from being altered.
    """
    stripped = line.strip()
    return stripped.startswith("|") and stripped.endswith("|")


def clean_text(text: str) -> str:
    """Normalize raw extracted text while preserving meaningful structure.

    Steps, in order:
      1. Unicode NFKC normalization (ligatures, full-width forms, etc.).
      2. Normalize line endings to ``\\n``.
      3. Rejoin words hyphenated across line breaks.
      4. Remove control, replacement, and zero-width characters.
      5. Per line: preserve leading indentation, collapse internal whitespace
         runs to a single space, and strip trailing whitespace. Markdown table
         rows keep their pipe delimiters.
      6. Cap consecutive blank lines at one.

    Structure preserved: nested-list / subsection indentation (leading
    whitespace per line), line breaks (single newlines), and Markdown tables.

    Args:
        text: Raw text as returned by the document parser.

    Returns:
        Cleaned text. Returns an empty string for empty/whitespace-only input.
    """
    if not text or not text.strip():
        return ""

    # 1. Canonical Unicode form.
    text = unicodedata.normalize("NFKC", text)

    # 2. Normalize line endings before any newline-sensitive step.
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # 3. Rejoin hyphenated line breaks from wrapped prose. Done before removing
    #    control chars so the newline is still present to match.
    text = _HYPHEN_LINEBREAK.sub(r"\1\2", text)

    # 4. Drop invisible/control characters that carry no meaning.
    text = _CONTROL_CHARS.sub("", text)
    text = _ZERO_WIDTH.sub("", text)

    # 5. Clean each line while preserving its leading indentation. This keeps
    #    nested lists and subsection layout intact; only whitespace *within*
    #    the line content is collapsed.
    cleaned_lines = []
    for line in text.split("\n"):
        match = _LEADING_WS.match(line)
        indent_raw, body = match.group(1), match.group(2)
        # Normalize indent characters (tabs/nbsp) to plain spaces so depth is
        # consistent, but keep the *amount* of indentation.
        indent = _INDENT_UNIFY.sub(" ", indent_raw)
        body = _HORIZONTAL_WS.sub(" ", body).rstrip()
        cleaned_lines.append(f"{indent}{body}" if body else "")
    text = "\n".join(cleaned_lines)

    # 6. Cap consecutive blank lines at a single paragraph break.
    text = _EXCESS_BLANK_LINES.sub("\n\n", text)

    # Remove only surrounding blank lines. ``str.strip()`` would erase the
    # first line's legal indentation, which is hierarchy evidence.
    return text.strip("\n")

def clean_parsed_document(document: ParsedDocument) -> ParsedDocument:
    """Clean each block without flattening pages or changing raw source.

    A deep copy is returned so callers can retain the exact parser output for
    diagnostics. Coordinates, page boundaries, table rows, and source spans
    are preserved; only ``clean_text`` and normalized table-cell values are
    populated on the copy.

    Parameters
    ----------
    document:
        Page-aware parser output.
    """
    cleaned = document.model_copy(deep=True)
    for page in cleaned.pages:
        for block in page.blocks:
            block.clean_text = clean_text(block.raw_text)
            if block.table_header:
                block.table_header = [clean_text(cell) for cell in block.table_header]
            if block.table_rows:
                block.table_rows = [
                    [clean_text(cell) for cell in row] for row in block.table_rows
                ]
    return cleaned
