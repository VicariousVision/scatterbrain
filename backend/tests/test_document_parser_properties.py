"""
Property-based tests for the document parser service.

**Validates: Requirements 2.5**
"""

from __future__ import annotations

import io

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from docx import Document as DocxDocument

from backend.services.document_parser import parse_document


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_docx_bytes(paragraphs: list[str]) -> bytes:
    """Build an in-memory DOCX file containing the given paragraphs."""
    doc = DocxDocument()
    for text in paragraphs:
        doc.add_paragraph(text)
    buf = io.BytesIO()
    doc.save(buf)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Property 3: Paragraph boundaries are preserved
# ---------------------------------------------------------------------------

# Strategy: lists of 2 or more non-empty, non-whitespace-only text strings.
# We exclude strings that are purely whitespace so that python-docx doesn't
# silently collapse them, and we keep the alphabet printable to avoid
# control characters that could confuse the DOCX XML layer.
_paragraph_text = st.text(
    alphabet=st.characters(
        whitelist_categories=("Lu", "Ll", "Nd", "Zs", "Po", "Pd"),
        whitelist_characters="abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 .,!?-",
    ),
    min_size=1,
).filter(lambda s: s.strip())  # ensure at least one non-whitespace character

_paragraph_list = st.lists(_paragraph_text, min_size=2, max_size=20)


@given(paragraphs=_paragraph_list)
@settings(max_examples=50)
def test_paragraph_boundaries_preserved(paragraphs: list[str]) -> None:
    """
    Property 3: Paragraph boundaries are preserved.

    For any DOCX document built from 2+ non-empty paragraphs, the text
    returned by parse_document must contain at least one newline character,
    confirming that paragraph boundaries are preserved.

    **Validates: Requirements 2.5**
    """
    docx_bytes = _build_docx_bytes(paragraphs)
    result = parse_document("test.docx", docx_bytes)

    assert "\n" in result, (
        f"Expected at least one '\\n' in parsed output for {len(paragraphs)} "
        f"paragraphs, but got: {result!r}"
    )
