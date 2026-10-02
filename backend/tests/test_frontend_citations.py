"""Pure Streamlit citation-label compatibility tests."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "frontend"))

from citation_format import format_citation_label, unique_citation_labels  # noqa: E402


def test_legal_label_uses_printed_page_and_pdf_fallback() -> None:
    assert format_citation_label({"clause_path": "B.4(A)(i)", "printed_page_start": "98", "pdf_page_start": 98}) == "B.4(A)(i), p. 98"
    label = format_citation_label({"section_id": "I.3", "printed_page_start": "94", "pdf_page_start": 101, "page_revisions": ["6/2026"]})
    assert label == "I.3, p. 94 (PDF 101) · rev. 6/2026"


def test_legacy_and_duplicate_payloads_are_safe() -> None:
    citations = [{"filename": "notes.txt"}, {"filename": "notes.txt"}]
    assert unique_citation_labels(citations) == ["notes.txt"]
    assert unique_citation_labels([]) == []
