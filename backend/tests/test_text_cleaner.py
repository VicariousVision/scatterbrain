"""Structured cleaning tests."""

from services.text_cleaner import clean_parsed_document, clean_text
from backend.tests.fixtures.legal_manual_snippets import manual_document


def test_clean_parsed_document_does_not_mutate_raw_text_or_layout() -> None:
    raw = "B.4  Rule\n  inter-\nnational\u200b text"
    document = manual_document(raw)
    cleaned = clean_parsed_document(document)
    original = document.pages[0].blocks[0]
    result = cleaned.pages[0].blocks[0]
    assert original.clean_text == raw
    assert result.raw_text == raw
    assert "international" in result.clean_text
    assert result.bbox == original.bbox
    assert result.source_spans == original.source_spans


def test_legacy_clean_text_keeps_indentation() -> None:
    assert clean_text("  (a)  one\n\n\nnext") == "  (a) one\n\nnext"
