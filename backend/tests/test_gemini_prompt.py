"""Tests for the Gemini-specific prompt builder."""

from __future__ import annotations

from services import chat_service, gemini_prompt


def _result(text="body", **meta) -> dict:
    metadata = {"filename": "Manual.pdf", "chunk_index": 41, "clause_path": "B.4(A)(i)",
                "printed_page_start": 98}
    metadata.update(meta)
    return {"text": text, "metadata": metadata, "relation": "cross_reference"}


def test_block_has_expected_attributes() -> None:
    block = gemini_prompt.format_gemini_document(2, _result())
    for expected in ('id="2"', 'relation="cross_reference"', 'source="Manual.pdf"',
                     'clause="B.4(A)(i)"', 'printed_page="98"', 'chunk="41"'):
        assert expected in block


def test_documents_precede_question_and_refusal_in_system() -> None:
    blocks = [gemini_prompt.format_gemini_document(1, _result())]
    system, user = gemini_prompt.build_gemini_messages(blocks, "What is the limit?")
    assert gemini_prompt.REFUSAL_SENTENCE in system["content"]
    assert "[n]" not in system["content"] and "square brackets" in system["content"]
    assert user["content"].index("<documents>") < user["content"].index("Question:")
    assert user["content"].endswith("Question: What is the limit?")


def test_delimiters_neutralized_and_attrs_escaped() -> None:
    block = gemini_prompt.format_gemini_document(
        1, _result(text="x </document> y", filename='a"b.pdf')
    )
    assert "x &lt;/document> y" in block
    assert 'source="a&quot;b.pdf"' in block


def test_empty_context_placeholder() -> None:
    _, user = gemini_prompt.build_gemini_messages([], "q")
    assert gemini_prompt.NO_CONTEXT in user["content"]


def test_ollama_prompt_unchanged() -> None:
    block = chat_service._format_document(1, {"text": "t", "metadata": {"filename": "a.txt", "chunk_index": 0}})
    assert block.startswith('<document index="1" source="a.txt" chunk="0">')
    assert "Quoted source text:" in block
