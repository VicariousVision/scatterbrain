"""Round-trip tests for structured transfer models."""

from models.content import ChunkRecord, ExtractedBlock, ExtractedPage, ParsedDocument, SourceSpan


def test_source_and_embedding_context_remain_distinct() -> None:
    span = SourceSpan(pdf_page=98, printed_page="98", page_revision="6/2026")
    record = ChunkRecord(
        source_filename="manual.pdf",
        source_sha256="a" * 64,
        child_id="c_one",
        parent_id="p_one",
        clause_path="B.4(A)(i)",
        breadcrumb="B.4 > (A) > (i)",
        source_text="Residents may use the allowance.",
        raw_text="Residents may use the allowance.",
        embedding_text="B.4 > (A) > (i)\nResidents may use the allowance.",
        source_spans=[span],
    )
    restored = ChunkRecord.model_validate_json(record.model_dump_json())
    assert restored == record
    assert restored.source_text not in restored.breadcrumb
    assert restored.source_spans[0].printed_page == "98"


def test_parsed_document_round_trip_preserves_pages_and_blocks() -> None:
    document = ParsedDocument(
        source_filename="x.pdf",
        source_sha256="b" * 64,
        extraction_method="fixture",
        pages=[
            ExtractedPage(
                pdf_page=1,
                blocks=[ExtractedBlock(raw_text="raw", pdf_page=1)],
            )
        ],
    )
    restored = ParsedDocument.model_validate(document.model_dump(mode="json"))
    assert restored.pages[0].blocks[0].raw_text == "raw"
    assert restored.flatten() == "raw"
