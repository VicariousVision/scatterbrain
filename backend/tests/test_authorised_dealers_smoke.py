"""Smoke harness orchestration/cleanup test with fake local clients."""

import asyncio
from pathlib import Path

from models.content import ExtractedBlock, ExtractedPage, ParsedDocument, SourceSpan
from tools import authorised_dealers_smoke as smoke


class FakeProvider:
    provider_name = "fake"
    model_name = "fake"
    embedding_dimension = 3

    async def generate_embedding(self, text):
        return [1.0, 0.0, 0.0]

    async def generate_query_embedding(self, text):
        return [1.0, 0.0, 0.0]

    async def generate_embeddings(self, texts):
        return [[1.0, 0.0, 0.0] for _ in texts]


class FakeChat:
    async def chat(self, messages, think=None):
        return "The allowance is R2 million; transfers above it require verification and proof."


def _document() -> ParsedDocument:
    text = (
        "B.4  Single discretionary allowance\n"
        "(A)  Per calendar year\n"
        "(i) Residents aged 18 or older have an overall annual R2 million allowance. "
        "Current transfers above that amount require verification and proof."
    )
    span = SourceSpan(pdf_page=98, printed_page="98", page_revision="6/2026")
    block = ExtractedBlock(
        raw_text=text,
        clean_text=text,
        pdf_page=98,
        printed_page="98",
        page_revision="6/2026",
        section_marker="B.4",
        source_spans=[span],
        extraction_method="fixture",
    )
    return ParsedDocument(
        source_filename="manual.pdf",
        source_sha256="a" * 64,
        document_title="Currency and Exchanges Manual for Authorised Dealers",
        extraction_method="fixture",
        pages=[
            ExtractedPage(
                pdf_page=98,
                printed_page="98",
                page_revision="6/2026",
                section_marker="B.4",
                blocks=[block],
            )
        ],
        is_legal=True,
    )


def test_smoke_uses_temp_db_writes_report_and_cleans_up(tmp_path, monkeypatch) -> None:
    pdf = tmp_path / "manual.pdf"
    pdf.write_bytes(b"fixture")
    report = tmp_path / "report.md"
    monkeypatch.setattr(smoke.settings, "sqlite_db_path", str(tmp_path / "user.db"))
    monkeypatch.setattr(smoke.settings, "ollama_base_url", "http://localhost:11434")
    monkeypatch.setattr(smoke.settings, "embedding_provider", "ollama")
    monkeypatch.setattr(smoke.settings, "ollama_model", smoke.EXPECTED_CHAT_MODEL)
    monkeypatch.setattr(smoke.settings, "ollama_embedding_model", smoke.EXPECTED_EMBEDDING_MODEL)
    monkeypatch.setattr(smoke.settings, "ollama_num_gpu", 0)
    monkeypatch.setattr(smoke.settings, "embedding_dimension", 768)
    monkeypatch.setattr(smoke, "parse_document_structured", lambda *args, **kwargs: _document())
    monkeypatch.setattr(smoke, "clean_parsed_document", lambda value: value)

    evidence = asyncio.run(
        smoke.run_smoke(
            pdf,
            report,
            ollama_client=FakeChat(),
            embedding_provider=FakeProvider(),
            preflight=False,
        )
    )

    assert report.is_file()
    assert evidence["temporary_directory_removed"] is True
    assert evidence["user_db_before"] == evidence["user_db_after"]
    assert "B.4(A)(i)" in report.read_text(encoding="utf-8")
