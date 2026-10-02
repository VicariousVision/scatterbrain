"""Structured ingestion orchestration tests without external services."""

import asyncio
from unittest.mock import AsyncMock, MagicMock

from services.document_service import DocumentService


def test_txt_ingestion_passes_structured_records_and_metadata() -> None:
    vector = MagicMock()
    vector.add_document = AsyncMock(return_value=1)
    db = MagicMock()
    service = DocumentService(vector, db)
    asyncio.run(service._process_document("doc", "notes.txt", b"ordinary local text"))
    records = vector.add_document.await_args.kwargs["chunks"]
    assert any(record.record_type == "parent" for record in records)
    assert any(record.record_type == "child" for record in records)
    db.update_status.assert_called()
    db.delete_by_filename_except.assert_called_once_with("notes.txt", "doc")


def test_failed_replacement_does_not_remove_prior_metadata() -> None:
    vector = MagicMock()
    vector.add_document = AsyncMock(side_effect=RuntimeError("embed failed"))
    db = MagicMock()
    service = DocumentService(vector, db)
    asyncio.run(service._process_document("doc", "notes.txt", b"ordinary local text"))
    db.delete_by_filename_except.assert_not_called()
    assert db.update_status.call_args.kwargs["status"] == "failed"
