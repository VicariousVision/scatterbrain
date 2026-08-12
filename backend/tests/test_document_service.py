"""Unit tests for DocumentService.

Tests cover:
- upload() returns a DocumentRecord with status 'processing' and a valid UUID
- list_documents() returns all stored records
- get_document() returns the correct record or None for unknown IDs
- Background processing pipeline updates status to 'completed' on success
- Background processing pipeline updates status to 'failed' on parse error
- Re-upload of same filename triggers deletion of stale graph data
"""

from __future__ import annotations

import asyncio
import uuid
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.models.document import DocumentRecord
from backend.services.document_service import DocumentService


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_graph_service() -> MagicMock:
    """Return a mock GraphService with no-op methods."""
    gs = MagicMock()
    gs.delete_by_document_id = MagicMock()
    gs.count_entities_for_document = MagicMock(return_value=3)
    return gs


def _make_ollama_client() -> MagicMock:
    """Return a mock OllamaClient."""
    client = MagicMock()
    client.generate_embedding = AsyncMock(return_value=[0.1] * 768)
    return client


# ---------------------------------------------------------------------------
# upload() tests
# ---------------------------------------------------------------------------


def test_upload_returns_processing_record():
    """upload() should immediately return a DocumentRecord with status 'processing'."""
    service = DocumentService(
        graph_service=_make_graph_service(),
        ollama_client=_make_ollama_client(),
    )

    async def run():
        with patch(
            "backend.services.document_service.asyncio.create_task"
        ) as mock_create_task:
            record = await service.upload("contract.txt", b"hello world")
            mock_create_task.assert_called_once()
        return record

    record = asyncio.run(run())

    assert isinstance(record, DocumentRecord)
    assert record.status == "processing"
    assert record.filename == "contract.txt"
    assert isinstance(record.uploaded_at, datetime)
    parsed = uuid.UUID(record.document_id, version=4)
    assert str(parsed) == record.document_id


def test_upload_stores_record_in_memory():
    """upload() should persist the record so list_documents() can return it."""
    service = DocumentService(
        graph_service=_make_graph_service(),
        ollama_client=_make_ollama_client(),
    )

    async def run():
        with patch("backend.services.document_service.asyncio.create_task"):
            record = await service.upload("doc.pdf", b"%PDF-1.4")
        return record

    record = asyncio.run(run())

    assert service.get_document(record.document_id) is not None
    assert len(service.list_documents()) == 1


# ---------------------------------------------------------------------------
# list_documents() tests
# ---------------------------------------------------------------------------


def test_list_documents_empty():
    """list_documents() returns an empty list when no documents have been uploaded."""
    service = DocumentService(
        graph_service=_make_graph_service(),
        ollama_client=_make_ollama_client(),
    )
    assert service.list_documents() == []


def test_list_documents_multiple():
    """list_documents() returns all uploaded documents."""
    service = DocumentService(
        graph_service=_make_graph_service(),
        ollama_client=_make_ollama_client(),
    )

    async def run():
        with patch("backend.services.document_service.asyncio.create_task"):
            r1 = await service.upload("a.txt", b"aaa")
            r2 = await service.upload("b.txt", b"bbb")
        return r1, r2

    r1, r2 = asyncio.run(run())

    docs = service.list_documents()
    ids = {d.document_id for d in docs}
    assert r1.document_id in ids
    assert r2.document_id in ids
    assert len(docs) == 2


# ---------------------------------------------------------------------------
# get_document() tests
# ---------------------------------------------------------------------------


def test_get_document_returns_record():
    """get_document() returns the correct record for a known document_id."""
    service = DocumentService(
        graph_service=_make_graph_service(),
        ollama_client=_make_ollama_client(),
    )

    async def run():
        with patch("backend.services.document_service.asyncio.create_task"):
            return await service.upload("test.txt", b"content")

    record = asyncio.run(run())

    fetched = service.get_document(record.document_id)
    assert fetched is not None
    assert fetched.document_id == record.document_id
    assert fetched.filename == "test.txt"


def test_get_document_returns_none_for_unknown_id():
    """get_document() returns None for an unknown document_id."""
    service = DocumentService(
        graph_service=_make_graph_service(),
        ollama_client=_make_ollama_client(),
    )
    assert service.get_document("nonexistent-id") is None


# ---------------------------------------------------------------------------
# Background processing pipeline tests
# ---------------------------------------------------------------------------


def test_processing_pipeline_sets_completed_on_success():
    """Background pipeline should update status to 'completed' after successful processing."""
    graph_service = _make_graph_service()
    service = DocumentService(
        graph_service=graph_service,
        ollama_client=_make_ollama_client(),
    )

    async def run():
        with (
            patch(
                "backend.services.document_service.parse_document",
                return_value="parsed text",
            ),
            patch.object(
                service,
                "_run_kg_pipeline",
                new_callable=AsyncMock,
            ) as mock_kg_pipeline,
        ):
            record = await service.upload("contract.txt", b"hello")
            await asyncio.sleep(0.1)
        return record, mock_kg_pipeline

    record, mock_kg_pipeline = asyncio.run(run())

    updated = service.get_document(record.document_id)
    assert updated is not None
    assert updated.status == "completed"
    assert updated.error is None
    mock_kg_pipeline.assert_awaited_once()


def test_processing_pipeline_sets_failed_on_parse_error():
    """Background pipeline should update status to 'failed' when parsing raises an exception."""
    graph_service = _make_graph_service()
    service = DocumentService(
        graph_service=graph_service,
        ollama_client=_make_ollama_client(),
    )

    async def run():
        with patch(
            "backend.services.document_service.parse_document",
            side_effect=ValueError("Unsupported file type"),
        ):
            record = await service.upload("bad.xyz", b"garbage")
            await asyncio.sleep(0.1)
        return record

    record = asyncio.run(run())

    updated = service.get_document(record.document_id)
    assert updated is not None
    assert updated.status == "failed"
    assert updated.error is not None
    assert "Unsupported file type" in updated.error


def test_processing_pipeline_sets_failed_when_no_entities_extracted():
    """Background pipeline should fail when the KG pipeline writes no entities."""
    graph_service = _make_graph_service()
    graph_service.count_entities_for_document = MagicMock(return_value=0)
    service = DocumentService(
        graph_service=graph_service,
        ollama_client=_make_ollama_client(),
    )

    async def run():
        with (
            patch(
                "backend.services.document_service.parse_document",
                return_value="parsed text",
            ),
            patch(
                "backend.services.document_service.SimpleKGPipeline",
            ) as mock_pipeline_cls,
        ):
            mock_pipeline = mock_pipeline_cls.return_value
            mock_pipeline.run_async = AsyncMock(return_value=MagicMock())
            record = await service.upload("contract.txt", b"hello")
            await asyncio.sleep(0.1)
        return record

    record = asyncio.run(run())

    updated = service.get_document(record.document_id)
    assert updated is not None
    assert updated.status == "failed"
    assert updated.error is not None
    assert "no entities were extracted" in updated.error.lower()


# ---------------------------------------------------------------------------
# Re-upload / stale data deletion tests
# ---------------------------------------------------------------------------


def test_reupload_deletes_stale_graph_data():
    """Re-uploading a file with the same filename should delete the old document's graph data."""
    graph_service = _make_graph_service()
    service = DocumentService(
        graph_service=graph_service,
        ollama_client=_make_ollama_client(),
    )

    old_id = str(uuid.uuid4())
    service._store[old_id] = DocumentRecord(
        document_id=old_id,
        filename="contract.txt",
        uploaded_at=datetime.now(tz=timezone.utc),
        status="completed",
    )

    async def run():
        with (
            patch(
                "backend.services.document_service.parse_document",
                return_value="new text",
            ),
            patch.object(
                service,
                "_run_kg_pipeline",
                new_callable=AsyncMock,
            ),
        ):
            new_record = await service.upload("contract.txt", b"new content")
            await asyncio.sleep(0.1)
        return new_record

    new_record = asyncio.run(run())

    graph_service.delete_by_document_id.assert_called_once_with(old_id)
    updated = service.get_document(new_record.document_id)
    assert updated is not None
    assert updated.status == "completed"
