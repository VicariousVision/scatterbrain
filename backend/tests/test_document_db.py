"""Additive document metadata migration tests."""

import sqlite3
from datetime import datetime, timezone

from models.content import SCHEMA_VERSION
from models.document import DocumentRecord
from services.document_db import DocumentDB


def test_legacy_document_schema_migrates_idempotently(tmp_path) -> None:
    path = tmp_path / "legacy.db"
    with sqlite3.connect(path) as conn:
        conn.execute("CREATE TABLE documents (document_id TEXT PRIMARY KEY, filename TEXT NOT NULL, uploaded_at TEXT NOT NULL, status TEXT NOT NULL, error TEXT)")
        conn.execute("INSERT INTO documents VALUES ('old', 'old.txt', ?, 'completed', NULL)", (datetime.now(timezone.utc).isoformat(),))
    DocumentDB(str(path))
    db = DocumentDB(str(path))
    assert db.get("old").needs_reingestion is True

    record = DocumentRecord(
        document_id="new",
        filename="new.pdf",
        uploaded_at=datetime.now(timezone.utc),
        status="processing",
        source_sha256="a" * 64,
        schema_version=SCHEMA_VERSION,
    )
    db.create(record)
    db.update_status("new", "completed", chunk_count=3, document_title="Manual", needs_reingestion=False)
    loaded = db.get("new")
    assert loaded.chunk_count == 3
    assert loaded.document_title == "Manual"
    assert loaded.needs_reingestion is False
