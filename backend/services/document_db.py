"""SQLite document-status and source-provenance store.

The synchronous metadata store shares the same SQLite file as the async vector
connection. Initialization only adds missing columns; existing completed flat
ingestions are marked as requiring re-ingestion for structured provenance.
"""

from __future__ import annotations

import logging
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from config import settings
from models.content import SCHEMA_VERSION
from models.document import DocumentRecord

logger = logging.getLogger(__name__)

_DOCUMENT_COLUMNS: dict[str, str] = {
    "chunk_count": "INTEGER DEFAULT 0",
    "source_sha256": "TEXT",
    "document_title": "TEXT",
    "document_version": "TEXT",
    "extraction_method": "TEXT",
    "parser_version": "TEXT",
    "schema_version": "INTEGER NOT NULL DEFAULT 1",
    "needs_reingestion": "INTEGER NOT NULL DEFAULT 1",
}


class DocumentDB:
    """Persistent document status/provenance records.

    Parameters
    ----------
    db_path:
        Optional explicit path; defaults to ``SQLITE_DB_PATH``.
    """

    def __init__(self, db_path: Optional[str] = None) -> None:
        self._db_path = db_path or settings.sqlite_db_path
        Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    @property
    def db_path(self) -> str:
        return self._db_path

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path, timeout=10)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_schema(self) -> None:
        """Create or additively migrate the document metadata table."""
        with self._get_connection() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS documents (
                    document_id TEXT PRIMARY KEY,
                    filename TEXT NOT NULL,
                    uploaded_at TEXT NOT NULL,
                    status TEXT NOT NULL,
                    error TEXT,
                    chunk_count INTEGER DEFAULT 0,
                    source_sha256 TEXT,
                    document_title TEXT,
                    document_version TEXT,
                    extraction_method TEXT,
                    parser_version TEXT,
                    schema_version INTEGER NOT NULL DEFAULT 1,
                    needs_reingestion INTEGER NOT NULL DEFAULT 1
                )
                """
            )
            present = {
                row[1] for row in conn.execute("PRAGMA table_info(documents)").fetchall()
            }
            for name, declaration in _DOCUMENT_COLUMNS.items():
                if name not in present:
                    conn.execute(
                        f'ALTER TABLE documents ADD COLUMN "{name}" {declaration}'
                    )
            conn.execute(
                """
                UPDATE documents
                SET needs_reingestion = 1
                WHERE status = 'completed' AND source_sha256 IS NULL
                """
            )
            conn.commit()

    def create(self, record: DocumentRecord) -> None:
        """Insert a processing/status record with source metadata."""
        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT INTO documents (
                    document_id, filename, uploaded_at, status, error,
                    chunk_count, source_sha256, document_title,
                    document_version, extraction_method, parser_version,
                    schema_version, needs_reingestion
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.document_id,
                    record.filename,
                    record.uploaded_at.isoformat(),
                    record.status,
                    record.error,
                    record.chunk_count,
                    record.source_sha256,
                    record.document_title,
                    record.document_version,
                    record.extraction_method,
                    record.parser_version,
                    record.schema_version,
                    int(record.needs_reingestion),
                ),
            )
            conn.commit()

    def get(self, document_id: str) -> Optional[DocumentRecord]:
        with self._get_connection() as conn:
            row = conn.execute(
                "SELECT * FROM documents WHERE document_id = ?", (document_id,)
            ).fetchone()
            return self._row_to_record(row) if row else None

    def get_by_filename(self, filename: str) -> List[DocumentRecord]:
        """Return records sharing a filename, newest first."""
        with self._get_connection() as conn:
            rows = conn.execute(
                "SELECT * FROM documents WHERE filename = ? ORDER BY uploaded_at DESC",
                (filename,),
            ).fetchall()
            return [self._row_to_record(row) for row in rows]

    def list_all(self) -> List[DocumentRecord]:
        with self._get_connection() as conn:
            rows = conn.execute(
                "SELECT * FROM documents ORDER BY uploaded_at DESC"
            ).fetchall()
            return [self._row_to_record(row) for row in rows]

    def update_status(
        self,
        document_id: str,
        status: str,
        error: Optional[str] = None,
        chunk_count: Optional[int] = None,
        *,
        document_title: str | None = None,
        document_version: str | None = None,
        extraction_method: str | None = None,
        parser_version: str | None = None,
        schema_version: int | None = None,
        needs_reingestion: bool | None = None,
    ) -> None:
        """Update status and optional detected ingestion metadata."""
        assignments = ["status = ?", "error = ?"]
        values: list[object] = [status, error]
        optional = {
            "chunk_count": chunk_count,
            "document_title": document_title,
            "document_version": document_version,
            "extraction_method": extraction_method,
            "parser_version": parser_version,
            "schema_version": schema_version,
            "needs_reingestion": (
                int(needs_reingestion) if needs_reingestion is not None else None
            ),
        }
        for name, value in optional.items():
            if value is not None:
                assignments.append(f"{name} = ?")
                values.append(value)
        values.append(document_id)
        with self._get_connection() as conn:
            conn.execute(
                f"UPDATE documents SET {', '.join(assignments)} WHERE document_id = ?",
                values,
            )
            conn.commit()

    def fail_stale_processing(
        self,
        error: str = "Ingestion did not finish before the server stopped.",
    ) -> int:
        with self._get_connection() as conn:
            cursor = conn.execute(
                "UPDATE documents SET status = 'failed', error = ? "
                "WHERE status = 'processing'",
                (error,),
            )
            conn.commit()
            return cursor.rowcount or 0

    def delete(self, document_id: str) -> None:
        with self._get_connection() as conn:
            conn.execute("DELETE FROM documents WHERE document_id = ?", (document_id,))
            conn.commit()

    def delete_by_filename_except(self, filename: str, document_id: str) -> int:
        """Remove superseded metadata only after vector replacement succeeds."""
        with self._get_connection() as conn:
            cursor = conn.execute(
                "DELETE FROM documents WHERE filename = ? AND document_id != ?",
                (filename, document_id),
            )
            conn.commit()
            return cursor.rowcount or 0

    def _row_to_record(self, row: sqlite3.Row) -> DocumentRecord:
        keys = set(row.keys())
        return DocumentRecord(
            document_id=row["document_id"],
            filename=row["filename"],
            uploaded_at=datetime.fromisoformat(row["uploaded_at"]),
            status=row["status"],
            error=row["error"],
            chunk_count=int(row["chunk_count"] or 0) if "chunk_count" in keys else 0,
            source_sha256=row["source_sha256"] if "source_sha256" in keys else None,
            document_title=row["document_title"] if "document_title" in keys else None,
            document_version=row["document_version"] if "document_version" in keys else None,
            extraction_method=row["extraction_method"] if "extraction_method" in keys else None,
            parser_version=row["parser_version"] if "parser_version" in keys else None,
            schema_version=int(row["schema_version"] or 1) if "schema_version" in keys else 1,
            needs_reingestion=bool(row["needs_reingestion"])
            if "needs_reingestion" in keys
            else True,
        )
