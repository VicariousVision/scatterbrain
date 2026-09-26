"""Document database service using SQLite for persistent storage.

Stores document metadata including upload status and error messages.
"""

from __future__ import annotations

import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

from backend.config import settings
from backend.models.document import DocumentRecord

logger = logging.getLogger(__name__)


class DocumentDB:
    """SQLite-based document metadata store.
    
    Provides CRUD operations for document records with automatic
    schema creation and connection management.
    """
    
    def __init__(self, db_path: Optional[str] = None) -> None:
        """Initialize document database.
        
        Args:
            db_path: Path to SQLite database file. Uses config setting if not provided.
        """
        self._db_path = db_path or settings.sqlite_db_path
        
        # Ensure parent directory exists
        Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize schema
        self._init_schema()
        
        logger.info("DocumentDB initialized at %s", self._db_path)
    
    def _get_connection(self) -> sqlite3.Connection:
        """Get a database connection with row factory."""
        conn = sqlite3.connect(self._db_path)
        conn.row_factory = sqlite3.Row
        return conn
    
    def _init_schema(self) -> None:
        """Create documents table if it doesn't exist."""
        conn = self._get_connection()
        try:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS documents (
                    document_id TEXT PRIMARY KEY,
                    filename TEXT NOT NULL,
                    uploaded_at TEXT NOT NULL,
                    status TEXT NOT NULL,
                    error TEXT,
                    chunk_count INTEGER DEFAULT 0
                )
            """)
            conn.commit()
            logger.info("Document schema initialized")
        finally:
            conn.close()
    
    def create(self, record: DocumentRecord) -> None:
        """Insert a new document record.
        
        Args:
            record: Document record to insert.
        """
        conn = self._get_connection()
        try:
            conn.execute(
                """
                INSERT INTO documents 
                (document_id, filename, uploaded_at, status, error, chunk_count)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    record.document_id,
                    record.filename,
                    record.uploaded_at.isoformat(),
                    record.status,
                    record.error,
                    0,
                ),
            )
            conn.commit()
            logger.info("Created document record: %s", record.document_id)
        finally:
            conn.close()
    
    def get(self, document_id: str) -> Optional[DocumentRecord]:
        """Retrieve a document record by ID.
        
        Args:
            document_id: Document ID to retrieve.
            
        Returns:
            DocumentRecord if found, None otherwise.
        """
        conn = self._get_connection()
        try:
            cursor = conn.execute(
                "SELECT * FROM documents WHERE document_id = ?",
                (document_id,),
            )
            row = cursor.fetchone()
            
            if row is None:
                return None
            
            return self._row_to_record(row)
        finally:
            conn.close()
    
    def list_all(self) -> List[DocumentRecord]:
        """List all document records ordered by upload time (newest first).
        
        Returns:
            List of all document records.
        """
        conn = self._get_connection()
        try:
            cursor = conn.execute(
                "SELECT * FROM documents ORDER BY uploaded_at DESC"
            )
            rows = cursor.fetchall()
            return [self._row_to_record(row) for row in rows]
        finally:
            conn.close()
    
    def update_status(
        self,
        document_id: str,
        status: str,
        error: Optional[str] = None,
        chunk_count: Optional[int] = None,
    ) -> None:
        """Update document status and optional error message.
        
        Args:
            document_id: Document ID to update.
            status: New status value.
            error: Optional error message.
            chunk_count: Optional number of chunks stored.
        """
        conn = self._get_connection()
        try:
            if chunk_count is not None:
                conn.execute(
                    """
                    UPDATE documents 
                    SET status = ?, error = ?, chunk_count = ?
                    WHERE document_id = ?
                    """,
                    (status, error, chunk_count, document_id),
                )
            else:
                conn.execute(
                    """
                    UPDATE documents 
                    SET status = ?, error = ?
                    WHERE document_id = ?
                    """,
                    (status, error, document_id),
                )
            conn.commit()
            logger.info(
                "Updated document %s: status=%s error=%s",
                document_id,
                status,
                error,
            )
        finally:
            conn.close()
    
    def delete(self, document_id: str) -> None:
        """Delete a document record.
        
        Args:
            document_id: Document ID to delete.
        """
        conn = self._get_connection()
        try:
            conn.execute(
                "DELETE FROM documents WHERE document_id = ?",
                (document_id,),
            )
            conn.commit()
            logger.info("Deleted document record: %s", document_id)
        finally:
            conn.close()
    
    def _row_to_record(self, row: sqlite3.Row) -> DocumentRecord:
        """Convert SQLite row to DocumentRecord.
        
        Args:
            row: SQLite row object.
            
        Returns:
            DocumentRecord instance.
        """
        return DocumentRecord(
            document_id=row["document_id"],
            filename=row["filename"],
            uploaded_at=datetime.fromisoformat(row["uploaded_at"]),
            status=row["status"],
            error=row["error"],
        )
