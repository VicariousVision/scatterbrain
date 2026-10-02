"""Asynchronous orchestration for structured document ingestion.

Pipeline: hash bytes -> page-aware parse -> block cleaning -> legal/generic
routing -> parent/child chunking -> atomic vector-store replacement -> metadata
status update. Parsing/chunking run in worker threads and Ollama remains async.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import uuid
from datetime import datetime, timezone
from typing import List, Optional

from models.content import SCHEMA_VERSION
from models.document import DocumentRecord
from services.document_db import DocumentDB
from services.document_parser import DocumentParsingError, parse_document_structured
from services.legal_chunker import chunk_parsed_document
from services.text_cleaner import clean_parsed_document
from services.vector_store import VectorStore

logger = logging.getLogger(__name__)


class DocumentService:
    """Orchestrate upload, serialized replacement, and deletion.

    Parameters
    ----------
    vector_store:
        Atomic structured persistence and embedding service.
    document_db:
        Upload status/provenance store.
    """

    def __init__(self, vector_store: VectorStore, document_db: DocumentDB) -> None:
        self._vector_store = vector_store
        self._document_db = document_db
        self._processing_semaphore = asyncio.Semaphore(1)

    async def upload(self, filename: str, content: bytes) -> DocumentRecord:
        """Persist a processing record and start ingestion in the background."""
        document_id = str(uuid.uuid4())
        record = DocumentRecord(
            document_id=document_id,
            filename=filename,
            uploaded_at=datetime.now(tz=timezone.utc),
            status="processing",
            source_sha256=hashlib.sha256(content).hexdigest(),
            schema_version=SCHEMA_VERSION,
            needs_reingestion=False,
        )
        self._document_db.create(record)
        asyncio.create_task(
            self._process_document(document_id, filename, content),
            name=f"process-{document_id}",
        )
        return record

    def list_documents(self) -> List[DocumentRecord]:
        return self._document_db.list_all()

    def get_document(self, document_id: str) -> Optional[DocumentRecord]:
        return self._document_db.get(document_id)

    async def delete_document(self, document_id: str) -> None:
        """Serialize deletion with ingestion and remove vectors before status."""
        async with self._processing_semaphore:
            await self._vector_store.delete_document(document_id)
            self._document_db.delete(document_id)

    async def _process_document(
        self, document_id: str, filename: str, content: bytes
    ) -> None:
        """Run the complete structured pipeline and preserve prior data on failure."""
        try:
            async with self._processing_semaphore:
                parsed = await asyncio.to_thread(
                    parse_document_structured, filename, content
                )
                cleaned = await asyncio.to_thread(clean_parsed_document, parsed)
                records = await asyncio.to_thread(
                    chunk_parsed_document,
                    cleaned,
                    document_id=document_id,
                )
                children = [record for record in records if record.record_type == "child"]
                if not children:
                    raise RuntimeError("No searchable chunks produced from document")

                # VectorStore computes embeddings before its transaction and
                # replaces the filename atomically. It does not delete the old
                # ingestion if parsing/embedding/insertion fails.
                chunk_count = await self._vector_store.add_document(
                    document_id=document_id,
                    filename=filename,
                    chunks=records,
                )
                self._document_db.update_status(
                    document_id,
                    "completed",
                    chunk_count=chunk_count,
                    document_title=cleaned.document_title,
                    document_version=cleaned.document_version,
                    extraction_method=cleaned.extraction_method,
                    parser_version=cleaned.parser_version,
                    schema_version=cleaned.schema_version,
                    needs_reingestion=False,
                )
                # Superseded metadata is removed only after successful vector
                # replacement and completion of the new row.
                self._document_db.delete_by_filename_except(filename, document_id)
                logger.info(
                    "Structured ingestion completed for %s (%d children)",
                    filename,
                    chunk_count,
                )
        except DocumentParsingError as exc:
            self._update_status(
                document_id, "failed", error=f"Document parsing failed: {exc}"
            )
        except Exception as exc:
            logger.error(
                "RAG ingestion failed for document_id=%s: %s",
                document_id,
                exc,
                exc_info=True,
            )
            self._update_status(document_id, "failed", error=str(exc))

    def _update_status(
        self,
        document_id: str,
        status: str,
        error: Optional[str] = None,
        chunk_count: Optional[int] = None,
    ) -> None:
        self._document_db.update_status(
            document_id,
            status=status,
            error=error,
            chunk_count=chunk_count,
        )
