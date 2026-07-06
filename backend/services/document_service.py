"""Document service orchestrating the full document processing pipeline.

Manages document upload, background processing, and status tracking using an
in-memory store keyed by ``document_id``.

Processing pipeline (run as a background asyncio task):
  1. Parse document text (``document_parser``)
  2. Extract entities and relationships (``entity_extractor``)
  3. Delete existing graph data for the same filename (re-upload case)
  4. Store entities and relationships in Neo4j (``graph_service``)
  5. Update document status to ``completed`` or ``failed``

Requirements: 1.4, 1.5, 1.6, 1.7, 4.4, 7.1, 7.3
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from datetime import datetime, timezone
from typing import Dict, List, Optional

from backend.models.document import DocumentRecord
from backend.services.document_parser import parse_document
from backend.services.entity_extractor import extract_entities
from backend.services.graph_service import GraphService
from backend.services.ollama_client import OllamaClient
from backend.services.text_chunker import chunk_text

logger = logging.getLogger(__name__)


class DocumentService:
    """Orchestrates document upload and the asynchronous processing pipeline.

    Maintains an in-memory dictionary of :class:`~backend.models.document.DocumentRecord`
    objects keyed by ``document_id``.  This store resets on server restart; a
    SQLite-backed store can be substituted without changing the public interface.

    Parameters
    ----------
    graph_service:
        A :class:`~backend.services.graph_service.GraphService` instance used
        to delete stale graph data and persist newly extracted entities and
        relationships.
    ollama_client:
        An :class:`~backend.services.ollama_client.OllamaClient` instance used
        by the entity extractor to call the LLM.
    """

    def __init__(
        self,
        graph_service: GraphService,
        ollama_client: OllamaClient,
    ) -> None:
        self._graph_service = graph_service
        self._ollama_client = ollama_client
        # In-memory store: document_id → DocumentRecord
        self._store: Dict[str, DocumentRecord] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def upload(self, filename: str, content: bytes) -> DocumentRecord:
        """Accept an uploaded file, create a tracking record, and start processing.

        Generates a UUID4 ``document_id``, stores a :class:`DocumentRecord`
        with status ``processing``, dispatches the background processing task,
        and returns the record immediately (202 Accepted pattern).

        Requirements: 1.4

        Parameters
        ----------
        filename:
            Original filename including extension (e.g. ``contract.pdf``).
        content:
            Raw file bytes.

        Returns
        -------
        DocumentRecord
            The newly created record with status ``processing``.
        """
        document_id = str(uuid.uuid4())
        record = DocumentRecord(
            document_id=document_id,
            filename=filename,
            uploaded_at=datetime.now(tz=timezone.utc),
            status="processing",
        )
        self._store[document_id] = record
        logger.info(
            "Document uploaded: document_id=%s filename=%s", document_id, filename
        )

        # Dispatch background processing without blocking the caller.
        asyncio.create_task(
            self._process_document(document_id, filename, content),
            name=f"process-{document_id}",
        )

        return record

    def list_documents(self) -> List[DocumentRecord]:
        """Return all document records in the in-memory store.

        Requirements: 7.1, 7.3

        Returns
        -------
        List[DocumentRecord]
            All tracked documents, in insertion order (Python 3.7+ dict ordering).
        """
        return list(self._store.values())

    def get_document(self, document_id: str) -> Optional[DocumentRecord]:
        """Return the record for a single document, or ``None`` if not found.

        Requirements: 7.3

        Parameters
        ----------
        document_id:
            The UUID of the document to retrieve.

        Returns
        -------
        DocumentRecord or None
            The matching record, or ``None`` if ``document_id`` is unknown.
        """
        return self._store.get(document_id)

    # ------------------------------------------------------------------
    # Background processing pipeline
    # ------------------------------------------------------------------

    async def _process_document(
        self, document_id: str, filename: str, content: bytes
    ) -> None:
        """Run the full processing pipeline for an uploaded document.

        Steps:
          1. Parse document text.
          2. Extract entities and relationships via LLM.
          3. Delete existing graph data for the same filename (re-upload case).
          4. Store entities and relationships in Neo4j.
          5. Update status to ``completed`` or ``failed``.

        Requirements: 1.5, 1.6, 1.7, 4.4

        Parameters
        ----------
        document_id:
            UUID of the document being processed.
        filename:
            Original filename (used to find and delete stale graph data).
        content:
            Raw file bytes to parse.
        """
        logger.info("Starting processing pipeline for document_id=%s", document_id)
        try:
            # Step 1: Parse document text.
            logger.debug("Parsing document: document_id=%s filename=%s", document_id, filename)
            text = parse_document(filename, content)

            # Step 1.5: Chunk text and generate embeddings.
            logger.debug("Chunking document: document_id=%s", document_id)
            chunks = chunk_text(text)
            logger.debug("Generating embeddings for %d chunks: document_id=%s", len(chunks), document_id)
            chunk_records = []
            for i, chunk_text_content in enumerate(chunks):
                embedding = await self._ollama_client.generate_embedding(chunk_text_content)
                chunk_records.append({
                    "id": f"{document_id}-chunk-{i}",
                    "text": chunk_text_content,
                    "document_id": document_id,
                    "embedding": embedding
                })

            # Step 2: Extract entities and relationships via LLM.
            logger.debug("Extracting entities: document_id=%s", document_id)
            extraction_result = await extract_entities(
                text=text,
                document_id=document_id,
                client=self._ollama_client,
            )

            # Step 3: Delete existing graph data for the same filename.
            # Find any previous document_id(s) that share the same filename.
            logger.debug(
                "Deleting stale graph data for filename=%s (new document_id=%s)",
                filename,
                document_id,
            )
            await self._delete_stale_graph_data(filename, document_id)

            # Step 4: Store entities, relationships, and chunks in Neo4j.
            logger.debug(
                "Storing %d entities, %d relationships, and %d chunks for document_id=%s",
                len(extraction_result.entities),
                len(extraction_result.relationships),
                len(chunk_records),
                document_id,
            )
            self._graph_service.store_entities(extraction_result.entities)
            self._graph_service.store_relationships(extraction_result.relationships)
            self._graph_service.store_chunks(chunk_records)
            self._graph_service.link_chunks_to_entities(document_id)

            # Step 5: Update status to completed.
            self._update_status(document_id, "completed")
            logger.info(
                "Processing completed for document_id=%s (%d entities, %d relationships, %d chunks)",
                document_id,
                len(extraction_result.entities),
                len(extraction_result.relationships),
                len(chunk_records),
            )

        except Exception as exc:  # noqa: BLE001
            error_message = str(exc)
            logger.error(
                "Processing failed for document_id=%s: %s",
                document_id,
                error_message,
                exc_info=True,
            )
            self._update_status(document_id, "failed", error=error_message)

    async def _delete_stale_graph_data(
        self, filename: str, current_document_id: str
    ) -> None:
        """Delete graph data for any previous upload of the same filename.

        Iterates the in-memory store to find records with the same filename
        but a different ``document_id``, then calls
        :meth:`~backend.services.graph_service.GraphService.delete_by_document_id`
        for each one.

        Requirements: 4.4

        Parameters
        ----------
        filename:
            The filename to match against existing records.
        current_document_id:
            The ``document_id`` of the current upload (excluded from deletion).
        """
        stale_ids = [
            record.document_id
            for record in self._store.values()
            if record.filename == filename
            and record.document_id != current_document_id
        ]
        for stale_id in stale_ids:
            logger.info(
                "Deleting stale graph data for document_id=%s (filename=%s)",
                stale_id,
                filename,
            )
            self._graph_service.delete_by_document_id(stale_id)

    def _update_status(
        self,
        document_id: str,
        status: str,
        error: Optional[str] = None,
    ) -> None:
        """Update the status (and optional error) of a document record in-place.

        Parameters
        ----------
        document_id:
            The UUID of the document to update.
        status:
            New status value: ``"completed"`` or ``"failed"``.
        error:
            Optional error description (set when status is ``"failed"``).
        """
        record = self._store.get(document_id)
        if record is None:
            logger.warning(
                "Attempted to update status for unknown document_id=%s", document_id
            )
            return

        # Pydantic v2 models are immutable by default; use model_copy to update.
        self._store[document_id] = record.model_copy(
            update={"status": status, "error": error}
        )
        logger.debug(
            "Updated document_id=%s status=%s error=%s", document_id, status, error
        )
