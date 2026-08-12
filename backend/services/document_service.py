"""Document service orchestrating the full document processing pipeline.

Manages document upload, background processing, and status tracking using an
in-memory store keyed by ``document_id``.

Processing pipeline (run as a background asyncio task):
  1. Parse document text (``document_parser``)
  2. Build knowledge graph via neo4j-graphrag ``SimpleKGPipeline``
     - Splits text into chunks (FixedSizeSplitter)
     - Embeds each chunk (OllamaEmbedderAdapter → nomic-embed-text)
     - Extracts entities + relationships (OllamaLLMAdapter → mistral)
     - Writes Document, Chunk, and Entity nodes + edges to Neo4j
     - Runs entity resolution (SinglePropertyExactMatchResolver)
  3. Update document status to ``completed`` or ``failed``

Memory notes (GTX 960 / 2 GB VRAM):
  - Ollama is configured with ``num_gpu=0`` (CPU-only). Generation is slower
    (~1–4 tok/s on a modern CPU) but stable — no VRAM is consumed by the LLM.
  - Chunk size is kept small (300 tokens, 25 overlap) so each LLM call receives
    a short prompt, reducing peak RAM usage and avoiding context-length errors.
  - Documents are processed one at a time (sequential asyncio tasks). FastAPI
    may accept concurrent uploads, but each ``_process_document`` task awaits
    the full KG pipeline before the next one starts, preventing parallel Ollama
    calls from stacking up in memory.

Requirements: 1.4, 1.5, 1.6, 1.7, 4.4, 7.1, 7.3
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from datetime import datetime, timezone
from typing import Dict, List, Optional

from neo4j import GraphDatabase
from neo4j_graphrag.experimental.components.text_splitters.fixed_size_splitter import (
    FixedSizeSplitter,
)
from neo4j_graphrag.experimental.components.types import DocumentInfo
from neo4j_graphrag.experimental.pipeline.kg_builder import SimpleKGPipeline

from backend.config import settings
from backend.models.document import DocumentRecord
from backend.services.document_parser import parse_document
from backend.services.graph_service import GraphService
from backend.services.ollama_adapters import OllamaEmbedderAdapter, OllamaLLMAdapter
from backend.services.ollama_client import OllamaClient

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# SimpleKGPipeline Configuration
# ---------------------------------------------------------------------------


class DocumentService:
    """Orchestrates document upload and the asynchronous processing pipeline.

    Maintains an in-memory dictionary of :class:`~backend.models.document.DocumentRecord`
    objects keyed by ``document_id``.  This store resets on server restart.

    Parameters
    ----------
    graph_service:
        A :class:`~backend.services.graph_service.GraphService` instance used
        to delete stale graph data and provide graph summaries.
    ollama_client:
        An :class:`~backend.services.ollama_client.OllamaClient` instance used
        for LLM generation and embeddings.
    """

    def __init__(
        self,
        graph_service: GraphService,
        ollama_client: OllamaClient,
    ) -> None:
        self._graph_service = graph_service
        self._ollama_client = ollama_client
        self._store: Dict[str, DocumentRecord] = {}
        # Limit to one active KG pipeline at a time. Parallel pipelines would
        # issue concurrent Ollama requests, multiplying RAM/VRAM usage and
        # causing OOM errors on memory-constrained hardware (e.g. GTX 960 2 GB).
        self._processing_semaphore = asyncio.Semaphore(1)

        # Build neo4j-graphrag adapters once.
        self._llm_adapter = OllamaLLMAdapter(
            ollama_client=ollama_client,
            model_name=settings.ollama_model,
        )
        self._embedder_adapter = OllamaEmbedderAdapter(ollama_client=ollama_client)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def upload(self, filename: str, content: bytes) -> DocumentRecord:
        """Accept an uploaded file, create a tracking record, and start processing.

        Requirements: 1.4
        """
        document_id = str(uuid.uuid4())
        record = DocumentRecord(
            document_id=document_id,
            filename=filename,
            uploaded_at=datetime.now(tz=timezone.utc),
            status="processing",
        )
        self._store[document_id] = record
        logger.info("Document uploaded: document_id=%s filename=%s", document_id, filename)

        asyncio.create_task(
            self._process_document(document_id, filename, content),
            name=f"process-{document_id}",
        )
        return record

    def list_documents(self) -> List[DocumentRecord]:
        """Return all document records in the in-memory store.

        Requirements: 7.1, 7.3
        """
        return list(self._store.values())

    def get_document(self, document_id: str) -> Optional[DocumentRecord]:
        """Return the record for a single document, or ``None`` if not found.

        Requirements: 7.3
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
          2. Delete stale graph data for the same filename (re-upload case).
          3. Run SimpleKGPipeline (chunk → embed → extract → write → resolve).
          4. Update status to ``completed`` or ``failed``.

        Requirements: 1.5, 1.6, 1.7, 4.4
        """
        logger.info("Starting processing pipeline for document_id=%s", document_id)
        try:
            # Step 1: Parse document text.
            logger.debug("Parsing document: document_id=%s filename=%s", document_id, filename)
            text = parse_document(filename, content)

            # Step 2: Delete stale graph data for the same filename.
            await self._delete_stale_graph_data(filename, document_id)

            # Step 3: Build knowledge graph via SimpleKGPipeline.
            # Semaphore ensures only one pipeline runs at a time — prevents
            # concurrent Ollama calls from stacking up in memory.
            async with self._processing_semaphore:
                logger.debug(
                    "Acquired processing semaphore for document_id=%s", document_id
                )
                await self._run_kg_pipeline(text, document_id, filename)

            # Step 4: Mark as completed.
            self._update_status(document_id, "completed")
            logger.info("Processing completed for document_id=%s", document_id)

        except Exception as exc:  # noqa: BLE001
            error_message = str(exc)
            logger.error(
                "Processing failed for document_id=%s: %s",
                document_id,
                error_message,
                exc_info=True,
            )
            self._update_status(document_id, "failed", error=error_message)

    async def _run_kg_pipeline(
        self, text: str, document_id: str, filename: str
    ) -> None:
        """Build the knowledge graph for one document using SimpleKGPipeline.

        SimpleKGPipeline requires a *sync* Neo4j driver internally (it manages
        its own session lifecycle), so we open a dedicated driver here rather
        than reusing the one held by GraphService (which is also sync but
        already managed by that service).
        """
        logger.debug("Running SimpleKGPipeline for document_id=%s", document_id)

        # SimpleKGPipeline works with a synchronous neo4j driver.
        driver = GraphDatabase.driver(
            settings.neo4j_uri,
            auth=(settings.neo4j_username, settings.neo4j_password),
        )
        try:
            # Small chunks keep each LLM call's prompt short, reducing peak RAM.
            # chunk_size is in characters (FixedSizeSplitter default unit).
            text_splitter = FixedSizeSplitter(chunk_size=300, chunk_overlap=25)

            kg_pipeline = SimpleKGPipeline(
                llm=self._llm_adapter,
                driver=driver,
                embedder=self._embedder_adapter,
                text_splitter=text_splitter,
                from_file=False,
                schema="EXTRACTED",
                perform_entity_resolution=True,
                on_error="RAISE",
            )

            result = await kg_pipeline.run_async(
                text=text,
                file_path=filename,
                document_metadata={
                    "document_id": document_id,
                    "filename": filename,
                },
            )
            logger.info(
                "SimpleKGPipeline finished for document_id=%s: %s",
                document_id,
                result,
            )

            entity_count = self._graph_service.count_entities_for_document(document_id)
            if entity_count == 0:
                raise RuntimeError(
                    "Knowledge graph pipeline completed but no entities were "
                    "extracted. Check that Ollama is running and the LLM "
                    "returned valid extraction JSON."
                )
            logger.info(
                "Knowledge graph written for document_id=%s: %d entities",
                document_id,
                entity_count,
            )
        finally:
            driver.close()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    async def _delete_stale_graph_data(
        self, filename: str, current_document_id: str
    ) -> None:
        """Delete graph data for any previous upload of the same filename.

        Requirements: 4.4
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
        record = self._store.get(document_id)
        if record is None:
            logger.warning(
                "Attempted to update status for unknown document_id=%s", document_id
            )
            return
        self._store[document_id] = record.model_copy(
            update={"status": status, "error": error}
        )
        logger.debug(
            "Updated document_id=%s status=%s error=%s", document_id, status, error
        )
