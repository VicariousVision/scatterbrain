"""Document service orchestrating the full ingestion pipeline.

Processing pipeline (run as a background asyncio task):
  1. Parse document text (document_parser)
  2. Chunk on provision boundaries (provision_chunker)
  3. Write structural Provision/Definition nodes to Neo4j (Neo4jLoader)
  4. Extract entities/relationships per chunk via Mistral 7B (extraction_service)
  5. Write extraction edges to Neo4j (Neo4jLoader.apply_extraction)
  6. Update document status to completed or failed

Key design decisions:
  - neo4j-graphrag's SimpleKGPipeline and SchemaFromTextExtractor are NOT used.
    They were replaced because:
      (a) The library's built-in extractor uses a generic prompt, not the
          domain-specific SARB prompt required here.
      (b) The library's pipeline orchestrator fires embedding and LLM calls
          concurrently via asyncio.gather, which caused ConnectTimeout failures
          against a CPU-only Ollama server that cannot serve two model requests
          simultaneously.
    All concurrency is now controlled explicitly via asyncio.Semaphore in
    ollama_client.py (Ollama serialisation) and extraction_service.py
    (_EXTRACTION_CONCURRENCY task cap).
  - No embeddings, no vector index.  Retrieval is handled by
    Text2CypherRetriever in graph_query_service.py.

Requirements: 1.4, 1.5, 1.6, 1.7, 4.4, 7.1, 7.3
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from datetime import datetime, timezone
from typing import Dict, List, Optional

from neo4j import GraphDatabase

from backend.config import settings
from backend.models.document import DocumentRecord
from backend.services.document_parser import parse_document
from backend.services.extraction_service import Neo4jLoader, extract_and_load_all
from backend.services.graph_schema_service import GraphSchemaService
from backend.services.graph_service import GraphService
from backend.services.ollama_client import OllamaClient
from backend.services.provision_chunker import parse_provisions

logger = logging.getLogger(__name__)


class DocumentService:
    """Orchestrates document upload and the asynchronous ingestion pipeline.

    Parameters
    ----------
    graph_service:
        Used for Neo4j connectivity verification and stale-data cleanup.
    ollama_client:
        Used for Mistral 7B entity extraction calls.
    """

    def __init__(
        self,
        graph_service: GraphService,
        ollama_client: OllamaClient,
    ) -> None:
        self._graph_service = graph_service
        self._ollama_client = ollama_client
        self._store: Dict[str, DocumentRecord] = {}
        # One active ingestion at a time.  Parallel ingestions would queue
        # many extraction tasks, all blocked at the Ollama semaphore, wasting
        # RAM holding prompt strings in-flight.
        self._processing_semaphore = asyncio.Semaphore(1)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def upload(self, filename: str, content: bytes) -> DocumentRecord:
        """Accept an uploaded file, create a tracking record, and start processing."""
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
        """Return all document records."""
        return list(self._store.values())

    def get_document(self, document_id: str) -> Optional[DocumentRecord]:
        """Return the record for a single document, or None if not found."""
        return self._store.get(document_id)

    # ------------------------------------------------------------------
    # Background processing pipeline
    # ------------------------------------------------------------------

    async def _process_document(
        self, document_id: str, filename: str, content: bytes
    ) -> None:
        """Run the full ingestion pipeline for an uploaded document."""
        logger.info("Starting ingestion pipeline for document_id=%s", document_id)
        try:
            # Step 1: Parse raw text.
            text = parse_document(filename, content)
            logger.info(
                "Parsed document_id=%s: %d characters.", document_id, len(text)
            )

            # Step 2: Delete stale graph data for the same filename.
            await self._delete_stale_graph_data(filename, document_id)

            # Step 3–5: Chunk → extract → load.
            async with self._processing_semaphore:
                await self._run_ingestion_pipeline(text, document_id, filename)

            self._update_status(document_id, "completed")
            logger.info("Ingestion completed for document_id=%s", document_id)

        except Exception as exc:  # noqa: BLE001
            error_message = str(exc)
            logger.error(
                "Ingestion failed for document_id=%s: %s",
                document_id,
                error_message,
                exc_info=True,
            )
            self._update_status(document_id, "failed", error=error_message)

    async def _run_ingestion_pipeline(
        self, text: str, document_id: str, filename: str
    ) -> None:
        """Chunk, extract, and write the knowledge graph for one document."""
        # Step 2: Chunk on provision boundaries.
        provisions, definitions = parse_provisions(text)
        logger.info(
            "Chunked document_id=%s: %d provisions, %d definitions.",
            document_id,
            len(provisions),
            len(definitions),
        )

        if not provisions:
            raise RuntimeError(
                "Provision chunker produced zero provision chunks.  "
                "Check that the document is the SARB Currency and Exchanges Manual "
                "and that the PDF was not blank or a scanned image."
            )

        # Determine manual name from the filename (strip extension).
        import os
        manual_name = os.path.splitext(filename)[0]

        # Open a sync Neo4j driver for the loader (schema service also uses sync).
        driver = GraphDatabase.driver(
            settings.neo4j_uri,
            auth=(settings.neo4j_username, settings.neo4j_password),
        )
        try:
            # Ensure schema/constraints are applied (idempotent).
            schema_svc = GraphSchemaService(driver)
            schema_svc.create_schema()

            loader = Neo4jLoader(
                driver=driver,
                manual_name=manual_name,
                # Version/date could be extracted from the document header in a
                # future iteration; for now we use a sensible default.
                manual_version="current",
                manual_issue_date="unknown",
            )

            # Steps 3–5: write nodes, run extraction, write edges.
            await extract_and_load_all(
                provisions=provisions,
                definitions=definitions,
                client=self._ollama_client,
                loader=loader,
            )
        finally:
            driver.close()

        # Sanity check: confirm at least some nodes were written.
        provision_count = self._graph_service.count_provisions_for_manual(manual_name)
        if provision_count == 0:
            raise RuntimeError(
                f"Ingestion pipeline completed but no Provision nodes were written "
                f"for manual '{manual_name}'.  This likely indicates a Neo4j "
                f"connectivity problem during the write phase."
            )
        logger.info(
            "Knowledge graph written for manual '%s': %d provision nodes.",
            manual_name,
            provision_count,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    async def _delete_stale_graph_data(
        self, filename: str, current_document_id: str
    ) -> None:
        """Delete graph data for any previous upload of the same filename."""
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
