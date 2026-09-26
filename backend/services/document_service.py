"""Document service orchestrating the RAG ingestion pipeline.

Processing pipeline (run as a background asyncio task):
  1. Parse document text (PDF or TXT)
  2. Chunk text using recursive text splitter
  3. Generate embeddings and store in PostgreSQL + pgvector
  4. Track status in memory (will be persisted in future iteration)
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from datetime import datetime, timezone
from typing import Dict, List, Optional

from models.document import DocumentRecord
from services.document_parser import parse_document, DocumentParsingError
from services.text_chunker import chunk_text
from services.vector_store import VectorStore

logger = logging.getLogger(__name__)


class DocumentService:
    """Orchestrates document upload and the asynchronous RAG ingestion pipeline.
    
    Parameters
    ----------
    vector_store:
        Vector store for embedding and storing document chunks.
    """
    
    def __init__(
        self,
        vector_store: VectorStore,
    ) -> None:
        self._vector_store = vector_store
        self._store: Dict[str, DocumentRecord] = {}
        # One active ingestion at a time to avoid memory bloat
        self._processing_semaphore = asyncio.Semaphore(1)
    
    async def upload(
        self,
        filename: str,
        content: bytes,
    ) -> DocumentRecord:
        """Accept an uploaded file, create a tracking record, and start processing.
        
        Args:
            filename: Original filename including extension.
            content: Raw file bytes.
            
        Returns:
            DocumentRecord with status "processing".
        """
        document_id = str(uuid.uuid4())
        record = DocumentRecord(
            document_id=document_id,
            filename=filename,
            uploaded_at=datetime.now(tz=timezone.utc),
            status="processing",
        )
        
        # Store in memory
        self._store[document_id] = record
        
        logger.info(
            "Document uploaded: document_id=%s filename=%s",
            document_id,
            filename,
        )
        
        # Start background processing
        asyncio.create_task(
            self._process_document(document_id, filename, content),
            name=f"process-{document_id}",
        )
        
        return record
    
    def list_documents(self) -> List[DocumentRecord]:
        """Return all document records from memory.
        
        Returns:
            List of all document records ordered by upload time (newest first).
        """
        return sorted(
            self._store.values(),
            key=lambda r: r.uploaded_at,
            reverse=True,
        )
    
    def get_document(self, document_id: str) -> Optional[DocumentRecord]:
        """Return the record for a single document, or None if not found.
        
        Args:
            document_id: Document ID to retrieve.
            
        Returns:
            DocumentRecord if found, None otherwise.
        """
        return self._store.get(document_id)
    
    async def delete_document(self, document_id: str) -> None:
        """Delete a document from vector store and memory.
        
        Args:
            document_id: Document ID to delete.
        """
        # Delete from vector store
        await self._vector_store.delete_document(document_id)
        
        # Delete from memory
        self._store.pop(document_id, None)
        
        logger.info("Deleted document: %s", document_id)
    
    async def _process_document(
        self,
        document_id: str,
        filename: str,
        content: bytes,
    ) -> None:
        """Run the full RAG ingestion pipeline for an uploaded document.
        
        Args:
            document_id: Unique document identifier.
            filename: Original filename.
            content: Raw file bytes.
        """
        logger.info("Starting RAG pipeline for document_id=%s", document_id)
        
        try:
            # Step 1: Parse document text
            text = parse_document(filename, content)
            logger.info(
                "Parsed document_id=%s: %d characters",
                document_id,
                len(text),
            )
            
            # Step 2: Chunk text
            chunks = chunk_text(text)
            logger.info(
                "Chunked document_id=%s: %d chunks",
                document_id,
                len(chunks),
            )
            
            if not chunks:
                raise RuntimeError("No chunks produced from document")
            
            # Step 3: Embed and store in vector database
            async with self._processing_semaphore:
                # Delete any existing chunks for this filename to avoid duplicates
                await self._vector_store.delete_by_filename(filename)
                
                # Add new chunks
                chunk_count = await self._vector_store.add_document(
                    document_id=document_id,
                    filename=filename,
                    chunks=chunks,
                )
            
            # Step 4: Update status to completed
            self._update_status(document_id, "completed")
            
            logger.info(
                "RAG pipeline completed for document_id=%s (%d chunks)",
                document_id,
                chunk_count,
            )
        
        except DocumentParsingError as exc:
            error_message = f"Document parsing failed: {exc}"
            logger.error(
                "RAG pipeline failed for document_id=%s: %s",
                document_id,
                error_message,
            )
            self._update_status(document_id, "failed", error=error_message)
        
        except Exception as exc:
            error_message = str(exc)
            logger.error(
                "RAG pipeline failed for document_id=%s: %s",
                document_id,
                error_message,
                exc_info=True,
            )
            self._update_status(document_id, "failed", error=error_message)
    
    def _update_status(
        self,
        document_id: str,
        status: str,
        error: Optional[str] = None,
    ) -> None:
        """Update document status in memory."""
        record = self._store.get(document_id)
        if record is None:
            logger.warning(
                "Attempted to update status for unknown document_id=%s", document_id
            )
            return
        self._store[document_id] = record.model_copy(
            update={"status": status, "error": error}
        )

