"""
SQLite + sqlite-vec vector store for document chunks.

Provides document storage, embedding, and semantic search capabilities.
"""

from __future__ import annotations

import logging
import os
import struct
import sys
from typing import Any, Dict, List, Optional

import aiosqlite
import sqlite_vec

from config import settings
from services.embedding_provider import EmbeddingProvider

logger = logging.getLogger(__name__)


class VectorStore:
    """Manage document embeddings and semantic search with SQLite + sqlite-vec.

    Parameters
    ----------
    embedding_provider:
        Provider used to generate document and query embeddings. Its reported
        dimension is used to create the sqlite-vec virtual table.

    Notes
    -----
    sqlite-vec fixes a vector table's dimension at creation time. The store
    records the provider, model, and dimension in ``embedding_metadata`` and
    refuses to start if an existing database was built with another embedding
    configuration. Delete/rebuild the configured database when changing it.
    """

    def __init__(
        self,
        embedding_provider: EmbeddingProvider | None = None,
        *,
        ollama_client: EmbeddingProvider | None = None,
    ) -> None:
        # ``ollama_client`` remains accepted for callers using the previous
        # constructor while the generic provider name supports Hugging Face.
        self._embedding_provider = embedding_provider or ollama_client
        if self._embedding_provider is None:
            raise ValueError("An embedding provider is required")
        if self._embedding_provider.embedding_dimension <= 0:
            raise ValueError("Embedding dimension must be positive")

        self._db: Optional[aiosqlite.Connection] = None
        logger.info("VectorStore initialized (SQLite + sqlite-vec)")

    async def initialize(self) -> None:
        """Initialize the database connection and create the vector schema."""
        db_dir = os.path.dirname(os.path.abspath(settings.sqlite_db_path))
        if db_dir and not os.path.exists(db_dir):
            os.makedirs(db_dir, exist_ok=True)

        self._db = await aiosqlite.connect(settings.sqlite_db_path)

        # Normalize the Windows DLL path before loading through SQLite's SQL
        # function; this works with sqlite-vec's bundled native extension.
        vec_path = os.path.normpath(sqlite_vec.loadable_path()).replace("\\", "/")
        if not os.path.exists(vec_path):
            platform_suffix = ".dll" if os.name == "nt" else ".dylib" if sys.platform == "darwin" else ".so"
            suffixed_path = f"{vec_path}{platform_suffix}"
            if os.path.exists(suffixed_path):
                vec_path = suffixed_path
        if not os.path.exists(vec_path):
            raise RuntimeError(
                f"sqlite-vec extension was not found at {vec_path}. "
                "Reinstall the sqlite-vec backend dependency."
            )
        await self._db.enable_load_extension(True)
        try:
            await self._db.execute(f"SELECT load_extension('{vec_path}')")
        finally:
            await self._db.enable_load_extension(False)

        await self._db.execute(
            """
            CREATE TABLE IF NOT EXISTS document_chunks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                chunk_id TEXT UNIQUE NOT NULL,
                document_id TEXT NOT NULL,
                filename TEXT NOT NULL,
                chunk_index INTEGER NOT NULL,
                content TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        await self._db.execute(
            """
            CREATE TABLE IF NOT EXISTS embedding_metadata (
                id INTEGER PRIMARY KEY CHECK (id = 1),
                provider TEXT NOT NULL,
                model TEXT NOT NULL,
                dimension INTEGER NOT NULL
            )
            """
        )

        await self._validate_embedding_configuration()

        dimension = self._embedding_provider.embedding_dimension
        await self._db.execute(
            f"""
            CREATE VIRTUAL TABLE IF NOT EXISTS vec_chunks USING vec0(
                chunk_id TEXT PRIMARY KEY,
                embedding FLOAT[{dimension}]
            )
            """
        )
        await self._db.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_document_chunks_document_id
            ON document_chunks(document_id)
            """
        )

        await self._db.commit()
        logger.info(
            "Database schema initialized with sqlite-vec (%d dimensions)",
            dimension,
        )

    async def _validate_embedding_configuration(self) -> None:
        """Ensure stored vectors match the configured provider and dimension."""
        if not self._db:
            raise RuntimeError("VectorStore database is not initialized")

        provider = self._embedding_provider
        cursor = await self._db.execute(
            "SELECT provider, model, dimension FROM embedding_metadata WHERE id = 1"
        )
        metadata = await cursor.fetchone()

        if metadata is None:
            # Databases created before embedding_metadata used FLOAT[768].
            # Treat that as the legacy dimension and migrate its metadata.
            cursor = await self._db.execute(
                "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'vec_chunks'"
            )
            has_existing_vectors = await cursor.fetchone() is not None
            if has_existing_vectors and provider.embedding_dimension != 768:
                raise RuntimeError(
                    "Existing vector database uses 768 dimensions, but the "
                    f"configured {provider.provider_name} model uses "
                    f"{provider.embedding_dimension}. Delete the configured "
                    "SQLITE_DB_PATH database and re-ingest documents."
                )
            await self._db.execute(
                """
                INSERT INTO embedding_metadata (id, provider, model, dimension)
                VALUES (1, ?, ?, ?)
                """,
                (
                    provider.provider_name,
                    provider.model_name,
                    provider.embedding_dimension,
                ),
            )
            return

        stored_provider, stored_model, stored_dimension = metadata
        if (
            stored_provider != provider.provider_name
            or stored_model != provider.model_name
            or stored_dimension != provider.embedding_dimension
        ):
            raise RuntimeError(
                "Embedding configuration does not match the existing vector "
                f"database (stored: {stored_provider}/{stored_model}/"
                f"{stored_dimension}; configured: {provider.provider_name}/"
                f"{provider.model_name}/{provider.embedding_dimension}). "
                "Delete the configured SQLITE_DB_PATH database and re-ingest "
                "documents when changing embedding providers or models."
            )

    async def close(self) -> None:
        """Close the database connection."""
        if self._db:
            await self._db.close()
            self._db = None
            logger.info("Database connection closed")

    async def add_document(
        self,
        document_id: str,
        filename: str,
        chunks: List[str],
    ) -> int:
        """Embed and add document chunks to the vector store."""
        if not chunks:
            logger.warning("No chunks provided for document_id=%s", document_id)
            return 0
        if not self._db:
            raise RuntimeError("VectorStore not initialized. Call initialize() first.")

        embeddings = await self._embed_texts(chunks)
        if len(embeddings) != len(chunks):
            raise RuntimeError(
                f"Embedding provider returned {len(embeddings)} vectors for "
                f"{len(chunks)} chunks"
            )

        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            self._validate_embedding(embedding)
            chunk_id = f"{document_id}_chunk_{i}"

            await self._db.execute(
                """
                INSERT OR REPLACE INTO document_chunks
                (chunk_id, document_id, filename, chunk_index, content)
                VALUES (?, ?, ?, ?, ?)
                """,
                (chunk_id, document_id, filename, i, chunk),
            )
            embedding_blob = struct.pack(f"{self._embedding_provider.embedding_dimension}f", *embedding)
            await self._db.execute(
                """
                INSERT OR REPLACE INTO vec_chunks (chunk_id, embedding)
                VALUES (?, ?)
                """,
                (chunk_id, embedding_blob),
            )

        await self._db.commit()
        logger.info(
            "Added %d chunks for document_id=%s filename=%s",
            len(chunks),
            document_id,
            filename,
        )
        return len(chunks)

    async def search(
        self,
        query: str,
        top_k: Optional[int] = None,
        document_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Search document chunks using cosine similarity."""
        if top_k is None:
            top_k = settings.retrieval_top_k
        if not self._db:
            raise RuntimeError("VectorStore not initialized. Call initialize() first.")

        query_embedding = await self._embed_text(query)
        self._validate_embedding(query_embedding)
        query_blob = struct.pack(
            f"{self._embedding_provider.embedding_dimension}f", *query_embedding
        )

        if document_id:
            cursor = await self._db.execute(
                """
                SELECT dc.chunk_id, dc.content, dc.document_id, dc.filename,
                       dc.chunk_index,
                       vec_distance_cosine(vc.embedding, ?) AS distance
                FROM document_chunks dc
                JOIN vec_chunks vc ON dc.chunk_id = vc.chunk_id
                WHERE dc.document_id = ?
                ORDER BY distance ASC
                LIMIT ?
                """,
                (query_blob, document_id, top_k),
            )
        else:
            cursor = await self._db.execute(
                """
                SELECT dc.chunk_id, dc.content, dc.document_id, dc.filename,
                       dc.chunk_index,
                       vec_distance_cosine(vc.embedding, ?) AS distance
                FROM document_chunks dc
                JOIN vec_chunks vc ON dc.chunk_id = vc.chunk_id
                ORDER BY distance ASC
                LIMIT ?
                """,
                (query_blob, top_k),
            )

        rows = await cursor.fetchall()
        results = [
            {
                "id": row[0],
                "text": row[1],
                "metadata": {
                    "document_id": row[2],
                    "filename": row[3],
                    "chunk_index": row[4],
                },
                "similarity": 1.0 - row[5],
            }
            for row in rows
        ]
        logger.info("Search query returned %d results (top_k=%d)", len(results), top_k)
        return results

    async def delete_document(self, document_id: str) -> None:
        """Delete all chunks for a document."""
        if not self._db:
            raise RuntimeError("VectorStore not initialized. Call initialize() first.")
        try:
            cursor = await self._db.execute(
                "SELECT chunk_id FROM document_chunks WHERE document_id = ?",
                (document_id,),
            )
            chunk_ids = [row[0] for row in await cursor.fetchall()]
            await self._db.execute(
                "DELETE FROM document_chunks WHERE document_id = ?", (document_id,)
            )
            for chunk_id in chunk_ids:
                await self._db.execute(
                    "DELETE FROM vec_chunks WHERE chunk_id = ?", (chunk_id,)
                )
            await self._db.commit()
            logger.info("Deleted %d chunks for document_id=%s", len(chunk_ids), document_id)
        except Exception as exc:
            logger.error("Failed to delete chunks for document_id=%s: %s", document_id, exc)
            raise

    async def delete_by_filename(self, filename: str) -> None:
        """Delete all chunks for documents with a specific filename."""
        if not self._db:
            raise RuntimeError("VectorStore not initialized. Call initialize() first.")
        try:
            cursor = await self._db.execute(
                "SELECT chunk_id FROM document_chunks WHERE filename = ?", (filename,)
            )
            chunk_ids = [row[0] for row in await cursor.fetchall()]
            await self._db.execute(
                "DELETE FROM document_chunks WHERE filename = ?", (filename,)
            )
            for chunk_id in chunk_ids:
                await self._db.execute(
                    "DELETE FROM vec_chunks WHERE chunk_id = ?", (chunk_id,)
                )
            await self._db.commit()
            logger.info("Deleted %d chunks for filename=%s", len(chunk_ids), filename)
        except Exception as exc:
            logger.error("Failed to delete chunks for filename=%s: %s", filename, exc)
            raise

    async def get_stats(self) -> Dict[str, Any]:
        """Return total chunk and document counts."""
        if not self._db:
            raise RuntimeError("VectorStore not initialized. Call initialize() first.")
        cursor = await self._db.execute(
            """
            SELECT COUNT(*) AS total_chunks,
                   COUNT(DISTINCT document_id) AS total_documents
            FROM document_chunks
            """
        )
        row = await cursor.fetchone()
        return {"total_chunks": row[0], "total_documents": row[1]}

    def _validate_embedding(self, embedding: List[float]) -> None:
        """Reject vectors that cannot be stored or searched in this schema."""
        expected = self._embedding_provider.embedding_dimension
        if len(embedding) != expected:
            raise ValueError(
                f"Embedding provider returned {len(embedding)} dimensions; "
                f"expected {expected} for {self._embedding_provider.model_name!r}."
            )

    async def _embed_text(self, text: str) -> List[float]:
        """Generate a query embedding, applying provider-specific instructions."""
        query_generator = getattr(
            self._embedding_provider, "generate_query_embedding", None
        )
        if query_generator is not None:
            return await query_generator(text)
        return await self._embedding_provider.generate_embedding(text)

    async def _embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Generate one embedding vector per input text."""
        return [
            await self._embedding_provider.generate_embedding(text) for text in texts
        ]
