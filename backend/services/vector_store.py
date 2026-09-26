"""Vector store service using SQLite with sqlite-vec extension.

Provides document storage, embedding, and semantic search capabilities.
"""

from __future__ import annotations

import logging
import struct
from typing import Dict, List, Optional

import aiosqlite
import sqlite_vec

from config import settings
from services.ollama_client import OllamaClient

logger = logging.getLogger(__name__)


class VectorStore:
    """Manages document embeddings and semantic search using SQLite + sqlite-vec.
    
    Parameters
    ----------
    ollama_client:
        Client for generating embeddings via Ollama.
    """
    
    def __init__(
        self,
        ollama_client: OllamaClient,
    ) -> None:
        self._ollama_client = ollama_client
        self._db: Optional[aiosqlite.Connection] = None
        logger.info("VectorStore initialized (SQLite + sqlite-vec)")
    
    async def initialize(self) -> None:
        """Initialize database connection and create schema."""
        # Ensure database directory exists
        import os
        db_dir = os.path.dirname(os.path.abspath(settings.sqlite_db_path))
        if db_dir and not os.path.exists(db_dir):
            os.makedirs(db_dir, exist_ok=True)
        
        # Connect to SQLite database
        self._db = await aiosqlite.connect(settings.sqlite_db_path)
        
        # Load extension using SQL (normalize path for Windows)
        import os
        vec_path = os.path.normpath(sqlite_vec.loadable_path()).replace('\\', '/')
        await self._db.enable_load_extension(True)
        await self._db.execute(f"SELECT load_extension('{vec_path}')")
        await self._db.enable_load_extension(False)
        
        # Create table for document chunks
        await self._db.execute("""
            CREATE TABLE IF NOT EXISTS document_chunks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                chunk_id TEXT UNIQUE NOT NULL,
                document_id TEXT NOT NULL,
                filename TEXT NOT NULL,
                chunk_index INTEGER NOT NULL,
                content TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Create virtual table for vector embeddings (768 dimensions for nomic-embed-text)
        await self._db.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS vec_chunks USING vec0(
                chunk_id TEXT PRIMARY KEY,
                embedding FLOAT[768]
            )
        """)
        
        # Create indexes
        await self._db.execute("""
            CREATE INDEX IF NOT EXISTS idx_document_chunks_document_id 
            ON document_chunks(document_id)
        """)
        
        await self._db.commit()
        logger.info("Database schema initialized with sqlite-vec extension")
    
    async def close(self) -> None:
        """Close database connection."""
        if self._db:
            await self._db.close()
            logger.info("Database connection closed")
    
    async def add_document(
        self,
        document_id: str,
        filename: str,
        chunks: List[str],
    ) -> int:
        """Add document chunks to the vector store.
        
        Args:
            document_id: Unique document identifier.
            filename: Original filename.
            chunks: List of text chunks to embed and store.
            
        Returns:
            Number of chunks added.
        """
        if not chunks:
            logger.warning("No chunks provided for document_id=%s", document_id)
            return 0
        
        if not self._db:
            raise RuntimeError("VectorStore not initialized. Call initialize() first.")
        
        # Generate embeddings for all chunks
        embeddings = await self._embed_texts(chunks)
        
        # Insert into database
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            chunk_id = f"{document_id}_chunk_{i}"
            
            # Insert text content
            await self._db.execute(
                """
                INSERT OR REPLACE INTO document_chunks 
                (chunk_id, document_id, filename, chunk_index, content)
                VALUES (?, ?, ?, ?, ?)
                """,
                (chunk_id, document_id, filename, i, chunk),
            )
            
            # Insert embedding vector
            # sqlite-vec expects blob format
            embedding_blob = struct.pack(f'{len(embedding)}f', *embedding)
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
    ) -> List[Dict[str, any]]:
        """Search for relevant document chunks using cosine similarity.
        
        Args:
            query: Search query text.
            top_k: Number of results to return (defaults to config setting).
            document_id: Optional filter to search within specific document.
            
        Returns:
            List of search results with text, metadata, and similarity scores.
        """
        if top_k is None:
            top_k = settings.retrieval_top_k
        
        if not self._db:
            raise RuntimeError("VectorStore not initialized. Call initialize() first.")
        
        # Generate query embedding
        query_embedding = await self._embed_text(query)
        query_blob = struct.pack(f'{len(query_embedding)}f', *query_embedding)
        
        # Search using sqlite-vec distance function
        if document_id:
            cursor = await self._db.execute(
                """
                SELECT 
                    dc.chunk_id,
                    dc.content,
                    dc.document_id,
                    dc.filename,
                    dc.chunk_index,
                    vec_distance_cosine(vc.embedding, ?) as distance
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
                SELECT 
                    dc.chunk_id,
                    dc.content,
                    dc.document_id,
                    dc.filename,
                    dc.chunk_index,
                    vec_distance_cosine(vc.embedding, ?) as distance
                FROM document_chunks dc
                JOIN vec_chunks vc ON dc.chunk_id = vc.chunk_id
                ORDER BY distance ASC
                LIMIT ?
                """,
                (query_blob, top_k),
            )
        
        rows = await cursor.fetchall()
        
        # Format results (convert cosine distance to similarity score)
        results = [
            {
                "id": row[0],
                "text": row[1],
                "metadata": {
                    "document_id": row[2],
                    "filename": row[3],
                    "chunk_index": row[4],
                },
                "similarity": 1.0 - row[5],  # Convert distance to similarity
            }
            for row in rows
        ]
        
        logger.info(
            "Search query returned %d results (top_k=%d)",
            len(results),
            top_k,
        )
        
        return results
    
    async def delete_document(self, document_id: str) -> None:
        """Delete all chunks for a document.
        
        Args:
            document_id: Document identifier to delete.
        """
        if not self._db:
            raise RuntimeError("VectorStore not initialized. Call initialize() first.")
        
        try:
            # Get all chunk IDs for this document
            cursor = await self._db.execute(
                "SELECT chunk_id FROM document_chunks WHERE document_id = ?",
                (document_id,),
            )
            chunk_ids = [row[0] for row in await cursor.fetchall()]
            
            # Delete from both tables
            await self._db.execute(
                "DELETE FROM document_chunks WHERE document_id = ?",
                (document_id,),
            )
            
            for chunk_id in chunk_ids:
                await self._db.execute(
                    "DELETE FROM vec_chunks WHERE chunk_id = ?",
                    (chunk_id,),
                )
            
            await self._db.commit()
            logger.info("Deleted %d chunks for document_id=%s", len(chunk_ids), document_id)
        except Exception as exc:
            logger.error(
                "Failed to delete chunks for document_id=%s: %s",
                document_id,
                exc,
            )
            raise
    
    async def delete_by_filename(self, filename: str) -> None:
        """Delete all chunks for documents with a specific filename.
        
        Args:
            filename: Filename to match for deletion.
        """
        if not self._db:
            raise RuntimeError("VectorStore not initialized. Call initialize() first.")
        
        try:
            # Get all chunk IDs for this filename
            cursor = await self._db.execute(
                "SELECT chunk_id FROM document_chunks WHERE filename = ?",
                (filename,),
            )
            chunk_ids = [row[0] for row in await cursor.fetchall()]
            
            # Delete from both tables
            await self._db.execute(
                "DELETE FROM document_chunks WHERE filename = ?",
                (filename,),
            )
            
            for chunk_id in chunk_ids:
                await self._db.execute(
                    "DELETE FROM vec_chunks WHERE chunk_id = ?",
                    (chunk_id,),
                )
            
            await self._db.commit()
            logger.info("Deleted %d chunks for filename=%s", len(chunk_ids), filename)
        except Exception as exc:
            logger.error(
                "Failed to delete chunks for filename=%s: %s",
                filename,
                exc,
            )
            raise
    
    async def get_stats(self) -> Dict[str, any]:
        """Get collection statistics.
        
        Returns:
            Dictionary with total chunks and document count.
        """
        if not self._db:
            raise RuntimeError("VectorStore not initialized. Call initialize() first.")
        
        cursor = await self._db.execute("""
            SELECT 
                COUNT(*) as total_chunks,
                COUNT(DISTINCT document_id) as total_documents
            FROM document_chunks
        """)
        
        row = await cursor.fetchone()
        
        return {
            "total_chunks": row[0],
            "total_documents": row[1],
        }
    
    async def _embed_text(self, text: str) -> List[float]:
        """Generate embedding for a single text.
        
        Args:
            text: Input text.
            
        Returns:
            Embedding vector as list of floats.
        """
        embedding = await self._ollama_client.generate_embedding(text)
        return embedding
    
    async def _embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts.
        
        Args:
            texts: List of input texts.
            
        Returns:
            List of embedding vectors.
        """
        embeddings = []
        for text in texts:
            embedding = await self._ollama_client.generate_embedding(text)
            embeddings.append(embedding)
        return embeddings
