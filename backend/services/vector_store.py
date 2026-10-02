"""SQLite/sqlite-vec persistence for structured parents and ranked children.

Schema initialization is additive and idempotent. Legacy flat rows/vectors are
retained and marked unstructured; legal parents live in a normal table and
only child identifiers are ever inserted into the sqlite-vec table.
"""

from __future__ import annotations

import json
import logging
import os
import re
import struct
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import aiosqlite
import sqlite_vec

from config import settings
from models.content import ChunkRecord, SourceSpan
from services.embedding_provider import EmbeddingProvider

logger = logging.getLogger(__name__)

_CHUNK_COLUMNS: dict[str, str] = {
    "child_id": "TEXT",
    "source_sha256": "TEXT",
    "document_title": "TEXT",
    "document_version": "TEXT",
    "pdf_page_start": "INTEGER",
    "pdf_page_end": "INTEGER",
    "printed_page_start": "TEXT",
    "printed_page_end": "TEXT",
    "page_revisions": "TEXT DEFAULT '[]'",
    "source_spans": "TEXT DEFAULT '[]'",
    "section_path": "TEXT DEFAULT '[]'",
    "cross_references": "TEXT DEFAULT '[]'",
    "table_header": "TEXT DEFAULT '[]'",
    "section_id": "TEXT",
    "clause_path": "TEXT",
    "heading": "TEXT",
    "breadcrumb": "TEXT",
    "parent_id": "TEXT",
    "ancestor_parent_id": "TEXT",
    "child_order": "INTEGER DEFAULT 0",
    "content_type": "TEXT DEFAULT 'generic'",
    "continues_from": "TEXT",
    "continues_to": "TEXT",
    "table_id": "TEXT",
    "code": "TEXT",
    "defined_term": "TEXT",
    "extraction_method": "TEXT",
    "parser_version": "TEXT",
    "raw_text": "TEXT",
    "source_text": "TEXT",
    "embedding_text": "TEXT",
    "is_structured": "INTEGER NOT NULL DEFAULT 0",
}

_PARENT_COLUMNS: dict[str, str] = {
    "document_id": "TEXT",
    "filename": "TEXT",
    "record_type": "TEXT DEFAULT 'parent'",
    "source_sha256": "TEXT",
    "document_title": "TEXT",
    "document_version": "TEXT",
    "pdf_page_start": "INTEGER",
    "pdf_page_end": "INTEGER",
    "printed_page_start": "TEXT",
    "printed_page_end": "TEXT",
    "page_revisions": "TEXT DEFAULT '[]'",
    "source_spans": "TEXT DEFAULT '[]'",
    "section_path": "TEXT DEFAULT '[]'",
    "cross_references": "TEXT DEFAULT '[]'",
    "table_header": "TEXT DEFAULT '[]'",
    "section_id": "TEXT",
    "clause_path": "TEXT",
    "heading": "TEXT",
    "breadcrumb": "TEXT",
    "ancestor_parent_id": "TEXT",
    "parent_order": "INTEGER DEFAULT 0",
    "content_type": "TEXT DEFAULT 'generic'",
    "table_id": "TEXT",
    "code": "TEXT",
    "defined_term": "TEXT",
    "extraction_method": "TEXT",
    "parser_version": "TEXT",
    "raw_text": "TEXT DEFAULT ''",
    "source_text": "TEXT DEFAULT ''",
}

_METADATA_COLUMNS = [
    "child_id",
    "source_sha256",
    "document_title",
    "document_version",
    "pdf_page_start",
    "pdf_page_end",
    "printed_page_start",
    "printed_page_end",
    "page_revisions",
    "source_spans",
    "section_path",
    "cross_references",
    "table_header",
    "section_id",
    "clause_path",
    "heading",
    "breadcrumb",
    "parent_id",
    "ancestor_parent_id",
    "child_order",
    "content_type",
    "continues_from",
    "continues_to",
    "table_id",
    "code",
    "defined_term",
    "extraction_method",
    "parser_version",
    "raw_text",
    "source_text",
    "embedding_text",
    "is_structured",
]


class VectorStore:
    """Manage parent/child records and child vector search.

    Parameters
    ----------
    embedding_provider:
        Provider used for document and query embeddings.
    db_path:
        Optional explicit database path. Supplying it keeps tests/evaluations
        isolated from the configured user database.
    """

    def __init__(
        self,
        embedding_provider: EmbeddingProvider,
        db_path: str | os.PathLike[str] | None = None,
    ) -> None:
        self._embedding_provider = embedding_provider
        if self._embedding_provider.embedding_dimension <= 0:
            raise ValueError("Embedding dimension must be positive")
        self._db_path = str(db_path) if db_path is not None else settings.sqlite_db_path
        self._db: Optional[aiosqlite.Connection] = None
        logger.info("VectorStore initialized for %s", self._db_path)

    @property
    def db_path(self) -> str:
        """Database path used by this instance."""
        return self._db_path

    async def initialize(self) -> None:
        """Open SQLite, load sqlite-vec, and migrate schema additively."""
        Path(self._db_path).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)
        self._db = await aiosqlite.connect(self._db_path)
        self._db.row_factory = aiosqlite.Row
        await self._load_vec_extension()
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
        await self._add_missing_columns("document_chunks", _CHUNK_COLUMNS)
        await self._db.execute(
            """
            CREATE TABLE IF NOT EXISTS document_parents (
                parent_id TEXT PRIMARY KEY,
                document_id TEXT NOT NULL,
                filename TEXT NOT NULL,
                record_type TEXT NOT NULL,
                source_sha256 TEXT,
                document_title TEXT,
                document_version TEXT,
                pdf_page_start INTEGER,
                pdf_page_end INTEGER,
                printed_page_start TEXT,
                printed_page_end TEXT,
                page_revisions TEXT NOT NULL DEFAULT '[]',
                source_spans TEXT NOT NULL DEFAULT '[]',
                section_path TEXT NOT NULL DEFAULT '[]',
                cross_references TEXT NOT NULL DEFAULT '[]',
                table_header TEXT NOT NULL DEFAULT '[]',
                section_id TEXT,
                clause_path TEXT,
                heading TEXT,
                breadcrumb TEXT,
                ancestor_parent_id TEXT,
                parent_order INTEGER NOT NULL DEFAULT 0,
                content_type TEXT NOT NULL DEFAULT 'generic',
                table_id TEXT,
                code TEXT,
                defined_term TEXT,
                extraction_method TEXT,
                parser_version TEXT,
                raw_text TEXT NOT NULL DEFAULT '',
                source_text TEXT NOT NULL DEFAULT '',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        await self._add_missing_columns("document_parents", _PARENT_COLUMNS)
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
            "CREATE INDEX IF NOT EXISTS idx_document_chunks_document_id "
            "ON document_chunks(document_id)"
        )
        await self._db.execute(
            "CREATE INDEX IF NOT EXISTS idx_document_chunks_parent_id "
            "ON document_chunks(parent_id)"
        )
        await self._db.execute(
            "CREATE INDEX IF NOT EXISTS idx_document_chunks_clause_path "
            "ON document_chunks(clause_path)"
        )
        await self._db.execute(
            "CREATE INDEX IF NOT EXISTS idx_document_parents_document_id "
            "ON document_parents(document_id)"
        )
        # Compatibility values are filled in-place; no legacy row or vector is
        # deleted or rebuilt by migration.
        await self._db.execute(
            """
            UPDATE document_chunks
            SET child_id = COALESCE(child_id, chunk_id),
                raw_text = COALESCE(raw_text, content),
                source_text = COALESCE(source_text, content),
                embedding_text = COALESCE(embedding_text, content),
                content_type = COALESCE(content_type, 'generic'),
                is_structured = COALESCE(is_structured, 0)
            """
        )
        await self._db.commit()
        logger.info(
            "Database schema initialized additively (%d vector dimensions)", dimension
        )

    async def _load_vec_extension(self) -> None:
        if not self._db:
            raise RuntimeError("VectorStore database is not initialized")
        vec_path = os.path.normpath(sqlite_vec.loadable_path()).replace("\\", "/")
        if not os.path.exists(vec_path):
            suffix = ".dll" if os.name == "nt" else ".dylib" if sys.platform == "darwin" else ".so"
            if os.path.exists(f"{vec_path}{suffix}"):
                vec_path = f"{vec_path}{suffix}"
        if not os.path.exists(vec_path):
            raise RuntimeError(
                f"sqlite-vec extension was not found at {vec_path}. Reinstall sqlite-vec."
            )
        await self._db.enable_load_extension(True)
        try:
            escaped = vec_path.replace("'", "''")
            await self._db.execute(f"SELECT load_extension('{escaped}')")
        finally:
            await self._db.enable_load_extension(False)

    async def _add_missing_columns(
        self, table: str, columns: dict[str, str]
    ) -> None:
        if not self._db:
            raise RuntimeError("VectorStore database is not initialized")
        cursor = await self._db.execute(f"PRAGMA table_info({table})")
        present = {str(row[1]) for row in await cursor.fetchall()}
        for name, declaration in columns.items():
            if name not in present:
                await self._db.execute(
                    f'ALTER TABLE {table} ADD COLUMN "{name}" {declaration}'
                )

    async def _validate_embedding_configuration(self) -> None:
        """Refuse vector-model/dimension mismatch without touching user data."""
        if not self._db:
            raise RuntimeError("VectorStore database is not initialized")
        provider = self._embedding_provider
        cursor = await self._db.execute(
            "SELECT provider, model, dimension FROM embedding_metadata WHERE id = 1"
        )
        row = await cursor.fetchone()
        if row is None:
            cursor = await self._db.execute(
                "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = 'vec_chunks'"
            )
            vec_row = await cursor.fetchone()
            if vec_row is not None:
                schema_sql = str(vec_row["sql"] or "")
                dimension_match = re.search(r"FLOAT\[(\d+)\]", schema_sql, re.I)
                legacy_dimension = (
                    int(dimension_match.group(1)) if dimension_match else 768
                )
                if provider.embedding_dimension != legacy_dimension:
                    raise RuntimeError(
                        f"Existing legacy vector table uses {legacy_dimension} dimensions, "
                        f"but the configured model uses {provider.embedding_dimension}. "
                        "Reconfigure the model or explicitly re-ingest into a new database; "
                        "no data was changed."
                    )
            await self._db.execute(
                "INSERT INTO embedding_metadata (id, provider, model, dimension) "
                "VALUES (1, ?, ?, ?)",
                (provider.provider_name, provider.model_name, provider.embedding_dimension),
            )
            return
        if (
            row["provider"] != provider.provider_name
            or row["model"] != provider.model_name
            or row["dimension"] != provider.embedding_dimension
        ):
            raise RuntimeError(
                "Embedding configuration does not match the existing vector database "
                f"(stored: {row['provider']}/{row['model']}/{row['dimension']}; "
                f"configured: {provider.provider_name}/{provider.model_name}/"
                f"{provider.embedding_dimension}). No data was changed; use a compatible "
                "configuration or a new database and re-ingest."
            )

    async def close(self) -> None:
        """Close the database connection."""
        if self._db:
            await self._db.close()
            self._db = None

    async def add_document(
        self,
        document_id: str,
        filename: str,
        chunks: Sequence[str | ChunkRecord],
    ) -> int:
        """Embed children, then atomically replace all records for a filename.

        Embeddings are computed before ``BEGIN IMMEDIATE``. Any persistence
        failure rolls back parent, child, and vector replacement together, so
        a failed re-ingestion leaves the prior filename ingestion intact.
        """
        if not self._db:
            raise RuntimeError("VectorStore not initialized. Call initialize() first.")
        if not chunks:
            logger.warning("No chunks provided for document_id=%s", document_id)
            return 0

        records = self._coerce_records(document_id, filename, chunks)
        parents = [record for record in records if record.record_type != "child"]
        children = [record for record in records if record.record_type == "child"]
        if not children:
            logger.warning("No searchable children provided for document_id=%s", document_id)
            return 0
        embedding_texts = [record.embedding_text or record.source_text for record in children]
        embeddings = await self._embed_texts(embedding_texts)
        if len(embeddings) != len(children):
            raise RuntimeError(
                f"Embedding provider returned {len(embeddings)} vectors for "
                f"{len(children)} children"
            )
        for embedding in embeddings:
            self._validate_embedding(embedding)

        try:
            await self._db.execute("BEGIN IMMEDIATE")
            await self._delete_matching_uncommitted("filename = ?", (filename,))
            for parent in parents:
                await self._insert_parent(parent, document_id, filename)
            for index, (child, embedding) in enumerate(zip(children, embeddings)):
                await self._insert_child(child, document_id, filename, index)
                blob = struct.pack(
                    f"{self._embedding_provider.embedding_dimension}f", *embedding
                )
                await self._db.execute(
                    "INSERT INTO vec_chunks (chunk_id, embedding) VALUES (?, ?)",
                    (child.child_id, blob),
                )
            await self._db.commit()
        except Exception:
            await self._db.rollback()
            logger.exception("Atomic document replacement failed for %s", filename)
            raise
        logger.info(
            "Atomically stored %d parents and %d children for %s",
            len(parents),
            len(children),
            filename,
        )
        return len(children)

    def _coerce_records(
        self,
        document_id: str,
        filename: str,
        chunks: Sequence[str | ChunkRecord],
    ) -> list[ChunkRecord]:
        if chunks and isinstance(chunks[0], ChunkRecord):
            records = [record.model_copy(deep=True) for record in chunks]  # type: ignore[union-attr]
            for record in records:
                record.document_id = document_id
                record.source_filename = filename
            return records
        records: list[ChunkRecord] = []
        parent_id = f"{document_id}_parent"
        whole = "\n\n".join(str(chunk) for chunk in chunks)
        records.append(
            ChunkRecord(
                record_type="parent",
                document_id=document_id,
                source_filename=filename,
                source_sha256="",
                parent_id=parent_id,
                source_text=whole,
                raw_text=whole,
                content_type="generic",
                is_structured=False,
            )
        )
        for index, chunk in enumerate(chunks):
            text = str(chunk)
            records.append(
                ChunkRecord(
                    record_type="child",
                    document_id=document_id,
                    source_filename=filename,
                    source_sha256="",
                    parent_id=parent_id,
                    child_id=f"{document_id}_chunk_{index}",
                    child_order=index,
                    raw_text=text,
                    source_text=text,
                    embedding_text=text,
                    content_type="generic",
                    is_structured=False,
                )
            )
        return records

    async def _insert_parent(
        self, record: ChunkRecord, document_id: str, filename: str
    ) -> None:
        assert self._db is not None
        await self._db.execute(
            """
            INSERT INTO document_parents (
                parent_id, document_id, filename, record_type, source_sha256,
                document_title, document_version, pdf_page_start, pdf_page_end,
                printed_page_start, printed_page_end, page_revisions, source_spans,
                section_path, cross_references, table_header, section_id,
                clause_path, heading, breadcrumb, ancestor_parent_id, parent_order,
                content_type, table_id, code, defined_term, extraction_method,
                parser_version, raw_text, source_text
            ) VALUES (
                ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
            )
            """,
            (
                record.parent_id,
                document_id,
                filename,
                record.record_type,
                record.source_sha256,
                record.document_title,
                record.document_version,
                record.pdf_page_start,
                record.pdf_page_end,
                record.printed_page_start,
                record.printed_page_end,
                _json(record.page_revisions),
                _json([span.model_dump(mode="json") for span in record.source_spans]),
                _json(record.section_path),
                _json(record.cross_references),
                _json(record.table_header),
                record.section_id,
                record.clause_path,
                record.heading,
                record.breadcrumb,
                record.ancestor_parent_id,
                record.parent_order,
                record.content_type,
                record.table_id,
                record.code,
                record.defined_term,
                record.extraction_method,
                record.parser_version,
                record.raw_text,
                record.source_text,
            ),
        )

    async def _insert_child(
        self,
        record: ChunkRecord,
        document_id: str,
        filename: str,
        fallback_index: int,
    ) -> None:
        assert self._db is not None
        await self._db.execute(
            """
            INSERT INTO document_chunks (
                chunk_id, document_id, filename, chunk_index, content, child_id,
                source_sha256, document_title, document_version, pdf_page_start,
                pdf_page_end, printed_page_start, printed_page_end, page_revisions,
                source_spans, section_path, cross_references, table_header,
                section_id, clause_path, heading, breadcrumb, parent_id,
                ancestor_parent_id, child_order, content_type, continues_from,
                continues_to, table_id, code, defined_term, extraction_method,
                parser_version, raw_text, source_text, embedding_text, is_structured
            ) VALUES (
                ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
            )
            """,
            (
                record.child_id,
                document_id,
                filename,
                fallback_index,
                record.source_text,
                record.child_id,
                record.source_sha256,
                record.document_title,
                record.document_version,
                record.pdf_page_start,
                record.pdf_page_end,
                record.printed_page_start,
                record.printed_page_end,
                _json(record.page_revisions),
                _json([span.model_dump(mode="json") for span in record.source_spans]),
                _json(record.section_path),
                _json(record.cross_references),
                _json(record.table_header),
                record.section_id,
                record.clause_path,
                record.heading,
                record.breadcrumb,
                record.parent_id,
                record.ancestor_parent_id,
                record.child_order,
                record.content_type,
                record.continues_from,
                record.continues_to,
                record.table_id,
                record.code,
                record.defined_term,
                record.extraction_method,
                record.parser_version,
                record.raw_text,
                record.source_text,
                record.embedding_text,
                int(record.is_structured),
            ),
        )

    async def search_candidates(
        self,
        query: str,
        top_k: Optional[int] = None,
        document_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Return vector-ranked child candidates with additive metadata."""
        if not self._db:
            raise RuntimeError("VectorStore not initialized. Call initialize() first.")
        limit = top_k or settings.retrieval_candidate_pool
        query_embedding = await self._embed_text(query)
        self._validate_embedding(query_embedding)
        blob = struct.pack(
            f"{self._embedding_provider.embedding_dimension}f", *query_embedding
        )
        selected = ", ".join(f"dc.{name}" for name in _METADATA_COLUMNS)
        sql = f"""
            SELECT dc.chunk_id, dc.content, dc.document_id, dc.filename,
                   dc.chunk_index, {selected},
                   vec_distance_cosine(vc.embedding, ?) AS distance
            FROM document_chunks dc
            JOIN vec_chunks vc ON dc.chunk_id = vc.chunk_id
        """
        params: list[Any] = [blob]
        if document_id:
            sql += " WHERE dc.document_id = ?"
            params.append(document_id)
        sql += " ORDER BY distance ASC, dc.chunk_id ASC LIMIT ?"
        params.append(limit)
        cursor = await self._db.execute(sql, params)
        return [self._row_to_result(row) for row in await cursor.fetchall()]

    async def search(
        self,
        query: str,
        top_k: Optional[int] = None,
        document_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Backward-compatible child vector search facade."""
        return await self.search_candidates(
            query=query,
            top_k=top_k or settings.retrieval_top_k,
            document_id=document_id,
        )

    def _row_to_result(self, row: aiosqlite.Row) -> dict[str, Any]:
        offset = 5
        raw_meta = {
            name: row[offset + index] for index, name in enumerate(_METADATA_COLUMNS)
        }
        distance = float(row[offset + len(_METADATA_COLUMNS)])
        for name in (
            "page_revisions",
            "source_spans",
            "section_path",
            "cross_references",
            "table_header",
        ):
            raw_meta[name] = _loads(raw_meta.get(name), [])
        raw_meta.update(
            {
                "document_id": row[2],
                "filename": row[3],
                "chunk_index": row[4],
                "is_structured": bool(raw_meta.get("is_structured")),
            }
        )
        return {
            "id": row[0],
            "text": raw_meta.get("source_text") or row[1],
            "embedding_text": raw_meta.get("embedding_text") or row[1],
            "metadata": raw_meta,
            "similarity": 1.0 - distance,
        }

    async def get_parent_excerpt(
        self, parent_id: str, max_chars: int = 600
    ) -> dict[str, Any] | None:
        """Load a short governing parent excerpt; parents are never ranked."""
        if not self._db:
            raise RuntimeError("VectorStore not initialized")
        cursor = await self._db.execute(
            "SELECT * FROM document_parents WHERE parent_id = ?", (parent_id,)
        )
        row = await cursor.fetchone()
        if row is None:
            return None
        text = str(row["source_text"] or "")
        return {
            "parent_id": parent_id,
            "text": text[:max_chars],
            "metadata": {
                "section_id": row["section_id"],
                "clause_path": row["clause_path"],
                "heading": row["heading"],
                "breadcrumb": row["breadcrumb"],
            },
        }

    async def get_adjacent_children(self, child_id: str) -> List[Dict[str, Any]]:
        """Load explicit continuation and immediate same-parent neighbors."""
        if not self._db:
            raise RuntimeError("VectorStore not initialized")
        cursor = await self._db.execute(
            "SELECT parent_id, child_order, continues_from, continues_to "
            "FROM document_chunks WHERE chunk_id = ?",
            (child_id,),
        )
        anchor = await cursor.fetchone()
        if anchor is None:
            return []
        cursor = await self._db.execute(
            """
            SELECT * FROM document_chunks
            WHERE chunk_id IN (?, ?)
               OR (parent_id = ? AND child_order BETWEEN ? AND ? AND chunk_id != ?)
            ORDER BY child_order, chunk_id
            """,
            (
                anchor["continues_from"] or "",
                anchor["continues_to"] or "",
                anchor["parent_id"],
                int(anchor["child_order"] or 0) - 1,
                int(anchor["child_order"] or 0) + 1,
                child_id,
            ),
        )
        return [self._plain_child_row(row) for row in await cursor.fetchall()]

    async def get_children_by_clause_paths(
        self,
        clause_paths: Sequence[str],
        document_id: str | None = None,
    ) -> List[Dict[str, Any]]:
        """Resolve canonical cross-reference targets to actual child source."""
        if not self._db or not clause_paths:
            return []
        normalized = {_normalize_clause(path) for path in clause_paths if path}
        where = "clause_path IS NOT NULL"
        params: list[Any] = []
        if document_id:
            where += " AND document_id = ?"
            params.append(document_id)
        cursor = await self._db.execute(
            f"SELECT * FROM document_chunks WHERE {where} ORDER BY child_order, chunk_id",
            params,
        )
        return [
            self._plain_child_row(row)
            for row in await cursor.fetchall()
            if _normalize_clause(str(row["clause_path"] or "")) in normalized
        ]

    def _plain_child_row(self, row: aiosqlite.Row) -> dict[str, Any]:
        metadata: dict[str, Any] = {
            "document_id": row["document_id"],
            "filename": row["filename"],
            "chunk_index": row["chunk_index"],
        }
        for name in _METADATA_COLUMNS:
            value = row[name]
            if name in {
                "page_revisions",
                "source_spans",
                "section_path",
                "cross_references",
                "table_header",
            }:
                value = _loads(value, [])
            elif name == "is_structured":
                value = bool(value)
            metadata[name] = value
        return {
            "id": row["chunk_id"],
            "text": row["source_text"] or row["content"],
            "embedding_text": row["embedding_text"] or row["content"],
            "metadata": metadata,
            "similarity": 0.0,
        }

    async def delete_document(self, document_id: str) -> None:
        """Atomically remove a document's parents, children, and vectors."""
        await self._delete_atomic("document_id = ?", (document_id,))

    async def delete_by_filename(self, filename: str) -> None:
        """Atomically remove every record/vector for one filename."""
        await self._delete_atomic("filename = ?", (filename,))

    async def _delete_atomic(self, where: str, params: tuple[Any, ...]) -> None:
        if not self._db:
            raise RuntimeError("VectorStore not initialized. Call initialize() first.")
        try:
            await self._db.execute("BEGIN IMMEDIATE")
            count = await self._delete_matching_uncommitted(where, params)
            await self._db.commit()
            logger.info("Deleted %d child vectors where %s", count, where)
        except Exception:
            await self._db.rollback()
            raise

    async def _delete_matching_uncommitted(
        self, where: str, params: tuple[Any, ...]
    ) -> int:
        assert self._db is not None
        cursor = await self._db.execute(
            f"SELECT chunk_id FROM document_chunks WHERE {where}", params
        )
        ids = [str(row[0]) for row in await cursor.fetchall()]
        for child_id in ids:
            await self._db.execute(
                "DELETE FROM vec_chunks WHERE chunk_id = ?", (child_id,)
            )
        await self._db.execute(f"DELETE FROM document_chunks WHERE {where}", params)
        await self._db.execute(f"DELETE FROM document_parents WHERE {where}", params)
        return len(ids)

    def _validate_embedding(self, embedding: List[float]) -> None:
        expected = self._embedding_provider.embedding_dimension
        if len(embedding) != expected:
            raise ValueError(
                f"Embedding provider returned {len(embedding)} dimensions; expected "
                f"{expected} for {self._embedding_provider.model_name!r}."
            )

    async def _embed_text(self, text: str) -> List[float]:
        query_generator = getattr(
            self._embedding_provider, "generate_query_embedding", None
        )
        if query_generator is not None:
            return await query_generator(text)
        return await self._embedding_provider.generate_embedding(text)

    async def _embed_texts(self, texts: Sequence[str]) -> List[List[float]]:
        if hasattr(type(self._embedding_provider), "generate_embeddings"):
            return await self._embedding_provider.generate_embeddings(list(texts))
        return [
            await self._embedding_provider.generate_embedding(text) for text in texts
        ]


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _loads(value: Any, default: Any) -> Any:
    if value in (None, ""):
        return default
    try:
        return json.loads(str(value))
    except (TypeError, ValueError, json.JSONDecodeError):
        return default


def _normalize_clause(value: str) -> str:
    return re_sub_whitespace(value).rstrip(".").lower()


def re_sub_whitespace(value: str) -> str:
    return "".join(value.split())
