"""Real temporary SQLite/sqlite-vec migration and atomicity tests."""

import asyncio
import sqlite3
import struct
from unittest.mock import AsyncMock

import pytest

from models.content import ChunkRecord, SourceSpan
from services.vector_store import VectorStore


class FakeProvider:
    provider_name = "fake"
    model_name = "fake-3d"
    embedding_dimension = 3

    async def generate_embedding(self, text):
        return [1.0, float(len(text) % 3), 0.0]

    async def generate_query_embedding(self, text):
        return [1.0, float(len(text) % 3), 0.0]

    async def generate_embeddings(self, texts):
        return [await self.generate_embedding(text) for text in texts]


def _records(tag: str, document_id: str = "doc") -> list[ChunkRecord]:
    parent_id = f"p_{tag}"
    common = dict(
        document_id=document_id,
        source_filename="manual.pdf",
        source_sha256=tag * 64,
        document_version="1.0",
        pdf_page_start=98,
        pdf_page_end=98,
        printed_page_start="98",
        printed_page_end="98",
        page_revisions=["6/2026"],
        section_path=["B.4", "(A)", "(i)"],
        section_id="B.4",
        clause_path="B.4(A)(i)",
        breadcrumb="B.4 > (A) > (i)",
        parent_id=parent_id,
        content_type="provision",
        source_spans=[SourceSpan(pdf_page=98, printed_page="98")],
    )
    return [
        ChunkRecord(record_type="parent", raw_text="parent", source_text="parent", **common),
        ChunkRecord(record_type="child", child_id=f"c_{tag}", raw_text=f"source {tag}", source_text=f"source {tag}", embedding_text=f"breadcrumb source {tag}", cross_references=["I.3(B)"], **common),
    ]


def test_structured_metadata_parent_not_vector_and_atomic_delete(tmp_path) -> None:
    async def run() -> None:
        store = VectorStore(FakeProvider(), tmp_path / "structured.db")
        await store.initialize()
        assert await store.add_document("doc", "manual.pdf", _records("a")) == 1
        results = await store.search("source", top_k=5)
        assert results[0]["metadata"]["clause_path"] == "B.4(A)(i)"
        assert results[0]["metadata"]["cross_references"] == ["I.3(B)"]
        resolved = await store.get_children_by_clause_paths(
            ["b.4(a)(i)"], document_id="doc"
        )
        assert [item["id"] for item in resolved] == ["c_a"]
        cursor = await store._db.execute("SELECT count(*) FROM document_parents")
        assert (await cursor.fetchone())[0] == 1
        cursor = await store._db.execute("SELECT count(*) FROM vec_chunks")
        assert (await cursor.fetchone())[0] == 1
        await store.delete_document("doc")
        for table in ("document_parents", "document_chunks", "vec_chunks"):
            cursor = await store._db.execute(f"SELECT count(*) FROM {table}")
            assert (await cursor.fetchone())[0] == 0
        await store.close()
    asyncio.run(run())


def test_legacy_schema_migrates_twice_without_losing_row(tmp_path) -> None:
    path = tmp_path / "legacy.db"
    with sqlite3.connect(path) as conn:
        conn.execute("CREATE TABLE document_chunks (id INTEGER PRIMARY KEY AUTOINCREMENT, chunk_id TEXT UNIQUE NOT NULL, document_id TEXT NOT NULL, filename TEXT NOT NULL, chunk_index INTEGER NOT NULL, content TEXT NOT NULL, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)")
        conn.execute("INSERT INTO document_chunks (chunk_id, document_id, filename, chunk_index, content) VALUES ('old_chunk', 'old', 'old.txt', 0, 'legacy source')")

    async def run() -> None:
        store = VectorStore(FakeProvider(), path)
        await store.initialize()
        blob = struct.pack("3f", 1.0, 0.0, 0.0)
        await store._db.execute("INSERT INTO vec_chunks (chunk_id, embedding) VALUES (?, ?)", ("old_chunk", blob))
        await store._db.commit()
        await store.close()
        store = VectorStore(FakeProvider(), path)
        await store.initialize()
        result = (await store.search("legacy", top_k=1))[0]
        assert result["text"] == "legacy source"
        assert result["metadata"]["is_structured"] is False
        await store.close()
    asyncio.run(run())


def test_failed_replacement_rolls_back_old_filename(tmp_path) -> None:
    async def run() -> None:
        store = VectorStore(FakeProvider(), tmp_path / "rollback.db")
        await store.initialize()
        await store.add_document("old", "manual.pdf", _records("a", "old"))
        original_insert = store._insert_child
        store._insert_child = AsyncMock(side_effect=RuntimeError("injected"))
        with pytest.raises(RuntimeError):
            await store.add_document("new", "manual.pdf", _records("b", "new"))
        store._insert_child = original_insert
        rows = await store.search("source", top_k=5)
        assert [row["id"] for row in rows] == ["c_a"]
        await store.close()
    asyncio.run(run())
