"""Unit tests for chat evaluation data (mocked vector store and LLM)."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from config import settings
from models.chat import ChatResponse
from models.content import RetrievedContext
from services.chat_service import ChatService, cosine_similarity, extract_cited_markers


def test_cosine_similarity() -> None:
    assert cosine_similarity([1, 0], [1, 0]) == pytest.approx(1.0)
    assert cosine_similarity([1, 0], [0, 1]) == pytest.approx(0.0)
    assert cosine_similarity([1, 0], [-1, 0]) == pytest.approx(-1.0)
    assert cosine_similarity([0, 0], [1, 0]) == 0.0


def test_extract_cited_markers() -> None:
    answer = "a [2][7]\n\nSources: [2], [7] and [1, 3] and [99]"
    assert extract_cited_markers(answer, 9) == {1, 2, 3, 7}
    assert extract_cited_markers("x [2][7]", 3) == {2}


def _contexts() -> list[RetrievedContext]:
    def ctx(i: int, clause: str, text: str, sim: float) -> RetrievedContext:
        return RetrievedContext(
            child_id=f"c{i}",
            source_text=text,
            similarity=sim,
            metadata={
                "document_id": "d",
                "filename": "m.pdf",
                "clause_path": clause,
                "parent_id": f"p{i}",
                "printed_page_start": "1",
                "printed_page_end": "1",
            },
        )

    return [ctx(1, "A(i)", "t" * 3000, 0.8), ctx(2, "B(i)", "two", 0.5), ctx(3, "A(i)", "three", 0.7)]


def _service(embed: AsyncMock | None) -> ChatService:
    retrieval = MagicMock()
    retrieval.retrieve = AsyncMock(return_value=_contexts())
    vector_store = MagicMock()
    vector_store.embedding_model_name = "test/model"
    vector_store.embed_documents = embed
    client = MagicMock()
    client.chat = AsyncMock(return_value="x [1][3]")
    return ChatService(vector_store=vector_store, llm_client=client, retrieval_service=retrieval)


def test_evaluation_items_and_citation_markers() -> None:
    embed = AsyncMock(return_value=[[1, 0], [1, 0], [0, 1], [1, 0]])
    result = asyncio.run(_service(embed).query("q"))

    ev = result.evaluation
    assert ev is not None and ev.query_embedding_model == "test/model"
    assert [i.marker for i in ev.retrieved] == [1, 2, 3]
    assert [i.cited for i in ev.retrieved] == [True, False, True]
    assert [i.answer_similarity for i in ev.retrieved] == pytest.approx([1.0, 0.0, 1.0])
    assert [i.similarity for i in ev.retrieved] == [0.8, 0.5, 0.7]
    assert ev.retrieved[0].chunk_text == "t" * 3000
    assert ev.retrieved[0].parent_id == "p1"
    assert [c.markers for c in result.citations] == [[1, 3], [2]]
    embed.assert_awaited_once()


def test_gate_flag_disables_evaluation(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "include_eval_data", False)
    result = asyncio.run(_service(AsyncMock(return_value=[]))._build_evaluation("a", []))
    assert result is None


def test_embed_failure_gives_none_answer_similarity() -> None:
    result = asyncio.run(_service(AsyncMock(side_effect=RuntimeError("boom"))).query("q"))
    assert result.evaluation is not None
    assert all(i.answer_similarity is None for i in result.evaluation.retrieved)


def test_response_serialization_and_backward_compat() -> None:
    embed = AsyncMock(return_value=[[1, 0]] * 4)
    ev = asyncio.run(_service(embed).query("q")).evaluation
    resp = ChatResponse(response="a", history=[], evaluation=ev)
    assert ChatResponse.model_validate_json(resp.model_dump_json()).evaluation == ev
    assert ChatResponse(response="a", history=[]).evaluation is None
