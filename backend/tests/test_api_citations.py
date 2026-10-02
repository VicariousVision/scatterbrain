"""Backward-compatible chat/search model and router citation tests."""

import asyncio
from unittest.mock import AsyncMock, MagicMock

from models.chat import ChatResponse
from models.content import ChatResult, Citation, RetrievedContext
from models.document import SearchResultItem
from routers import chat


def test_old_payload_models_still_deserialize() -> None:
    response = ChatResponse(response="a", history=[], retrieved_chunks=1)
    assert response.citations == []
    old_search = SearchResultItem(chunk_id="c", document_id="d", filename="x", chunk_index=0, text="t", similarity=0.5)
    assert old_search.clause_path is None


def test_chat_router_serializes_structured_citations() -> None:
    service = MagicMock()
    service.query = AsyncMock(
        return_value=ChatResult(
            answer="R2 million",
            contexts=[RetrievedContext(child_id="c", source_text="source")],
            citations=[Citation(filename="manual.pdf", clause_path="B.4(A)(i)", printed_page_start="98", label="B.4(A)(i), p. 98")],
        )
    )
    chat.set_services(service)
    result = asyncio.run(chat.chat_query(chat.ChatRequest(query="q")))
    assert result.response == "R2 million"
    assert result.retrieved_chunks == 1
    assert result.citations[0].label == "B.4(A)(i), p. 98"
