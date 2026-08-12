"""Adapters that bridge OllamaClient to neo4j-graphrag interfaces.

OllamaLLMAdapter is used by ChatService for final answer generation.

OllamaEmbedderAdapter is retained as a no-op stub so any external code that
imports it does not break.  It is NOT used in the ingestion or query pipeline —
this pipeline uses Cypher traversal, not vector embeddings.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import logging
from typing import Any, List, Optional, Union

from neo4j_graphrag.llm import LLMInterface, LLMResponse
from neo4j_graphrag.message_history import MessageHistory
from neo4j_graphrag.types import LLMMessage

from backend.services.ollama_client import OllamaClient

logger = logging.getLogger(__name__)

# Serialise all LLM calls — the OllamaClient semaphore already does this for
# async callers, but the sync invoke() path uses a thread so we also guard here.
_LLM_SEMAPHORE = asyncio.Semaphore(1)


def _run_async(coro: Any) -> Any:
    """Run an async coroutine from a sync context, even inside a running loop."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(asyncio.run, coro)
            return future.result()
    else:
        return asyncio.run(coro)


class OllamaLLMAdapter(LLMInterface):
    """Wraps OllamaClient so it can be used anywhere LLMInterface is expected.

    Parameters
    ----------
    ollama_client:
        A live OllamaClient instance.
    model_name:
        Model name forwarded to LLMInterface; informational only.
    """

    def __init__(self, ollama_client: OllamaClient, model_name: str) -> None:
        super().__init__(model_name=model_name)
        self._client = ollama_client

    def invoke(
        self,
        input: str,
        message_history: Optional[Union[List[LLMMessage], MessageHistory]] = None,
        system_instruction: Optional[str] = None,
    ) -> LLMResponse:
        """Synchronous text generation via Ollama."""
        text = _run_async(self._client.generate(input))
        return LLMResponse(content=text)

    async def ainvoke(
        self,
        input: str,
        message_history: Optional[Union[List[LLMMessage], MessageHistory]] = None,
        system_instruction: Optional[str] = None,
    ) -> LLMResponse:
        """Async text generation via Ollama."""
        async with _LLM_SEMAPHORE:
            text = await self._client.generate(input)
        return LLMResponse(content=text)


class OllamaEmbedderAdapter:
    """No-op stub retained for import compatibility.

    This pipeline does not use embeddings.  If you see this class being
    instantiated somewhere in active code, that code path should be removed.
    """

    def __init__(self, ollama_client: OllamaClient) -> None:
        self._client = ollama_client
        logger.warning(
            "OllamaEmbedderAdapter instantiated — this pipeline does not use "
            "embeddings.  Check whether the caller should be updated."
        )

    def embed_query(self, text: str) -> list[float]:
        logger.warning("OllamaEmbedderAdapter.embed_query called — returning empty vector.")
        return []
