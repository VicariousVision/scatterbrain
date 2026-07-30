"""Adapters that bridge the existing OllamaClient to the neo4j-graphrag
LLMInterface and Embedder interfaces so the library can use the local Ollama
server without any cloud credentials.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import logging
from typing import Any, List, Optional, Union

from neo4j_graphrag.embeddings.base import Embedder
from neo4j_graphrag.llm import LLMInterface, LLMResponse
from neo4j_graphrag.message_history import MessageHistory
from neo4j_graphrag.types import LLMMessage

from backend.services.ollama_client import OllamaClient

logger = logging.getLogger(__name__)

# Limit concurrent Ollama LLM requests to 1 to prevent OOM crashes.
# The neo4j-graphrag pipeline calls ainvoke in parallel for every chunk;
# without this guard the model is loaded/run multiple times simultaneously.
_LLM_SEMAPHORE = asyncio.Semaphore(1)


def _run_async(coro: Any) -> Any:
    """Run an async coroutine from a sync context, even inside a running loop."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        # We're already inside an async event loop (e.g. FastAPI / asyncio).
        # Spin up a thread with its own loop so we don't block the main one.
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(asyncio.run, coro)
            return future.result()
    else:
        return asyncio.run(coro)


class OllamaLLMAdapter(LLMInterface):
    """Wraps :class:`~backend.services.ollama_client.OllamaClient` so it can
    be used anywhere ``neo4j_graphrag`` expects an ``LLMInterface``.

    Parameters
    ----------
    ollama_client:
        A live :class:`~backend.services.ollama_client.OllamaClient` instance.
    model_name:
        Model name forwarded to ``LLMInterface``; informational only.
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
        """Async text generation via Ollama, serialised via semaphore.

        The neo4j-graphrag pipeline fires ainvoke concurrently for every chunk.
        Running multiple llama3 inferences in parallel exhausts RAM/VRAM and
        causes the runner to crash with exit code 2.  The semaphore ensures
        only one Ollama request is in-flight at a time.
        """
        async with _LLM_SEMAPHORE:
            text = await self._client.generate(input)
        return LLMResponse(content=text)


class OllamaEmbedderAdapter(Embedder):
    """Wraps :class:`~backend.services.ollama_client.OllamaClient` so it can
    be used anywhere ``neo4j_graphrag`` expects an ``Embedder``.

    Parameters
    ----------
    ollama_client:
        A live :class:`~backend.services.ollama_client.OllamaClient` instance.
    """

    def __init__(self, ollama_client: OllamaClient) -> None:
        self._client = ollama_client

    def embed_query(self, text: str) -> list[float]:
        """Synchronous embedding via Ollama."""
        return _run_async(self._client.generate_embedding(text))
