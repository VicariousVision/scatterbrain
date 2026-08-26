"""LLMInterface adapters for DeepSeek and OpenRouter.

These adapters wrap the async DeepSeekClient / OpenRouterClient so they can be
plugged into neo4j-graphrag's Text2CypherRetriever, which expects a sync
``invoke()`` and async ``ainvoke()`` compatible with LLMInterface.

The adapters are used only when the user selects "DeepSeek" or "OpenRouter" as
the Q&A backend from the Chat UI.  The legacy auto-select path
(graph_query_service._build_paid_llm) is unchanged.

Requirements: external LLM backend (Chat UI)
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import logging
from typing import List, Optional, Union

from neo4j_graphrag.llm import LLMInterface, LLMResponse
from neo4j_graphrag.message_history import MessageHistory
from neo4j_graphrag.types import LLMMessage

from backend.services.external_llm_client import DeepSeekClient, OpenRouterClient
from backend.services.graph_query_service import _sanitize_cypher

logger = logging.getLogger(__name__)


def _run_async(coro) -> any:
    """Run *coro* from a sync context, even inside a running event loop."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None
    if loop and loop.is_running():
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(asyncio.run, coro).result()
    return asyncio.run(coro)


class DeepSeekLLMAdapter(LLMInterface):
    """Wraps DeepSeekClient for use with Text2CypherRetriever.

    The ``invoke`` / ``ainvoke`` methods receive the assembled prompt from
    Text2CypherRetriever and forward it as the user message with the compact
    _CYPHER_SYSTEM prompt applied inside DeepSeekClient.generate_cypher.

    Because Text2CypherRetriever assembles the full prompt (schema + examples +
    question) itself and passes it as a single string, we forward it verbatim
    as the user message.  The system prompt inside DeepSeekClient is kept
    minimal to avoid double-counting tokens.
    """

    def __init__(self, client: DeepSeekClient, model_name: str = "deepseek-chat") -> None:
        super().__init__(model_name=model_name)
        self._client = client

    def _call_api(self, prompt: str) -> str:
        """Forward the assembled prompt to DeepSeek with a minimal wrapper."""
        import httpx
        payload = {
            "model": self._client._model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a Neo4j Cypher expert. Output ONLY a single valid "
                        "Cypher query. No explanations, no markdown fences."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            "temperature": 0,
            "max_tokens": 512,
        }
        return _run_async(self._post_cypher(payload))

    async def _post_cypher(self, payload: dict) -> str:
        import httpx
        headers = {
            "Authorization": f"Bearer {self._client._api_key}",
            "Content-Type": "application/json",
        }
        async with httpx.AsyncClient(
            timeout=httpx.Timeout(connect=10.0, read=60.0, write=10.0, pool=10.0)
        ) as client:
            response = await client.post(
                "https://api.deepseek.com/v1/chat/completions",
                json=payload,
                headers=headers,
            )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"].strip()

    def invoke(
        self,
        input: str,
        message_history: Optional[Union[List[LLMMessage], MessageHistory]] = None,
        system_instruction: Optional[str] = None,
    ) -> LLMResponse:
        text = self._call_api(input)
        text = _sanitize_cypher(text)
        logger.debug("DeepSeekLLMAdapter.invoke output (first 500): %s", text[:500])
        return LLMResponse(content=text)

    async def ainvoke(
        self,
        input: str,
        message_history: Optional[Union[List[LLMMessage], MessageHistory]] = None,
        system_instruction: Optional[str] = None,
    ) -> LLMResponse:
        text = await self._post_cypher(
            {
                "model": self._client._model,
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "You are a Neo4j Cypher expert. Output ONLY a single valid "
                            "Cypher query. No explanations, no markdown fences."
                        ),
                    },
                    {"role": "user", "content": input},
                ],
                "temperature": 0,
                "max_tokens": 512,
            }
        )
        text = _sanitize_cypher(text)
        logger.debug("DeepSeekLLMAdapter.ainvoke output (first 500): %s", text[:500])
        return LLMResponse(content=text)


class OpenRouterLLMAdapter(LLMInterface):
    """Wraps OpenRouterClient for use with Text2CypherRetriever.

    Free-tier models are tried first (round-robin); paid fallback used only
    when all free tiers are rate-limited or unavailable.
    """

    def __init__(self, client: OpenRouterClient, model_name: str = "openrouter") -> None:
        super().__init__(model_name=model_name)
        self._client = client

    def invoke(
        self,
        input: str,
        message_history: Optional[Union[List[LLMMessage], MessageHistory]] = None,
        system_instruction: Optional[str] = None,
    ) -> LLMResponse:
        text = _run_async(self._ainvoke_inner(input))
        text = _sanitize_cypher(text)
        logger.debug("OpenRouterLLMAdapter.invoke output (first 500): %s", text[:500])
        return LLMResponse(content=text)

    async def ainvoke(
        self,
        input: str,
        message_history: Optional[Union[List[LLMMessage], MessageHistory]] = None,
        system_instruction: Optional[str] = None,
    ) -> LLMResponse:
        text = await self._ainvoke_inner(input)
        text = _sanitize_cypher(text)
        logger.debug("OpenRouterLLMAdapter.ainvoke output (first 500): %s", text[:500])
        return LLMResponse(content=text)

    async def _ainvoke_inner(self, prompt: str) -> str:
        """Forward prompt to OpenRouter with free-tier rotation."""
        import httpx
        import itertools
        from backend.services.external_llm_client import (
            _OPENROUTER_FREE_TIER_CYPHER_MODELS,
            _OPENROUTER_PAID_FALLBACK_MODEL,
            _free_cypher_cycle,
        )

        headers = self._client._headers()
        payload_base = {
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a Neo4j Cypher expert. Output ONLY a single valid "
                        "Cypher query. No explanations, no markdown fences."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            "temperature": 0,
            "max_tokens": 512,
        }

        attempted: set[str] = set()
        for _ in range(len(_OPENROUTER_FREE_TIER_CYPHER_MODELS)):
            model = next(_free_cypher_cycle)
            if model in attempted:
                continue
            attempted.add(model)
            payload = {**payload_base, "model": model}
            async with httpx.AsyncClient(
                timeout=httpx.Timeout(connect=10.0, read=60.0, write=10.0, pool=10.0)
            ) as client:
                response = await client.post(
                    self._client._BASE_URL, json=payload, headers=headers
                )
            if response.status_code in (404, 429, 503):
                logger.warning("OpenRouterLLMAdapter: %s returned %s.", model, response.status_code)
                continue
            response.raise_for_status()
            logger.info("OpenRouterLLMAdapter: used free model %s.", model)
            return response.json()["choices"][0]["message"]["content"].strip()

        # Paid fallback.
        payload = {**payload_base, "model": _OPENROUTER_PAID_FALLBACK_MODEL}
        async with httpx.AsyncClient(
            timeout=httpx.Timeout(connect=10.0, read=60.0, write=10.0, pool=10.0)
        ) as client:
            response = await client.post(
                self._client._BASE_URL, json=payload, headers=headers
            )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"].strip()
