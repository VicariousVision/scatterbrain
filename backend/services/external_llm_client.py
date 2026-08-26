"""Async HTTP clients for DeepSeek Chat API and OpenRouter API.

Both clients are used for the explicit Q&A backend path (selected via the
Chat UI dropdown).  They are NOT involved in the legacy auto-select
Text2Cypher path — that path is handled by graph_query_service._build_paid_llm.

Token-cost optimisations applied throughout:
  - System prompts are as short as possible while preserving grounding.
  - Cypher-generation prompts include only the schema + examples needed.
  - Answer-generation prompts trim history to the last 6 exchanges.
  - ``max_tokens`` is capped for Cypher (512) and answers (1024).
  - ``temperature=0`` for Cypher generation (deterministic, no sampling).
  - ``temperature=0.3`` for answer generation (slightly creative but grounded).

OpenRouter free-tier staggering:
  - A module-level round-robin counter cycles through the FREE_TIER_MODELS list.
  - If a free-tier model returns a 429 (rate-limited) or 503, the next model in
    the list is tried automatically before falling back to the paid tier.

Requirements: external LLM backend (Chat UI)
"""

from __future__ import annotations

import asyncio
import itertools
import logging
from typing import Optional

import httpx

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Timeouts
# ---------------------------------------------------------------------------
_CYPHER_TIMEOUT = httpx.Timeout(connect=10.0, read=60.0, write=10.0, pool=10.0)
_ANSWER_TIMEOUT = httpx.Timeout(connect=10.0, read=120.0, write=10.0, pool=10.0)

# ---------------------------------------------------------------------------
# OpenRouter free-tier model rotation
# Models are tried in round-robin order.  If one rate-limits, the next is used.
# All listed models have confirmed free tiers on OpenRouter as of 2026-08.
# Run `openrouter.ai/api/v1/models` and filter for `:free` to refresh this list.
# ---------------------------------------------------------------------------
_OPENROUTER_FREE_TIER_MODELS: list[str] = [
    "nvidia/nemotron-3-super-120b-a12b:free",
    "google/gemma-4-31b-it:free",
    "google/gemma-4-26b-a4b-it:free",
    "nvidia/nemotron-3-ultra-550b-a55b:free",
    "thinkingmachines/inkling:free",
    "z-ai/glm-5.2:free",
]

# Cypher generation benefits from instruction-following accuracy — prefer
# larger / code-capable models from the free tier.
_OPENROUTER_FREE_TIER_CYPHER_MODELS: list[str] = [
    "nvidia/nemotron-3-super-120b-a12b:free",
    "google/gemma-4-31b-it:free",
    "nvidia/nemotron-3-ultra-550b-a55b:free",
    "google/gemma-4-26b-a4b-it:free",
    "cohere/north-mini-code:free",
    "thinkingmachines/inkling:free",
]

# Paid-tier fallback for OpenRouter (used only after all free tiers are exhausted).
_OPENROUTER_PAID_FALLBACK_MODEL: str = "deepseek/deepseek-chat"

# Round-robin iterators (module-level, persists across requests).
_free_answer_cycle = itertools.cycle(_OPENROUTER_FREE_TIER_MODELS)
_free_cypher_cycle = itertools.cycle(_OPENROUTER_FREE_TIER_CYPHER_MODELS)

# ---------------------------------------------------------------------------
# Shared system prompt fragments
# ---------------------------------------------------------------------------
_CYPHER_SYSTEM = (
    "You are a Neo4j Cypher expert for a legal document knowledge graph. "
    "Output ONLY a single valid Cypher query. No explanations, no markdown fences, "
    "no commentary. The query must be executable as-is."
)

_ANSWER_SYSTEM = (
    "You are a legal document analysis assistant for the South African Reserve Bank "
    "Currency and Exchanges Manual for Authorised Dealers. "
    "Answer using ONLY the provided graph context. "
    "If the context is insufficient, say so. Do not fabricate facts. "
    "Cite provision paths (e.g. B.4(B)(iv)) where possible."
)


# ---------------------------------------------------------------------------
# DeepSeek client
# ---------------------------------------------------------------------------


class DeepSeekClient:
    """Async client for the DeepSeek Chat API (OpenAI-compatible).

    Parameters
    ----------
    api_key:  Your DeepSeek API key.
    model:    Model ID — defaults to ``deepseek-chat`` (DeepSeek-V3).
    """

    _BASE_URL = "https://api.deepseek.com/v1/chat/completions"

    def __init__(self, api_key: str, model: str = "deepseek-chat") -> None:
        self._api_key = api_key
        self._model = model

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

    async def generate_cypher(self, schema: str, examples: str, question: str) -> str:
        """Generate a Cypher query for *question* given *schema* and *examples*.

        Token optimisations:
          - Temperature 0 (greedy — deterministic Cypher).
          - max_tokens 512 (Cypher queries are short).
          - Minimal system prompt.
          - Schema and examples sent once in the user turn.
        """
        # Compact prompt: schema + up to 5 examples + question.
        user_content = (
            f"Schema:\n{schema}\n\n"
            f"Examples:\n{examples}\n\n"
            f"Question: {question}\n\n"
            "Cypher query:"
        )
        payload = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": _CYPHER_SYSTEM},
                {"role": "user", "content": user_content},
            ],
            "temperature": 0,
            "max_tokens": 512,
        }
        async with httpx.AsyncClient(timeout=_CYPHER_TIMEOUT) as client:
            response = await client.post(
                self._BASE_URL, json=payload, headers=self._headers()
            )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"].strip()

    async def generate_answer(
        self,
        question: str,
        context: str,
        history: list[dict],
    ) -> str:
        """Generate a grounded answer given graph *context* and *history*.

        Token optimisations:
          - History trimmed to last 6 exchanges (12 messages).
          - max_tokens 1024.
          - Temperature 0.3.
          - Context injected in the system turn, not repeated per message.
        """
        trimmed = history[-12:] if len(history) > 12 else history
        messages: list[dict] = [
            {
                "role": "system",
                "content": f"{_ANSWER_SYSTEM}\n\nCONTEXT:\n{context}",
            }
        ]
        messages.extend(trimmed)
        messages.append({"role": "user", "content": question})

        payload = {
            "model": self._model,
            "messages": messages,
            "temperature": 0.3,
            "max_tokens": 1024,
        }
        async with httpx.AsyncClient(timeout=_ANSWER_TIMEOUT) as client:
            response = await client.post(
                self._BASE_URL, json=payload, headers=self._headers()
            )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"].strip()


# ---------------------------------------------------------------------------
# OpenRouter client
# ---------------------------------------------------------------------------


class OpenRouterClient:
    """Async client for the OpenRouter API.

    Free-tier models are tried in round-robin order.  On a 429 / 503 the next
    free model is attempted until all are exhausted, then the paid fallback is
    used.

    Parameters
    ----------
    api_key:    Your OpenRouter API key.
    site_url:   Your site URL forwarded in the ``HTTP-Referer`` header.
    site_name:  Your site name forwarded in the ``X-Title`` header.
    """

    _BASE_URL = "https://openrouter.ai/api/v1/chat/completions"

    def __init__(
        self,
        api_key: str,
        site_url: str = "http://localhost:3000",
        site_name: str = "Scatterbrain",
    ) -> None:
        self._api_key = api_key
        self._site_url = site_url
        self._site_name = site_name

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self._api_key}",
            "HTTP-Referer": self._site_url,
            "X-Title": self._site_name,
            "Content-Type": "application/json",
        }

    async def _post(
        self,
        payload: dict,
        timeout: httpx.Timeout,
    ) -> httpx.Response:
        async with httpx.AsyncClient(timeout=timeout) as client:
            return await client.post(
                self._BASE_URL, json=payload, headers=self._headers()
            )

    async def _try_models(
        self,
        free_cycle: "itertools.cycle[str]",
        free_pool: list[str],
        payload_factory,  # callable(model) -> dict
        timeout: httpx.Timeout,
    ) -> str:
        """Try each free-tier model in rotation, falling back to paid on exhaustion.

        Raises
        ------
        PermissionError
            On a 401 Unauthorized response — the API key is invalid.
        httpx.HTTPStatusError
            On any non-retriable HTTP error from the paid fallback.
        """
        attempted: set[str] = set()

        # Attempt each free-tier model once (round-robin start from cycle state).
        for _ in range(len(free_pool)):
            model = next(free_cycle)
            if model in attempted:
                continue
            attempted.add(model)
            payload = payload_factory(model)
            try:
                response = await self._post(payload, timeout)
                if response.status_code == 401:
                    raise PermissionError(
                        "OpenRouter API key is invalid or expired (401 Unauthorized). "
                        "Please update OPENROUTER_API_KEY in your .env file."
                    )
                if response.status_code == 404:
                    logger.warning(
                        "OpenRouter free model %s returned 404 (model not found) — trying next.",
                        model,
                    )
                    continue
                if response.status_code in (429, 503):
                    logger.warning(
                        "OpenRouter free model %s returned %s — trying next.",
                        model,
                        response.status_code,
                    )
                    await asyncio.sleep(0.5)
                    continue
                response.raise_for_status()
                content = response.json()["choices"][0]["message"]["content"]
                if content is None:
                    logger.warning(
                        "OpenRouter free model %s returned null content — trying next.", model
                    )
                    continue
                logger.info("OpenRouter: used free model %s.", model)
                return content.strip()
            except httpx.HTTPStatusError as exc:
                if exc.response.status_code == 401:
                    raise PermissionError(
                        "OpenRouter API key is invalid or expired (401 Unauthorized). "
                        "Please update OPENROUTER_API_KEY in your .env file."
                    ) from exc
                if exc.response.status_code in (404, 429, 503):
                    logger.warning(
                        "OpenRouter free model %s HTTP error %s — trying next.",
                        model,
                        exc.response.status_code,
                    )
                    if exc.response.status_code != 404:
                        await asyncio.sleep(0.5)
                    continue
                raise

        # All free tiers exhausted — use paid fallback.
        logger.info(
            "All OpenRouter free models exhausted — using paid fallback %s.",
            _OPENROUTER_PAID_FALLBACK_MODEL,
        )
        payload = payload_factory(_OPENROUTER_PAID_FALLBACK_MODEL)
        try:
            response = await self._post(payload, timeout)
            if response.status_code == 401:
                raise PermissionError(
                    "OpenRouter API key is invalid or expired (401 Unauthorized). "
                    "Please update OPENROUTER_API_KEY in your .env file."
                )
            if response.status_code in (403, 402, 429, 503):
                logger.warning(
                    "OpenRouter paid fallback %s returned %s — all models exhausted.",
                    _OPENROUTER_PAID_FALLBACK_MODEL,
                    response.status_code,
                )
                raise RuntimeError(
                    f"All OpenRouter models are currently unavailable "
                    f"(free tiers rate-limited, paid fallback returned {response.status_code}). "
                    "Please try again shortly or switch to a different backend."
                )
            response.raise_for_status()
            content = response.json()["choices"][0]["message"]["content"]
            if content is None:
                raise RuntimeError(
                    "OpenRouter paid fallback returned null content. "
                    "Please try again shortly."
                )
            return content.strip()
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 401:
                raise PermissionError(
                    "OpenRouter API key is invalid or expired (401 Unauthorized). "
                    "Please update OPENROUTER_API_KEY in your .env file."
                ) from exc
            raise RuntimeError(
                f"OpenRouter paid fallback failed with HTTP {exc.response.status_code}. "
                "Please try again shortly or switch to a different backend."
            ) from exc

    async def generate_cypher(self, schema: str, examples: str, question: str) -> str:
        """Generate a Cypher query, trying free-tier models first."""
        user_content = (
            f"Schema:\n{schema}\n\n"
            f"Examples:\n{examples}\n\n"
            f"Question: {question}\n\n"
            "Cypher query:"
        )

        def make_payload(model: str) -> dict:
            return {
                "model": model,
                "messages": [
                    {"role": "system", "content": _CYPHER_SYSTEM},
                    {"role": "user", "content": user_content},
                ],
                "temperature": 0,
                "max_tokens": 512,
            }

        return await self._try_models(
            _free_cypher_cycle,
            _OPENROUTER_FREE_TIER_CYPHER_MODELS,
            make_payload,
            _CYPHER_TIMEOUT,
        )

    async def generate_answer(
        self,
        question: str,
        context: str,
        history: list[dict],
    ) -> str:
        """Generate a grounded answer, trying free-tier models first."""
        trimmed = history[-12:] if len(history) > 12 else history
        base_messages: list[dict] = [
            {
                "role": "system",
                "content": f"{_ANSWER_SYSTEM}\n\nCONTEXT:\n{context}",
            }
        ]
        base_messages.extend(trimmed)
        base_messages.append({"role": "user", "content": question})

        def make_payload(model: str) -> dict:
            return {
                "model": model,
                "messages": base_messages,
                "temperature": 0.3,
                "max_tokens": 1024,
            }

        return await self._try_models(
            _free_answer_cycle,
            _OPENROUTER_FREE_TIER_MODELS,
            make_payload,
            _ANSWER_TIMEOUT,
        )
