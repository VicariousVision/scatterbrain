"""Adapters that let RAGAS metrics run on Scatterbrain's local services.

RAGAS 0.4 "collections" metrics need two components:

* an ``InstructorBaseRagasLLM`` that turns a prompt into a validated Pydantic
  object, and
* a ``BaseRagasEmbedding`` for metrics that compare texts (answer relevancy).

Both are implemented here on top of the project's own ``OllamaClient`` and
``EmbeddingProvider`` so evaluation stays fully local and reuses the same
concurrency guard, retries, and embedding model as the application.

Why not ``ragas.llms.llm_factory`` with Ollama's OpenAI-compatible endpoint?
Small local models (e.g. ``qwen3.5:0.8b``) wrap JSON in Markdown fences or
echo the JSON Schema instead of filling it, so Instructor's validation fails
and retries until it gives up. Ollama's native ``format=<json schema>`` uses
grammar-constrained decoding, which always yields schema-shaped JSON.
"""

from __future__ import annotations

import asyncio
import re
import typing as t

from pydantic import BaseModel, ValidationError
from ragas.embeddings.base import BaseRagasEmbedding
from ragas.llms.base import InstructorBaseRagasLLM

from services.embedding_provider import EmbeddingProvider
from services.ollama_client import OllamaClient

ModelT = t.TypeVar("ModelT", bound=BaseModel)

_FENCE_RE = re.compile(r"^\s*```(?:json)?\s*(.*?)\s*```\s*$", re.DOTALL)


class JudgeOutputError(RuntimeError):
    """Raised when the judge LLM output cannot be validated against the schema."""


def _run_sync(coro: t.Awaitable[t.Any]) -> t.Any:
    """Run ``coro`` from synchronous code; refuse inside a running loop."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)  # type: ignore[arg-type]
    raise RuntimeError(
        "Synchronous RAGAS calls are not supported inside a running event "
        "loop; use the async API (ascore / agenerate) instead."
    )


def parse_structured_output(raw: str, response_model: t.Type[ModelT]) -> ModelT:
    """Validate judge output, tolerating a Markdown code fence around the JSON.

    Parameters
    ----------
    raw:
        Text returned by the LLM.
    response_model:
        Pydantic model the text must validate against.
    """
    try:
        return response_model.model_validate_json(raw)
    except ValidationError as first_error:
        match = _FENCE_RE.match(raw)
        if match:
            try:
                return response_model.model_validate_json(match.group(1))
            except ValidationError:
                pass
        raise JudgeOutputError(
            f"Judge output did not match {response_model.__name__}: {first_error}"
        ) from first_error


class OllamaRagasLLM(InstructorBaseRagasLLM):
    """RAGAS judge LLM backed by the local Ollama server.

    Parameters
    ----------
    ollama_client:
        Client whose ``model`` is the judge model.
    max_tokens:
        Output token cap per judge call.
    temperature:
        Sampling temperature; ``0`` keeps scores as repeatable as possible.
    max_attempts:
        Attempts per call when the output fails schema validation.
    """

    def __init__(
        self,
        ollama_client: OllamaClient,
        max_tokens: int = 2048,
        temperature: float = 0.0,
        max_attempts: int = 2,
    ) -> None:
        self._client = ollama_client
        self._max_tokens = max_tokens
        self._temperature = temperature
        self._max_attempts = max(1, max_attempts)

    @property
    def model(self) -> str:
        """Name of the Ollama judge model."""
        return self._client.model

    async def agenerate(self, prompt: str, response_model: t.Type[ModelT]) -> ModelT:
        """Generate a schema-constrained response and validate it."""
        schema = response_model.model_json_schema()
        last_error: JudgeOutputError | None = None
        for _ in range(self._max_attempts):
            raw = await self._client.generate(
                prompt,
                json_schema=schema,
                max_tokens=self._max_tokens,
                # Thinking tokens would eat the budget before any JSON appears.
                think=False,
                temperature=self._temperature,
            )
            try:
                return parse_structured_output(raw, response_model)
            except JudgeOutputError as exc:
                last_error = exc
        assert last_error is not None
        raise last_error

    def generate(self, prompt: str, response_model: t.Type[ModelT]) -> ModelT:
        """Synchronous wrapper around :meth:`agenerate`."""
        return _run_sync(self.agenerate(prompt, response_model))

    def __repr__(self) -> str:
        return f"OllamaRagasLLM(model={self.model!r})"


class ProviderRagasEmbeddings(BaseRagasEmbedding):
    """RAGAS embeddings backed by a Scatterbrain ``EmbeddingProvider``.

    Parameters
    ----------
    provider:
        The embedding provider used by the application (Hugging Face or
        Ollama). All texts are embedded with the document (non-query) path so
        that compared vectors share one embedding space.
    """

    def __init__(self, provider: EmbeddingProvider) -> None:
        super().__init__(cache=None)
        self._provider = provider

    async def aembed_text(self, text: str, **kwargs: t.Any) -> t.List[float]:
        """Embed one text."""
        return await self._provider.generate_embedding(text)

    async def aembed_texts(
        self, texts: t.List[str], **kwargs: t.Any
    ) -> t.List[t.List[float]]:
        """Embed many texts, batching when the provider supports it."""
        if hasattr(type(self._provider), "generate_embeddings"):
            return await self._provider.generate_embeddings(list(texts))
        return [await self._provider.generate_embedding(text) for text in texts]

    def embed_text(self, text: str, **kwargs: t.Any) -> t.List[float]:
        """Synchronous wrapper around :meth:`aembed_text`."""
        return _run_sync(self.aembed_text(text))

    def embed_texts(self, texts: t.List[str], **kwargs: t.Any) -> t.List[t.List[float]]:
        """Synchronous wrapper around :meth:`aembed_texts`."""
        return _run_sync(self.aembed_texts(texts))
