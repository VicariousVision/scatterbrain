"""Embedding provider abstractions and the local Hugging Face backend."""

from __future__ import annotations

import asyncio
from typing import List, Protocol


class EmbeddingProvider(Protocol):
    """Interface required by the SQLite vector store."""

    provider_name: str
    model_name: str
    embedding_dimension: int

    async def generate_embedding(self, text: str) -> List[float]:
        """Generate one document embedding for ``text``."""
        ...

    async def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate one document embedding per input text, in input order."""
        ...


class HuggingFaceEmbeddingProvider:
    """Generate embeddings with a Hugging Face Sentence Transformers model.

    Parameters
    ----------
    model_name:
        Hugging Face model ID or a local Sentence Transformers model path.
    device:
        Torch device used by Sentence Transformers, such as ``cpu`` or ``cuda``.
    normalize_embeddings:
        Whether to L2-normalize vectors before storing them.
    query_prefix:
        Optional instruction prepended to query text. The litil-embed model
        requires ``Instruct: Retrieve text based on user query.\\nQuery: ``.

    batch_size:
        Number of texts encoded per forward pass in ``generate_embeddings``.
        Larger batches use CPU cores more efficiently at the cost of memory.

    Notes
    -----
    Model loading is synchronous and happens during application startup. Text
    encoding is moved to a worker thread so the async API event loop remains
    responsive while embeddings are generated. Document embeddings are encoded
    without the query prefix; query embeddings use it when configured.
    ``generate_embeddings`` encodes a whole list in one batched call, which is
    substantially faster on CPU than encoding each chunk separately.
    """

    provider_name = "huggingface"

    def __init__(
        self,
        model_name: str,
        device: str = "cpu",
        normalize_embeddings: bool = True,
        query_prefix: str = "",
        batch_size: int = 32,
    ) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "Hugging Face embeddings require sentence-transformers. "
                "Install backend requirements with: pip install -r requirements.txt"
            ) from exc

        self.model_name = model_name
        self._normalize_embeddings = normalize_embeddings
        self._query_prefix = query_prefix
        self._batch_size = max(1, batch_size)
        self._model = SentenceTransformer(model_name, device=device)
        # Newer sentence-transformers renamed this method; fall back for
        # older installs that only have the original name.
        get_dimension = getattr(self._model, "get_embedding_dimension", None) or (
            self._model.get_sentence_embedding_dimension
        )
        dimension = get_dimension()
        if dimension is None or dimension <= 0:
            raise RuntimeError(
                f"Could not determine embedding dimension for Hugging Face model "
                f"{model_name!r}."
            )
        self.embedding_dimension = int(dimension)

    async def generate_embedding(self, text: str) -> List[float]:
        """Encode document text without a query instruction."""
        vectors = await self._encode([text])
        return vectors[0]

    async def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Encode many document chunks in one batched model call.

        Sentence Transformers encodes the whole list in a single ``encode``
        call using ``batch_size`` internally, which keeps all CPU cores busy
        and avoids the per-call overhead of encoding one chunk at a time.
        Vectors are returned in the same order as ``texts``.
        """
        if not texts:
            return []
        return await self._encode(list(texts))

    async def generate_query_embedding(self, text: str) -> List[float]:
        """Encode query text with the configured model-specific instruction."""
        vectors = await self._encode([f"{self._query_prefix}{text}"])
        return vectors[0]

    async def _encode(self, texts: List[str]) -> List[List[float]]:
        """Run Sentence Transformers off the async event loop.

        Always encodes a list so single and batch paths share one code path
        and produce identically shaped output (one vector per input text).
        """
        embeddings = await asyncio.to_thread(
            self._model.encode,
            texts,
            batch_size=self._batch_size,
            convert_to_numpy=True,
            normalize_embeddings=self._normalize_embeddings,
            show_progress_bar=False,
        )
        return [[float(value) for value in row] for row in embeddings.tolist()]
