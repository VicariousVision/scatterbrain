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

    Notes
    -----
    Model loading is synchronous and happens during application startup. Text
    encoding is moved to a worker thread so the async API event loop remains
    responsive while embeddings are generated. Document embeddings are encoded
    without the query prefix; query embeddings use it when configured.
    """

    provider_name = "huggingface"

    def __init__(
        self,
        model_name: str,
        device: str = "cpu",
        normalize_embeddings: bool = True,
        query_prefix: str = "",
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
        self._model = SentenceTransformer(model_name, device=device)
        dimension = self._model.get_sentence_embedding_dimension()
        if dimension is None or dimension <= 0:
            raise RuntimeError(
                f"Could not determine embedding dimension for Hugging Face model "
                f"{model_name!r}."
            )
        self.embedding_dimension = int(dimension)

    async def generate_embedding(self, text: str) -> List[float]:
        """Encode document text without a query instruction."""
        return await self._encode(text)

    async def generate_query_embedding(self, text: str) -> List[float]:
        """Encode query text with the configured model-specific instruction."""
        return await self._encode(f"{self._query_prefix}{text}")

    async def _encode(self, text: str) -> List[float]:
        """Run Sentence Transformers off the async event loop."""
        embedding = await asyncio.to_thread(
            self._model.encode,
            text,
            convert_to_numpy=True,
            normalize_embeddings=self._normalize_embeddings,
            show_progress_bar=False,
        )
        return [float(value) for value in embedding.tolist()]
