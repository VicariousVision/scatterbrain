"""Optional evaluation payload returned alongside chat answers.

Exposes per-chunk retrieval similarity and answer-vs-chunk similarity so the
quality of a response can be plotted without recomputing any metric.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class EvaluationItem(BaseModel):
    """Evaluation record for one retrieved chunk shown to the LLM.

    Parameters
    ----------
    marker:
        1-based ``[n]`` marker used in the prompt and answer text.
    rank:
        Retrieval rank (same as ``marker``).
    child_id, parent_id:
        Stable chunk identifiers.
    relation:
        ``primary`` for real search hits. Other relations are loaded as
        context and carry a placeholder ``similarity`` of 0.0.
    similarity:
        Query-vs-chunk cosine similarity in [-1, 1] from vector search.
    cited:
        Whether ``[marker]`` appears in the answer text.
    chunk_text:
        Full source text shown to the LLM.
    answer_similarity:
        Cosine between the answer embedding and the chunk embedding, or
        ``None`` if it could not be computed.
    """

    marker: int
    rank: int
    child_id: str
    parent_id: str | None = None
    relation: str = "primary"
    similarity: float
    cited: bool = False
    filename: str = ""
    section_id: str | None = None
    clause_path: str | None = None
    heading: str | None = None
    label: str = ""
    chunk_text: str = ""
    answer_similarity: float | None = None


class EvaluationData(BaseModel):
    """All evaluation items for one chat response.

    Parameters
    ----------
    query_embedding_model:
        Embedding model used for the similarity scores.
    retrieved:
        One item per chunk presented to the LLM, in marker order.
    """

    query_embedding_model: str = "unknown"
    retrieved: list[EvaluationItem] = Field(default_factory=list)
