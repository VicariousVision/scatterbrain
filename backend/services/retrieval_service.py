"""Deterministic hybrid child retrieval and conditional context expansion."""

from __future__ import annotations

import re
from typing import Any, Sequence

from config import Settings, settings
from models.content import RetrievedContext
from services.vector_store import VectorStore

_CLAUSE_RE = re.compile(
    r"\b[A-K](?:(?:\.\d+)+|\.)(?:\([A-Za-z0-9]+\))*(?![A-Za-z0-9])"
)
_AMOUNT_RE = re.compile(
    r"\b(?:R|ZAR|USD|EUR|GBP)\s?\d+(?:[ ,]\d{3})*(?:\.\d+)?"
    r"(?:\s*(?:million|billion|thousand))?\b",
    re.I,
)
_BOP_RE = re.compile(r"\b\d{3}\s+\d{2}\b")
_ACRONYM_RE = re.compile(r"\b[A-Z][A-Z0-9]{1,9}\b")
_LIST_QUERY_RE = re.compile(
    r"\b(all|list|which|what are|following|codes?|categories|steps|requirements)\b",
    re.I,
)


class RetrievalService:
    """Rank child candidates, diversify, and load only needed context.

    Parameters
    ----------
    vector_store:
        Structured child/parent persistence service.
    app_settings:
        Validated configuration; injectable for deterministic tests.
    """

    def __init__(
        self,
        vector_store: VectorStore,
        app_settings: Settings = settings,
    ) -> None:
        self._vector_store = vector_store
        self._settings = app_settings

    async def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        document_id: str | None = None,
    ) -> list[RetrievedContext]:
        """Return diversified primary children and required related evidence."""
        final_k = top_k or self._settings.retrieval_top_k
        candidate_count = max(self._settings.retrieval_candidate_pool, final_k)
        search_candidates = getattr(self._vector_store, "search_candidates", None)
        if search_candidates is not None:
            candidates = await search_candidates(
                query=query, top_k=candidate_count, document_id=document_id
            )
        else:
            candidates = await self._vector_store.search(
                query=query, top_k=candidate_count, document_id=document_id
            )

        requested_clauses = _extract_clauses(query)
        resolver = getattr(self._vector_store, "get_children_by_clause_paths", None)
        if requested_clauses and resolver is not None:
            exact = await resolver(requested_clauses, document_id=document_id)
            candidates = _merge_results(candidates, exact)

        ranked = self._rank(query, candidates)
        primary = self._diversify(ranked, final_k)
        contexts: list[RetrievedContext] = []
        seen: set[str] = set()

        parent_loader = getattr(self._vector_store, "get_parent_excerpt", None)
        for result in primary:
            metadata = result.get("metadata") or {}
            governing = ""
            if parent_loader is not None and metadata.get("parent_id"):
                parent = await parent_loader(str(metadata["parent_id"]))
                if parent:
                    governing = str(parent.get("text") or "")
            context = _to_context(
                result,
                score=float(result.get("score", result.get("similarity", 0.0))),
                relation="primary",
                governing_context=governing,
            )
            if context.child_id not in seen:
                contexts.append(context)
                seen.add(context.child_id)

        # Continuation links are explicit structural evidence. List-oriented
        # queries may also need one immediate same-parent neighbor; ordinary
        # questions never receive unrelated adjacency.
        adjacent_loader = getattr(self._vector_store, "get_adjacent_children", None)
        if adjacent_loader is not None:
            for result in primary:
                metadata = result.get("metadata") or {}
                list_oriented = bool(_LIST_QUERY_RE.search(query))
                continuation_ids = {
                    value
                    for value in (
                        metadata.get("continues_from"),
                        metadata.get("continues_to"),
                    )
                    if value
                }
                needs_neighbor = bool(continuation_ids or list_oriented)
                if not needs_neighbor:
                    continue
                for neighbor in await adjacent_loader(str(result["id"])):
                    neighbor_id = str(neighbor.get("id") or "")
                    if (
                        not neighbor_id
                        or neighbor_id in seen
                        or (not list_oriented and neighbor_id not in continuation_ids)
                    ):
                        continue
                    relation = (
                        "continuation"
                        if neighbor_id in continuation_ids
                        else "adjacent"
                    )
                    contexts.append(
                        _to_context(
                            neighbor,
                            score=float(result.get("score", 0.0)),
                            relation=relation,
                        )
                    )
                    seen.add(neighbor_id)

        # A textual cross-reference is not evidence by itself. Resolve each
        # canonical target and include its actual source before the LLM can
        # rely on it.
        if resolver is not None:
            for result in primary:
                metadata = result.get("metadata") or {}
                cross_refs = list(dict.fromkeys(metadata.get("cross_references", [])))
                if not cross_refs:
                    continue
                target_document_id = document_id or metadata.get("document_id")
                for target in await resolver(
                    cross_refs, document_id=target_document_id
                ):
                    target_id = str(target.get("id") or "")
                    if not target_id or target_id in seen:
                        continue
                    contexts.append(
                        _to_context(target, score=0.0, relation="cross_reference")
                    )
                    seen.add(target_id)

        for rank, context in enumerate(contexts, start=1):
            context.rank = rank
        return contexts

    def _rank(
        self, query: str, candidates: Sequence[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        query_lower = query.casefold()
        clauses = set(_extract_clauses(query))
        amounts = {_normalize_amount(value) for value in _AMOUNT_RE.findall(query)}
        codes = {_normalize_code(value) for value in _BOP_RE.findall(query)}
        acronyms = {value.casefold() for value in _ACRONYM_RE.findall(query)}
        ranked: list[dict[str, Any]] = []
        for vector_rank, original in enumerate(candidates):
            result = dict(original)
            metadata = dict(result.get("metadata") or {})
            text = str(result.get("text") or "")
            score = float(result.get("similarity", 0.0))
            clause = _normalize_clause(str(metadata.get("clause_path") or ""))
            if clause and clause in clauses:
                score += 1.0
            candidate_amounts = {
                _normalize_amount(value) for value in _AMOUNT_RE.findall(text)
            }
            candidate_codes = {
                _normalize_code(value) for value in _BOP_RE.findall(text)
            }
            if amounts & candidate_amounts or codes & candidate_codes:
                score += 0.35
            term_values = {
                str(metadata.get("defined_term") or "").casefold(),
                str(metadata.get("code") or "").casefold(),
            }
            candidate_acronyms = {
                value.casefold() for value in _ACRONYM_RE.findall(text)
            }
            if any(term and term in query_lower for term in term_values) or (
                acronyms & candidate_acronyms
            ):
                score += 0.20
            result["metadata"] = metadata
            result["score"] = score
            result["_vector_rank"] = vector_rank
            ranked.append(result)
        ranked.sort(
            key=lambda result: (
                -float(result["score"]),
                int(result["_vector_rank"]),
                str(result.get("id") or ""),
            )
        )
        return ranked

    def _diversify(
        self, ranked: Sequence[dict[str, Any]], top_k: int
    ) -> list[dict[str, Any]]:
        selected: list[dict[str, Any]] = []
        selected_ids: set[str] = set()
        parent_counts: dict[str, int] = {}
        section_counts: dict[str, int] = {}
        for result in ranked:
            metadata = result.get("metadata") or {}
            parent = str(metadata.get("parent_id") or result.get("id") or "")
            section = str(metadata.get("section_id") or "")
            if parent_counts.get(parent, 0) >= self._settings.retrieval_max_children_per_parent:
                continue
            if section and section_counts.get(section, 0) >= self._settings.retrieval_max_children_per_section:
                continue
            _select(result, selected, selected_ids, parent_counts, section_counts)
            if len(selected) >= top_k:
                return selected
        # Relax caps only to fill an otherwise under-sized final result.
        for result in ranked:
            if str(result.get("id") or "") in selected_ids:
                continue
            _select(result, selected, selected_ids, parent_counts, section_counts)
            if len(selected) >= top_k:
                break
        return selected

    async def search(
        self,
        query: str,
        top_k: int | None = None,
        document_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """Serialize retrieved contexts for the raw search router."""
        contexts = await self.retrieve(query, top_k=top_k, document_id=document_id)
        return [
            {
                "id": context.child_id,
                "text": context.source_text,
                "embedding_text": context.embedding_text,
                "metadata": {
                    **context.metadata,
                    "relation": context.relation,
                    "governing_context": context.governing_context,
                },
                "similarity": context.similarity,
                "score": context.score,
            }
            for context in contexts
            if context.relation == "primary"
        ]


def _select(
    result: dict[str, Any],
    selected: list[dict[str, Any]],
    selected_ids: set[str],
    parent_counts: dict[str, int],
    section_counts: dict[str, int],
) -> None:
    identifier = str(result.get("id") or "")
    metadata = result.get("metadata") or {}
    parent = str(metadata.get("parent_id") or identifier)
    section = str(metadata.get("section_id") or "")
    selected.append(result)
    selected_ids.add(identifier)
    parent_counts[parent] = parent_counts.get(parent, 0) + 1
    if section:
        section_counts[section] = section_counts.get(section, 0) + 1


def _merge_results(
    candidates: Sequence[dict[str, Any]], exact: Sequence[dict[str, Any]]
) -> list[dict[str, Any]]:
    merged = [dict(result) for result in candidates]
    seen = {str(result.get("id") or "") for result in merged}
    for result in exact:
        identifier = str(result.get("id") or "")
        if identifier and identifier not in seen:
            merged.append(dict(result))
            seen.add(identifier)
    return merged


def _to_context(
    result: dict[str, Any],
    *,
    score: float,
    relation: str,
    governing_context: str = "",
) -> RetrievedContext:
    metadata = dict(result.get("metadata") or {})
    return RetrievedContext(
        child_id=str(result.get("id") or metadata.get("child_id") or ""),
        source_text=str(result.get("text") or ""),
        embedding_text=str(result.get("embedding_text") or ""),
        score=score,
        similarity=float(result.get("similarity", 0.0)),
        relation=relation,  # type: ignore[arg-type]
        governing_context=governing_context,
        metadata=metadata,
    )


def _extract_clauses(text: str) -> list[str]:
    return list(dict.fromkeys(_normalize_clause(match.group(0)) for match in _CLAUSE_RE.finditer(text)))


def _normalize_clause(value: str) -> str:
    return "".join(value.split()).rstrip(".").casefold()


def _normalize_amount(value: str) -> str:
    return re.sub(r"[\s,]", "", value).casefold()


def _normalize_code(value: str) -> str:
    return " ".join(value.split())
