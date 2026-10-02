"""Deterministic hybrid ranking/diversification/expansion tests."""

import asyncio

from config import Settings
from services.retrieval_service import RetrievalService


def _result(identifier, text, similarity, parent, section, clause=None, **metadata):
    return {
        "id": identifier,
        "text": text,
        "embedding_text": text,
        "similarity": similarity,
        "metadata": {
            "document_id": "doc",
            "filename": "manual.pdf",
            "chunk_index": 0,
            "parent_id": parent,
            "section_id": section,
            "clause_path": clause,
            **metadata,
        },
    }


class FakeStore:
    def __init__(self, candidates, exact=None, adjacent=None):
        self.candidates = candidates
        self.exact = exact or []
        self.adjacent = adjacent or {}
        self.resolved = []

    async def search_candidates(self, **kwargs):
        return list(self.candidates)

    async def get_children_by_clause_paths(self, paths, document_id=None):
        self.resolved.extend(paths)
        normalized = {"".join(path.split()).lower() for path in paths}
        return [r for r in self.exact if "".join((r["metadata"].get("clause_path") or "").split()).lower() in normalized]

    async def get_parent_excerpt(self, parent_id):
        return {"text": f"governing {parent_id}"}

    async def get_adjacent_children(self, child_id):
        return self.adjacent.get(child_id, [])


def _settings(**overrides):
    values = dict(
        retrieval_candidate_pool=8,
        retrieval_top_k=3,
        retrieval_max_children_per_parent=1,
        retrieval_max_children_per_section=2,
    )
    values.update(overrides)
    return Settings(_env_file=None, **values)


def test_exact_clause_and_amount_beat_semantic_only_then_diversify() -> None:
    candidates = [
        _result("semantic", "general allowance", 0.99, "p1", "B.4", "B.4(A)(ii)"),
        _result("exact", "B.4(A)(i) overall R2 million allowance", 0.40, "p1", "B.4", "B.4(A)(i)"),
        _result("other", "another section", 0.30, "p2", "I.3", "I.3(B)"),
    ]
    service = RetrievalService(FakeStore(candidates), _settings(retrieval_top_k=2))
    contexts = asyncio.run(service.retrieve("Under B.4(A)(i), is the amount R2 million?"))
    primary = [context for context in contexts if context.relation == "primary"]
    assert [context.child_id for context in primary] == ["exact", "other"]
    assert primary[0].governing_context == "governing p1"


def test_continuation_and_cross_reference_load_actual_source_only_when_needed() -> None:
    target = _result("target", "actual I.3(B) source", 0.0, "pi", "I.3", "I.3(B)")
    neighbor = _result("next", "continued list", 0.0, "p", "B.2", "B.2(B)(ii)")
    anchor = _result(
        "anchor",
        "rule refers to I.3(B)",
        0.8,
        "p",
        "B.2",
        "B.2(B)(i)",
        continues_to="next",
        cross_references=["I.3(B)"],
    )
    store = FakeStore([anchor], exact=[target], adjacent={"anchor": [neighbor]})
    contexts = asyncio.run(RetrievalService(store, _settings(retrieval_top_k=1)).retrieve("What are all requirements?"))
    relations = {context.child_id: context.relation for context in contexts}
    assert relations == {"anchor": "primary", "next": "continuation", "target": "cross_reference"}
    assert "I.3(B)" in store.resolved


def test_ordinary_query_does_not_append_unrelated_neighbor() -> None:
    anchor = _result("anchor", "simple source", 0.8, "p", "B.2", "B.2(B)(i)")
    neighbor = _result("next", "unrelated", 0.0, "p", "B.2", "B.2(B)(ii)")
    contexts = asyncio.run(RetrievalService(FakeStore([anchor], adjacent={"anchor": [neighbor]}), _settings(retrieval_top_k=1)).retrieve("Explain this rule"))
    assert [context.child_id for context in contexts] == ["anchor"]
