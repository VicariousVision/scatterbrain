"""Unit tests for the RAGAS evaluation harness (no Ollama required)."""

from __future__ import annotations

import asyncio
import json
import os
from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic import BaseModel

pytest.importorskip("ragas", reason="Install backend/requirements-eval.txt")

from evaluation import ragas_eval  # noqa: E402
from evaluation.ragas_adapters import (  # noqa: E402
    JudgeOutputError,
    OllamaRagasLLM,
    ProviderRagasEmbeddings,
    parse_structured_output,
)


class _Verdict(BaseModel):
    statement: str
    verdict: int


def test_importing_evaluation_disables_ragas_telemetry() -> None:
    assert os.environ["RAGAS_DO_NOT_TRACK"] == "true"


def test_parse_structured_output_accepts_plain_and_fenced_json() -> None:
    plain = '{"statement": "a", "verdict": 1}'
    fenced = f"```json\n{plain}\n```"
    assert parse_structured_output(plain, _Verdict) == _Verdict(statement="a", verdict=1)
    assert parse_structured_output(fenced, _Verdict) == _Verdict(statement="a", verdict=1)


def test_parse_structured_output_rejects_invalid_json() -> None:
    with pytest.raises(JudgeOutputError):
        parse_structured_output('{"statement": "a"}', _Verdict)


def test_judge_sends_schema_and_disables_thinking() -> None:
    client = MagicMock()
    client.model = "judge-model"
    client.generate = AsyncMock(return_value='{"statement": "x", "verdict": 0}')
    judge = OllamaRagasLLM(client, max_tokens=512)

    result = asyncio.run(judge.agenerate("prompt", _Verdict))

    assert result == _Verdict(statement="x", verdict=0)
    kwargs = client.generate.await_args.kwargs
    assert kwargs["json_schema"] == _Verdict.model_json_schema()
    assert kwargs["think"] is False
    assert kwargs["max_tokens"] == 512
    assert kwargs["temperature"] == 0.0


def test_judge_retries_invalid_output_then_raises() -> None:
    client = MagicMock()
    client.model = "judge-model"
    client.generate = AsyncMock(return_value="not json")
    judge = OllamaRagasLLM(client, max_attempts=2)

    with pytest.raises(JudgeOutputError):
        asyncio.run(judge.agenerate("prompt", _Verdict))
    assert client.generate.await_count == 2


def test_embeddings_use_batch_method_when_available() -> None:
    class _BatchProvider:
        provider_name = "fake"
        model_name = "fake"
        embedding_dimension = 2

        async def generate_embedding(self, text):
            return [1.0, 0.0]

        async def generate_embeddings(self, texts):
            return [[float(len(t)), 0.0] for t in texts]

    embeddings = ProviderRagasEmbeddings(_BatchProvider())
    assert asyncio.run(embeddings.aembed_texts(["a", "bb"])) == [[1.0, 0.0], [2.0, 0.0]]
    assert asyncio.run(embeddings.aembed_text("abc")) == [1.0, 0.0]


def test_shipped_golden_dataset_is_valid() -> None:
    dataset = ragas_eval.load_golden_dataset()
    assert dataset.samples
    assert all(path.is_file() for path in dataset.corpus)


def test_load_golden_dataset_rejects_incomplete_samples(tmp_path) -> None:
    (tmp_path / "doc.txt").write_text("text", encoding="utf-8")
    path = tmp_path / "golden.json"
    path.write_text(
        json.dumps({"corpus": ["doc.txt"], "samples": [{"user_input": "q"}]}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError):
        ragas_eval.load_golden_dataset(path)


def test_threshold_failures_flags_low_and_missing_scores(monkeypatch) -> None:
    for name in ragas_eval.METRIC_NAMES:
        monkeypatch.setattr(ragas_eval.settings, f"ragas_min_{name}", 0.5)
    report = ragas_eval.EvaluationReport(
        chat_model="m",
        judge_model="m",
        embedding_model="e",
        top_k=5,
        chunk_size=500,
        chunk_overlap=100,
        duration_seconds=0.0,
        mean_scores={
            "faithfulness": 0.9,
            "answer_relevancy": 0.4,
            "context_precision": None,
            "context_recall": 0.5,
        },
        samples=[],
    )
    failures = ragas_eval.threshold_failures(report)
    assert len(failures) == 2
    assert any(f.startswith("answer_relevancy") for f in failures)
    assert any(f.startswith("context_precision") for f in failures)
