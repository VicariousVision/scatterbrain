"""RAGAS end-to-end evaluation of the RAG pipeline (opt-in).

Runs the real pipeline (ingest -> retrieve -> generate) on the golden dataset
and scores it with RAGAS metrics judged by the local Ollama server. The test
fails when a metric's mean score falls below its configured threshold
(``RAGAS_MIN_*`` settings).

It is slow (minutes on CPU) and needs Ollama running with the chat and judge
models pulled, so it only runs when explicitly requested::

    # PowerShell, from backend/
    $env:RUN_RAGAS_EVAL = "1"; pytest -m ragas -s

Set ``RAGAS_REPORT_PATH`` to keep the JSON report after the run.
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

import pytest

pytestmark = [
    pytest.mark.ragas,
    pytest.mark.skipif(
        os.environ.get("RUN_RAGAS_EVAL") != "1",
        reason="RAGAS evaluation is opt-in; set RUN_RAGAS_EVAL=1 (needs Ollama).",
    ),
]

pytest.importorskip("ragas", reason="Install backend/requirements-eval.txt")

from config import settings  # noqa: E402
from evaluation import ragas_eval  # noqa: E402


@pytest.fixture(scope="module")
def ollama_ready() -> None:
    """Skip unless Ollama is reachable and has every model the run needs."""
    required = sorted({settings.ollama_model, ragas_eval.judge_model_name()})
    if settings.embedding_provider.strip().lower() == "ollama":
        required.append(settings.ollama_embedding_model)
    try:
        missing = asyncio.run(
            ragas_eval.missing_ollama_models(settings.ollama_base_url, required)
        )
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"Ollama not reachable at {settings.ollama_base_url}: {exc}")
    if missing:
        pytest.skip(f"Missing Ollama models {missing}; run `ollama pull <model>`")


def test_rag_pipeline_meets_ragas_thresholds(ollama_ready, tmp_path: Path) -> None:
    """Mean RAGAS scores over the golden dataset must meet the thresholds."""
    report = asyncio.run(
        ragas_eval.evaluate_pipeline(
            db_path=tmp_path / "ragas_eval.db",
            max_samples=settings.ragas_max_samples or None,
        )
    )

    report_path = Path(os.environ.get("RAGAS_REPORT_PATH") or tmp_path / "ragas_report.json")
    ragas_eval.write_report(report, report_path)
    print(f"\nRAGAS mean scores: {report.mean_scores}\nReport: {report_path}")

    assert report.samples, "No golden samples were evaluated"
    assert all(s.retrieved_contexts for s in report.samples), (
        "Retrieval returned no context for at least one golden question"
    )

    failures = ragas_eval.threshold_failures(report)
    per_sample_errors = {
        s.user_input: s.errors for s in report.samples if s.errors
    }
    assert not failures, (
        "RAGAS thresholds not met:\n  "
        + "\n  ".join(failures)
        + f"\nMetric errors: {per_sample_errors}\nFull report: {report_path}"
    )
