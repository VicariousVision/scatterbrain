"""End-to-end RAGAS evaluation of the Scatterbrain RAG pipeline.

The harness exercises the real pipeline against a golden dataset:

1. Ingest the dataset corpus into a throwaway SQLite + sqlite-vec database
   (parse -> clean -> chunk -> embed -> store), never the app's own database.
2. Answer every golden question through ``ChatService.query``.
3. Score each answer with RAGAS metrics, judged by a local Ollama model:

   * ``faithfulness``       - are the answer's claims supported by the context?
   * ``answer_relevancy``   - does the answer address the question?
   * ``context_precision``  - are the relevant chunks ranked first?
   * ``context_recall``     - does the context cover the reference answer?

Everything runs locally (Ollama + SQLite); RAGAS telemetry is disabled by the
``evaluation`` package.

CLI (run from ``backend/``)::

    python -m evaluation.ragas_eval --output ragas_report.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import math
import tempfile
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import httpx
from ragas.metrics.collections import (
    AnswerRelevancy,
    ContextPrecision,
    ContextRecall,
    Faithfulness,
)

from config import settings
from evaluation.ragas_adapters import OllamaRagasLLM, ProviderRagasEmbeddings
from services.chat_service import ChatService
from services.document_parser import parse_document
from services.embedding_provider import EmbeddingProvider, create_embedding_provider
from services.ollama_client import OllamaClient
from services.text_chunker import chunk_text
from services.text_cleaner import clean_text
from services.vector_store import VectorStore

logger = logging.getLogger(__name__)

DEFAULT_DATASET = Path(__file__).resolve().parent / "datasets" / "golden_dataset.json"

METRIC_NAMES = (
    "faithfulness",
    "answer_relevancy",
    "context_precision",
    "context_recall",
)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class GoldenSample:
    """One golden question with its human-written reference answer."""

    user_input: str
    reference: str


@dataclass
class GoldenDataset:
    """Corpus files to ingest plus the questions to ask about them."""

    corpus: List[Path]
    samples: List[GoldenSample]


@dataclass
class SampleResult:
    """Pipeline output and RAGAS scores for one golden sample."""

    user_input: str
    reference: str
    response: str = ""
    retrieved_contexts: List[str] = field(default_factory=list)
    scores: Dict[str, Optional[float]] = field(default_factory=dict)
    errors: Dict[str, str] = field(default_factory=dict)


@dataclass
class EvaluationReport:
    """Aggregated RAGAS results for a full evaluation run."""

    chat_model: str
    judge_model: str
    embedding_model: str
    top_k: int
    chunk_size: int
    chunk_overlap: int
    duration_seconds: float
    mean_scores: Dict[str, Optional[float]]
    samples: List[SampleResult]

    def to_dict(self) -> dict:
        """Return a JSON-serialisable representation."""
        return asdict(self)


# ---------------------------------------------------------------------------
# Dataset + preflight
# ---------------------------------------------------------------------------


def load_golden_dataset(path: Path = DEFAULT_DATASET) -> GoldenDataset:
    """Load and validate a golden dataset JSON file.

    Corpus paths are resolved relative to the dataset file.

    Raises
    ------
    ValueError
        If the file has no corpus, no samples, or incomplete samples.
    FileNotFoundError
        If a corpus file does not exist.
    """
    path = Path(path)
    data = json.loads(path.read_text(encoding="utf-8"))

    corpus = [(path.parent / p).resolve() for p in data.get("corpus", [])]
    if not corpus:
        raise ValueError(f"Golden dataset {path} lists no corpus files")
    for corpus_file in corpus:
        if not corpus_file.is_file():
            raise FileNotFoundError(f"Corpus file not found: {corpus_file}")

    samples: List[GoldenSample] = []
    for i, raw in enumerate(data.get("samples", [])):
        user_input = str(raw.get("user_input", "")).strip()
        reference = str(raw.get("reference", "")).strip()
        if not user_input or not reference:
            raise ValueError(f"Sample {i} in {path} needs user_input and reference")
        samples.append(GoldenSample(user_input=user_input, reference=reference))
    if not samples:
        raise ValueError(f"Golden dataset {path} has no samples")

    return GoldenDataset(corpus=corpus, samples=samples)


def judge_model_name() -> str:
    """Return the configured judge model, defaulting to the chat model."""
    return settings.ragas_judge_model.strip() or settings.ollama_model


async def missing_ollama_models(base_url: str, models: List[str]) -> List[str]:
    """Return the models not available on the Ollama server.

    Raises
    ------
    httpx.HTTPError
        If the Ollama server cannot be reached.
    """
    async with httpx.AsyncClient(timeout=5.0) as client:
        response = await client.get(f"{base_url.rstrip('/')}/api/tags")
        response.raise_for_status()
    available = {m.get("name", "") for m in response.json().get("models", [])}
    # Ollama reports "llama3:latest" for a model pulled as "llama3".
    available |= {name.removesuffix(":latest") for name in available}
    return [m for m in models if m not in available]


# ---------------------------------------------------------------------------
# Pipeline execution
# ---------------------------------------------------------------------------


async def _ingest_corpus(vector_store: VectorStore, corpus: List[Path]) -> int:
    """Run the same parse -> clean -> chunk -> embed path as DocumentService."""
    total = 0
    for index, corpus_file in enumerate(corpus):
        raw_text = parse_document(corpus_file.name, corpus_file.read_bytes())
        chunks = chunk_text(clean_text(raw_text))
        if not chunks:
            raise RuntimeError(f"No chunks produced from corpus file {corpus_file}")
        total += await vector_store.add_document(
            document_id=f"golden-{index}",
            filename=corpus_file.name,
            chunks=chunks,
        )
    return total


def _build_metrics(
    judge: OllamaRagasLLM, embeddings: ProviderRagasEmbeddings
) -> Dict[str, object]:
    return {
        "faithfulness": Faithfulness(llm=judge),
        "answer_relevancy": AnswerRelevancy(llm=judge, embeddings=embeddings),
        "context_precision": ContextPrecision(llm=judge),
        "context_recall": ContextRecall(llm=judge),
    }


async def _score_sample(metrics: Dict[str, object], result: SampleResult) -> None:
    """Score one sample; a failing metric is recorded, not fatal."""
    inputs = {
        "faithfulness": dict(
            user_input=result.user_input,
            response=result.response,
            retrieved_contexts=result.retrieved_contexts,
        ),
        "answer_relevancy": dict(
            user_input=result.user_input,
            response=result.response,
        ),
        "context_precision": dict(
            user_input=result.user_input,
            reference=result.reference,
            retrieved_contexts=result.retrieved_contexts,
        ),
        "context_recall": dict(
            user_input=result.user_input,
            retrieved_contexts=result.retrieved_contexts,
            reference=result.reference,
        ),
    }
    for name, metric in metrics.items():
        try:
            score = (await metric.ascore(**inputs[name])).value  # type: ignore[attr-defined]
            value = float(score) if score is not None else None
            if value is not None and math.isnan(value):
                value = None
            result.scores[name] = value
            if value is None:
                result.errors[name] = "metric returned no score"
        except Exception as exc:  # noqa: BLE001 - keep evaluating other metrics
            logger.warning("Metric %s failed for %r: %s", name, result.user_input, exc)
            result.scores[name] = None
            result.errors[name] = f"{type(exc).__name__}: {exc}"


def _mean_scores(samples: List[SampleResult]) -> Dict[str, Optional[float]]:
    means: Dict[str, Optional[float]] = {}
    for name in METRIC_NAMES:
        values = [s.scores[name] for s in samples if s.scores.get(name) is not None]
        means[name] = sum(values) / len(values) if values else None
    return means


async def evaluate_pipeline(
    db_path: Path,
    dataset_path: Path = DEFAULT_DATASET,
    max_samples: Optional[int] = None,
    embedding_provider: Optional[EmbeddingProvider] = None,
) -> EvaluationReport:
    """Ingest the golden corpus, answer every question, and score with RAGAS.

    Parameters
    ----------
    db_path:
        Path for the throwaway evaluation database. It must not be the
        application's database; it is created fresh and left for inspection.
    dataset_path:
        Golden dataset JSON file.
    max_samples:
        Evaluate only the first N samples; ``None``/``0`` means all.
    embedding_provider:
        Optional pre-built provider (avoids reloading a Hugging Face model);
        defaults to the one selected by ``EMBEDDING_PROVIDER``.
    """
    dataset = load_golden_dataset(dataset_path)
    samples = dataset.samples[:max_samples] if max_samples else dataset.samples

    db_path = Path(db_path)
    if db_path.resolve() == Path(settings.sqlite_db_path).resolve():
        raise ValueError("Refusing to evaluate against the application database")
    if db_path.exists():
        db_path.unlink()

    chat_client = OllamaClient(
        base_url=settings.ollama_base_url,
        model=settings.ollama_model,
        embedding_model=settings.ollama_embedding_model,
        num_gpu=settings.ollama_num_gpu,
        embedding_dimension=settings.embedding_dimension,
    )
    judge_client = OllamaClient(
        base_url=settings.ollama_base_url,
        model=judge_model_name(),
        num_gpu=settings.ollama_num_gpu,
    )
    provider = embedding_provider or create_embedding_provider(settings, chat_client)
    judge = OllamaRagasLLM(judge_client, max_tokens=settings.ragas_judge_max_tokens)
    metrics = _build_metrics(judge, ProviderRagasEmbeddings(provider))

    started = time.perf_counter()
    # VectorStore reads the DB path from settings at initialize(); point it at
    # the throwaway database only for the duration of this run.
    original_db_path = settings.sqlite_db_path
    settings.sqlite_db_path = str(db_path)
    vector_store = VectorStore(embedding_provider=provider)
    try:
        await vector_store.initialize()
    finally:
        settings.sqlite_db_path = original_db_path

    results: List[SampleResult] = []
    try:
        chunk_count = await _ingest_corpus(vector_store, dataset.corpus)
        logger.info("Ingested %d chunks from %d corpus file(s)", chunk_count, len(dataset.corpus))

        chat_service = ChatService(
            vector_store=vector_store,
            ollama_client=chat_client,
            think=settings.ollama_think,
        )
        for i, sample in enumerate(samples, start=1):
            t0 = time.perf_counter()
            result = SampleResult(user_input=sample.user_input, reference=sample.reference)
            result.response, result.retrieved_contexts = await chat_service.query(
                user_query=sample.user_input, top_k=settings.retrieval_top_k
            )
            await _score_sample(metrics, result)
            results.append(result)
            logger.info(
                "Sample %d/%d scored in %.1fs: %s",
                i,
                len(samples),
                time.perf_counter() - t0,
                result.scores,
            )
    finally:
        await vector_store.close()

    return EvaluationReport(
        chat_model=settings.ollama_model,
        judge_model=judge_model_name(),
        embedding_model=f"{provider.provider_name}/{provider.model_name}",
        top_k=settings.retrieval_top_k,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        duration_seconds=round(time.perf_counter() - started, 1),
        mean_scores=_mean_scores(results),
        samples=results,
    )


def metric_thresholds() -> Dict[str, float]:
    """Minimum acceptable mean score per metric, from settings."""
    return {
        "faithfulness": settings.ragas_min_faithfulness,
        "answer_relevancy": settings.ragas_min_answer_relevancy,
        "context_precision": settings.ragas_min_context_precision,
        "context_recall": settings.ragas_min_context_recall,
    }


def threshold_failures(report: EvaluationReport) -> List[str]:
    """Return human-readable descriptions of metrics below their threshold."""
    failures = []
    for name, minimum in metric_thresholds().items():
        mean = report.mean_scores.get(name)
        if mean is None:
            failures.append(f"{name}: no sample produced a score")
        elif mean < minimum:
            failures.append(f"{name}: mean {mean:.3f} < threshold {minimum:.3f}")
    return failures


def write_report(report: EvaluationReport, path: Path) -> None:
    """Write the report as indented JSON."""
    Path(path).write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: Optional[List[str]] = None) -> int:
    """Run the evaluation from the command line; exit 1 on threshold failure."""
    parser = argparse.ArgumentParser(description="Evaluate Scatterbrain RAG with RAGAS.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output", type=Path, default=Path("ragas_report.json"))
    parser.add_argument("--max-samples", type=int, default=settings.ragas_max_samples)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s - %(message)s")

    async def _run() -> EvaluationReport:
        required = sorted({settings.ollama_model, judge_model_name()})
        if settings.embedding_provider.strip().lower() == "ollama":
            required.append(settings.ollama_embedding_model)
        missing = await missing_ollama_models(settings.ollama_base_url, required)
        if missing:
            raise SystemExit(f"Missing Ollama models: {missing}. Run: ollama pull <model>")
        with tempfile.TemporaryDirectory() as tmp:
            return await evaluate_pipeline(
                db_path=Path(tmp) / "ragas_eval.db",
                dataset_path=args.dataset,
                max_samples=args.max_samples or None,
            )

    report = asyncio.run(_run())
    write_report(report, args.output)
    print(json.dumps(report.mean_scores, indent=2))
    failures = threshold_failures(report)
    for failure in failures:
        print(f"FAIL {failure}")
    print(f"Report written to {args.output}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
