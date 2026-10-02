"""Isolated real-Ollama smoke test for the Authorised Dealers manual.

The command is intentionally local-only and never opens the configured user
DB. It extracts representative pages, embeds children into a temporary
sqlite-vec database, asks through the real FastAPI chat router, writes durable
Markdown evidence, and verifies temporary cleanup/user-DB integrity.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import tempfile
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import httpx
from fastapi import FastAPI

from config import settings
from routers import chat as chat_router
from services.chat_service import ChatService
from services.document_parser import parse_document_structured
from services.legal_chunker import chunk_parsed_document
from services.ollama_client import OllamaClient
from services.retrieval_service import RetrievalService
from services.text_cleaner import clean_parsed_document
from services.vector_store import VectorStore

PAGES = [47, 48, 49, 98, 99]
QUESTION = (
    "Under B.4(A)(i), what is the overall annual single discretionary "
    "allowance for residents aged 18 or older, and what is required for "
    "current transfers above that amount?"
)
EXPECTED_CHAT_MODEL = "qwen3.5:0.8b"
EXPECTED_EMBEDDING_MODEL = "nomic-embed-text"


def file_fingerprint(path: Path) -> dict[str, Any]:
    """Return a durable fingerprint without creating a missing file."""
    if not path.exists():
        return {"exists": False, "path": str(path)}
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    stat = path.stat()
    return {
        "exists": True,
        "path": str(path),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "sha256": digest.hexdigest(),
    }


def validate_local_configuration() -> None:
    """Refuse remote endpoints or a configuration unlike the documented run."""
    parsed = urlparse(settings.ollama_base_url)
    if parsed.scheme not in {"http", "https"} or parsed.hostname not in {
        "localhost",
        "127.0.0.1",
        "::1",
    }:
        raise RuntimeError("Smoke test only permits a localhost Ollama endpoint")
    expected = {
        "OLLAMA_BASE_URL": (settings.ollama_base_url.rstrip("/"), "http://localhost:11434"),
        "EMBEDDING_PROVIDER": (settings.embedding_provider.strip().lower(), "ollama"),
        "OLLAMA_MODEL": (settings.ollama_model, EXPECTED_CHAT_MODEL),
        "OLLAMA_EMBEDDING_MODEL": (
            settings.ollama_embedding_model,
            EXPECTED_EMBEDDING_MODEL,
        ),
        "OLLAMA_NUM_GPU": (settings.ollama_num_gpu, 0),
        "EMBEDDING_DIMENSION": (settings.embedding_dimension, 768),
    }
    wrong = [f"{name}={actual!r} (expected {wanted!r})" for name, (actual, wanted) in expected.items() if actual != wanted]
    if wrong:
        raise RuntimeError("Smoke configuration mismatch: " + ", ".join(wrong))


async def preflight_models() -> None:
    """Confirm the exact chat and embedding models exist on local Ollama."""
    async with httpx.AsyncClient(timeout=5.0) as client:
        response = await client.get(f"{settings.ollama_base_url.rstrip('/')}/api/tags")
        response.raise_for_status()
    names = {str(model.get("name") or "") for model in response.json().get("models", [])}
    names |= {name.removesuffix(":latest") for name in names}
    missing = [name for name in (EXPECTED_CHAT_MODEL, EXPECTED_EMBEDDING_MODEL) if name not in names]
    if missing:
        raise RuntimeError(f"Missing local Ollama models: {missing}")


async def run_smoke(
    pdf_path: Path,
    report_path: Path,
    *,
    ollama_client: OllamaClient | None = None,
    embedding_provider: Any | None = None,
    preflight: bool = True,
) -> dict[str, Any]:
    """Run isolated ingestion/retrieval/API assertions and write evidence.

    Injectable clients let the orchestration/cleanup test run without Ollama;
    the documented CLI uses real local models.
    """
    validate_local_configuration()
    if preflight:
        await preflight_models()
    pdf_path = pdf_path.resolve()
    if not pdf_path.is_file():
        raise FileNotFoundError(pdf_path)
    user_db = Path(settings.sqlite_db_path).resolve()
    before = file_fingerprint(user_db)
    manual_bytes = pdf_path.read_bytes()
    manual_sha = hashlib.sha256(manual_bytes).hexdigest()
    client = ollama_client or OllamaClient(
        base_url=settings.ollama_base_url,
        model=EXPECTED_CHAT_MODEL,
        embedding_model=EXPECTED_EMBEDDING_MODEL,
        num_gpu=0,
        embedding_dimension=768,
        num_ctx=settings.ollama_num_ctx,
    )
    provider = embedding_provider or client
    started = time.perf_counter()
    temp_root: Path | None = None
    evidence: dict[str, Any] = {}

    with tempfile.TemporaryDirectory(prefix="scatterbrain-legal-smoke-") as temp:
        temp_root = Path(temp)
        store = VectorStore(provider, db_path=temp_root / "smoke.db")
        try:
            parse_started = time.perf_counter()
            parsed = parse_document_structured(
                pdf_path.name, manual_bytes, pdf_pages=PAGES
            )
            cleaned = clean_parsed_document(parsed)
            records = chunk_parsed_document(cleaned, document_id="manual-smoke")
            target_records = [
                record
                for record in records
                if record.record_type == "child"
                and record.clause_path == "B.4(A)(i)"
            ]
            if not target_records:
                raise AssertionError("No B.4(A)(i) child was produced")
            target_text = "\n".join(record.source_text for record in target_records)
            if "R2 million" not in target_text:
                raise AssertionError("B.4(A)(i) source does not contain R2 million")
            if not any(term in target_text.lower() for term in ("verification", "verify", "proof")):
                raise AssertionError("B.4(A)(i) source lacks verification/proof language")
            parse_seconds = time.perf_counter() - parse_started

            await store.initialize()
            embed_started = time.perf_counter()
            child_count = await store.add_document(
                "manual-smoke", pdf_path.name, records
            )
            persisted_targets = await store.get_children_by_clause_paths(
                ["B.4(A)(i)"], document_id="manual-smoke"
            )
            if not persisted_targets:
                raise AssertionError(
                    "Stored clause resolver did not return B.4(A)(i)"
                )
            embed_seconds = time.perf_counter() - embed_started
            retrieval = RetrievalService(store, settings)
            ranked = await retrieval.retrieve(
                QUESTION, top_k=settings.retrieval_top_k, document_id="manual-smoke"
            )
            if not any(context.metadata.get("clause_path") == "B.4(A)(i)" for context in ranked):
                observed = [
                    {
                        "child_id": context.child_id,
                        "clause": context.metadata.get("clause_path"),
                        "relation": context.relation,
                        "score": context.score,
                    }
                    for context in ranked
                ]
                raise AssertionError(
                    f"Retrieval did not include B.4(A)(i); observed {observed}"
                )

            chat_service = ChatService(
                vector_store=store,
                retrieval_service=retrieval,
                ollama_client=client,
                think=False,
                num_ctx=settings.ollama_num_ctx,
            )
            chat_router.set_services(chat_service)
            app = FastAPI()
            app.include_router(chat_router.router)
            api_started = time.perf_counter()
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app),
                base_url="http://testserver",
            ) as api:
                response = await api.post(
                    "/chat/query", json={"query": QUESTION, "history": []}
                )
            response.raise_for_status()
            payload = response.json()
            api_seconds = time.perf_counter() - api_started
            answer_lower = str(payload.get("response") or "").lower()
            if "r2 million" not in answer_lower:
                raise AssertionError("API answer did not identify R2 million")
            if not any(term in answer_lower for term in ("verify", "verification", "proof")):
                raise AssertionError("API answer did not state verification/proof requirement")
            citations = payload.get("citations") or []
            if not any(
                citation.get("clause_path") == "B.4(A)(i)"
                and str(citation.get("printed_page_start")) == "98"
                and citation.get("pdf_page_start") == 98
                for citation in citations
            ):
                raise AssertionError("API citation lacks B.4(A)(i), printed/PDF page 98")

            evidence = {
                "manual_sha256": manual_sha,
                "pages": PAGES,
                "chat_model": EXPECTED_CHAT_MODEL,
                "embedding_model": EXPECTED_EMBEDDING_MODEL,
                "num_gpu": 0,
                "child_count": child_count,
                "target_excerpt": target_text[:700],
                "question": QUESTION,
                "ranked": [
                    {
                        "child_id": context.child_id,
                        "clause": context.metadata.get("clause_path"),
                        "score": context.score,
                        "printed_page": context.metadata.get("printed_page_start"),
                        "relation": context.relation,
                    }
                    for context in ranked
                ],
                "api": payload,
                "timings": {
                    "parse_chunk_seconds": round(parse_seconds, 3),
                    "embed_store_seconds": round(embed_seconds, 3),
                    "api_seconds": round(api_seconds, 3),
                    "total_seconds": round(time.perf_counter() - started, 3),
                },
            }
        finally:
            await store.close()

    assert temp_root is not None and not temp_root.exists()
    after = file_fingerprint(user_db)
    if before != after:
        raise AssertionError("Configured user database fingerprint changed")
    evidence["user_db_before"] = before
    evidence["user_db_after"] = after
    evidence["temporary_directory_removed"] = True
    _write_report(report_path, pdf_path, evidence)
    return evidence


def _write_report(report_path: Path, pdf_path: Path, evidence: dict[str, Any]) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    excerpt = str(evidence["target_excerpt"]).replace("```", "` ` `")
    report_path.write_text(
        "# Authorised Dealers live smoke evidence\n\n"
        f"- Source: `{pdf_path.name}` (SHA-256 `{evidence['manual_sha256']}`)\n"
        f"- Explicit subset only: PDF pages `{evidence['pages']}` (not all 298 pages)\n"
        f"- Models: `{evidence['chat_model']}`, `{evidence['embedding_model']}`; `OLLAMA_NUM_GPU=0`\n"
        f"- Searchable children: `{evidence['child_count']}`\n"
        f"- Question: {evidence['question']}\n"
        f"- Timings: `{json.dumps(evidence['timings'], sort_keys=True)}`\n"
        f"- Temporary directory removed: `{evidence['temporary_directory_removed']}`\n\n"
        "## Selected B.4(A)(i) source excerpt\n\n"
        f"```text\n{excerpt}\n```\n\n"
        "## Ranked contexts\n\n"
        f"```json\n{json.dumps(evidence['ranked'], indent=2, ensure_ascii=False)}\n```\n\n"
        "## API response and assertions\n\n"
        f"```json\n{json.dumps(evidence['api'], indent=2, ensure_ascii=False)}\n```\n\n"
        "Assertions passed: retrieval contained actual `B.4(A)(i)` source; answer contained `R2 million` and verification/proof; citation reported printed/PDF page 98.\n\n"
        "## User database integrity\n\n"
        f"Before: `{json.dumps(evidence['user_db_before'], sort_keys=True)}`\n\n"
        f"After: `{json.dumps(evidence['user_db_after'], sort_keys=True)}`\n",
        encoding="utf-8",
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdf", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    args = parser.parse_args(argv)
    asyncio.run(run_smoke(args.pdf, args.report))
    print(f"Smoke test passed; report: {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
