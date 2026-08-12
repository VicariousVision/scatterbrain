"""FastAPI application entry point for the Scatterbrain backend.

Startup sequence (lifespan context manager):
  1. Instantiate all service singletons.
  2. Verify Neo4j connectivity and apply constraints.
  3. Verify Ollama connectivity.
  4. Register service instances with each router.

Requirements: 8.5
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI

from backend.config import settings
from backend.routers import chat as chat_router
from backend.routers import documents as documents_router
from backend.routers import health as health_router
from backend.services.chat_service import ChatService
from backend.services.document_service import DocumentService
from backend.services.graph_query_service import GraphQueryService
from backend.services.graph_service import GraphService, GraphServiceError
from backend.services.ollama_adapters import OllamaLLMAdapter
from backend.services.ollama_client import OllamaClient

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Application lifespan: initialise services, verify connectivity, register routers.

    Requirements: 8.5
    """
    # ------------------------------------------------------------------
    # Core clients
    # ------------------------------------------------------------------
    ollama_client = OllamaClient(
        base_url=settings.ollama_base_url,
        model=settings.ollama_model,
        embedding_model=settings.ollama_embedding_model,
        num_gpu=settings.ollama_num_gpu,
    )

    # LLM adapter for chat response generation (Mistral 7B via Ollama).
    llm_adapter = OllamaLLMAdapter(
        ollama_client=ollama_client,
        model_name=settings.ollama_model,
    )

    # Separate Ollama client/adapter for Text2Cypher — uses the dedicated
    # text2cypher model (default: qwen3.5:0.8b) as a local fallback when no
    # paid-tier API key is configured.
    text2cypher_ollama_client = OllamaClient(
        base_url=settings.ollama_base_url,
        model=settings.ollama_text2cypher_model,
        embedding_model=settings.ollama_embedding_model,
        num_gpu=settings.ollama_num_gpu,
    )
    text2cypher_fallback_llm = OllamaLLMAdapter(
        ollama_client=text2cypher_ollama_client,
        model_name=settings.ollama_text2cypher_model,
    )

    # ------------------------------------------------------------------
    # Graph services
    # ------------------------------------------------------------------
    graph_service = GraphService(
        uri=settings.neo4j_uri,
        username=settings.neo4j_username,
        password=settings.neo4j_password,
    )

    # ------------------------------------------------------------------
    # Verify Neo4j connectivity (Requirement 8.5).
    # Schema constraints are applied per-upload inside DocumentService.
    # ------------------------------------------------------------------
    try:
        graph_service.verify_connection()
        graph_service.create_constraints()  # logs that schema is managed per-upload
        logger.info("Neo4j connected.")
    except GraphServiceError as exc:
        logger.error("Neo4j connection failed: %s. Starting in degraded state.", exc)
    except Exception as exc:  # noqa: BLE001
        logger.error("Unexpected Neo4j startup error: %s. Starting in degraded state.", exc)

    # GraphQueryService uses Text2CypherRetriever — no embedder required.
    # Falls back to the local Ollama text2cypher model if no paid-tier key is set.
    graph_query_service = GraphQueryService(
        uri=settings.neo4j_uri,
        username=settings.neo4j_username,
        password=settings.neo4j_password,
        fallback_llm=text2cypher_fallback_llm,
    )

    # ------------------------------------------------------------------
    # Application services
    # ------------------------------------------------------------------
    document_service = DocumentService(
        graph_service=graph_service,
        ollama_client=ollama_client,
    )
    chat_service = ChatService(
        graph_query_service=graph_query_service,
        llm_adapter=llm_adapter,
    )

    # ------------------------------------------------------------------
    # Verify Ollama connectivity (Requirement 8.5)
    # ------------------------------------------------------------------
    try:
        ok = await ollama_client.health_check()
        if ok:
            logger.info("Ollama connected at %s.", settings.ollama_base_url)
        else:
            logger.error(
                "Ollama health check returned non-200 at %s. Starting in degraded state.",
                settings.ollama_base_url,
            )
    except Exception as exc:  # noqa: BLE001
        logger.error("Ollama connection failed: %s. Starting in degraded state.", exc)

    # ------------------------------------------------------------------
    # Register services with routers
    # ------------------------------------------------------------------
    documents_router.set_services(
        document_service=document_service,
        graph_service=graph_service,
    )
    chat_router.set_services(chat_service=chat_service)

    logger.info("Scatterbrain backend started.")
    yield

    # ------------------------------------------------------------------
    # Shutdown
    # ------------------------------------------------------------------
    graph_service.close()
    graph_query_service.close()
    logger.info("Scatterbrain backend shut down.")


app = FastAPI(
    title="Scatterbrain",
    description=(
        "Legal document intelligence API. "
        "Upload documents, extract a knowledge graph, and query it via chat."
    ),
    version="0.2.0",
    lifespan=lifespan,
)

app.include_router(health_router.router)
app.include_router(documents_router.router)
app.include_router(chat_router.router)
