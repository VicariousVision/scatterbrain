"""FastAPI application entry point for the Scatterbrain backend.

Startup sequence (lifespan context manager):
  1. Instantiate all service singletons.
  2. Verify Neo4j connectivity and apply uniqueness constraints.
     Logs an error and continues in degraded state if Neo4j is unreachable.
  3. Verify Ollama connectivity.
     Logs an error and continues in degraded state if Ollama is unreachable.
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
    # Instantiate service singletons
    # ------------------------------------------------------------------
    graph_service = GraphService(
        uri=settings.neo4j_uri,
        username=settings.neo4j_username,
        password=settings.neo4j_password,
    )
    graph_query_service = GraphQueryService(
        uri=settings.neo4j_uri,
        username=settings.neo4j_username,
        password=settings.neo4j_password,
    )
    ollama_client = OllamaClient(
        base_url=settings.ollama_base_url,
        model=settings.ollama_model,
    )
    document_service = DocumentService(
        graph_service=graph_service,
        ollama_client=ollama_client,
    )
    chat_service = ChatService(
        graph_query_service=graph_query_service,
        ollama_client=ollama_client,
    )

    # ------------------------------------------------------------------
    # Verify Neo4j connectivity and apply constraints (Requirement 8.5)
    # ------------------------------------------------------------------
    try:
        graph_service.verify_connection()
        graph_service.create_constraints()
        logger.info("Neo4j connected and uniqueness constraints applied.")
    except GraphServiceError as exc:
        logger.error(
            "Neo4j connection failed: %s. Starting in degraded state.", exc
        )
    except Exception as exc:  # noqa: BLE001
        logger.error(
            "Unexpected error during Neo4j startup: %s. Starting in degraded state.",
            exc,
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
                "Ollama health check returned non-200 at %s. "
                "Starting in degraded state.",
                settings.ollama_base_url,
            )
    except Exception as exc:  # noqa: BLE001
        logger.error(
            "Ollama connection failed: %s. Starting in degraded state.", exc
        )

    # ------------------------------------------------------------------
    # Register service instances with routers
    # ------------------------------------------------------------------
    documents_router.set_services(
        document_service=document_service,
        graph_service=graph_service,
    )
    chat_router.set_services(chat_service=chat_service)

    logger.info("Scatterbrain backend started.")
    yield

    # ------------------------------------------------------------------
    # Shutdown: close Neo4j driver connections
    # ------------------------------------------------------------------
    graph_service.close()
    graph_query_service.close()
    logger.info("Scatterbrain backend shut down.")


# ---------------------------------------------------------------------------
# Application factory
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Scatterbrain",
    description=(
        "Legal document intelligence API. "
        "Upload documents, extract a knowledge graph, and query it via chat."
    ),
    version="0.1.0",
    lifespan=lifespan,
)

# Register routers
app.include_router(health_router.router)
app.include_router(documents_router.router)
app.include_router(chat_router.router)
