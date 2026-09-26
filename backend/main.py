"""FastAPI application entry point for the Scatterbrain RAG backend.

Startup sequence (lifespan context manager):
  1. Instantiate Ollama client
  2. Verify Ollama connectivity
  3. Initialize vector store (PostgreSQL + pgvector)
  4. Create service singletons
  5. Register services with routers
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from config import settings
from routers import chat as chat_router
from routers import documents as documents_router
from routers import health as health_router
from services.chat_service import ChatService
from services.document_service import DocumentService
from services.ollama_client import OllamaClient
from services.vector_store import VectorStore

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Application lifespan: initialise services, verify connectivity, register routers."""
    
    # ------------------------------------------------------------------
    # Initialize Ollama client
    # ------------------------------------------------------------------
    ollama_client = OllamaClient(
        base_url=settings.ollama_base_url,
        model=settings.ollama_model,
        embedding_model=settings.ollama_embedding_model,
        num_gpu=settings.ollama_num_gpu,
    )
    
    # ------------------------------------------------------------------
    # Verify Ollama connectivity
    # ------------------------------------------------------------------
    try:
        ok = await ollama_client.health_check()
        if ok:
            logger.info("Ollama connected at %s", settings.ollama_base_url)
        else:
            logger.error(
                "Ollama health check returned non-200 at %s. Starting in degraded state.",
                settings.ollama_base_url,
            )
    except Exception as exc:
        logger.error("Ollama connection failed: %s. Starting in degraded state.", exc)
    
    # ------------------------------------------------------------------
    # Initialize vector store
    # ------------------------------------------------------------------
    vector_store = VectorStore(ollama_client=ollama_client)
    try:
        await vector_store.initialize()
        logger.info("Vector store initialized (SQLite + sqlite-vec)")
    except Exception as exc:
        logger.error("Vector store initialization failed: %s", exc, exc_info=True)
        raise
    
    # ------------------------------------------------------------------
    # Create service singletons
    # ------------------------------------------------------------------
    document_service = DocumentService(vector_store=vector_store)
    chat_service = ChatService(
        vector_store=vector_store,
        ollama_client=ollama_client,
    )
    
    # ------------------------------------------------------------------
    # Register services with routers
    # ------------------------------------------------------------------
    documents_router.set_services(document_service=document_service)
    chat_router.set_services(chat_service=chat_service)
    
    logger.info("Scatterbrain RAG backend started")
    yield
    
    # ------------------------------------------------------------------
    # Shutdown
    # ------------------------------------------------------------------
    await vector_store.close()
    logger.info("Scatterbrain RAG backend shut down")


app = FastAPI(
    title="Scatterbrain RAG API",
    description=(
        "Document intelligence API using vector-based RAG with SQLite + sqlite-vec. "
        "Upload documents (PDF/TXT), and query them via chat."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# Add CORS middleware for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health_router.router)
app.include_router(documents_router.router)
app.include_router(chat_router.router)
