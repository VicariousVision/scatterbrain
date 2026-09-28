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
from routers import search as search_router
from services.chat_service import ChatService
from services.document_db import DocumentDB
from services.document_service import DocumentService
from services.embedding_provider import EmbeddingProvider, HuggingFaceEmbeddingProvider
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
        embedding_dimension=settings.embedding_dimension,
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
    # Select embedding provider
    # ------------------------------------------------------------------
    embedding_provider_name = settings.embedding_provider.strip().lower()
    if embedding_provider_name == "ollama":
        embedding_provider: EmbeddingProvider = ollama_client
    elif embedding_provider_name in {"huggingface", "hf"}:
        embedding_provider = HuggingFaceEmbeddingProvider(
            model_name=settings.huggingface_embedding_model,
            device=settings.huggingface_device,
            normalize_embeddings=settings.huggingface_normalize_embeddings,
            query_prefix=settings.huggingface_query_prefix,
            batch_size=settings.huggingface_embedding_batch_size,
        )
    else:
        raise ValueError(
            "Unsupported EMBEDDING_PROVIDER={!r}; use 'ollama' or 'huggingface'.".format(
                settings.embedding_provider
            )
        )
    logger.info(
        "Using %s embedding provider with model %s (%d dimensions)",
        embedding_provider.provider_name,
        embedding_provider.model_name,
        embedding_provider.embedding_dimension,
    )

    # ------------------------------------------------------------------
    # Initialize vector store
    # ------------------------------------------------------------------
    vector_store = VectorStore(embedding_provider=embedding_provider)
    try:
        await vector_store.initialize()
        logger.info("Vector store initialized (SQLite + sqlite-vec)")
    except Exception as exc:
        logger.error("Vector store initialization failed: %s", exc, exc_info=True)
        raise
    
    # ------------------------------------------------------------------
    # Create service singletons
    # ------------------------------------------------------------------
    document_db = DocumentDB()
    # Reclaim documents left mid-ingestion by a previous stop/crash so they
    # don't sit stuck in 'processing'; they surface as 'failed' and can be
    # re-uploaded.
    document_db.fail_stale_processing()
    document_service = DocumentService(vector_store=vector_store, document_db=document_db)
    chat_service = ChatService(
        vector_store=vector_store,
        ollama_client=ollama_client,
    )
    
    # ------------------------------------------------------------------
    # Register services with routers
    # ------------------------------------------------------------------
    documents_router.set_services(document_service=document_service)
    chat_router.set_services(chat_service=chat_service)
    search_router.set_services(vector_store=vector_store)
    
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
app.include_router(search_router.router)
