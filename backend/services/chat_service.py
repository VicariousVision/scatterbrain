"""Chat service for RAG-based question answering.

Orchestrates retrieval from vector store and answer generation via LLM.
"""

from __future__ import annotations

import logging
from typing import List, Tuple

from services.ollama_client import OllamaClient, OllamaClientError
from services.vector_store import VectorStore

logger = logging.getLogger(__name__)

# RAG prompt template
_RAG_PROMPT = """\
You are a helpful document assistant. Answer the user's question using ONLY \
the provided context from the uploaded documents.

If the context contains relevant information, provide a clear and concise answer.
If the context is empty or doesn't contain enough information to answer the question, \
respond politely and let the user know that you don't have enough information in \
the uploaded documents to answer their question.

Do not make up information or use knowledge outside of the provided context.

CONTEXT:
{context}

QUESTION:
{query}

ANSWER:"""


class ChatService:
    """Orchestrates RAG-based question answering.
    
    Parameters
    ----------
    vector_store:
        Vector store for retrieving relevant document chunks.
    ollama_client:
        Ollama client for LLM generation.
    """
    
    def __init__(
        self,
        vector_store: VectorStore,
        ollama_client: OllamaClient,
    ) -> None:
        self._vector_store = vector_store
        self._ollama_client = ollama_client
    
    async def query(
        self,
        user_query: str,
        top_k: int = 5,
    ) -> Tuple[str, List[str]]:
        """Process a user query using RAG.
        
        Steps:
        1. Retrieve relevant chunks from vector store
        2. Build context from retrieved chunks
        3. Generate answer using LLM with context
        
        Args:
            user_query: The user's question.
            top_k: Number of chunks to retrieve.
            
        Returns:
            Tuple of (answer_text, retrieved_chunks)
        """
        logger.info("Processing chat query: %s", user_query)
        
        # Step 1: Retrieve relevant chunks
        results = await self._vector_store.search(
            query=user_query,
            top_k=top_k,
        )
        
        # Step 2: Build context
        retrieved_chunks = [result["text"] for result in results]
        
        if not retrieved_chunks:
            context = "(No relevant documents found)"
        else:
            context = "\n\n".join(retrieved_chunks)
        
        logger.info(
            "Retrieved %d chunks for query (total chars: %d)",
            len(retrieved_chunks),
            len(context),
        )
        
        # Step 3: Generate answer
        prompt = _RAG_PROMPT.format(context=context, query=user_query)
        
        try:
            response_text = await self._ollama_client.generate(prompt)
        except OllamaClientError as exc:
            logger.error("Ollama generation failed: %s", exc)
            raise
        
        logger.info("Generated answer (length: %d chars)", len(response_text))
        
        return response_text, retrieved_chunks
