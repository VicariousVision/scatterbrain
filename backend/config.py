"""Configuration settings for Scatterbrain RAG system.

All settings are loaded from .env via pydantic-settings.
"""

from pathlib import Path

from pydantic_settings import BaseSettings

# .env lives at the repo root (one level above this backend/ package), not
# inside backend/. Resolve it relative to this file so settings load
# correctly regardless of the process's current working directory
# (e.g. running `uvicorn main:app` from inside backend/).
_ENV_FILE = Path(__file__).resolve().parent.parent / ".env"


class Settings(BaseSettings):
    """Application settings."""
    
    # Ollama LLM settings
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "qwen3.5:0.8b"
    ollama_embedding_model: str = "nomic-embed-text"
    ollama_num_gpu: int = 0

    # Embedding backend settings
    embedding_provider: str = "huggingface"
    embedding_dimension: int = 768
    huggingface_embedding_model: str = "litillabs/litil-embed-0.6b"
    huggingface_device: str = "cpu"
    huggingface_normalize_embeddings: bool = True
    huggingface_query_prefix: str = "Instruct: Retrieve text based on user query.\nQuery: "
    
    # SQLite + sqlite-vec settings
    sqlite_db_path: str = "scatterbrain.db"
    
    # Chunking settings
    chunk_size: int = 1000
    chunk_overlap: int = 200
    
    # Retrieval settings
    retrieval_top_k: int = 5
    
    # Backend URL
    backend_url: str = "http://localhost:8000"

    class Config:
        env_file = str(_ENV_FILE)
        env_prefix = ""


settings = Settings()
