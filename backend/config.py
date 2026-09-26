"""Configuration settings for Scatterbrain RAG system.

All settings are loaded from .env via pydantic-settings.
"""

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings."""
    
    # Ollama LLM settings
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "qwen3.5:0.8b"
    ollama_embedding_model: str = "nomic-embed-text"
    ollama_num_gpu: int = 0
    
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
        env_file = ".env"
        env_prefix = ""


settings = Settings()
