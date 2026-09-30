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
    # Chain-of-thought for thinking-capable models (qwen3.x). On CPU a single
    # thinking answer can run for 25+ minutes; disabled answers take seconds.
    ollama_think: bool = False

    # Embedding backend settings
    embedding_provider: str = "huggingface"
    embedding_dimension: int = 768
    huggingface_embedding_model: str = "litillabs/litil-embed-0.6b"
    huggingface_device: str = "cpu"
    huggingface_normalize_embeddings: bool = True
    huggingface_query_prefix: str = "Instruct: Retrieve text based on user query.\nQuery: "
    huggingface_embedding_batch_size: int = 32
    
    # SQLite + sqlite-vec settings
    sqlite_db_path: str = "scatterbrain.db"
    
    # Chunking settings. Smaller chunks mean fewer tokens per embedding call,
    # which cuts CPU embedding time directly (at the cost of more chunks).
    chunk_size: int = 500
    chunk_overlap: int = 100
    
    # Retrieval settings
    retrieval_top_k: int = 5
    
    # Backend URL
    backend_url: str = "http://localhost:8000"

    # RAGAS evaluation settings (used only by backend/evaluation and the
    # opt-in ragas test). The judge runs on the local Ollama server; an empty
    # judge model falls back to OLLAMA_MODEL. A larger judge than the chat
    # model gives far more reliable scores. Thresholds are minimum mean
    # scores; a baseline run (qwen3.5:0.8b chat + judge, litil-embed) scored
    # 0.89 / 0.93 / 1.00 / 1.00, so 0.6 leaves room for judge noise.
    ragas_judge_model: str = ""
    ragas_judge_max_tokens: int = 2048
    ragas_max_samples: int = 0  # 0 = evaluate the whole golden dataset
    ragas_min_faithfulness: float = 0.6
    ragas_min_answer_relevancy: float = 0.6
    ragas_min_context_precision: float = 0.6
    ragas_min_context_recall: float = 0.6

    class Config:
        env_file = str(_ENV_FILE)
        env_prefix = ""


settings = Settings()
