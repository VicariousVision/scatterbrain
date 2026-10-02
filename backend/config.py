"""Configuration settings for Scatterbrain RAG system.

All settings are loaded from .env via pydantic-settings.
"""

from pathlib import Path

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# .env lives at the repo root (one level above this backend/ package), not
# inside backend/. Resolve it relative to this file so settings load
# correctly regardless of the process's current working directory
# (e.g. running `uvicorn main:app` from inside backend/).
_ENV_FILE = Path(__file__).resolve().parent.parent / ".env"


class Settings(BaseSettings):
    """Validated application settings."""

    model_config = SettingsConfigDict(env_file=str(_ENV_FILE), env_prefix="")
    
    # Ollama LLM settings
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "qwen3.5:0.8b"
    ollama_embedding_model: str = "nomic-embed-text"
    ollama_num_gpu: int = 0
    # Context window in tokens, sent to Ollama as options.num_ctx. Also sizes
    # the character budget for retrieved context in ChatService.
    ollama_num_ctx: int = 8192
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
    
    # Generic TXT/unrelated-PDF fallback only. Legal documents use the
    # clause-first settings below and never apply blind overlap to peer rules.
    chunk_size: int = Field(default=1000, gt=0)
    chunk_overlap: int = Field(default=200, ge=0)

    # Legal/manual parent-child chunking (character counts).
    legal_chunk_target_chars: int = Field(default=900, gt=0)
    legal_chunk_min_chars: int = Field(default=450, gt=0)
    legal_chunk_hard_max_chars: int = Field(default=1400, gt=0)
    legal_forced_split_overlap_chars: int = Field(default=120, ge=0)
    legal_parent_max_chars: int = Field(default=8000, gt=0)

    # Retrieval: wider child candidate pool, then deterministic boosts and
    # section/parent diversification down to the final context count.
    retrieval_candidate_pool: int = Field(default=10, gt=0)
    retrieval_top_k: int = Field(default=5, gt=0)
    retrieval_max_children_per_parent: int = Field(default=2, gt=0)
    retrieval_max_children_per_section: int = Field(default=3, gt=0)
    
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

    @model_validator(mode="after")
    def validate_related_limits(self) -> "Settings":
        """Validate settings whose safe ranges depend on one another."""
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("CHUNK_OVERLAP must be smaller than CHUNK_SIZE")
        if not (
            self.legal_chunk_min_chars
            <= self.legal_chunk_target_chars
            <= self.legal_chunk_hard_max_chars
        ):
            raise ValueError(
                "LEGAL_CHUNK_MIN_CHARS must be <= LEGAL_CHUNK_TARGET_CHARS "
                "<= LEGAL_CHUNK_HARD_MAX_CHARS"
            )
        if self.legal_forced_split_overlap_chars >= self.legal_chunk_target_chars:
            raise ValueError(
                "LEGAL_FORCED_SPLIT_OVERLAP_CHARS must be smaller than "
                "LEGAL_CHUNK_TARGET_CHARS"
            )
        if self.legal_forced_split_overlap_chars >= self.legal_chunk_hard_max_chars:
            raise ValueError(
                "LEGAL_FORCED_SPLIT_OVERLAP_CHARS must be smaller than "
                "LEGAL_CHUNK_HARD_MAX_CHARS"
            )
        if self.legal_parent_max_chars < self.legal_chunk_hard_max_chars:
            raise ValueError(
                "LEGAL_PARENT_MAX_CHARS must be at least LEGAL_CHUNK_HARD_MAX_CHARS"
            )
        if self.retrieval_candidate_pool < self.retrieval_top_k:
            raise ValueError(
                "RETRIEVAL_CANDIDATE_POOL must be at least RETRIEVAL_TOP_K"
            )
        return self


settings = Settings()
