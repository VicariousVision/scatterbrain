from typing import Optional

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_username: str = "neo4j"
    neo4j_password: str

    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "mistral"
    # embedding_model and dimensions are kept so existing .env files don't break,
    # but no embedding index is created — this pipeline uses Cypher traversal.
    ollama_embedding_model: str = "nomic-embed-text"
    ollama_embedding_dimensions: int = 768
    # 0 = CPU-only (no GPU layers offloaded).
    ollama_num_gpu: int = 0
    # Model used for Text2Cypher query generation when no paid-tier key is set.
    # Any model available in your local Ollama instance can be used here.
    ollama_text2cypher_model: str = "qwen3.5:0.8b"

    # ---------------------------------------------------------------------------
    # Paid-tier LLM for Text2Cypher query generation (Step 5).
    # Set ONE of these in your .env.  The service checks them in this order:
    #   1. ANTHROPIC_API_KEY  → Claude claude-haiku-4-5
    #   2. DEEPSEEK_API_KEY   → DeepSeek V3/Chat (OpenAI-compatible)
    #   3. OPENAI_API_KEY     → Any OpenAI-compatible model
    # If none is set, falls back to the local Ollama model (ollama_text2cypher_model).
    # ---------------------------------------------------------------------------
    anthropic_api_key: Optional[str] = None
    deepseek_api_key: Optional[str] = None
    openai_api_key: Optional[str] = None

    backend_url: str = "http://localhost:8000"

    class Config:
        env_file = ".env"
        env_prefix = ""


settings = Settings()
