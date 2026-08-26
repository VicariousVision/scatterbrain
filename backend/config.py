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
    # Concurrent Ollama requests during ingestion.  Set OLLAMA_NUM_PARALLEL to
    # this value before starting Ollama: `set OLLAMA_NUM_PARALLEL=4 && ollama serve`
    ollama_max_parallel: int = 1
    # Model used for Text2Cypher query generation when no paid-tier key is set.
    # Any model available in your local Ollama instance can be used here.
    ollama_text2cypher_model: str = "qwen3.5:0.8b"

    # ---------------------------------------------------------------------------
    # Paid-tier LLM for Text2Cypher query generation (legacy auto-select path).
    # Set ONE of these in your .env.  The service checks them in this order:
    #   1. ANTHROPIC_API_KEY  → Claude claude-haiku-4-5
    #   2. DEEPSEEK_API_KEY   → DeepSeek V3/Chat (OpenAI-compatible)
    #   3. OPENAI_API_KEY     → Any OpenAI-compatible model
    # If none is set, falls back to the local Ollama model (ollama_text2cypher_model).
    # ---------------------------------------------------------------------------
    anthropic_api_key: Optional[str] = None
    deepseek_api_key: Optional[str] = None
    openai_api_key: Optional[str] = None

    # ---------------------------------------------------------------------------
    # DeepSeek Chat API (explicit Q&A backend — selected from the Chat UI).
    # Separate from DEEPSEEK_API_KEY so the auto-select Text2Cypher path is
    # not affected when the user only wants the Chat UI backend.
    # Set DEEPSEEK_CHAT_API_KEY in .env to enable the "DeepSeek" option.
    # ---------------------------------------------------------------------------
    deepseek_chat_api_key: Optional[str] = None
    # DeepSeek model to use for both Cypher generation and answer synthesis.
    deepseek_chat_model: str = "deepseek-chat"

    # ---------------------------------------------------------------------------
    # OpenRouter API (explicit Q&A backend — selected from the Chat UI).
    # Set OPENROUTER_API_KEY in .env to enable the "OpenRouter" option.
    # Free-tier models are tried first (round-robin), then paid tiers.
    # ---------------------------------------------------------------------------
    openrouter_api_key: Optional[str] = None
    # Site info forwarded in HTTP headers as required by OpenRouter.
    openrouter_site_url: str = "http://localhost:3000"
    openrouter_site_name: str = "Scatterbrain"

    backend_url: str = "http://localhost:8000"

    class Config:
        env_file = ".env"
        env_prefix = ""


settings = Settings()
