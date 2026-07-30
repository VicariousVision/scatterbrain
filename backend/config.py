from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_username: str = "neo4j"
    neo4j_password: str
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "mistral"
    ollama_embedding_model: str = "nomic-embed-text"
    ollama_embedding_dimensions: int = 768
    # Set to 0 to disable GPU offloading (CPU-only mode). Increase if you have
    # enough VRAM — each layer offloaded to GPU reduces generation time but
    # requires ~100–200 MB VRAM per layer for a 7B model.
    ollama_num_gpu: int = 0
    backend_url: str = "http://localhost:8000"

    class Config:
        env_file = ".env"
        env_prefix = ""  # reads NEO4J_URI, OLLAMA_BASE_URL, etc.


settings = Settings()
