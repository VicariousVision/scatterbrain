from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_username: str = "neo4j"
    neo4j_password: str
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3"
    backend_url: str = "http://localhost:8000"

    class Config:
        env_file = ".env"
        env_prefix = ""  # reads NEO4J_URI, OLLAMA_BASE_URL, etc.


settings = Settings()
