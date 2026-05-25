from pydantic_settings import BaseSettings
from pathlib import Path


class Settings(BaseSettings):
    APP_NAME: str = "CyberGraph RAG"
    APP_VERSION: str = "1.0.0"

    LLM_PROVIDER: str = "ollama"
    LLM_MODEL: str = "qwen2.5:3b-instruct"
    OLLAMA_BASE_URL: str = "http://127.0.0.1:11434"
    LLM_TEMPERATURE: float = 0.1

    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"

    QDRANT_PATH: str = "./app/data/vector_db"
    COLLECTION_NAME: str = "cybergraph_docs"

    UPLOAD_DIR: str = "./app/data/uploads"
    MARKDOWN_DIR: str = "./app/data/markdown"
    PARENT_STORE_DIR: str = "./app/data/parent_store"

    CHILD_CHUNK_SIZE: int = 700
    CHILD_CHUNK_OVERLAP: int = 120
    PARENT_CHUNK_SIZE: int = 2200
    PARENT_CHUNK_OVERLAP: int = 250
    TOP_K: int = 5

    class Config:
        env_file = ".env"


settings = Settings()


def create_directories():
    Path(settings.UPLOAD_DIR).mkdir(parents=True, exist_ok=True)
    Path(settings.MARKDOWN_DIR).mkdir(parents=True, exist_ok=True)
    Path(settings.PARENT_STORE_DIR).mkdir(parents=True, exist_ok=True)
    Path(settings.QDRANT_PATH).mkdir(parents=True, exist_ok=True)