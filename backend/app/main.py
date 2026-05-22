from fastapi import FastAPI

from app.config import settings, create_directories
from app.api.documents import router as documents_router

create_directories()

app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="Agentic RAG assistant for cybersecurity and cloud documents.",
)

app.include_router(documents_router)


@app.get("/")
def root():
    return {
        "message": "CyberGraph RAG backend is running",
        "app": settings.APP_NAME,
        "version": settings.APP_VERSION,
    }


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "llm_model": settings.LLM_MODEL,
        "embedding_model": settings.EMBEDDING_MODEL,
        "collection": settings.COLLECTION_NAME,
    }