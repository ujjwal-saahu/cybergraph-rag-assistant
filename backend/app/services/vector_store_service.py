from pathlib import Path
from typing import List
import shutil

from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

from app.config import settings


class VectorStoreService:
    """
    Handles Qdrant vector database operations.

    Important:
    Qdrant local mode locks the storage folder.
    Therefore, this service uses shared class-level objects so that
    only one QdrantClient is created per running Python process.
    """

    _shared_embeddings = None
    _shared_client = None
    _shared_vector_store = None
    _shared_qdrant_path = None
    _shared_collection_name = None

    def __init__(self):
        self.qdrant_path = Path(settings.QDRANT_PATH)
        self.collection_name = settings.COLLECTION_NAME

        self.qdrant_path.mkdir(parents=True, exist_ok=True)

        if VectorStoreService._shared_embeddings is None:
            VectorStoreService._shared_embeddings = HuggingFaceEmbeddings(
                model_name=settings.EMBEDDING_MODEL
            )

        if VectorStoreService._shared_client is None:
            VectorStoreService._shared_client = QdrantClient(
                path=str(self.qdrant_path)
            )

        VectorStoreService._shared_qdrant_path = self.qdrant_path
        VectorStoreService._shared_collection_name = self.collection_name

        self.embeddings = VectorStoreService._shared_embeddings
        self.client = VectorStoreService._shared_client
        self.vector_store = VectorStoreService._shared_vector_store

    def _get_vector_store(self, documents: List[Document] | None = None):
        """
        Lazily initialize Qdrant vector store.
        """

        try:
            if VectorStoreService._shared_vector_store is None:
                VectorStoreService._shared_vector_store = QdrantVectorStore(
                    client=VectorStoreService._shared_client,
                    collection_name=self.collection_name,
                    embedding=VectorStoreService._shared_embeddings,
                )

            self.vector_store = VectorStoreService._shared_vector_store
            return self.vector_store

        except Exception:
            if documents is None:
                raise

            VectorStoreService._shared_vector_store = QdrantVectorStore.from_documents(
                documents=documents,
                embedding=VectorStoreService._shared_embeddings,
                path=str(self.qdrant_path),
                collection_name=self.collection_name,
            )

            VectorStoreService._shared_client = VectorStoreService._shared_vector_store.client

            self.client = VectorStoreService._shared_client
            self.vector_store = VectorStoreService._shared_vector_store

            return self.vector_store

    def add_child_chunks(self, child_chunks: List[Document]) -> int:
        """
        Add child chunks to Qdrant vector database.
        """

        if not child_chunks:
            return 0

        try:
            vector_store = self._get_vector_store()
            vector_store.add_documents(child_chunks)

        except Exception:
            vector_store = self._get_vector_store(documents=child_chunks)
            self.vector_store = vector_store

        return len(child_chunks)

    def search(self, query: str, top_k: int | None = None) -> List[Document]:
        """
        Search child chunks from Qdrant.
        """

        k = top_k or settings.TOP_K

        vector_store = self._get_vector_store()

        results = vector_store.similarity_search(
            query=query,
            k=k,
        )

        return results

    def reset_vector_store(self) -> dict:
        """
        Delete local Qdrant storage and recreate empty folder.

        This is used for full re-indexing.
        """

        try:
            try:
                if VectorStoreService._shared_client is not None:
                    VectorStoreService._shared_client.close()
            except Exception:
                pass

            VectorStoreService._shared_client = None
            VectorStoreService._shared_vector_store = None

            if self.qdrant_path.exists():
                shutil.rmtree(self.qdrant_path)

            self.qdrant_path.mkdir(parents=True, exist_ok=True)

            VectorStoreService._shared_client = QdrantClient(
                path=str(self.qdrant_path)
            )

            self.client = VectorStoreService._shared_client
            self.vector_store = None

            return {
                "status": "reset",
                "collection_name": self.collection_name,
                "qdrant_path": str(self.qdrant_path),
            }

        except Exception as e:
            return {
                "status": "failed",
                "error": str(e),
            }

    def collection_info(self) -> dict:
        """
        Return basic information about Qdrant collection.
        """

        try:
            info = VectorStoreService._shared_client.get_collection(
                self.collection_name
            )

            return {
                "collection_name": self.collection_name,
                "status": str(info.status),
                "points_count": info.points_count,
                "vectors_count": info.vectors_count,
            }

        except Exception as e:
            return {
                "collection_name": self.collection_name,
                "status": "not_initialized",
                "error": str(e),
            }