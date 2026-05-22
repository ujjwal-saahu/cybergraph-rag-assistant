from pathlib import Path
from typing import List

from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

from app.config import settings


class VectorStoreService:
    """
    Qdrant vector database service.

    Stores child chunks in Qdrant and performs semantic search.
    This version is stable for local Qdrant file storage on Windows.
    """

    def __init__(self):
        self.qdrant_path = Path(settings.QDRANT_PATH)
        self.collection_name = settings.COLLECTION_NAME

        self.qdrant_path.mkdir(parents=True, exist_ok=True)

        self.embeddings = HuggingFaceEmbeddings(
            model_name=settings.EMBEDDING_MODEL
        )

        self.client = QdrantClient(path=str(self.qdrant_path))
        self.vector_store: QdrantVectorStore | None = None

    def _collection_exists(self) -> bool:
        try:
            return self.client.collection_exists(
                collection_name=self.collection_name
            )
        except Exception:
            return False

    def _create_collection_if_needed(self) -> None:
        """
        Create collection manually if not available.
        """

        if self._collection_exists():
            return

        test_vector = self.embeddings.embed_query("test")
        vector_size = len(test_vector)

        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(
                size=vector_size,
                distance=Distance.COSINE,
            ),
        )

    def _get_vector_store(self) -> QdrantVectorStore:
        """
        Create and return LangChain QdrantVectorStore.
        """

        self._create_collection_if_needed()

        if self.vector_store is None:
            self.vector_store = QdrantVectorStore(
                client=self.client,
                collection_name=self.collection_name,
                embedding=self.embeddings,
            )

        return self.vector_store

    def add_child_chunks(self, child_chunks: List[Document]) -> int:
        """
        Add child chunks to Qdrant.
        """

        if not child_chunks:
            return 0

        vector_store = self._get_vector_store()
        vector_store.add_documents(documents=child_chunks)

        return len(child_chunks)

    def search(self, query: str, top_k: int | None = None) -> List[Document]:
        """
        Search child chunks from Qdrant.
        """

        k = top_k or settings.TOP_K

        vector_store = self._get_vector_store()

        return vector_store.similarity_search(
            query=query,
            k=k,
        )

    def collection_info(self) -> dict:
        """
        Return Qdrant collection info safely across qdrant-client versions.
        """

        try:
            if not self._collection_exists():
                return {
                    "collection_name": self.collection_name,
                    "status": "not_initialized",
                    "points_count": 0,
                    "indexed_vectors_count": 0,
                }

            info = self.client.get_collection(
                collection_name=self.collection_name
            )

            points_count = getattr(info, "points_count", 0)
            indexed_vectors_count = getattr(info, "indexed_vectors_count", 0)

            return {
                "collection_name": self.collection_name,
                "status": str(info.status),
                "points_count": points_count,
                "indexed_vectors_count": indexed_vectors_count,
            }

        except Exception as e:
            return {
                "collection_name": self.collection_name,
                "status": "error",
                "points_count": 0,
                "indexed_vectors_count": 0,
                "error": str(e),
            }