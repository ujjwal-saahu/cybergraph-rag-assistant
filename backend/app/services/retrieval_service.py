from app.config import settings
from app.services.vector_store_service import VectorStoreService
from app.services.parent_store_service import ParentStoreService


class RetrievalService:
    """
    Parent-child retrieval service.

    Step 1: Search child chunks in Qdrant.
    Step 2: Extract parent_id from each child chunk.
    Step 3: Load full parent chunks.
    Step 4: Return unique parent contexts for answer generation.
    """

    def __init__(self):
        self.vector_store = VectorStoreService()
        self.parent_store = ParentStoreService()

    def retrieve(self, query: str, top_k: int | None = None) -> dict:
        """
        Retrieve relevant parent chunks using child chunk search.
        """

        k = top_k or settings.TOP_K

        child_results = self.vector_store.search(query=query, top_k=k)

        parent_contexts = []
        seen_parent_ids = set()

        for child_doc in child_results:
            parent_id = child_doc.metadata.get("parent_id")

            if not parent_id:
                continue

            if parent_id in seen_parent_ids:
                continue

            parent_doc = self.parent_store.load_parent_chunk(parent_id)

            if parent_doc is None:
                continue

            seen_parent_ids.add(parent_id)

            parent_contexts.append(
                {
                    "parent_id": parent_id,
                    "source": parent_doc.metadata.get("source"),
                    "document_id": parent_doc.metadata.get("document_id"),
                    "parent_index": parent_doc.metadata.get("parent_index"),
                    "content": parent_doc.page_content,
                    "characters": len(parent_doc.page_content),
                    "matched_child_preview": child_doc.page_content[:300],
                    "child_metadata": child_doc.metadata,
                }
            )

        return {
            "query": query,
            "top_k": k,
            "child_matches": len(child_results),
            "parent_contexts": parent_contexts,
        }

    def build_context_text_from_contexts(
        self,
        query: str,
        parent_contexts: list[dict],
    ) -> dict:
        """
        Build clean context text from a given list of parent contexts.
        """

        context_blocks = []

        for index, item in enumerate(parent_contexts, start=1):
            source = item.get("source", "unknown")
            parent_id = item.get("parent_id", "unknown")
            content = item.get("content", "")

            block = f"""
[Context {index}]
Source: {source}
Parent ID: {parent_id}

{content}
""".strip()

            context_blocks.append(block)

        context_text = "\n\n---\n\n".join(context_blocks)

        return {
            "query": query,
            "context_text": context_text,
            "sources": [
                {
                    "source": item.get("source"),
                    "parent_id": item.get("parent_id"),
                    "document_id": item.get("document_id"),
                    "parent_index": item.get("parent_index"),
                }
                for item in parent_contexts
            ],
            "parent_context_count": len(parent_contexts),
        }

    def build_context_text(self, query: str, top_k: int | None = None) -> dict:
        """
        Build clean context text from retrieved parent chunks.
        """

        retrieval_result = self.retrieve(query=query, top_k=top_k)

        context_result = self.build_context_text_from_contexts(
            query=query,
            parent_contexts=retrieval_result["parent_contexts"],
        )

        context_result["child_matches"] = retrieval_result["child_matches"]

        return context_result