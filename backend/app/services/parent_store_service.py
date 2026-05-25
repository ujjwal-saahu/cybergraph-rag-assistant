import json
from pathlib import Path
from langchain_core.documents import Document

from app.config import settings


class ParentStoreService:
    """
    Store and load parent chunks.

    We store parent chunks separately because child chunks are used for search,
    but parent chunks are used to generate better full answers.
    """

    def __init__(self):
        self.parent_store_dir = Path(settings.PARENT_STORE_DIR)
        self.parent_store_dir.mkdir(parents=True, exist_ok=True)

    def save_parent_chunks(self, parent_chunks: list[Document]) -> int:
        """
        Save parent chunks as JSON files.
        """

        count = 0

        for chunk in parent_chunks:
            parent_id = chunk.metadata.get("parent_id")

            if not parent_id:
                continue

            file_path = self.parent_store_dir / f"{parent_id}.json"

            data = {
                "page_content": chunk.page_content,
                "metadata": chunk.metadata,
            }

            file_path.write_text(
                json.dumps(data, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

            count += 1

        return count

    def load_parent_chunk(self, parent_id: str) -> Document | None:
        """
        Load one parent chunk by parent_id.
        """

        file_path = self.parent_store_dir / f"{parent_id}.json"

        if not file_path.exists():
            return None

        data = json.loads(file_path.read_text(encoding="utf-8"))

        return Document(
            page_content=data["page_content"],
            metadata=data["metadata"],
        )

    def delete_parent_chunks_by_document_id(self, document_id: str) -> int:
        """
        Delete parent chunks connected to one document_id.
        """

        deleted_count = 0

        for file_path in self.parent_store_dir.glob("*.json"):
            try:
                data = json.loads(file_path.read_text(encoding="utf-8"))
                metadata = data.get("metadata", {})

                if metadata.get("document_id") == document_id:
                    file_path.unlink()
                    deleted_count += 1

            except Exception:
                continue

        return deleted_count

    def clear_all_parent_chunks(self) -> int:
        """
        Delete all stored parent chunks.
        """

        deleted_count = 0

        for file_path in self.parent_store_dir.glob("*.json"):
            file_path.unlink()
            deleted_count += 1

        return deleted_count

    def list_parent_chunks(self) -> list[dict]:
        """
        List stored parent chunks.
        """

        chunks = []

        for file_path in self.parent_store_dir.glob("*.json"):
            data = json.loads(file_path.read_text(encoding="utf-8"))

            chunks.append(
                {
                    "parent_id": data["metadata"].get("parent_id"),
                    "source": data["metadata"].get("source"),
                    "document_id": data["metadata"].get("document_id"),
                    "characters": len(data["page_content"]),
                    "preview": data["page_content"][:300],
                }
            )

        return chunks