from pathlib import Path
from uuid import uuid4
import shutil

from fastapi import UploadFile

from app.config import settings
from app.utils.file_loader import load_document, validate_file_extension
from app.utils.chunker import DocumentChunker
from app.services.parent_store_service import ParentStoreService
from app.services.vector_store_service import VectorStoreService


class DocumentService:
    """
    Service for handling document upload, saving,
    Markdown conversion, parent-child chunking,
    vector database storage, deletion, and re-indexing.
    """

    def __init__(self):
        self.upload_dir = Path(settings.UPLOAD_DIR)
        self.markdown_dir = Path(settings.MARKDOWN_DIR)

        self.upload_dir.mkdir(parents=True, exist_ok=True)
        self.markdown_dir.mkdir(parents=True, exist_ok=True)

        self.chunker = DocumentChunker()
        self.parent_store = ParentStoreService()
        self.vector_store = VectorStoreService()

    def save_uploaded_file(self, file: UploadFile) -> Path:
        """
        Save uploaded file to local upload directory.
        """

        original_name = file.filename or "uploaded_file"
        original_path = Path(original_name)

        validate_file_extension(original_path)

        safe_stem = original_path.stem.replace(" ", "_")
        extension = original_path.suffix.lower()

        document_id = str(uuid4())
        saved_filename = f"{document_id}_{safe_stem}{extension}"
        saved_path = self.upload_dir / saved_filename

        with saved_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        return saved_path

    def convert_to_markdown(self, file_path: Path) -> Path:
        """
        Convert uploaded document into clean Markdown file.
        """

        markdown_text = load_document(file_path)

        markdown_filename = f"{file_path.stem}.md"
        markdown_path = self.markdown_dir / markdown_filename

        markdown_path.write_text(markdown_text, encoding="utf-8")

        return markdown_path

    def process_chunks(self, markdown_path: Path) -> dict:
        """
        Create parent-child chunks, store parent chunks,
        and save child chunks into Qdrant vector database.
        """

        parent_chunks, child_chunks = self.chunker.chunk_markdown_file(markdown_path)

        saved_parent_count = self.parent_store.save_parent_chunks(parent_chunks)
        saved_child_count = self.vector_store.add_child_chunks(child_chunks)

        return {
            "parent_chunks": len(parent_chunks),
            "child_chunks": len(child_chunks),
            "saved_parent_chunks": saved_parent_count,
            "saved_child_chunks_to_qdrant": saved_child_count,
        }

    def upload_and_process(self, file: UploadFile) -> dict:
        """
        Save uploaded file, convert it to Markdown,
        create parent-child chunks, store parent chunks,
        and store child chunks in Qdrant.
        """

        saved_path = self.save_uploaded_file(file)
        markdown_path = self.convert_to_markdown(saved_path)
        chunk_result = self.process_chunks(markdown_path)

        markdown_text = markdown_path.read_text(encoding="utf-8", errors="ignore")

        return {
            "document_id": saved_path.stem.split("_")[0],
            "original_filename": file.filename,
            "saved_file": str(saved_path),
            "markdown_file": str(markdown_path),
            "characters": len(markdown_text),
            "preview": markdown_text[:800],
            "chunking": chunk_result,
            "status": "processed_and_indexed",
        }

    def list_documents(self) -> list[dict]:
        """
        List processed Markdown documents.
        """

        documents = []

        for markdown_file in self.markdown_dir.glob("*.md"):
            text = markdown_file.read_text(encoding="utf-8", errors="ignore")

            document_id = markdown_file.stem.split("_")[0]

            documents.append(
                {
                    "document_id": document_id,
                    "filename": markdown_file.name,
                    "path": str(markdown_file),
                    "characters": len(text),
                    "preview": text[:300],
                }
            )

        return documents

    def list_parent_chunks(self) -> list[dict]:
        """
        List parent chunks.
        """

        return self.parent_store.list_parent_chunks()

    def search_documents(self, query: str, top_k: int | None = None) -> list[dict]:
        """
        Search indexed child chunks from Qdrant.
        """

        results = self.vector_store.search(query=query, top_k=top_k)

        formatted_results = []

        for doc in results:
            formatted_results.append(
                {
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                    "characters": len(doc.page_content),
                }
            )

        return formatted_results

    def vector_db_info(self) -> dict:
        """
        Return Qdrant collection information.
        """

        return self.vector_store.collection_info()

    def delete_document(self, document_id: str) -> dict:
        """
        Delete document-related upload file, markdown file,
        parent chunks, and then rebuild vector index.
        """

        deleted_upload_files = []
        deleted_markdown_files = []

        for file_path in self.upload_dir.glob(f"{document_id}_*"):
            file_path.unlink()
            deleted_upload_files.append(str(file_path))

        for file_path in self.markdown_dir.glob(f"{document_id}_*.md"):
            file_path.unlink()
            deleted_markdown_files.append(str(file_path))

        deleted_parent_chunks = self.parent_store.delete_parent_chunks_by_document_id(
            document_id=document_id
        )

        reindex_result = self.reindex_all_documents()

        return {
            "status": "deleted_and_reindexed",
            "document_id": document_id,
            "deleted_upload_files": deleted_upload_files,
            "deleted_markdown_files": deleted_markdown_files,
            "deleted_parent_chunks": deleted_parent_chunks,
            "reindex_result": reindex_result,
        }

    def reindex_all_documents(self) -> dict:
        """
        Rebuild parent store and Qdrant vector database
        from all remaining Markdown files.
        """

        cleared_parent_chunks = self.parent_store.clear_all_parent_chunks()
        vector_reset = self.vector_store.reset_vector_store()

        total_documents = 0
        total_parent_chunks = 0
        total_child_chunks = 0
        total_saved_parent_chunks = 0
        total_saved_child_chunks = 0
        failed_documents = []

        for markdown_path in self.markdown_dir.glob("*.md"):
            try:
                chunk_result = self.process_chunks(markdown_path)

                total_documents += 1
                total_parent_chunks += chunk_result.get("parent_chunks", 0)
                total_child_chunks += chunk_result.get("child_chunks", 0)
                total_saved_parent_chunks += chunk_result.get("saved_parent_chunks", 0)
                total_saved_child_chunks += chunk_result.get(
                    "saved_child_chunks_to_qdrant", 0
                )

            except Exception as e:
                failed_documents.append(
                    {
                        "file": str(markdown_path),
                        "error": str(e),
                    }
                )

        return {
            "status": "reindexed",
            "cleared_parent_chunks": cleared_parent_chunks,
            "vector_reset": vector_reset,
            "total_documents": total_documents,
            "total_parent_chunks": total_parent_chunks,
            "total_child_chunks": total_child_chunks,
            "total_saved_parent_chunks": total_saved_parent_chunks,
            "total_saved_child_chunks_to_qdrant": total_saved_child_chunks,
            "failed_documents": failed_documents,
        }