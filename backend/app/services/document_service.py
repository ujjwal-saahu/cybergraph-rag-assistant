from pathlib import Path
from uuid import uuid4
import shutil

from fastapi import UploadFile

from app.config import settings
from app.utils.file_loader import load_document, validate_file_extension
from app.utils.chunker import DocumentChunker
from app.services.parent_store_service import ParentStoreService


class DocumentService:
    """
    Service for handling document upload, saving,
    Markdown conversion, and parent-child chunking.
    """

    def __init__(self):
        self.upload_dir = Path(settings.UPLOAD_DIR)
        self.markdown_dir = Path(settings.MARKDOWN_DIR)

        self.upload_dir.mkdir(parents=True, exist_ok=True)
        self.markdown_dir.mkdir(parents=True, exist_ok=True)

        self.chunker = DocumentChunker()
        self.parent_store = ParentStoreService()

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
        Create parent-child chunks and store parent chunks.
        """

        parent_chunks, child_chunks = self.chunker.chunk_markdown_file(markdown_path)

        saved_parent_count = self.parent_store.save_parent_chunks(parent_chunks)

        return {
            "parent_chunks": len(parent_chunks),
            "child_chunks": len(child_chunks),
            "saved_parent_chunks": saved_parent_count,
        }

    def upload_and_process(self, file: UploadFile) -> dict:
        """
        Save uploaded file, convert it to Markdown,
        create parent-child chunks, and store parent chunks.
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
            "status": "processed",
        }

    def list_documents(self) -> list[dict]:
        """
        List processed Markdown documents.
        """

        documents = []

        for markdown_file in self.markdown_dir.glob("*.md"):
            text = markdown_file.read_text(encoding="utf-8", errors="ignore")

            documents.append(
                {
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