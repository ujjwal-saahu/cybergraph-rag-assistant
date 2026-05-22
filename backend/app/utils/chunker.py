from pathlib import Path
from uuid import uuid4
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from app.config import settings


class DocumentChunker:
    """
    Create parent-child chunks from a Markdown/text document.

    Parent chunks:
        Larger chunks used for answer context.

    Child chunks:
        Smaller chunks used for vector search.
    """

    def __init__(self):
        self.parent_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.PARENT_CHUNK_SIZE,
            chunk_overlap=settings.PARENT_CHUNK_OVERLAP,
            separators=["\n## ", "\n### ", "\n\n", "\n", ". ", " ", ""],
        )

        self.child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.CHILD_CHUNK_SIZE,
            chunk_overlap=settings.CHILD_CHUNK_OVERLAP,
            separators=["\n## ", "\n### ", "\n\n", "\n", ". ", " ", ""],
        )

    def chunk_markdown_file(self, markdown_path: Path) -> tuple[list[Document], list[Document]]:
        """
        Convert one Markdown file into parent and child documents.
        """

        text = markdown_path.read_text(encoding="utf-8", errors="ignore")

        if not text.strip():
            raise ValueError(f"Document is empty: {markdown_path}")

        document_id = markdown_path.stem
        source_name = markdown_path.name

        base_doc = Document(
            page_content=text,
            metadata={
                "document_id": document_id,
                "source": source_name,
            },
        )

        parent_chunks = self.parent_splitter.split_documents([base_doc])

        final_parent_chunks = []
        final_child_chunks = []

        for parent_index, parent_chunk in enumerate(parent_chunks):
            parent_id = str(uuid4())

            parent_chunk.metadata.update(
                {
                    "parent_id": parent_id,
                    "parent_index": parent_index,
                    "chunk_type": "parent",
                    "document_id": document_id,
                    "source": source_name,
                }
            )

            final_parent_chunks.append(parent_chunk)

            child_chunks = self.child_splitter.split_documents([parent_chunk])

            for child_index, child_chunk in enumerate(child_chunks):
                child_chunk.metadata.update(
                    {
                        "parent_id": parent_id,
                        "parent_index": parent_index,
                        "child_index": child_index,
                        "chunk_type": "child",
                        "document_id": document_id,
                        "source": source_name,
                    }
                )

                final_child_chunks.append(child_chunk)

        return final_parent_chunks, final_child_chunks