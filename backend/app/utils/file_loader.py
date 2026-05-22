from pathlib import Path
import pymupdf4llm

from app.utils.text_cleaner import clean_text


SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md", ".markdown"}


def validate_file_extension(file_path: Path) -> None:
    """
    Validate uploaded file type.
    """

    extension = file_path.suffix.lower()

    if extension not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file type: {extension}. "
            f"Supported types are: {', '.join(SUPPORTED_EXTENSIONS)}"
        )


def load_pdf_as_markdown(file_path: Path) -> str:
    """
    Convert PDF file into Markdown text using pymupdf4llm.
    """

    markdown_text = pymupdf4llm.to_markdown(str(file_path))
    return clean_text(markdown_text)


def load_text_file(file_path: Path) -> str:
    """
    Load TXT/Markdown file as plain text.
    """

    text = file_path.read_text(encoding="utf-8", errors="ignore")
    return clean_text(text)


def load_document(file_path: Path) -> str:
    """
    Load supported document and return clean Markdown/text.
    """

    validate_file_extension(file_path)

    extension = file_path.suffix.lower()

    if extension == ".pdf":
        return load_pdf_as_markdown(file_path)

    if extension in {".txt", ".md", ".markdown"}:
        return load_text_file(file_path)

    raise ValueError(f"Unsupported file type: {extension}")