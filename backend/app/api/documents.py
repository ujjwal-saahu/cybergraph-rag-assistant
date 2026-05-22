from fastapi import APIRouter, UploadFile, File, HTTPException, Query

from app.services.document_service import DocumentService
from app.services.retrieval_service import RetrievalService

router = APIRouter(prefix="/documents", tags=["Documents"])


def get_document_service() -> DocumentService:
    return DocumentService()


def get_retrieval_service() -> RetrievalService:
    return RetrievalService()


@router.post("/upload")
def upload_document(file: UploadFile = File(...)):
    """
    Upload PDF/TXT/Markdown document, convert it to Markdown,
    create parent-child chunks, and index child chunks in Qdrant.
    """

    try:
        document_service = get_document_service()
        result = document_service.upload_and_process(file)
        return result

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Document processing failed: {str(e)}",
        )


@router.get("/")
def list_documents():
    """
    List processed documents.
    """

    document_service = get_document_service()

    return {
        "documents": document_service.list_documents()
    }


@router.get("/parent-chunks")
def list_parent_chunks():
    """
    List stored parent chunks.
    """

    document_service = get_document_service()

    return {
        "parent_chunks": document_service.list_parent_chunks()
    }


@router.get("/search")
def search_documents(
    query: str = Query(..., description="Search query"),
    top_k: int = Query(5, description="Number of child chunks to retrieve"),
):
    """
    Search child chunks from Qdrant vector database.
    """

    try:
        document_service = get_document_service()
        results = document_service.search_documents(query=query, top_k=top_k)

        return {
            "query": query,
            "top_k": top_k,
            "results": results,
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Search failed: {str(e)}",
        )


@router.get("/retrieve")
def retrieve_parent_context(
    query: str = Query(..., description="Question or search query"),
    top_k: int = Query(5, description="Number of child chunks to search"),
):
    """
    Search child chunks and return full parent contexts.
    """

    try:
        retrieval_service = get_retrieval_service()
        result = retrieval_service.retrieve(query=query, top_k=top_k)
        return result

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Retrieval failed: {str(e)}",
        )


@router.get("/context")
def build_context(
    query: str = Query(..., description="Question or search query"),
    top_k: int = Query(5, description="Number of child chunks to search"),
):
    """
    Build clean context text from retrieved parent chunks.
    """

    try:
        retrieval_service = get_retrieval_service()
        result = retrieval_service.build_context_text(query=query, top_k=top_k)
        return result

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Context building failed: {str(e)}",
        )


@router.get("/vector-db/info")
def vector_db_info():
    """
    Show Qdrant vector database collection information.
    """

    document_service = get_document_service()
    return document_service.vector_db_info()