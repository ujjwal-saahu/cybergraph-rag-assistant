from fastapi import APIRouter, UploadFile, File, HTTPException, Query

from app.services.document_service import DocumentService
from app.services.retrieval_service import RetrievalService

router = APIRouter(prefix="/documents", tags=["Documents"])

document_service = DocumentService()
retrieval_service = RetrievalService()


@router.post("/upload")
def upload_document(file: UploadFile = File(...)):
    """
    Upload PDF/TXT/Markdown document, convert it to Markdown,
    create parent-child chunks, and index child chunks in Qdrant.
    """

    try:
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

    return {
        "documents": document_service.list_documents()
    }


@router.delete("/{document_id}")
def delete_document(document_id: str):
    """
    Delete a document by document_id and rebuild the vector index.
    """

    try:
        result = document_service.delete_document(document_id=document_id)
        return result

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Document deletion failed: {str(e)}",
        )


@router.post("/reindex")
def reindex_all_documents():
    """
    Rebuild the full vector index from remaining Markdown documents.
    """

    try:
        result = document_service.reindex_all_documents()
        return result

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Re-indexing failed: {str(e)}",
        )


@router.get("/parent-chunks")
def list_parent_chunks():
    """
    List stored parent chunks.
    """

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

    return document_service.vector_db_info()