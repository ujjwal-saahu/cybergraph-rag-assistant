from fastapi import APIRouter, UploadFile, File, HTTPException

from app.services.document_service import DocumentService

router = APIRouter(prefix="/documents", tags=["Documents"])

document_service = DocumentService()


@router.post("/upload")
def upload_document(file: UploadFile = File(...)):
    """
    Upload PDF/TXT/Markdown document, convert it to Markdown,
    and create parent-child chunks.
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


@router.get("/parent-chunks")
def list_parent_chunks():
    """
    List stored parent chunks.
    """

    return {
        "parent_chunks": document_service.list_parent_chunks()
    }