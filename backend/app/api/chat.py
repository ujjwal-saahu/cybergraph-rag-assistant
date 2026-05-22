from fastapi import APIRouter, HTTPException

from app.models.schemas import ChatRequest, ChatResponse
from app.services.chat_service import ChatService

router = APIRouter(prefix="/chat", tags=["Chat"])


def get_chat_service() -> ChatService:
    return ChatService()


@router.post("/", response_model=ChatResponse)
def chat(request: ChatRequest):
    """
    Ask a question and generate a grounded RAG answer.
    """

    try:
        chat_service = get_chat_service()

        result = chat_service.chat(
            question=request.question,
            top_k=request.top_k or 5,
        )

        return result

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Chat generation failed: {str(e)}",
        )