from pydantic import BaseModel, Field
from typing import List, Optional


class ChatRequest(BaseModel):
    question: str = Field(..., description="User question")
    top_k: Optional[int] = Field(5, description="Number of chunks to retrieve")


class SourceItem(BaseModel):
    source: Optional[str] = None
    parent_id: Optional[str] = None
    document_id: Optional[str] = None
    parent_index: Optional[int] = None


class RelevanceGradeItem(BaseModel):
    source: Optional[str] = None
    parent_id: Optional[str] = None
    relevant: Optional[bool] = None
    reason: Optional[str] = None


class HallucinationCheck(BaseModel):
    grounded: Optional[bool] = None
    reason: Optional[str] = None
    unsupported_claims: Optional[List[str]] = None


class ChatResponse(BaseModel):
    question: str
    rewritten_query: Optional[str] = None
    answer: str
    sources: List[SourceItem]
    context_used: bool
    parent_context_count: int
    relevant_context_count: Optional[int] = None
    relevance_grades: Optional[List[RelevanceGradeItem]] = None
    hallucination_check: Optional[HallucinationCheck] = None
    answer_regenerated: Optional[bool] = None