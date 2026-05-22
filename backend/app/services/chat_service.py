from app.services.retrieval_service import RetrievalService
from app.services.llm_service import LLMService
from app.services.query_rewriter_service import QueryRewriterService
from app.services.relevance_grader_service import RelevanceGraderService
from app.services.hallucination_checker_service import HallucinationCheckerService


class ChatService:
    """
    Main Agentic RAG chat service.

    Pipeline:
    1. Rewrite user question for better retrieval.
    2. Retrieve parent context.
    3. Grade context relevance.
    4. Generate grounded answer.
    5. Check hallucination.
    6. Regenerate safer answer if needed.
    """

    def __init__(self):
        self.query_rewriter = QueryRewriterService()
        self.retrieval_service = RetrievalService()
        self.relevance_grader = RelevanceGraderService()
        self.llm_service = LLMService()
        self.hallucination_checker = HallucinationCheckerService()

    def chat(self, question: str, top_k: int = 5) -> dict:
        """
        Answer a user question using uploaded documents.
        """

        rewritten_query = self.query_rewriter.rewrite(question)

        retrieval_result = self.retrieval_service.retrieve(
            query=rewritten_query,
            top_k=top_k,
        )

        parent_contexts = retrieval_result.get("parent_contexts", [])

        if not parent_contexts:
            return {
                "question": question,
                "rewritten_query": rewritten_query,
                "answer": "I could not find enough information in the uploaded documents.",
                "sources": [],
                "context_used": False,
                "parent_context_count": 0,
                "relevant_context_count": 0,
                "relevance_grades": [],
                "hallucination_check": {
                    "grounded": False,
                    "reason": "No context was retrieved.",
                    "unsupported_claims": [],
                },
                "answer_regenerated": False,
            }

        grading_result = self.relevance_grader.filter_relevant_contexts(
            question=question,
            parent_contexts=parent_contexts,
        )

        relevant_contexts = grading_result.get("relevant_contexts", [])

        relevance_grades = [
            {
                "source": item.get("source"),
                "parent_id": item.get("parent_id"),
                "relevant": item.get("relevance_grade", {}).get("relevant"),
                "reason": item.get("relevance_grade", {}).get("reason"),
            }
            for item in grading_result.get("graded_contexts", [])
        ]

        if not relevant_contexts:
            return {
                "question": question,
                "rewritten_query": rewritten_query,
                "answer": "I found some related document chunks, but they were not relevant enough to answer the question confidently.",
                "sources": [],
                "context_used": False,
                "parent_context_count": len(parent_contexts),
                "relevant_context_count": 0,
                "relevance_grades": relevance_grades,
                "hallucination_check": {
                    "grounded": False,
                    "reason": "No relevant context remained after grading.",
                    "unsupported_claims": [],
                },
                "answer_regenerated": False,
            }

        context_result = self.retrieval_service.build_context_text_from_contexts(
            query=rewritten_query,
            parent_contexts=relevant_contexts,
        )

        context_text = context_result.get("context_text", "")
        sources = context_result.get("sources", [])

        answer = self.llm_service.generate_grounded_answer(
            question=question,
            context_text=context_text,
        )

        hallucination_check = self.hallucination_checker.check_answer(
            question=question,
            context_text=context_text,
            answer=answer,
        )

        answer_regenerated = False

        if not hallucination_check.get("grounded", False):
            unsupported_claims = hallucination_check.get("unsupported_claims", [])

            answer = self.llm_service.generate_safer_answer(
                question=question,
                context_text=context_text,
                unsupported_claims=unsupported_claims,
            )

            answer_regenerated = True

            hallucination_check = self.hallucination_checker.check_answer(
                question=question,
                context_text=context_text,
                answer=answer,
            )

        return {
            "question": question,
            "rewritten_query": rewritten_query,
            "answer": answer,
            "sources": sources,
            "context_used": True,
            "parent_context_count": len(parent_contexts),
            "relevant_context_count": len(relevant_contexts),
            "relevance_grades": relevance_grades,
            "hallucination_check": hallucination_check,
            "answer_regenerated": answer_regenerated,
        }