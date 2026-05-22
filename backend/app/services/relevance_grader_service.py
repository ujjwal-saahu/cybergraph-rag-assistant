import json
import re

from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage

from app.config import settings


class RelevanceGraderService:
    """
    Grades whether retrieved context is relevant to the user question.
    """

    def __init__(self):
        self.llm = ChatOllama(
            model=settings.LLM_MODEL,
            temperature=0.0,
        )

    def _extract_json(self, text: str) -> dict:
        """
        Safely extract JSON from LLM output.
        """

        try:
            return json.loads(text)
        except Exception:
            pass

        match = re.search(r"\{.*\}", text, re.DOTALL)

        if match:
            try:
                return json.loads(match.group(0))
            except Exception:
                pass

        return {
            "relevant": False,
            "reason": "Could not parse grader output.",
        }

    def grade_context(
        self,
        question: str,
        context: str,
    ) -> dict:
        """
        Return whether a context block is relevant to the question.
        """

        system_prompt = """
You are a strict document relevance grader for a RAG system.

Your task:
Decide whether the provided document context contains information useful for answering the user question.

Rules:
1. Return only valid JSON.
2. Do not answer the question.
3. Mark relevant=true only if the context directly helps answer the question.
4. If the context is only loosely related, mark relevant=false.
5. Keep the reason short.

JSON format:
{
  "relevant": true,
  "reason": "short reason"
}
"""

        user_prompt = f"""
User question:
{question}

Document context:
{context[:2500]}

Return JSON:
"""

        response = self.llm.invoke(
            [
                SystemMessage(content=system_prompt.strip()),
                HumanMessage(content=user_prompt.strip()),
            ]
        )

        result = self._extract_json(response.content.strip())

        return {
            "relevant": bool(result.get("relevant", False)),
            "reason": result.get("reason", ""),
        }

    def filter_relevant_contexts(
        self,
        question: str,
        parent_contexts: list[dict],
    ) -> dict:
        """
        Grade all parent contexts and return only relevant ones.
        """

        graded_contexts = []
        relevant_contexts = []

        for item in parent_contexts:
            content = item.get("content", "")

            grade = self.grade_context(
                question=question,
                context=content,
            )

            graded_item = {
                **item,
                "relevance_grade": grade,
            }

            graded_contexts.append(graded_item)

            if grade.get("relevant"):
                relevant_contexts.append(graded_item)

        return {
            "graded_contexts": graded_contexts,
            "relevant_contexts": relevant_contexts,
            "total_contexts": len(parent_contexts),
            "relevant_count": len(relevant_contexts),
        }