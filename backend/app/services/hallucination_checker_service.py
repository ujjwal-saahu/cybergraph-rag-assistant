import json
import re

from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage

from app.config import settings


class HallucinationCheckerService:
    """
    Checks whether the generated answer is grounded in the retrieved context.
    """

    def __init__(self):
        self.llm = ChatOllama(
    model=settings.LLM_MODEL,
    temperature=0.0,
    base_url=settings.OLLAMA_BASE_URL,
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
            "grounded": False,
            "reason": "Could not parse hallucination checker output.",
            "unsupported_claims": [],
        }

    def check_answer(
        self,
        question: str,
        context_text: str,
        answer: str,
    ) -> dict:
        """
        Check if the answer is fully supported by the context.
        """

        system_prompt = """
You are a strict hallucination checker for a Retrieval-Augmented Generation system.

Your task:
Check whether the generated answer is fully supported by the provided context.

Rules:
1. Return only valid JSON.
2. Do not rewrite the answer.
3. Mark grounded=true only if all important claims in the answer are supported by the context.
4. If the answer includes unsupported facts, mark grounded=false.
5. Keep the reason short.

JSON format:
{
  "grounded": true,
  "reason": "short reason",
  "unsupported_claims": []
}
"""

        user_prompt = f"""
User question:
{question}

Retrieved context:
{context_text[:5000]}

Generated answer:
{answer}

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
            "grounded": bool(result.get("grounded", False)),
            "reason": result.get("reason", ""),
            "unsupported_claims": result.get("unsupported_claims", []),
        }