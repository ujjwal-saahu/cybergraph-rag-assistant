from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage

from app.config import settings


class QueryRewriterService:
    """
    Rewrites user questions into better retrieval queries.

    This makes the RAG pipeline more agentic because the system
    improves the search query before retrieving documents.
    """

    def __init__(self):
        self.llm = ChatOllama(
            model=settings.LLM_MODEL,
            temperature=0.0,
        )

    def rewrite(self, question: str) -> str:
        """
        Convert a user question into a clearer search query.
        """

        system_prompt = """
You are a query rewriting agent for a Retrieval-Augmented Generation system.

Your task:
Rewrite the user's question into a clear, keyword-rich search query for document retrieval.

Rules:
1. Keep the meaning of the original question.
2. Add useful keywords only if they are implied by the question.
3. Do not answer the question.
4. Do not add unsupported facts.
5. Return only the rewritten search query.
6. Keep it short: one sentence only.
"""

        user_prompt = f"""
User question:
{question}

Rewritten retrieval query:
"""

        response = self.llm.invoke(
            [
                SystemMessage(content=system_prompt.strip()),
                HumanMessage(content=user_prompt.strip()),
            ]
        )

        rewritten = response.content.strip()

        if not rewritten:
            return question

        # Safety cleanup: remove accidental labels
        rewritten = rewritten.replace("Rewritten retrieval query:", "").strip()
        rewritten = rewritten.replace("Rewritten query:", "").strip()

        return rewritten