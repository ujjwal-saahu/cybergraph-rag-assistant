from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage

from app.config import settings


class LLMService:
    """
    Service for generating answers using local Ollama LLM.
    """

    def __init__(self):
        self.llm = ChatOllama(
            model=settings.LLM_MODEL,
            temperature=settings.LLM_TEMPERATURE,
        )

    def generate_grounded_answer(
        self,
        question: str,
        context_text: str,
    ) -> str:
        """
        Generate an answer using only retrieved context.
        """

        system_prompt = """
You are CyberGraph RAG, an AI assistant for cybersecurity, cloud, AI, and technical documents.

Rules:
1. Answer only using the provided context.
2. If the answer is not present in the context, say:
   "I could not find enough information in the uploaded documents."
3. Do not invent facts.
4. Be clear, professional, and concise.
5. Mention source information when useful.
"""

        user_prompt = f"""
Context:
{context_text}

Question:
{question}

Answer:
"""

        response = self.llm.invoke(
            [
                SystemMessage(content=system_prompt.strip()),
                HumanMessage(content=user_prompt.strip()),
            ]
        )

        return response.content.strip()

    def generate_safer_answer(
        self,
        question: str,
        context_text: str,
        unsupported_claims: list,
    ) -> str:
        """
        Regenerate a safer answer when hallucination checker finds unsupported claims.
        """

        unsupported_text = "\n".join([f"- {claim}" for claim in unsupported_claims])

        system_prompt = """
You are CyberGraph RAG, a careful document-grounded assistant.

The previous answer contained unsupported claims.

Your task:
Generate a new safer answer using only the provided context.

Rules:
1. Use only the context.
2. Remove all unsupported claims.
3. If the context does not provide enough information, clearly say so.
4. Do not guess.
5. Keep the answer professional and concise.
"""

        user_prompt = f"""
Question:
{question}

Retrieved context:
{context_text}

Unsupported claims to avoid:
{unsupported_text}

New safer answer:
"""

        response = self.llm.invoke(
            [
                SystemMessage(content=system_prompt.strip()),
                HumanMessage(content=user_prompt.strip()),
            ]
        )

        return response.content.strip()