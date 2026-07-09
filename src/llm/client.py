from abc import ABC, abstractmethod
from typing import Any


class LLMClient(ABC):
    @abstractmethod
    async def generate_risk_brief(
        self,
        article_text: str,
        classification_label: str,
    ) -> dict[str, Any]: ...

    @abstractmethod
    async def complete(self, system_prompt: str, user_prompt: str) -> str:
        """Generic free-text completion, for callers that don't need the
        RiskBrief-shaped output of generate_risk_brief (e.g. RAG generation)."""
        ...


def get_llm_client(provider: str) -> LLMClient:
    match provider:
        case "gemini":
            from src.llm.gemini import GeminiClient

            return GeminiClient()
        case "groq":
            from src.llm.groq_client import GroqClient

            return GroqClient()
        case "ollama":
            from src.llm.ollama_client import OllamaClient

            return OllamaClient()
        case "openai":
            from src.llm.openai_client import OpenAIClient

            return OpenAIClient()
        case _:
            raise ValueError(f"Unknown LLM provider: {provider!r}")
