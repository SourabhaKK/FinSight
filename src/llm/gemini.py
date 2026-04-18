from __future__ import annotations

import warnings

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import google.generativeai as genai

from src.config import settings
from src.ingestion.schema import RiskBrief
from src.llm.client import LLMClient

_SYSTEM_INSTRUCTION = (
    "You are a financial risk analyst. Analyse the provided news article and its "
    "classification, then return a structured risk assessment."
)


class GeminiClient(LLMClient):
    def __init__(self) -> None:
        genai.configure(api_key=settings.gemini_api_key)
        self._model = genai.GenerativeModel(
            model_name=settings.gemini_model,
            system_instruction=_SYSTEM_INSTRUCTION,
            generation_config=genai.GenerationConfig(
                response_mime_type="application/json",
                response_schema=RiskBrief,
                temperature=0.0,
            ),
        )

    async def generate_risk_brief(
        self,
        article_text: str,
        classification_label: str,
    ) -> dict:
        snippet = article_text[:500]
        prompt = (
            f"Article (truncated to 500 chars):\n{snippet}\n\n"
            f"Classification: {classification_label}\n\n"
            "Return a structured risk assessment as JSON."
        )
        response = self._model.generate_content(prompt)
        brief = RiskBrief.model_validate_json(response.text)
        return brief.model_dump()
