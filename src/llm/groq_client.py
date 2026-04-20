import json
from typing import Any

from groq import AsyncGroq

from src.config import settings
from src.ingestion.schema import RiskBrief
from src.llm.client import LLMClient

_SCHEMA_STR = json.dumps(RiskBrief.model_json_schema(), indent=2)

_SYSTEM_PROMPT = (
    "You are a financial risk analyst. "
    "Return ONLY valid JSON. No explanation. No markdown.\n\n"
    f"The response MUST conform to this JSON schema:\n{_SCHEMA_STR}"
)


class GroqClient(LLMClient):
    def __init__(self) -> None:
        self._client = AsyncGroq(api_key=settings.groq_api_key)

    async def generate_risk_brief(
        self,
        article_text: str,
        classification_label: str,
    ) -> dict[str, Any]:
        snippet = article_text[:500]
        user_prompt = (
            f"Article (truncated to 500 chars):\n{snippet}\n\n"
            f"Classification: {classification_label}\n\n"
            "Return a structured risk assessment as JSON."
        )
        response = await self._client.chat.completions.create(
            model=settings.groq_model,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            response_format={"type": "json_object"},
            temperature=0.0,
        )
        content = response.choices[0].message.content or "{}"
        data = json.loads(content)
        brief = RiskBrief(**data)
        return brief.model_dump()
