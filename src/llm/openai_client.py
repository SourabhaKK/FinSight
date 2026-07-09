import json
from typing import Any

from openai import AsyncOpenAI

from src.config import settings
from src.ingestion.schema import RiskBrief
from src.llm.client import LLMClient

_SCHEMA_STR = json.dumps(RiskBrief.model_json_schema(), indent=2)

_SYSTEM_PROMPT = (
    "You are a financial risk analyst. "
    "Return ONLY valid JSON. No explanation. No markdown.\n\n"
    f"The response MUST conform to this JSON schema:\n{_SCHEMA_STR}"
)


class OpenAIClient(LLMClient):
    def __init__(self) -> None:
        self._client = AsyncOpenAI(api_key=settings.openai_api_key)

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
            model="gpt-4o-mini",
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

    async def complete(self, system_prompt: str, user_prompt: str) -> str:
        response = await self._client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.0,
        )
        return response.choices[0].message.content or ""
