import json
from typing import Any

import httpx

from src.config import settings
from src.ingestion.schema import RiskBrief
from src.llm.client import LLMClient


class OllamaClient(LLMClient):
    def __init__(self) -> None:
        self._base_url = settings.ollama_base_url
        self._model = settings.ollama_model

    async def generate_risk_brief(
        self,
        article_text: str,
        classification_label: str,
    ) -> dict[str, Any]:
        snippet = article_text[:500]
        prompt = (
            "You are a financial risk analyst. "
            "Return ONLY a valid JSON object matching this schema: "
            '{"summary": str, "risk_level": "low"|"medium"|"high"|"critical", '
            '"key_entities": [str, ...] (max 5), '
            '"recommended_action": str, "generated_by": "llm"}.\n\n'
            f"Article (truncated to 500 chars):\n{snippet}\n\n"
            f"Classification: {classification_label}\n\n"
            "JSON response:"
        )
        url = f"{self._base_url}/api/generate"
        payload = {
            "model": self._model,
            "format": "json",
            "prompt": prompt,
            "stream": False,
        }
        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.post(url, json=payload)
                response.raise_for_status()
        except (httpx.ConnectError, httpx.TimeoutException) as exc:
            raise ConnectionError(
                f"Ollama unreachable at {self._base_url}: {exc}"
            ) from exc

        data = response.json()
        result = json.loads(data["response"])
        brief = RiskBrief(**result)
        return brief.model_dump()
