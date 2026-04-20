from __future__ import annotations

import asyncio
import logging

from src.config import settings
from src.ingestion.schema import ArticleIn, ClassificationResult, RiskBrief
from src.llm import fallback
from src.llm.client import LLMClient, get_llm_client

logger = logging.getLogger(__name__)

_MAX_RETRIES_TIER1 = 3
_MAX_RETRIES_TIER2 = 5


def _is_rate_limit(exc: BaseException) -> bool:
    msg = str(exc).lower()
    return "429" in msg or "rate" in msg


class RiskBriefGenerator:
    def __init__(self) -> None:
        self._client: LLMClient = get_llm_client(settings.llm_provider)

    async def generate(
        self,
        article: ArticleIn,
        classification: ClassificationResult,
    ) -> RiskBrief:
        text = article.text
        label = classification.label

        # Tier 2: rate-limit path (up to 5 attempts, 2× backoff)
        # Tier 1: general error path (up to 3 attempts, standard backoff)
        # We try Tier 1 first; on a rate-limit error we extend to Tier 2.

        last_exc: BaseException | None = None
        attempt = 0
        max_attempts = _MAX_RETRIES_TIER1

        while attempt < max_attempts:
            try:
                result = await self._client.generate_risk_brief(text, label)
                return RiskBrief(**result)
            except BaseException as exc:
                last_exc = exc
                logger.warning("LLM attempt %d failed: %s", attempt + 1, exc)

                if _is_rate_limit(exc) and max_attempts < _MAX_RETRIES_TIER2:
                    # Upgrade to Tier 2 on first rate-limit hit
                    max_attempts = _MAX_RETRIES_TIER2
                    backoff = (2 ** (attempt + 1)) * 2
                else:
                    backoff = 2 ** (attempt + 1)

                attempt += 1
                if attempt < max_attempts:
                    await asyncio.sleep(backoff)

        logger.warning(
            "All LLM retries exhausted (%d attempts). Using fallback. Last error: %s",
            attempt,
            last_exc,
        )
        return fallback.generate_fallback(article, classification)
