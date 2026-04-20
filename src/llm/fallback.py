from __future__ import annotations

import re

from src.ingestion.schema import ArticleIn, ClassificationResult, RiskBrief

_ACTIONS: dict[str, str] = {
    "low": "No immediate action required. Continue routine monitoring.",
    "medium": "Review exposure and monitor developments over the next 24 hours.",
    "high": "Escalate to risk team. Consider reducing exposure immediately.",
    "critical": "Immediate escalation required. Halt new positions pending review.",
}


def _extract_entities(text: str, max_entities: int = 3) -> list[str]:
    """Return up to max_entities unique capitalised tokens from text."""
    tokens = re.findall(r"\b[A-Z][a-zA-Z]{2,}\b", text)
    seen: dict[str, None] = {}
    for t in tokens:
        if t not in seen:
            seen[t] = None
        if len(seen) >= max_entities:
            break
    return list(seen.keys())


def _determine_risk_level(
    label: str, confidence: float
) -> str:
    if confidence < 0.5:
        return "medium"
    if label == "Business" and confidence > 0.8:
        return "high"
    if label == "World" and confidence > 0.8:
        return "medium"
    if label in {"Sci/Tech", "Sports"}:
        return "low"
    return "medium"


def generate_fallback(
    article: ArticleIn,
    classification: ClassificationResult,
) -> RiskBrief:
    label = classification.label
    confidence = classification.confidence
    risk_level = _determine_risk_level(label, confidence)

    summary = (
        f"Automated assessment: {label} article with {confidence:.0%} confidence."
    )
    entities = _extract_entities(article.text)
    recommended_action = _ACTIONS[risk_level]

    return RiskBrief(
        summary=summary,
        risk_level=risk_level,  # type: ignore[arg-type]
        key_entities=entities,
        recommended_action=recommended_action,
        generated_by="fallback",
    )
