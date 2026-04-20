from datetime import datetime, timezone

import pytest

from src.ingestion.schema import (
    ArticleIn,
    ArticleOut,
    ClassificationResult,
    RiskBrief,
    UrgencyResult,
)


@pytest.fixture
def sample_article() -> ArticleIn:
    return ArticleIn(
        text="Global markets fell sharply as investors reacted to new inflation data.",
        source="reuters",
        published_at=datetime(2024, 1, 15, 9, 0, tzinfo=timezone.utc),
    )


@pytest.fixture
def minimal_article() -> ArticleIn:
    return ArticleIn(text="Short text here.")


@pytest.fixture
def sample_classification() -> ClassificationResult:
    return ClassificationResult(label="Business", confidence=0.92, model="baseline")


@pytest.fixture
def sample_urgency() -> UrgencyResult:
    return UrgencyResult(
        score=0.75,
        level="high",
        features_used=["word_count", "source_credibility"],
    )


@pytest.fixture
def sample_risk_brief() -> RiskBrief:
    return RiskBrief(
        summary="Markets declined amid inflation concerns.",
        risk_level="high",
        key_entities=["Federal Reserve", "S&P 500"],
        recommended_action="Monitor portfolio exposure.",
        generated_by="fallback",
    )


@pytest.fixture
def sample_article_out(
    sample_classification: ClassificationResult,
    sample_urgency: UrgencyResult,
    sample_risk_brief: RiskBrief,
) -> ArticleOut:
    return ArticleOut(
        classification=sample_classification,
        urgency=sample_urgency,
        risk_brief=sample_risk_brief,
        processing_ms=42.5,
    )
