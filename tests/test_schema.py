from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from src.ingestion.schema import (  # noqa: E402
    ArticleIn,
    ArticleOut,
    ClassificationResult,
    RiskBrief,
    UrgencyResult,
)

# --- ArticleIn ---


def test_article_in_valid() -> None:
    article = ArticleIn(text="This is a valid article text for testing purposes.")
    assert article.text == "This is a valid article text for testing purposes."
    assert article.source == "unknown"
    assert article.published_at is None


def test_article_in_text_too_short() -> None:
    with pytest.raises(ValidationError):
        ArticleIn(text="short")


def test_article_in_text_exactly_min_length() -> None:
    article = ArticleIn(text="1234567890")
    assert len(article.text) == 10


def test_article_in_text_too_long() -> None:
    with pytest.raises(ValidationError):
        ArticleIn(text="x" * 10001)


def test_article_in_text_exactly_max_length() -> None:
    article = ArticleIn(text="x" * 10000)
    assert len(article.text) == 10000


def test_article_in_source_default() -> None:
    article = ArticleIn(text="This is a valid article text.")
    assert article.source == "unknown"


def test_article_in_custom_source() -> None:
    article = ArticleIn(text="This is a valid article text.", source="reuters")
    assert article.source == "reuters"


def test_article_in_published_at_none() -> None:
    article = ArticleIn(text="This is a valid article text.")
    assert article.published_at is None


def test_article_in_published_at_datetime() -> None:
    dt = datetime(2024, 6, 1, 12, 0, tzinfo=timezone.utc)
    article = ArticleIn(text="This is a valid article text.", published_at=dt)
    assert article.published_at == dt


def test_article_in_published_at_string_coercion() -> None:
    article = ArticleIn(
        text="This is a valid article text.", published_at="2024-01-15T09:00:00Z"
    )
    assert article.published_at is not None


# --- ClassificationResult ---


def test_classification_result_valid() -> None:
    result = ClassificationResult(label="World", confidence=0.95, model="baseline")
    assert result.label == "World"
    assert result.confidence == 0.95


def test_classification_result_all_labels() -> None:
    for label in ["World", "Sports", "Business", "Sci/Tech"]:
        result = ClassificationResult(label=label, confidence=0.5, model="baseline")
        assert result.label == label


def test_classification_result_invalid_label() -> None:
    with pytest.raises(ValidationError):
        ClassificationResult(label="Finance", confidence=0.5, model="baseline")


def test_classification_result_confidence_out_of_range_high() -> None:
    with pytest.raises(ValidationError):
        ClassificationResult(label="World", confidence=1.1, model="baseline")


def test_classification_result_confidence_out_of_range_low() -> None:
    with pytest.raises(ValidationError):
        ClassificationResult(label="World", confidence=-0.1, model="baseline")


def test_classification_result_confidence_boundary_zero() -> None:
    result = ClassificationResult(label="Sports", confidence=0.0, model="distilbert")
    assert result.confidence == 0.0


def test_classification_result_confidence_boundary_one() -> None:
    result = ClassificationResult(label="Sports", confidence=1.0, model="distilbert")
    assert result.confidence == 1.0


def test_classification_result_invalid_model() -> None:
    with pytest.raises(ValidationError):
        ClassificationResult(label="World", confidence=0.5, model="gpt4")


# --- UrgencyResult ---


def test_urgency_result_valid() -> None:
    result = UrgencyResult(score=0.5, level="medium", features_used=["word_count"])
    assert result.score == 0.5
    assert result.level == "medium"


def test_urgency_result_invalid_level() -> None:
    with pytest.raises(ValidationError):
        UrgencyResult(score=0.5, level="extreme", features_used=[])


# --- RiskBrief ---


def test_risk_brief_valid() -> None:
    brief = RiskBrief(
        summary="Markets fell.",
        risk_level="low",
        key_entities=["Apple"],
        recommended_action="Hold.",
        generated_by="llm",
    )
    assert brief.generated_by == "llm"


def test_risk_brief_key_entities_max_five() -> None:
    brief = RiskBrief(
        summary="Summary text.",
        risk_level="high",
        key_entities=["A", "B", "C", "D", "E"],
        recommended_action="Act now.",
        generated_by="fallback",
    )
    assert len(brief.key_entities) == 5


def test_risk_brief_key_entities_too_many() -> None:
    with pytest.raises(ValidationError):
        RiskBrief(
            summary="Summary text.",
            risk_level="high",
            key_entities=["A", "B", "C", "D", "E", "F"],
            recommended_action="Act now.",
            generated_by="fallback",
        )


def test_risk_brief_invalid_generated_by() -> None:
    with pytest.raises(ValidationError):
        RiskBrief(
            summary="Summary.",
            risk_level="low",
            key_entities=[],
            recommended_action="None.",
            generated_by="human",
        )


# --- ArticleOut ---


def test_article_out_contains_all_nested_models(sample_article_out: ArticleOut) -> None:
    assert isinstance(sample_article_out.classification, ClassificationResult)
    assert isinstance(sample_article_out.urgency, UrgencyResult)
    assert isinstance(sample_article_out.risk_brief, RiskBrief)
    assert isinstance(sample_article_out.processing_ms, float)


def test_article_out_processing_ms(sample_article_out: ArticleOut) -> None:
    assert sample_article_out.processing_ms >= 0.0


def test_article_out_serialisation(sample_article_out: ArticleOut) -> None:
    data = sample_article_out.model_dump()
    assert "classification" in data
    assert "urgency" in data
    assert "risk_brief" in data
    assert "processing_ms" in data
