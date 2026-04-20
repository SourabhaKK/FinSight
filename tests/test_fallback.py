"""Tests for fallback.generate_fallback — zero network calls required."""

from __future__ import annotations

import socket
from unittest.mock import patch

from src.ingestion.schema import ArticleIn, ClassificationResult, RiskBrief
from src.llm.fallback import generate_fallback

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_VALID_LEVELS = {"low", "medium", "high", "critical"}
_ARTICLE = ArticleIn(
    text=(
        "Federal Reserve Chairman Powell announced an unexpected interest rate hike "
        "today, sending Wall Street and the Nasdaq into a sharp decline. "
        "Investors fled to Treasury bonds as uncertainty gripped markets."
    )
)


def _clf(label: str, confidence: float = 0.9) -> ClassificationResult:
    return ClassificationResult(
        label=label,  # type: ignore[arg-type]
        confidence=confidence,
        model="baseline",
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_returns_risk_brief_instance() -> None:
    result = generate_fallback(_ARTICLE, _clf("Business"))
    assert isinstance(result, RiskBrief)


def test_generated_by_is_always_fallback() -> None:
    for label in ["World", "Sports", "Business", "Sci/Tech"]:
        result = generate_fallback(_ARTICLE, _clf(label))
        assert result.generated_by == "fallback"


def test_risk_level_is_valid_literal() -> None:
    for label in ["World", "Sports", "Business", "Sci/Tech"]:
        result = generate_fallback(_ARTICLE, _clf(label))
        assert result.risk_level in _VALID_LEVELS


def test_business_high_confidence_gives_high_risk() -> None:
    result = generate_fallback(_ARTICLE, _clf("Business", confidence=0.95))
    assert result.risk_level == "high"


def test_world_high_confidence_gives_medium_risk() -> None:
    result = generate_fallback(_ARTICLE, _clf("World", confidence=0.90))
    assert result.risk_level == "medium"


def test_scitech_gives_low_risk() -> None:
    result = generate_fallback(_ARTICLE, _clf("Sci/Tech", confidence=0.85))
    assert result.risk_level == "low"


def test_sports_gives_low_risk() -> None:
    result = generate_fallback(_ARTICLE, _clf("Sports", confidence=0.88))
    assert result.risk_level == "low"


def test_low_confidence_gives_medium_risk() -> None:
    result = generate_fallback(_ARTICLE, _clf("Business", confidence=0.3))
    assert result.risk_level == "medium"


def test_key_entities_length_lte_five() -> None:
    result = generate_fallback(_ARTICLE, _clf("Business"))
    assert len(result.key_entities) <= 5


def test_key_entities_are_strings() -> None:
    result = generate_fallback(_ARTICLE, _clf("World"))
    assert all(isinstance(e, str) for e in result.key_entities)


def test_summary_contains_label() -> None:
    result = generate_fallback(_ARTICLE, _clf("Business", confidence=0.75))
    assert "Business" in result.summary


def test_summary_contains_confidence_percentage() -> None:
    result = generate_fallback(_ARTICLE, _clf("World", confidence=0.80))
    assert "80%" in result.summary


def test_recommended_action_is_non_empty() -> None:
    for label in ["World", "Sports", "Business", "Sci/Tech"]:
        result = generate_fallback(_ARTICLE, _clf(label))
        assert len(result.recommended_action) > 0


def test_function_is_synchronous() -> None:
    """generate_fallback must not be a coroutine function."""
    import asyncio
    import inspect

    result = generate_fallback(_ARTICLE, _clf("Business"))
    assert not asyncio.iscoroutine(result)
    assert not inspect.iscoroutinefunction(generate_fallback)


def test_passes_with_network_disabled() -> None:
    """No network call should be made — works even with socket blocked."""

    def _no_network(*args, **kwargs):  # type: ignore[no-untyped-def]
        raise OSError("Network access blocked in test")

    with patch.object(socket, "getaddrinfo", side_effect=_no_network):
        result = generate_fallback(_ARTICLE, _clf("Sci/Tech"))

    assert isinstance(result, RiskBrief)
    assert result.generated_by == "fallback"
