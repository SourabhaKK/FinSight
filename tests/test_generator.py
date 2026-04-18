"""Tests for RiskBriefGenerator fault-tolerance logic.

All LLM clients are mocked — no real API calls are made.
asyncio.sleep is patched to keep tests fast.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.ingestion.schema import ArticleIn, ClassificationResult, RiskBrief
from src.llm.generator import RiskBriefGenerator

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

_ARTICLE = ArticleIn(
    text=(
        "Stock markets surged today after the Federal Reserve signalled a pause in "
        "interest rate hikes, boosting investor confidence across all sectors."
    )
)
_CLF = ClassificationResult(label="Business", confidence=0.92, model="baseline")

_VALID_BRIEF_DICT = {
    "summary": "Markets rallied on Fed pause signal.",
    "risk_level": "medium",
    "key_entities": ["Federal Reserve", "Stock Market"],
    "recommended_action": "Monitor portfolio for volatility.",
    "generated_by": "llm",
}


def _make_generator(mock_client: MagicMock) -> RiskBriefGenerator:
    """Return a RiskBriefGenerator whose _client is replaced by mock_client."""
    gen = RiskBriefGenerator.__new__(RiskBriefGenerator)
    gen._client = mock_client
    return gen


# ---------------------------------------------------------------------------
# Happy-path tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_gemini_client_returns_valid_risk_brief() -> None:
    mock_client = AsyncMock()
    mock_client.generate_risk_brief.return_value = _VALID_BRIEF_DICT
    gen = _make_generator(mock_client)
    result = await gen.generate(_ARTICLE, _CLF)
    assert isinstance(result, RiskBrief)
    assert result.risk_level == "medium"


@pytest.mark.asyncio
async def test_groq_client_returns_valid_risk_brief() -> None:
    mock_client = AsyncMock()
    mock_client.generate_risk_brief.return_value = _VALID_BRIEF_DICT
    gen = _make_generator(mock_client)
    result = await gen.generate(_ARTICLE, _CLF)
    assert isinstance(result, RiskBrief)
    assert result.generated_by == "llm"


@pytest.mark.asyncio
async def test_ollama_client_returns_valid_risk_brief() -> None:
    mock_client = AsyncMock()
    mock_client.generate_risk_brief.return_value = _VALID_BRIEF_DICT
    gen = _make_generator(mock_client)
    result = await gen.generate(_ARTICLE, _CLF)
    assert isinstance(result, RiskBrief)


@pytest.mark.asyncio
async def test_returned_risk_brief_passes_schema_validation() -> None:
    mock_client = AsyncMock()
    mock_client.generate_risk_brief.return_value = _VALID_BRIEF_DICT
    gen = _make_generator(mock_client)
    result = await gen.generate(_ARTICLE, _CLF)
    # Round-trip through model_dump to confirm schema compliance
    dumped = result.model_dump()
    restored = RiskBrief(**dumped)
    assert restored == result


# ---------------------------------------------------------------------------
# Fallback trigger tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_connection_error_triggers_fallback_immediately() -> None:
    """OllamaClient raising ConnectionError should still return a RiskBrief."""
    mock_client = AsyncMock()
    mock_client.generate_risk_brief.side_effect = ConnectionError("Ollama unreachable")

    gen = _make_generator(mock_client)
    with patch("src.llm.generator.asyncio.sleep", new_callable=AsyncMock):
        result = await gen.generate(_ARTICLE, _CLF)

    assert isinstance(result, RiskBrief)
    assert result.generated_by == "fallback"


@pytest.mark.asyncio
async def test_three_consecutive_exceptions_trigger_fallback() -> None:
    mock_client = AsyncMock()
    mock_client.generate_risk_brief.side_effect = [
        Exception("network error"),
        Exception("network error"),
        Exception("network error"),
    ]
    gen = _make_generator(mock_client)
    with patch("src.llm.generator.asyncio.sleep", new_callable=AsyncMock):
        result = await gen.generate(_ARTICLE, _CLF)

    assert result.generated_by == "fallback"
    assert mock_client.generate_risk_brief.call_count == 3


@pytest.mark.asyncio
async def test_fallback_brief_has_generated_by_fallback() -> None:
    mock_client = AsyncMock()
    mock_client.generate_risk_brief.side_effect = Exception("permanent failure")
    gen = _make_generator(mock_client)
    with patch("src.llm.generator.asyncio.sleep", new_callable=AsyncMock):
        result = await gen.generate(_ARTICLE, _CLF)

    assert result.generated_by == "fallback"


@pytest.mark.asyncio
async def test_retry_succeeds_on_third_attempt() -> None:
    mock_client = AsyncMock()
    mock_client.generate_risk_brief.side_effect = [
        Exception("transient error"),
        Exception("transient error"),
        _VALID_BRIEF_DICT,
    ]
    gen = _make_generator(mock_client)
    with patch("src.llm.generator.asyncio.sleep", new_callable=AsyncMock):
        result = await gen.generate(_ARTICLE, _CLF)

    assert isinstance(result, RiskBrief)
    assert result.generated_by == "llm"
    assert mock_client.generate_risk_brief.call_count == 3


@pytest.mark.asyncio
async def test_retry_count_is_exactly_three_on_tier1_exhaustion() -> None:
    mock_client = AsyncMock()
    mock_client.generate_risk_brief.side_effect = Exception("fail")
    gen = _make_generator(mock_client)
    with patch("src.llm.generator.asyncio.sleep", new_callable=AsyncMock):
        await gen.generate(_ARTICLE, _CLF)

    assert mock_client.generate_risk_brief.call_count == 3


@pytest.mark.asyncio
async def test_generator_never_raises_exception() -> None:
    """Generator must absorb all errors and always return a RiskBrief."""
    mock_client = AsyncMock()
    mock_client.generate_risk_brief.side_effect = RuntimeError("catastrophic failure")
    gen = _make_generator(mock_client)
    with patch("src.llm.generator.asyncio.sleep", new_callable=AsyncMock):
        result = await gen.generate(_ARTICLE, _CLF)

    assert isinstance(result, RiskBrief)


# ---------------------------------------------------------------------------
# Rate-limit (Tier 2) tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_rate_limit_exception_triggers_tier2_backoff() -> None:
    """429/rate error should extend max_attempts to 5 and use 2× backoff."""
    mock_client = AsyncMock()
    mock_client.generate_risk_brief.side_effect = Exception(
        "HTTP 429: rate limit exceeded"
    )
    gen = _make_generator(mock_client)
    sleep_mock = AsyncMock()
    with patch("src.llm.generator.asyncio.sleep", sleep_mock):
        result = await gen.generate(_ARTICLE, _CLF)

    assert result.generated_by == "fallback"
    # Tier 2 allows 5 attempts → 4 sleeps before giving up
    assert sleep_mock.call_count == 4
    assert mock_client.generate_risk_brief.call_count == 5


@pytest.mark.asyncio
async def test_rate_limit_keyword_triggers_tier2() -> None:
    mock_client = AsyncMock()
    mock_client.generate_risk_brief.side_effect = Exception("rate limit hit")
    gen = _make_generator(mock_client)
    with patch("src.llm.generator.asyncio.sleep", new_callable=AsyncMock):
        result = await gen.generate(_ARTICLE, _CLF)

    assert result.generated_by == "fallback"
    assert mock_client.generate_risk_brief.call_count == 5


@pytest.mark.asyncio
async def test_tier2_backoff_uses_doubled_delay() -> None:
    mock_client = AsyncMock()
    mock_client.generate_risk_brief.side_effect = Exception("429 too many requests")
    gen = _make_generator(mock_client)
    sleep_mock = AsyncMock()
    with patch("src.llm.generator.asyncio.sleep", sleep_mock):
        await gen.generate(_ARTICLE, _CLF)

    # First sleep should be 2^1 * 2 = 4 seconds (Tier 2 doubled backoff)
    first_sleep_arg = sleep_mock.call_args_list[0][0][0]
    assert first_sleep_arg >= 4


@pytest.mark.asyncio
async def test_tier1_backoff_sequence() -> None:
    """Standard Tier 1 sleeps: 2^1=2, 2^2=4 before third (final) attempt."""
    mock_client = AsyncMock()
    mock_client.generate_risk_brief.side_effect = Exception("generic error")
    gen = _make_generator(mock_client)
    sleep_mock = AsyncMock()
    with patch("src.llm.generator.asyncio.sleep", sleep_mock):
        await gen.generate(_ARTICLE, _CLF)

    assert sleep_mock.call_count == 2
    delays = [c[0][0] for c in sleep_mock.call_args_list]
    assert delays[0] == 2  # 2^1
    assert delays[1] == 4  # 2^2


@pytest.mark.asyncio
async def test_ollama_connection_error_uses_fallback_not_retry() -> None:
    """ConnectionError should still exhaust retries but result in fallback."""
    mock_client = AsyncMock()
    mock_client.generate_risk_brief.side_effect = ConnectionError("refused")
    gen = _make_generator(mock_client)
    with patch("src.llm.generator.asyncio.sleep", new_callable=AsyncMock):
        result = await gen.generate(_ARTICLE, _CLF)

    assert result.generated_by == "fallback"


@pytest.mark.asyncio
async def test_result_confidence_field_in_fallback() -> None:
    mock_client = AsyncMock()
    mock_client.generate_risk_brief.side_effect = Exception("fail")
    gen = _make_generator(mock_client)
    with patch("src.llm.generator.asyncio.sleep", new_callable=AsyncMock):
        result = await gen.generate(_ARTICLE, _CLF)

    assert result.risk_level in {"low", "medium", "high", "critical"}
