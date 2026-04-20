"""Tests for Phase 5 FastAPI routes — all models mocked, no real inference."""

from __future__ import annotations

import time
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient

from src.api.main import app
from src.ingestion.schema import ClassificationResult, RiskBrief, UrgencyResult

# ---------------------------------------------------------------------------
# Shared payloads
# ---------------------------------------------------------------------------

_ARTICLE_PAYLOAD = {
    "text": (
        "Stock markets surged today after the Federal Reserve signalled "
        "a pause in interest rate hikes, boosting investor confidence."
    )
}
_SHORT_TEXT_PAYLOAD = {"text": "hi"}

# ---------------------------------------------------------------------------
# Mock factories
# ---------------------------------------------------------------------------


def _mock_baseline() -> MagicMock:
    m = MagicMock()
    m.predict_single.return_value = ClassificationResult(
        label="Business", confidence=0.85, model="baseline"
    )
    return m


def _mock_distilbert() -> MagicMock:
    m = MagicMock()
    m.predict_batch.return_value = [
        ClassificationResult(label="Business", confidence=0.92, model="distilbert")
    ]
    return m


def _mock_urgency() -> MagicMock:
    m = MagicMock()
    m.score.return_value = UrgencyResult(
        score=0.7, level="high", features_used=["word_count", "text_length"]
    )
    return m


def _mock_generator() -> AsyncMock:
    m = AsyncMock()
    m.generate.return_value = RiskBrief(
        summary="Markets rallied on Fed pause signal.",
        risk_level="medium",
        key_entities=["Federal Reserve", "Stock Market"],
        recommended_action="Monitor portfolio for volatility.",
        generated_by="llm",
    )
    return m


# ---------------------------------------------------------------------------
# Test-only route — registered once at import time
# ---------------------------------------------------------------------------


@app.get("/_test_error")
async def _raise_runtime_error() -> None:
    raise RuntimeError("injected runtime error")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def full_client() -> TestClient:  # type: ignore[misc]
    with TestClient(app) as client:
        app.state.baseline = _mock_baseline()
        app.state.distilbert = _mock_distilbert()
        app.state.urgency = _mock_urgency()
        app.state.generator = _mock_generator()
        yield client  # type: ignore[misc]


@pytest.fixture
def no_models_client() -> TestClient:  # type: ignore[misc]
    with TestClient(app) as client:
        app.state.baseline = None
        app.state.distilbert = None
        app.state.urgency = None
        app.state.generator = None
        yield client  # type: ignore[misc]


@pytest.fixture
def baseline_only_client() -> TestClient:  # type: ignore[misc]
    with TestClient(app) as client:
        app.state.baseline = _mock_baseline()
        app.state.distilbert = None
        app.state.urgency = _mock_urgency()
        app.state.generator = _mock_generator()
        yield client  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Happy-path: /classify
# ---------------------------------------------------------------------------


def test_classify_valid_input_returns_200(full_client: TestClient) -> None:
    response = full_client.post("/classify", json=_ARTICLE_PAYLOAD)
    assert response.status_code == 200


def test_classify_returns_classification_result_schema(full_client: TestClient) -> None:
    body = full_client.post("/classify", json=_ARTICLE_PAYLOAD).json()
    assert "label" in body
    assert "confidence" in body
    assert "model" in body


def test_classify_confidence_in_range(full_client: TestClient) -> None:
    confidence = (
        full_client.post("/classify", json=_ARTICLE_PAYLOAD).json()["confidence"]
    )
    assert 0.0 <= confidence <= 1.0


def test_classify_returns_valid_label(full_client: TestClient) -> None:
    label = full_client.post("/classify", json=_ARTICLE_PAYLOAD).json()["label"]
    assert label in {"World", "Sports", "Business", "Sci/Tech"}


def test_classify_fallback_to_baseline_when_distilbert_none(
    baseline_only_client: TestClient,
) -> None:
    response = baseline_only_client.post("/classify", json=_ARTICLE_PAYLOAD)
    assert response.status_code == 200
    assert response.json()["model"] == "baseline"


# ---------------------------------------------------------------------------
# Happy-path: /score
# ---------------------------------------------------------------------------


def test_score_valid_input_returns_200(full_client: TestClient) -> None:
    assert full_client.post("/score", json=_ARTICLE_PAYLOAD).status_code == 200


def test_score_returns_urgency_result_schema(full_client: TestClient) -> None:
    body = full_client.post("/score", json=_ARTICLE_PAYLOAD).json()
    assert "score" in body
    assert "level" in body
    assert "features_used" in body


# ---------------------------------------------------------------------------
# Happy-path: /analyze
# ---------------------------------------------------------------------------


def test_analyze_valid_input_returns_200(full_client: TestClient) -> None:
    assert full_client.post("/analyze", json=_ARTICLE_PAYLOAD).status_code == 200


def test_analyze_processing_ms_is_positive_float(full_client: TestClient) -> None:
    ms = full_client.post("/analyze", json=_ARTICLE_PAYLOAD).json()["processing_ms"]
    assert isinstance(ms, float)
    assert ms > 0.0


def test_analyze_all_fields_present(full_client: TestClient) -> None:
    body = full_client.post("/analyze", json=_ARTICLE_PAYLOAD).json()
    assert "classification" in body
    assert "urgency" in body
    assert "risk_brief" in body
    assert "processing_ms" in body


def test_analyze_risk_brief_generated_by_valid(full_client: TestClient) -> None:
    body = full_client.post("/analyze", json=_ARTICLE_PAYLOAD).json()
    generated_by = body["risk_brief"]["generated_by"]
    assert generated_by in {"llm", "fallback"}


# ---------------------------------------------------------------------------
# Happy-path: /health and /ready
# ---------------------------------------------------------------------------


def test_health_returns_200_ok(full_client: TestClient) -> None:
    response = full_client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_ready_all_models_loaded_returns_true(full_client: TestClient) -> None:
    body = full_client.get("/ready").json()
    assert body["models_loaded"] is True


# ---------------------------------------------------------------------------
# Error / 422 tests
# ---------------------------------------------------------------------------


def test_classify_text_too_short_returns_422(full_client: TestClient) -> None:
    assert full_client.post("/classify", json=_SHORT_TEXT_PAYLOAD).status_code == 422


def test_classify_missing_body_returns_422(full_client: TestClient) -> None:
    assert full_client.post("/classify").status_code == 422


def test_422_response_has_detail_key(full_client: TestClient) -> None:
    response = full_client.post("/classify", json=_SHORT_TEXT_PAYLOAD)
    assert "detail" in response.json()


# ---------------------------------------------------------------------------
# 503 tests — missing models
# ---------------------------------------------------------------------------


def test_classify_no_model_returns_503(no_models_client: TestClient) -> None:
    assert no_models_client.post("/classify", json=_ARTICLE_PAYLOAD).status_code == 503


def test_score_no_urgency_model_returns_503(no_models_client: TestClient) -> None:
    assert no_models_client.post("/score", json=_ARTICLE_PAYLOAD).status_code == 503


def test_analyze_no_classification_model_returns_503(
    no_models_client: TestClient,
) -> None:
    assert no_models_client.post("/analyze", json=_ARTICLE_PAYLOAD).status_code == 503


def test_ready_all_models_none_returns_false(no_models_client: TestClient) -> None:
    body = no_models_client.get("/ready").json()
    assert body["models_loaded"] is False


def test_ready_partial_models_returns_false(baseline_only_client: TestClient) -> None:
    body = baseline_only_client.get("/ready").json()
    assert body["models_loaded"] is False


# ---------------------------------------------------------------------------
# Global exception handler
# ---------------------------------------------------------------------------


def test_global_exception_handler_returns_error_key(full_client: TestClient) -> None:
    response = full_client.get("/_test_error")
    assert response.status_code == 500
    assert "error" in response.json()


# ---------------------------------------------------------------------------
# Benchmark — tagged so pytest -m "not benchmark" skips it
# ---------------------------------------------------------------------------


@pytest.mark.benchmark
def test_benchmark_classify_baseline_latency() -> None:
    with TestClient(app) as client:
        app.state.baseline = _mock_baseline()
        app.state.distilbert = None
        app.state.urgency = _mock_urgency()
        app.state.generator = _mock_generator()

        latencies: list[float] = []
        for _ in range(100):
            t0 = time.perf_counter()
            resp = client.post("/classify", json=_ARTICLE_PAYLOAD)
            latencies.append((time.perf_counter() - t0) * 1000)
            assert resp.status_code == 200

        latencies.sort()
        p50 = latencies[49]
        p99 = latencies[98]
        print(f"\n/classify baseline — p50={p50:.1f}ms  p99={p99:.1f}ms")
        assert p50 < 50
        assert p99 < 200
