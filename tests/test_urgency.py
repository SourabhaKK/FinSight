import os
import random
import tempfile

import pytest

from src.ingestion.schema import UrgencyResult
from src.models.urgency import _FEATURE_NAMES, UrgencyScorer

# ---------------------------------------------------------------------------
# Synthetic data helpers
# ---------------------------------------------------------------------------

_RNG = random.Random(42)


def _make_feature_row(label: int) -> dict[str, float]:
    """Generate a synthetic feature dict correlated with urgency label."""
    base_word_count = [50.0, 120.0, 250.0, 400.0][label]
    base_exclamation = [0.0, 1.0, 3.0, 6.0][label]
    return {
        "word_count": base_word_count + _RNG.uniform(-10, 10),
        "avg_word_length": 4.5 + _RNG.uniform(-0.5, 0.5),
        "digit_ratio": 0.05 + _RNG.uniform(0.0, 0.05),
        "uppercase_ratio": 0.03 + label * 0.02 + _RNG.uniform(0.0, 0.01),
        "exclamation_count": max(0.0, base_exclamation + _RNG.uniform(-1, 1)),
        "question_count": _RNG.uniform(0.0, 2.0),
        "text_length": base_word_count * 5.5 + _RNG.uniform(-50, 50),
    }


def _make_dataset(
    n_per_class: int = 20,
) -> tuple[list[dict[str, float]], list[int]]:
    features: list[dict[str, float]] = []
    labels: list[int] = []
    for label in range(4):
        for _ in range(n_per_class):
            features.append(_make_feature_row(label))
            labels.append(label)
    return features, labels


@pytest.fixture(scope="module")
def fitted_scorer() -> UrgencyScorer:
    scorer = UrgencyScorer()
    x_train, y_train = _make_dataset(n_per_class=20)
    scorer.fit(x_train, y_train)
    return scorer


@pytest.fixture(scope="module")
def sample_features() -> dict[str, float]:
    return _make_feature_row(label=2)  # "high" urgency class


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_score_returns_urgency_result(
    fitted_scorer: UrgencyScorer, sample_features: dict[str, float]
) -> None:
    result = fitted_scorer.score(sample_features)
    assert isinstance(result, UrgencyResult)


def test_score_level_is_valid_literal(
    fitted_scorer: UrgencyScorer, sample_features: dict[str, float]
) -> None:
    result = fitted_scorer.score(sample_features)
    assert result.level in {"low", "medium", "high", "critical"}


def test_score_value_between_zero_and_one(
    fitted_scorer: UrgencyScorer, sample_features: dict[str, float]
) -> None:
    result = fitted_scorer.score(sample_features)
    assert 0.0 <= result.score <= 1.0


def test_features_used_lists_all_seven_names(
    fitted_scorer: UrgencyScorer, sample_features: dict[str, float]
) -> None:
    result = fitted_scorer.score(sample_features)
    assert result.features_used == _FEATURE_NAMES
    assert len(result.features_used) == 7


def test_score_on_multiple_samples(fitted_scorer: UrgencyScorer) -> None:
    x_test, _ = _make_dataset(n_per_class=5)
    for feat in x_test:
        result = fitted_scorer.score(feat)
        assert isinstance(result, UrgencyResult)
        assert 0.0 <= result.score <= 1.0


def test_high_word_count_and_exclamation_tend_to_high_urgency(
    fitted_scorer: UrgencyScorer,
) -> None:
    high_urgency_feat = {
        "word_count": 350.0,
        "avg_word_length": 5.0,
        "digit_ratio": 0.05,
        "uppercase_ratio": 0.08,
        "exclamation_count": 7.0,
        "question_count": 1.0,
        "text_length": 1800.0,
    }
    result = fitted_scorer.score(high_urgency_feat)
    assert result.level in {"high", "critical"}


def test_save_creates_file(fitted_scorer: UrgencyScorer) -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "urgency.joblib")
        fitted_scorer.save(path)
        assert os.path.exists(path)
        assert os.path.getsize(path) > 0


def test_load_roundtrip_produces_same_result(
    fitted_scorer: UrgencyScorer, sample_features: dict[str, float]
) -> None:
    original = fitted_scorer.score(sample_features)

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "urgency.joblib")
        fitted_scorer.save(path)
        loaded = UrgencyScorer.load(path)

    reloaded = loaded.score(sample_features)
    assert original.level == reloaded.level
    assert abs(original.score - reloaded.score) < 1e-6


def test_load_returns_urgency_scorer_instance(
    fitted_scorer: UrgencyScorer,
) -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "urgency.joblib")
        fitted_scorer.save(path)
        loaded = UrgencyScorer.load(path)
    assert isinstance(loaded, UrgencyScorer)


def test_fit_on_larger_dataset_then_score() -> None:
    scorer = UrgencyScorer()
    x_train, y_train = _make_dataset(n_per_class=20)
    scorer.fit(x_train, y_train)
    x_test, _ = _make_dataset(n_per_class=5)
    for feat in x_test:
        result = scorer.score(feat)
        assert result.level in {"low", "medium", "high", "critical"}
        assert 0.0 <= result.score <= 1.0
