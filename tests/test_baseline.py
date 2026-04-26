import os
import tempfile

import pytest
from sklearn.exceptions import NotFittedError
from sklearn.metrics import roc_auc_score

from src.ingestion.schema import ClassificationResult
from src.models.baseline import BaselineClassifier

# ---------------------------------------------------------------------------
# Synthetic data helpers
# ---------------------------------------------------------------------------

_CLASS_TEXTS: dict[int, list[str]] = {
    0: [  # Politics
        "world leaders met in geneva to discuss climate change and global policy",
        "united nations summit addressed international security concerns today",
        "european union foreign ministers convened for emergency trade talks",
        "nato alliance expanded membership to include new eastern european nations",
        "global pandemic response required coordinated international health efforts",
    ],
    1: [  # Business
        "stock market rally pushed the dow jones to record highs on friday",
        "federal reserve raised interest rates to combat persistent inflation",
        "technology company reported quarterly earnings exceeding analyst forecasts",
        "merger acquisition deal valued at billions reshapes the banking sector",
        "oil prices surged following opec production cut announcement today",
    ],
    2: [  # Entertainment
        "the film won three academy awards including best picture of the year",
        "pop star released new album that broke streaming records globally",
        "television drama series finale drew the largest audience of the decade",
        "celebrity couple announced engagement after two years of dating publicly",
        "box office results showed record opening weekend for superhero sequel",
    ],
    3: [  # Wellness
        "researchers found mediterranean diet reduces heart disease risk significantly",
        "new study links regular exercise to improved mental health outcomes",
        "mindfulness meditation practice shown to lower cortisol levels in adults",
        "nutritionists recommend increasing plant based foods for long term wellness",
        "sleep quality improvements linked to better cognitive performance daily",
    ],
}


def _make_dataset(n_per_class: int = 50) -> tuple[list[str], list[int]]:
    texts: list[str] = []
    labels: list[int] = []
    base_texts = list(_CLASS_TEXTS.items())
    for label, samples in base_texts:
        for i in range(n_per_class):
            t = samples[i % len(samples)]
            texts.append(f"{t} article number {i} additional context words here")
            labels.append(label)
    return texts, labels


@pytest.fixture(scope="module")
def fitted_classifier() -> BaselineClassifier:
    clf = BaselineClassifier()
    x_train, y_train = _make_dataset(n_per_class=50)
    clf.fit(x_train, y_train)
    return clf


@pytest.fixture(scope="module")
def test_data() -> tuple[list[str], list[int]]:
    return _make_dataset(n_per_class=12)


# ---------------------------------------------------------------------------
# Leakage guard tests
# ---------------------------------------------------------------------------


def test_vocabulary_absent_before_fit() -> None:
    clf = BaselineClassifier()
    tfidf = clf.pipeline.named_steps["tfidf"]
    assert not hasattr(tfidf, "vocabulary_")


def test_vocabulary_present_after_fit(fitted_classifier: BaselineClassifier) -> None:
    tfidf = fitted_classifier.pipeline.named_steps["tfidf"]
    assert hasattr(tfidf, "vocabulary_")
    assert len(tfidf.vocabulary_) > 0


def test_assert_not_fitted_raises_on_second_fit(
    fitted_classifier: BaselineClassifier,
) -> None:
    with pytest.raises(RuntimeError, match="already-fitted"):
        fitted_classifier.fit(["some text here to test"], [0])


# ---------------------------------------------------------------------------
# predict() contract
# ---------------------------------------------------------------------------


def test_predict_returns_list_of_classification_results(
    fitted_classifier: BaselineClassifier,
) -> None:
    texts = ["global summit on trade", "basketball finals tonight"]
    results = fitted_classifier.predict(texts)
    assert isinstance(results, list)
    assert len(results) == 2
    assert all(isinstance(r, ClassificationResult) for r in results)


def test_predict_model_field_is_baseline(
    fitted_classifier: BaselineClassifier,
) -> None:
    results = fitted_classifier.predict(["stock market dropped sharply today"])
    assert results[0].model == "baseline"


def test_predict_confidence_in_range(fitted_classifier: BaselineClassifier) -> None:
    x_test, _ = _make_dataset(n_per_class=5)
    results = fitted_classifier.predict(x_test)
    for r in results:
        assert 0.0 <= r.confidence <= 1.0


def test_predict_label_is_valid_literal(fitted_classifier: BaselineClassifier) -> None:
    valid_labels = {"Politics", "Business", "Entertainment", "Wellness"}
    x_test, _ = _make_dataset(n_per_class=5)
    results = fitted_classifier.predict(x_test)
    for r in results:
        assert r.label in valid_labels


def test_predict_single_returns_classification_result(
    fitted_classifier: BaselineClassifier,
) -> None:
    result = fitted_classifier.predict_single("nasa launched new satellite today")
    assert isinstance(result, ClassificationResult)
    assert result.model == "baseline"


def test_predict_length_matches_input(fitted_classifier: BaselineClassifier) -> None:
    texts = ["a " * 10, "b " * 10, "c " * 10]
    results = fitted_classifier.predict(texts)
    assert len(results) == len(texts)


# ---------------------------------------------------------------------------
# Performance test
# ---------------------------------------------------------------------------


def test_roc_auc_above_threshold(fitted_classifier: BaselineClassifier) -> None:
    x_test, y_test = _make_dataset(n_per_class=12)
    proba = fitted_classifier.pipeline.predict_proba(x_test)
    auc = roc_auc_score(y_test, proba, multi_class="ovr")
    assert auc > 0.70, f"ROC-AUC {auc:.4f} is below 0.70 threshold"


# ---------------------------------------------------------------------------
# Save / load round-trip
# ---------------------------------------------------------------------------


def test_save_creates_file(fitted_classifier: BaselineClassifier) -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "baseline.joblib")
        fitted_classifier.save(path)
        assert os.path.exists(path)
        assert os.path.getsize(path) > 0


def test_load_roundtrip_produces_identical_predictions(
    fitted_classifier: BaselineClassifier,
) -> None:
    texts = [
        "world leaders discuss climate policy",
        "stock market rally on friday",
        "nasa launches new telescope",
        "basketball championship final game",
    ]
    original_preds = fitted_classifier.predict(texts)

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "baseline.joblib")
        fitted_classifier.save(path)
        loaded = BaselineClassifier.load(path)

    loaded_preds = loaded.predict(texts)
    for orig, loaded_r in zip(original_preds, loaded_preds):
        assert orig.label == loaded_r.label
        assert abs(orig.confidence - loaded_r.confidence) < 1e-6


def test_load_returns_baseline_classifier_instance(
    fitted_classifier: BaselineClassifier,
) -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "baseline.joblib")
        fitted_classifier.save(path)
        loaded = BaselineClassifier.load(path)
    assert isinstance(loaded, BaselineClassifier)


def test_unfitted_classifier_raises_on_predict() -> None:
    clf = BaselineClassifier()
    with pytest.raises(NotFittedError):
        clf.predict(["some text here that has not been fitted yet"])
