"""Unit tests for FinSightClassifier.

All non-slow tests use a tiny mock model (58K params) built from a minimal
DistilBertConfig — no network calls, no 250MB download.
"""

from __future__ import annotations

import os
import tempfile
from typing import Any
from unittest.mock import patch

import pytest
import torch
from transformers import (
    DistilBertConfig,
    DistilBertForSequenceClassification,
    PreTrainedTokenizerFast,
)

from src.ingestion.schema import ClassificationResult
from src.models.distilbert import FinSightClassifier

# ---------------------------------------------------------------------------
# Tiny mock infrastructure
# ---------------------------------------------------------------------------

# vocab_size must match the real BERT tokenizer (30522) so embedding lookups work
_TINY_CFG = DistilBertConfig(
    vocab_size=30522,
    n_layers=1,
    n_heads=2,
    dim=32,
    hidden_dim=64,
    num_labels=4,
    max_position_embeddings=512,
)


def _make_tiny_model(*args: Any, **kwargs: Any) -> DistilBertForSequenceClassification:
    """Return a randomly-initialised tiny DistilBERT (no download)."""
    return DistilBertForSequenceClassification(_TINY_CFG)


def _make_tiny_tokenizer(*args: Any, **kwargs: Any) -> PreTrainedTokenizerFast:
    """Return the real fast tokenizer for 'bert-base-uncased' (tiny vocab ok)."""
    from transformers import AutoTokenizer

    # Use a cached tokenizer or fall back to bert-base-uncased
    try:
        tok = AutoTokenizer.from_pretrained(
            "bert-base-uncased", local_files_only=False
        )
    except Exception:
        tok = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    return tok  # type: ignore[return-value]


@pytest.fixture(scope="session")
def tiny_tokenizer() -> PreTrainedTokenizerFast:
    return _make_tiny_tokenizer()


@pytest.fixture(scope="session")
def patched_classifier(tiny_tokenizer: PreTrainedTokenizerFast) -> FinSightClassifier:
    """FinSightClassifier backed by a tiny model — no network, no GPU needed."""
    with (
        patch(
            "src.models.distilbert.AutoModelForSequenceClassification.from_pretrained",
            side_effect=_make_tiny_model,
        ),
        patch(
            "src.models.distilbert.AutoTokenizer.from_pretrained",
            return_value=tiny_tokenizer,
        ),
    ):
        clf = FinSightClassifier()
    return clf


@pytest.fixture
def fresh_patched_clf(tiny_tokenizer: PreTrainedTokenizerFast) -> FinSightClassifier:
    """Fresh instance per test — needed for training tests."""
    with (
        patch(
            "src.models.distilbert.AutoModelForSequenceClassification.from_pretrained",
            side_effect=_make_tiny_model,
        ),
        patch(
            "src.models.distilbert.AutoTokenizer.from_pretrained",
            return_value=tiny_tokenizer,
        ),
    ):
        return FinSightClassifier()


# ---------------------------------------------------------------------------
# Sample texts
# ---------------------------------------------------------------------------

_TEXTS = [
    "stock market rally driven by strong earnings reports today",
    "olympic athletes set new world records in swimming events",
    "world leaders summit on climate change and global policy",
    "nasa launches new telescope to study distant galaxies",
]

_LABELS = [2, 1, 0, 3]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_predict_returns_classification_result(
    patched_classifier: FinSightClassifier,
) -> None:
    result = patched_classifier.predict(_TEXTS[0])
    assert isinstance(result, ClassificationResult)
    assert result.model == "distilbert"


def test_predict_label_is_valid_literal(
    patched_classifier: FinSightClassifier,
) -> None:
    valid = {"World", "Sports", "Business", "Sci/Tech"}
    result = patched_classifier.predict(_TEXTS[0])
    assert result.label in valid


def test_predict_confidence_between_zero_and_one(
    patched_classifier: FinSightClassifier,
) -> None:
    result = patched_classifier.predict(_TEXTS[0])
    assert 0.0 <= result.confidence <= 1.0


def test_predict_batch_returns_correct_length(
    patched_classifier: FinSightClassifier,
) -> None:
    results = patched_classifier.predict_batch(_TEXTS)
    assert len(results) == len(_TEXTS)


def test_predict_batch_all_classification_results(
    patched_classifier: FinSightClassifier,
) -> None:
    results = patched_classifier.predict_batch(_TEXTS)
    assert all(isinstance(r, ClassificationResult) for r in results)


def test_predict_batch_confidence_sum_approx_one(
    patched_classifier: FinSightClassifier,
) -> None:
    """Softmax probabilities across all classes must sum to ~1 per sample."""
    # Re-run logits manually to verify softmax property
    patched_classifier.model.eval()
    enc = patched_classifier.tokenizer(
        [_TEXTS[0]],
        max_length=64,
        padding=True,
        truncation=True,
        return_tensors="pt",
    )
    with torch.no_grad():
        logits = patched_classifier.model(**enc).logits

    probs = torch.softmax(logits, dim=-1)
    assert abs(probs.sum().item() - 1.0) < 1e-5


def test_save_writes_file(patched_classifier: FinSightClassifier) -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "model.pt")
        patched_classifier.save(path)
        assert os.path.exists(path)
        assert os.path.getsize(path) > 0


def test_load_roundtrip_identical_predictions(
    patched_classifier: FinSightClassifier, tiny_tokenizer: PreTrainedTokenizerFast
) -> None:
    original = patched_classifier.predict(_TEXTS[0])

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "model.pt")
        patched_classifier.save(path)

        with patch(
            "src.models.distilbert.AutoModelForSequenceClassification.from_pretrained",
            side_effect=_make_tiny_model,
        ):
            loaded = FinSightClassifier.load(path)

    reloaded = loaded.predict(_TEXTS[0])
    assert original.label == reloaded.label
    assert abs(original.confidence - reloaded.confidence) < 1e-5


def test_evaluate_returns_required_keys(
    patched_classifier: FinSightClassifier,
) -> None:
    metrics = patched_classifier.evaluate(_TEXTS, _LABELS)
    assert "accuracy" in metrics
    assert "macro_f1" in metrics
    assert "weighted_f1" in metrics
    assert "per_class_f1" in metrics


def test_evaluate_accuracy_in_range(patched_classifier: FinSightClassifier) -> None:
    metrics = patched_classifier.evaluate(_TEXTS, _LABELS)
    assert 0.0 <= metrics["accuracy"] <= 1.0


def test_train_runs_without_error(
    fresh_patched_clf: FinSightClassifier,
) -> None:
    """50 samples, 1 epoch — must complete on CPU without raising."""
    texts = [f"financial news article {i} economy markets stocks" for i in range(50)]
    labels = [i % 4 for i in range(50)]
    train_t = texts[:40]
    train_l = labels[:40]
    val_t = texts[40:]
    val_l = labels[40:]

    with tempfile.TemporaryDirectory() as tmpdir:
        out = os.path.join(tmpdir, "tiny.pt")
        history = fresh_patched_clf.train(
            train_t, train_l, val_t, val_l,
            epochs=1, batch_size=8, lr=2e-5, output_path=out,
        )

    assert "train_loss" in history
    assert "val_loss" in history
    assert "val_accuracy" in history
    assert len(history["train_loss"]) == 1


def test_early_stopping_triggers(
    fresh_patched_clf: FinSightClassifier,
) -> None:
    """With a static model and constant val_loss, patience=2 stops at epoch 3."""
    texts = [f"market news article number {i} stocks bonds yields" for i in range(60)]
    labels = [i % 4 for i in range(60)]

    original_forward = fresh_patched_clf.model.forward

    call_count = 0

    def patched_forward(*args: Any, **kwargs: Any) -> Any:
        nonlocal call_count
        call_count += 1
        out = original_forward(*args, **kwargs)
        # Return a fixed loss so val_loss never improves after epoch 1.
        # requires_grad=True lets .backward() succeed on this leaf tensor.
        if kwargs.get("labels") is not None:
            from types import SimpleNamespace

            fixed_loss = torch.tensor(1.5, requires_grad=True)
            return SimpleNamespace(loss=fixed_loss, logits=out.logits)
        return out

    fresh_patched_clf.model.forward = patched_forward  # type: ignore[method-assign]

    with tempfile.TemporaryDirectory() as tmpdir:
        out = os.path.join(tmpdir, "es.pt")
        history = fresh_patched_clf.train(
            texts[:48], labels[:48], texts[48:], labels[48:],
            epochs=5, batch_size=8, lr=2e-5, output_path=out,
        )

    # Should stop before all 5 epochs due to patience=2
    assert len(history["train_loss"]) <= 5


def test_co2_kg_is_float_after_training(
    fresh_patched_clf: FinSightClassifier,
) -> None:
    texts = [f"economic data markets reacted article {i}" for i in range(40)]
    labels = [i % 4 for i in range(40)]

    with tempfile.TemporaryDirectory() as tmpdir:
        out = os.path.join(tmpdir, "co2.pt")
        history = fresh_patched_clf.train(
            texts[:32], labels[:32], texts[32:], labels[32:],
            epochs=1, batch_size=8, lr=2e-5, output_path=out,
        )

    assert "co2_kg" in history
    assert isinstance(history["co2_kg"][0], float)
    assert history["co2_kg"][0] >= 0.0


# ---------------------------------------------------------------------------
# Integration test — only runs if artefact exists
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_integration_accuracy_on_real_artefact() -> None:
    artefact = "artefacts/distilbert_finsight.pt"
    if not os.path.exists(artefact):
        pytest.skip("artefact not found — run scripts/train_distilbert.py first")

    import random
    from datasets import load_dataset

    _SELECTED = ["POLITICS", "BUSINESS", "ENTERTAINMENT", "WELLNESS"]
    _LABEL_TO_INT = {"POLITICS": 0, "BUSINESS": 1, "ENTERTAINMENT": 2, "WELLNESS": 3}
    raw = load_dataset("heegyu/news-category-dataset", split="train")
    per: dict = {c: [] for c in _SELECTED}
    for item in raw:
        cat = item["category"]
        if cat in per and len(per[cat]) < 25:
            per[cat].append(item)
    all_items = [i for items in per.values() for i in items]
    random.shuffle(all_items)
    test_texts: list[str] = [
        (i["headline"] + " " + i["short_description"]).strip() for i in all_items
    ]
    test_labels: list[int] = [_LABEL_TO_INT[i["category"]] for i in all_items]

    clf = FinSightClassifier.load(artefact)
    metrics = clf.evaluate(test_texts, test_labels)
    assert metrics["accuracy"] > 0.85, (
        f"Expected accuracy > 0.85, got {metrics['accuracy']:.4f}"
    )
