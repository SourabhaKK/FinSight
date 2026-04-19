"""Tests for Phase 6 drift detection — no real API calls, no network."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from src.monitoring.drift import DriftDetector, DriftReport

_VALID_STATUSES = {"stable", "warning", "critical"}
_FINSIGHT_ROOT = Path(__file__).parent.parent


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_balanced_data(n_per_class: int = 50) -> tuple[list[str], list[int]]:
    texts: list[str] = []
    labels: list[int] = []
    for label in range(4):
        for i in range(n_per_class):
            texts.append(" ".join(["word"] * (18 + i % 5)))
            labels.append(label)
    return texts, labels


def _fitted_detector(n_per_class: int = 50) -> DriftDetector:
    det = DriftDetector()
    texts, labels = _make_balanced_data(n_per_class)
    det.fit(texts, labels)
    return det


def _make_critical_current() -> tuple[list[str], list[int]]:
    """80% Business (label 2), 20% evenly split across others."""
    texts = [" ".join(["word"] * 20)] * 100
    labels = [2] * 80 + [0] * 7 + [1] * 7 + [3] * 6
    return texts, labels


# ---------------------------------------------------------------------------
# Happy-path tests
# ---------------------------------------------------------------------------


def test_identical_distributions_stable() -> None:
    texts, labels = _make_balanced_data()
    det = DriftDetector()
    det.fit(texts, labels)
    report = det.detect(texts, labels)
    assert report.status == "stable"
    assert report.psi < 0.05


def test_shifted_distribution_critical() -> None:
    ref_texts = [" ".join(["word"] * 20)] * 100
    ref_labels = [i % 4 for i in range(100)]

    cur_texts, cur_labels = _make_critical_current()
    det = DriftDetector()
    det.fit(ref_texts, ref_labels)
    report = det.detect(cur_texts, cur_labels)
    assert report.status == "critical"


def test_small_length_change_stable() -> None:
    rng = np.random.default_rng(42)
    n = 200
    # Wide uniform range — 5-word shift is too small for KS to detect at α=0.05
    ref_lengths = rng.integers(50, 150, size=n)
    cur_lengths = rng.integers(55, 155, size=n)
    labels = [i % 4 for i in range(n)]
    ref_texts = [" ".join(["w"] * int(ln)) for ln in ref_lengths]
    cur_texts = [" ".join(["w"] * int(ln)) for ln in cur_lengths]

    det = DriftDetector()
    det.fit(ref_texts, labels)
    report = det.detect(cur_texts, labels)
    assert report.status == "stable"


def test_large_length_change_warning_or_critical() -> None:
    n = 200
    labels = [i % 4 for i in range(n)]
    ref_texts = [" ".join(["w"] * 20)] * n
    cur_texts = [" ".join(["w"] * 40)] * n  # doubled mean

    det = DriftDetector()
    det.fit(ref_texts, labels)
    report = det.detect(cur_texts, labels)
    assert report.status in {"warning", "critical"}


def test_triggered_by_empty_on_stable() -> None:
    texts, labels = _make_balanced_data()
    det = DriftDetector()
    det.fit(texts, labels)
    report = det.detect(texts, labels)
    assert report.triggered_by == []


def test_triggered_by_contains_psi_on_distribution_shift() -> None:
    ref_texts = [" ".join(["word"] * 20)] * 100
    ref_labels = [i % 4 for i in range(100)]
    cur_texts, cur_labels = _make_critical_current()

    det = DriftDetector()
    det.fit(ref_texts, ref_labels)
    report = det.detect(cur_texts, cur_labels)
    assert "psi" in report.triggered_by


def test_save_creates_file_load_restores_distributions(tmp_path: Path) -> None:
    texts, labels = _make_balanced_data()
    det = DriftDetector()
    det.fit(texts, labels)

    path = str(tmp_path / "detector.joblib")
    det.save(path)
    assert Path(path).exists()

    loaded = DriftDetector.load(path)
    np.testing.assert_array_almost_equal(
        loaded.reference_topic_dist, det.reference_topic_dist
    )
    np.testing.assert_array_almost_equal(
        loaded.reference_lengths, det.reference_lengths
    )


def test_detect_raises_valueerror_if_not_fitted() -> None:
    det = DriftDetector()
    texts, labels = _make_balanced_data(n_per_class=10)
    with pytest.raises(ValueError):
        det.detect(texts, labels)


def test_psi_is_non_negative_float() -> None:
    det = _fitted_detector()
    cur_texts = [" ".join(["word"] * 20)] * 100
    cur_labels = [2] * 60 + [0] * 15 + [1] * 15 + [3] * 10
    report = det.detect(cur_texts, cur_labels)
    assert isinstance(report.psi, float)
    assert report.psi >= 0.0


def test_ks_statistic_between_0_and_1() -> None:
    det = _fitted_detector()
    texts, labels = _make_balanced_data()
    report = det.detect(texts, labels)
    assert 0.0 <= report.ks_statistic <= 1.0


def test_status_is_valid_literal() -> None:
    det = _fitted_detector()
    texts, labels = _make_balanced_data()
    report = det.detect(texts, labels)
    assert report.status in _VALID_STATUSES


def test_chi2_pvalue_between_0_and_1() -> None:
    det = _fitted_detector()
    texts, labels = _make_balanced_data()
    report = det.detect(texts, labels)
    assert 0.0 <= report.chi2_pvalue <= 1.0


def test_detect_30_samples_per_class_no_error() -> None:
    texts, labels = _make_balanced_data(n_per_class=30)  # 120 total
    det = DriftDetector()
    det.fit(texts, labels)
    report = det.detect(texts, labels)
    assert isinstance(report, DriftReport)


# ---------------------------------------------------------------------------
# CLI subprocess tests
# ---------------------------------------------------------------------------


def _run_cli(ref_path: str, cur_path: str) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "src.monitoring.alerts",
            "--reference",
            ref_path,
            "--current",
            cur_path,
        ],
        capture_output=True,
        cwd=str(_FINSIGHT_ROOT),
    )


def test_cli_exit_code_0_on_stable_data(tmp_path: Path) -> None:
    texts, labels = _make_balanced_data()
    det = DriftDetector()
    det.fit(texts, labels)

    ref_path = str(tmp_path / "detector.joblib")
    det.save(ref_path)

    cur_path = str(tmp_path / "current.json")
    with open(cur_path, "w") as f:
        json.dump({"texts": texts, "labels": labels}, f)

    result = _run_cli(ref_path, cur_path)
    assert result.returncode == 0


def test_cli_exit_code_2_on_critical_data(tmp_path: Path) -> None:
    ref_texts = [" ".join(["word"] * 20)] * 100
    ref_labels = [i % 4 for i in range(100)]
    det = DriftDetector()
    det.fit(ref_texts, ref_labels)

    ref_path = str(tmp_path / "detector.joblib")
    det.save(ref_path)

    cur_texts, cur_labels = _make_critical_current()
    cur_path = str(tmp_path / "current.json")
    with open(cur_path, "w") as f:
        json.dump({"texts": cur_texts, "labels": cur_labels}, f)

    result = _run_cli(ref_path, cur_path)
    assert result.returncode == 2
