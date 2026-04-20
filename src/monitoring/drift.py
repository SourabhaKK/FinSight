from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import joblib
import numpy as np
from scipy.stats import chisquare, ks_2samp

_PSI_WARN = 0.1
_PSI_CRIT = 0.2
_P_THRESHOLD = 0.05


@dataclass
class DriftReport:
    psi: float
    ks_statistic: float
    ks_pvalue: float
    chi2_statistic: float
    chi2_pvalue: float
    status: Literal["stable", "warning", "critical"]
    triggered_by: list[str]


class DriftDetector:
    def __init__(self) -> None:
        self.reference_topic_dist: np.ndarray | None = None
        self.reference_lengths: np.ndarray | None = None

    def fit(
        self,
        reference_texts: list[str],
        reference_labels: list[int],
    ) -> None:
        counts = np.zeros(4, dtype=float)
        for label in reference_labels:
            counts[label] += 1.0
        self.reference_topic_dist = counts / counts.sum()
        self.reference_lengths = np.array(
            [len(text.split()) for text in reference_texts], dtype=float
        )

    def detect(
        self,
        current_texts: list[str],
        current_labels: list[int],
    ) -> DriftReport:
        if self.reference_topic_dist is None or self.reference_lengths is None:
            raise ValueError("DriftDetector must be fitted before calling detect()")

        current_counts = np.zeros(4, dtype=float)
        for label in current_labels:
            current_counts[label] += 1.0
        current_dist = current_counts / current_counts.sum()

        eps = 1e-10
        ref = self.reference_topic_dist + eps
        cur = current_dist + eps
        psi = float(np.sum((cur - ref) * np.log(cur / ref)))

        current_lengths = np.array(
            [len(text.split()) for text in current_texts], dtype=float
        )
        ks_stat, ks_pval = ks_2samp(self.reference_lengths, current_lengths)

        expected = self.reference_topic_dist * current_counts.sum()
        chi2_stat, chi2_pval = chisquare(current_counts, f_exp=expected)

        triggered_by: list[str] = []
        is_critical = False

        if psi >= _PSI_CRIT:
            triggered_by.append("psi")
            is_critical = True
        elif psi >= _PSI_WARN:
            triggered_by.append("psi")

        if ks_pval < _P_THRESHOLD:
            triggered_by.append("ks")

        if chi2_pval < _P_THRESHOLD:
            triggered_by.append("chi2")

        if is_critical:
            status: Literal["stable", "warning", "critical"] = "critical"
        elif triggered_by:
            status = "warning"
        else:
            status = "stable"

        return DriftReport(
            psi=psi,
            ks_statistic=float(ks_stat),
            ks_pvalue=float(ks_pval),
            chi2_statistic=float(chi2_stat),
            chi2_pvalue=float(chi2_pval),
            status=status,
            triggered_by=triggered_by,
        )

    def save(self, path: str) -> None:
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str) -> DriftDetector:
        obj = joblib.load(path)
        if not isinstance(obj, cls):
            raise TypeError(f"Expected DriftDetector, got {type(obj)}")
        return obj
