from __future__ import annotations

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.ingestion.schema import UrgencyResult

_LEVEL_MAP: dict[int, str] = {
    0: "low",
    1: "medium",
    2: "high",
    3: "critical",
}

_FEATURE_NAMES: list[str] = [
    "word_count",
    "avg_word_length",
    "digit_ratio",
    "uppercase_ratio",
    "exclamation_count",
    "question_count",
    "text_length",
]


def _dict_to_array(features: dict[str, float]) -> list[list[float]]:
    return [[features[k] for k in _FEATURE_NAMES]]


class UrgencyScorer:
    def __init__(self) -> None:
        self.pipeline: Pipeline = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(max_iter=1000)),
            ]
        )

    def fit(self, x: list[dict[str, float]], y: list[int]) -> None:
        matrix = [[row[k] for k in _FEATURE_NAMES] for row in x]
        self.pipeline.fit(matrix, y)

    def score(self, features: dict[str, float]) -> UrgencyResult:
        row = _dict_to_array(features)
        proba = self.pipeline.predict_proba(row)[0]
        idx = int(np.argmax(proba))
        return UrgencyResult(
            score=float(proba[idx]),
            level=_LEVEL_MAP[idx],  # type: ignore[arg-type]
            features_used=_FEATURE_NAMES,
        )

    def save(self, path: str) -> None:
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str) -> UrgencyScorer:
        obj = joblib.load(path)
        if not isinstance(obj, cls):
            raise TypeError(f"Expected UrgencyScorer, got {type(obj)}")
        return obj
