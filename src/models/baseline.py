from __future__ import annotations

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.ingestion.schema import ClassificationResult

_LABEL_MAP: dict[int, str] = {
    0: "Politics",
    1: "Business",
    2: "Entertainment",
    3: "Wellness",
}


class BaselineClassifier:
    def __init__(self) -> None:
        self.pipeline: Pipeline = Pipeline(
            [
                (
                    "tfidf",
                    TfidfVectorizer(
                        max_features=10000,
                        ngram_range=(1, 2),
                        sublinear_tf=True,
                    ),
                ),
                (
                    "clf",
                    LogisticRegression(
                        max_iter=1000,
                        C=1.0,
                        solver="lbfgs",
                    ),
                ),
            ]
        )

    def _assert_not_fitted(self) -> None:
        tfidf: TfidfVectorizer = self.pipeline.named_steps["tfidf"]
        if hasattr(tfidf, "vocabulary_"):
            raise RuntimeError(
                "fit() called on an already-fitted pipeline — create a new "
                "BaselineClassifier instance to retrain."
            )

    def fit(self, x_train: list[str], y_train: list[int]) -> None:
        self._assert_not_fitted()
        self.pipeline.fit(x_train, y_train)

    def predict(self, texts: list[str]) -> list[ClassificationResult]:
        proba = self.pipeline.predict_proba(texts)
        results: list[ClassificationResult] = []
        for row in proba:
            idx = int(row.argmax())
            results.append(
                ClassificationResult(
                    label=_LABEL_MAP[idx],  # type: ignore[arg-type]
                    confidence=float(row[idx]),
                    model="baseline",
                )
            )
        return results

    def predict_single(self, text: str) -> ClassificationResult:
        return self.predict([text])[0]

    def save(self, path: str) -> None:
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str) -> BaselineClassifier:
        obj = joblib.load(path)
        if not isinstance(obj, cls):
            raise TypeError(f"Expected BaselineClassifier, got {type(obj)}")
        return obj
