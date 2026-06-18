"""Standalone training script for BaselineClassifier (TF-IDF + LogReg)."""

from __future__ import annotations

import random
import statistics
import sys
import time
from pathlib import Path

# allow `python scripts/train_baseline.py` from repo root
sys.path.insert(0, str(Path(__file__).parent.parent))

import mlflow
import numpy as np
from datasets import load_dataset
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

from src.models.baseline import BaselineClassifier

SELECTED_CATEGORIES = ["POLITICS", "BUSINESS", "ENTERTAINMENT", "WELLNESS"]
LABEL_TO_INT = {
    "POLITICS": 0, "BUSINESS": 1,
    "ENTERTAINMENT": 2, "WELLNESS": 3,
}
INT_TO_LABEL = {v: k.capitalize() for k, v in LABEL_TO_INT.items()}
N_PER_CLASS = 5_000
SEED = 42


def main() -> None:
    mlflow.set_tracking_uri("mlruns")
    mlflow.set_experiment("finsight")

    random.seed(SEED)
    np.random.seed(SEED)

    print("Loading HuffPost News Category Dataset (CC BY 4.0)...")
    print("Citation: Misra, R. (2022). arXiv:2209.11429")
    raw_ds = load_dataset("heegyu/news-category-dataset", split="train")

    per_class: dict[str, list] = {cat: [] for cat in SELECTED_CATEGORIES}
    for item in raw_ds:
        cat = item["category"]
        if cat in per_class and len(per_class[cat]) < N_PER_CLASS:
            per_class[cat].append(item)

    all_items = [item for items in per_class.values() for item in items]
    random.shuffle(all_items)
    all_texts = [
        (item["headline"] + " " + item["short_description"]).strip()
        for item in all_items
    ]
    all_labels = [LABEL_TO_INT[item["category"]] for item in all_items]

    n_train, n_val = 16000, 2000
    print(f"Full mode: {n_train} train / {n_val} val samples")

    x_train, x_tmp, y_train, y_tmp = train_test_split(
        all_texts, all_labels,
        train_size=n_train,
        stratify=all_labels,
        random_state=42,
    )
    x_val, test_texts, y_val, test_labels = train_test_split(
        x_tmp, y_tmp,
        train_size=n_val,
        stratify=y_tmp,
        random_state=42,
    )

    print(f"Train: {len(x_train)}, Val: {len(x_val)}, Test: {len(test_texts)}")

    artefact_path = "artefacts/baseline_pipeline.joblib"
    max_features = 10000
    ngram_range = (1, 2)
    c_param = 1.0

    with mlflow.start_run(run_name="tfidf-logreg-baseline"):
        mlflow.log_param("model_name", "tfidf-logreg")
        mlflow.log_param("max_features", max_features)
        mlflow.log_param("ngram_range", str(ngram_range))
        mlflow.log_param("C", c_param)
        mlflow.log_param("train_samples", len(x_train))
        mlflow.log_param("val_samples", len(x_val))
        mlflow.log_param("test_samples", len(test_texts))
        mlflow.log_param("seed", SEED)
        mlflow.log_param("dataset", "heegyu/news-category-dataset")
        mlflow.log_param("num_classes", 4)

        print("\nTraining BaselineClassifier (TF-IDF + LogReg)...")
        clf = BaselineClassifier()
        clf.fit(x_train, y_train)

        print("\nEvaluating on test set...")
        preds = clf.predict(test_texts)
        pred_ids = [
            next(k for k, v in INT_TO_LABEL.items() if v == r.label) for r in preds
        ]
        accuracy = accuracy_score(test_labels, pred_ids)
        macro_f1 = f1_score(test_labels, pred_ids, average="macro", zero_division=0)

        print("\nMeasuring inference latency (p50)...")
        latencies_ms: list[float] = []
        for text in test_texts[:200]:
            t0 = time.perf_counter()
            clf.predict([text])
            latencies_ms.append((time.perf_counter() - t0) * 1000)
        p50_latency = statistics.median(latencies_ms)

        print("\n=== Evaluation Report ===")
        print(f"Accuracy:         {accuracy:.4f}")
        print(f"Macro F1:         {macro_f1:.4f}")
        print(f"Inference p50:    {p50_latency:.4f} ms")

        mlflow.log_metric("test_accuracy", float(accuracy))
        mlflow.log_metric("test_macro_f1", float(macro_f1))
        mlflow.log_metric("inference_latency_ms_p50", p50_latency)

        Path("artefacts").mkdir(exist_ok=True)
        clf.save(artefact_path)
        mlflow.log_artifact(artefact_path)

        print(f"\nArtefact saved to: {artefact_path}")


if __name__ == "__main__":
    main()
