"""Standalone training script for FinSightClassifier on HuffPost News Category."""

from __future__ import annotations

import argparse
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

# allow `python scripts/train_distilbert.py` from repo root
sys.path.insert(0, str(Path(__file__).parent.parent))

import mlflow
import numpy as np
import sklearn
import torch
import transformers
from datasets import load_dataset
from sklearn.model_selection import train_test_split

from src.models.distilbert import FinSightClassifier

SELECTED_CATEGORIES = ["POLITICS", "BUSINESS", "ENTERTAINMENT", "WELLNESS"]
LABEL_TO_INT = {
    "POLITICS": 0, "BUSINESS": 1,
    "ENTERTAINMENT": 2, "WELLNESS": 3,
}
N_PER_CLASS = 5_000
SEED = 42


def main() -> None:
    mlflow.set_tracking_uri("mlruns")
    mlflow.set_experiment("finsight")

    device_name = (
        torch.cuda.get_device_name(0)
        if torch.cuda.is_available()
        else "CPU — WARNING: training will be slow"
    )
    print(f"Device: {device_name}")

    parser = argparse.ArgumentParser(
        description="Train FinSightClassifier on HuffPost News Category Dataset"
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Smoke-test mode: 500 train / 100 val samples",
    )
    args = parser.parse_args()

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

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

    if args.quick:
        n_train, n_val = 500, 100
        print(f"Quick mode: {n_train} train / {n_val} val samples")
    else:
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

    epochs = 3
    batch_size = 16
    lr = 2e-5
    artefact_path = "artefacts/distilbert_finsight.pt"
    meta_path = "artefacts/distilbert_finsight_meta.json"

    with mlflow.start_run(run_name="distilbert-finetune"):
        mlflow.log_param("model_name", "distilbert-base-uncased")
        mlflow.log_param("epochs", epochs)
        mlflow.log_param("batch_size", batch_size)
        mlflow.log_param("learning_rate", lr)
        mlflow.log_param("train_samples", len(x_train))
        mlflow.log_param("val_samples", len(x_val))
        mlflow.log_param("test_samples", len(test_texts))
        mlflow.log_param("seed", SEED)
        mlflow.log_param("dataset", "heegyu/news-category-dataset")
        mlflow.log_param("num_classes", 4)

        print("\nInitialising FinSightClassifier (distilbert-base-uncased)...")
        clf = FinSightClassifier()

        print("\nStarting training...")
        history = clf.train(
            train_texts=x_train,
            train_labels=y_train,
            val_texts=x_val,
            val_labels=y_val,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            output_path=artefact_path,
        )

        print("\nTraining history:")
        for i, (tl, vl, va) in enumerate(
            zip(history["train_loss"], history["val_loss"], history["val_accuracy"])
        ):
            print(
                f"  Epoch {i + 1}: train_loss={tl:.4f}, "
                f"val_loss={vl:.4f}, val_acc={va:.4f}"
            )
            mlflow.log_metric("train_loss", tl, step=i)
            mlflow.log_metric("val_loss", vl, step=i)
            mlflow.log_metric("val_acc", va, step=i)

        co2_kg = history["co2_kg"][0] if history.get("co2_kg") else 0.0

        print("\nEvaluating on test set...")
        metrics = clf.evaluate(test_texts, test_labels)

        print("\n=== Evaluation Report ===")
        print(f"Accuracy:    {metrics['accuracy']:.4f}")
        print(f"Macro F1:    {metrics['macro_f1']:.4f}")
        print(f"Weighted F1: {metrics['weighted_f1']:.4f}")
        print("\nPer-class F1:")
        for label, score in metrics["per_class_f1"].items():
            print(f"  {label:<12} {score:.4f}")

        mlflow.log_metric("test_accuracy", metrics["accuracy"])
        mlflow.log_metric("test_macro_f1", metrics["macro_f1"])
        mlflow.log_metric("test_weighted_f1", metrics["weighted_f1"])
        mlflow.log_metric("co2_kg", co2_kg)
        mlflow.log_metric("f1_politics", metrics["per_class_f1"]["Politics"])
        mlflow.log_metric("f1_business", metrics["per_class_f1"]["Business"])
        mlflow.log_metric("f1_entertainment", metrics["per_class_f1"]["Entertainment"])
        mlflow.log_metric("f1_wellness", metrics["per_class_f1"]["Wellness"])

        meta = {
            "model": "distilbert-base-uncased",
            "task": "4-class news classification",
            "classes": SELECTED_CATEGORIES,
            "dataset": "heegyu/news-category-dataset",
            "train_samples": len(x_train),
            "val_samples": len(x_val),
            "test_samples": len(test_texts),
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": lr,
            "seed": SEED,
            "accuracy": metrics["accuracy"],
            "macro_f1": metrics["macro_f1"],
            "training_date": datetime.now(timezone.utc).isoformat(),
            "sklearn_version": sklearn.__version__,
            "torch_version": torch.__version__,
            "transformers_version": transformers.__version__,
        }
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)

        mlflow.log_artifact(artefact_path)
        mlflow.log_artifact(meta_path)

        print(f"\nArtefact saved to: {artefact_path}")
        print(f"Metadata saved to: {meta_path}")


if __name__ == "__main__":
    main()
