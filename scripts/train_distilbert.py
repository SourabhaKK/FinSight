"""Standalone training script for FinSightClassifier on HuffPost News Category Dataset."""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

# allow `python scripts/train_distilbert.py` from repo root
sys.path.insert(0, str(Path(__file__).parent.parent))

from datasets import load_dataset
from sklearn.model_selection import train_test_split

from src.models.distilbert import FinSightClassifier

SELECTED_CATEGORIES = ["POLITICS", "BUSINESS", "ENTERTAINMENT", "WELLNESS"]
LABEL_TO_INT = {
    "POLITICS": 0, "BUSINESS": 1,
    "ENTERTAINMENT": 2, "WELLNESS": 3,
}
N_PER_CLASS = 5_000


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train FinSightClassifier on HuffPost News Category Dataset"
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Smoke-test mode: 500 train / 100 val samples",
    )
    args = parser.parse_args()

    print("Loading HuffPost News Category Dataset (CC BY 4.0)...")
    print("Citation: Misra, R. (2022). arXiv:2209.11429")
    raw_ds = load_dataset("heegyu/news-category-dataset", split="train")

    random.seed(42)
    per_class: dict[str, list] = {cat: [] for cat in SELECTED_CATEGORIES}
    for item in raw_ds:
        cat = item["category"]
        if cat in per_class and len(per_class[cat]) < N_PER_CLASS:
            per_class[cat].append(item)

    all_items = [item for items in per_class.values() for item in items]
    random.shuffle(all_items)
    all_texts  = [(item["headline"] + " " + item["short_description"]).strip() for item in all_items]
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

    print("\nInitialising FinSightClassifier (distilbert-base-uncased)...")
    clf = FinSightClassifier()

    print("\nStarting training...")
    history = clf.train(
        train_texts=x_train,
        train_labels=y_train,
        val_texts=x_val,
        val_labels=y_val,
        epochs=3,
        batch_size=16,
        lr=2e-5,
        output_path="artefacts/distilbert_finsight.pt",
    )

    print("\nTraining history:")
    for i, (tl, vl, va) in enumerate(
        zip(history["train_loss"], history["val_loss"], history["val_accuracy"])
    ):
        print(f"  Epoch {i + 1}: train_loss={tl:.4f}, val_loss={vl:.4f}, val_acc={va:.4f}")

    print("\nEvaluating on test set...")
    metrics = clf.evaluate(test_texts, test_labels)

    print("\n=== Evaluation Report ===")
    print(f"Accuracy:    {metrics['accuracy']:.4f}")
    print(f"Macro F1:    {metrics['macro_f1']:.4f}")
    print(f"Weighted F1: {metrics['weighted_f1']:.4f}")
    print("\nPer-class F1:")
    for label, score in metrics["per_class_f1"].items():
        print(f"  {label:<12} {score:.4f}")

    print(f"\nArtefact saved to: artefacts/distilbert_finsight.pt")


if __name__ == "__main__":
    main()
