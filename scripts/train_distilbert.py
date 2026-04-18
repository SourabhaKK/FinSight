"""Standalone training script for FinSightClassifier on AG News."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# allow `python scripts/train_distilbert.py` from repo root
sys.path.insert(0, str(Path(__file__).parent.parent))

from datasets import load_dataset
from sklearn.model_selection import train_test_split

from src.models.distilbert import FinSightClassifier


def main() -> None:
    parser = argparse.ArgumentParser(description="Train FinSightClassifier on AG News")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Smoke-test mode: 1000 train / 200 val samples",
    )
    args = parser.parse_args()

    print("Loading AG News dataset...")
    ds = load_dataset("ag_news")

    train_texts_full: list[str] = ds["train"]["text"]
    train_labels_full: list[int] = ds["train"]["label"]
    test_texts: list[str] = ds["test"]["text"]
    test_labels: list[int] = ds["test"]["label"]

    if args.quick:
        n_train, n_val = 1000, 200
        print(f"Quick mode: {n_train} train / {n_val} val samples")
    else:
        n_train, n_val = 20000, 2000
        print(f"Full mode: {n_train} train / {n_val} val samples")

    # stratified split from training set
    x_train, x_val, y_train, y_val = train_test_split(
        train_texts_full,
        train_labels_full,
        train_size=n_train,
        test_size=n_val,
        stratify=train_labels_full,
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
