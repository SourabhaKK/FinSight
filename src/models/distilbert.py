from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F  # noqa: N812
from sklearn.metrics import f1_score
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    get_linear_schedule_with_warmup,
)

from src.ingestion.schema import ClassificationResult

try:
    from codecarbon import EmissionsTracker

    _HAS_CODECARBON = True
except ImportError:
    _HAS_CODECARBON = False


class _TextDataset(Dataset[dict[str, Any]]):
    def __init__(
        self,
        texts: list[str],
        labels: list[int],
        tokenizer: PreTrainedTokenizerBase,
        max_length: int = 128,
    ) -> None:
        self.encodings = tokenizer(
            texts,
            max_length=max_length,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        item = {k: v[idx] for k, v in self.encodings.items()}
        item["labels"] = self.labels[idx]
        return item


class FinSightClassifier:
    MODEL_NAME: str = "distilbert-base-uncased"
    NUM_LABELS: int = 4
    LABEL_MAP: dict[int, str] = {
        0: "World",
        1: "Sports",
        2: "Business",
        3: "Sci/Tech",
    }

    def __init__(self, model_path: str | None = None) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer: PreTrainedTokenizerBase
        self.model: PreTrainedModel

        if model_path is not None:
            self._load_from_path(model_path)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(self.MODEL_NAME)
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.MODEL_NAME, num_labels=self.NUM_LABELS
            )
            self.model.to(self.device)  # type: ignore[arg-type]

    def _load_from_path(self, path: str) -> None:
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.tokenizer = checkpoint["tokenizer"]
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.MODEL_NAME,
            num_labels=self.NUM_LABELS,
        )
        # load_state_dict separately so mocked from_pretrained still restores weights
        self.model.load_state_dict(
            {k: v.to(self.device) for k, v in checkpoint["model_state_dict"].items()}
        )
        self.model.to(self.device)  # type: ignore[arg-type]

    def train(
        self,
        train_texts: list[str],
        train_labels: list[int],
        val_texts: list[str],
        val_labels: list[int],
        epochs: int = 3,
        batch_size: int = 16,
        lr: float = 2e-5,
        output_path: str = "artefacts/distilbert_finsight.pt",
    ) -> dict[str, list[float]]:
        train_dataset = _TextDataset(train_texts, train_labels, self.tokenizer)
        val_dataset = _TextDataset(val_texts, val_labels, self.tokenizer)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)

        total_steps = len(train_loader) * epochs
        warmup_steps = int(0.1 * total_steps)

        optimizer = AdamW(self.model.parameters(), lr=lr)
        scheduler = get_linear_schedule_with_warmup(  # type: ignore[no-untyped-call]
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )

        history: dict[str, list[float]] = {
            "train_loss": [],
            "val_loss": [],
            "val_accuracy": [],
        }

        best_val_loss = float("inf")
        best_state: dict[str, Any] = {}
        patience_counter = 0
        co2_kg = 0.0

        tracker = None
        if _HAS_CODECARBON:
            Path("artefacts").mkdir(exist_ok=True)
            tracker = EmissionsTracker(
                output_dir="artefacts/", log_level="error", save_to_file=True
            )
            tracker.start()

        for epoch in range(epochs):
            t0 = time.time()

            # --- training ---
            self.model.train()
            total_train_loss = 0.0
            for batch in train_loader:
                optimizer.zero_grad()
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )
                loss = outputs.loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                total_train_loss += loss.item()

            avg_train_loss = total_train_loss / len(train_loader)

            # --- validation ---
            self.model.eval()  # type: ignore[no-untyped-call]
            total_val_loss = 0.0
            correct = 0
            total = 0
            with torch.no_grad():
                for batch in val_loader:
                    input_ids = batch["input_ids"].to(self.device)
                    attention_mask = batch["attention_mask"].to(self.device)
                    labels = batch["labels"].to(self.device)
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels,
                    )
                    total_val_loss += outputs.loss.item()
                    preds = outputs.logits.argmax(dim=-1)
                    correct += (preds == labels).sum().item()
                    total += labels.size(0)

            avg_val_loss = total_val_loss / len(val_loader)
            val_acc = correct / total if total > 0 else 0.0
            elapsed = time.time() - t0

            history["train_loss"].append(avg_train_loss)
            history["val_loss"].append(avg_val_loss)
            history["val_accuracy"].append(val_acc)

            # early stopping: check improvement
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_state = {
                    k: v.cpu().clone() for k, v in self.model.state_dict().items()
                }
                patience_counter = 0
                # save best checkpoint
                Path(output_path).parent.mkdir(parents=True, exist_ok=True)
                torch.save(
                    {
                        "model_state_dict": best_state,
                        "tokenizer": self.tokenizer,
                    },
                    output_path,
                )
            else:
                patience_counter += 1

            print(
                f"Epoch {epoch + 1}/{epochs} | "
                f"train_loss={avg_train_loss:.4f} | "
                f"val_loss={avg_val_loss:.4f} | "
                f"val_acc={val_acc:.4f} | "
                f"elapsed={elapsed:.1f}s | "
                f"co2_kg={co2_kg:.6f}"
            )

            if patience_counter >= 2:
                print(f"Early stopping at epoch {epoch + 1}")
                break

        if tracker is not None:
            co2_kg = tracker.stop() or 0.0
            print(f"Total co2_kg_emitted={co2_kg:.6f}")

        # restore best weights
        if best_state:
            self.model.load_state_dict(
                {k: v.to(self.device) for k, v in best_state.items()}
            )

        history["co2_kg"] = [co2_kg]
        return history

    def predict(self, text: str) -> ClassificationResult:
        return self.predict_batch([text])[0]

    def predict_batch(self, texts: list[str]) -> list[ClassificationResult]:
        self.model.eval()
        results: list[ClassificationResult] = []
        batch_size = 32

        for i in range(0, len(texts), batch_size):
            chunk = texts[i : i + batch_size]
            enc = self.tokenizer(
                chunk,
                max_length=128,
                padding=True,
                truncation=True,
                return_tensors="pt",
            )
            enc = {k: v.to(self.device) for k, v in enc.items()}
            with torch.no_grad():
                logits = self.model(**enc).logits
            probs = F.softmax(logits, dim=-1)
            for row in probs:
                idx = int(row.argmax().item())
                results.append(
                    ClassificationResult(
                        label=self.LABEL_MAP[idx],  # type: ignore[arg-type]
                        confidence=float(row[idx].item()),
                        model="distilbert",
                    )
                )
        return results

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state_dict": {
                    k: v.cpu() for k, v in self.model.state_dict().items()
                },
                "tokenizer": self.tokenizer,
            },
            path,
        )

    @classmethod
    def load(cls, path: str) -> FinSightClassifier:
        obj = cls.__new__(cls)
        obj.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        obj._load_from_path(path)
        return obj

    def evaluate(
        self,
        test_texts: list[str],
        test_labels: list[int],
    ) -> dict[str, float]:
        preds_objs = self.predict_batch(test_texts)
        pred_ids = [
            next(k for k, v in self.LABEL_MAP.items() if v == r.label)
            for r in preds_objs
        ]
        correct = sum(p == t for p, t in zip(pred_ids, test_labels))
        accuracy = correct / len(test_labels) if test_labels else 0.0

        macro_f1 = float(
            f1_score(test_labels, pred_ids, average="macro", zero_division=0)
        )
        weighted_f1 = float(
            f1_score(test_labels, pred_ids, average="weighted", zero_division=0)
        )
        per_class = f1_score(
            test_labels, pred_ids, average=None, zero_division=0, labels=[0, 1, 2, 3]
        )
        per_class_f1 = {
            self.LABEL_MAP[i]: float(per_class[i]) for i in range(self.NUM_LABELS)
        }

        return {
            "accuracy": accuracy,
            "macro_f1": macro_f1,
            "weighted_f1": weighted_f1,
            "per_class_f1": per_class_f1,  # type: ignore[dict-item]
        }
