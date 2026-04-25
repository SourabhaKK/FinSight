import os
import sys
from pathlib import Path

# Add project root to path
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from datasets import load_dataset

PLOTS = ROOT / "notebooks" / "plots"
PLOTS.mkdir(exist_ok=True)

SELECTED_CATEGORIES = ["POLITICS", "BUSINESS", "ENTERTAINMENT", "WELLNESS"]
LABEL_NAMES = ["Politics", "Business", "Entertainment", "Wellness"]
LABEL_TO_INT = {
    "POLITICS": 0, "BUSINESS": 1,
    "ENTERTAINMENT": 2, "WELLNESS": 3,
}

import random
from collections import Counter

print("Loading HuffPost News Category Dataset (CC BY 4.0)...")
print("Citation: Misra, R. (2022). arXiv:2209.11429")
raw_ds = load_dataset("heegyu/news-category-dataset", split="train")

N_PER_CLASS = 5_000
random.seed(42)
per_class: dict = {cat: [] for cat in SELECTED_CATEGORIES}
for item in raw_ds:
    cat = item["category"]
    if cat in per_class and len(per_class[cat]) < N_PER_CLASS:
        per_class[cat].append(item)

all_items = [item for items in per_class.values() for item in items]
random.shuffle(all_items)
all_texts  = [(item["headline"] + " " + item["short_description"]).strip() for item in all_items]
all_labels = [LABEL_TO_INT[item["category"]] for item in all_items]

# ── Plot 1: Class distribution bar chart ──
print("Generating plot 1: class distribution...")
counts = Counter(all_labels)

fig, ax = plt.subplots(figsize=(8, 4))
ax.bar(LABEL_NAMES, [counts[i] for i in range(4)],
       color=["#4C72B0", "#DD8452", "#55A868", "#C44E52"], alpha=0.8)
ax.set_xlabel("Class", fontsize=11)
ax.set_ylabel("Article count", fontsize=11)
ax.set_title("HuffPost News Category Distribution (balanced 4-class subset)", fontsize=12)
ax.set_ylim(0, N_PER_CLASS * 1.15)
for i, v in enumerate([counts[j] for j in range(4)]):
    ax.text(i, v + 50, str(v), ha="center", va="bottom", fontsize=10)
plt.tight_layout()
plt.savefig(PLOTS / "01_class_distribution.png", bbox_inches="tight", dpi=150)
plt.close()
print("  Saved: 01_class_distribution.png")

# ── Plot 2: Article length boxplot ──
print("Generating plot 2: article length distribution...")
lengths_by_class = {name: [] for name in LABEL_NAMES}
for text, label in zip(all_texts, all_labels):
    lengths_by_class[LABEL_NAMES[label]].append(len(text.split()))

fig, ax = plt.subplots(figsize=(8, 4))
bp = ax.boxplot(
    [lengths_by_class[n] for n in LABEL_NAMES],
    tick_labels=LABEL_NAMES,
    patch_artist=True,
    medianprops={"color": "red", "linewidth": 2},
)
colours = ["#4C8CBF", "#F4845F", "#67B99A", "#9B72CF"]
for patch, colour in zip(bp["boxes"], colours):
    patch.set_facecolor(colour)
    patch.set_alpha(0.6)
ax.set_xlabel("Class", fontsize=11)
ax.set_ylabel("Article length (words)", fontsize=11)
ax.set_title("Article Length Distribution by Class (Train set)", fontsize=12)
plt.tight_layout()
plt.savefig(PLOTS / "02_article_length_distribution.png",
            bbox_inches="tight", dpi=150)
plt.close()
print("  Saved: 02_article_length_distribution.png")

# ── Plot 3: Baseline confusion matrix ──
print("Generating plot 3: baseline confusion matrix...")

BASELINE_PATH = ROOT / "artefacts" / "baseline_pipeline.joblib"

if BASELINE_PATH.exists():
    from sklearn.metrics import confusion_matrix, f1_score, accuracy_score
    sample_idx = random.sample(range(len(all_texts)), min(1000, len(all_texts)))
    texts  = [all_texts[i] for i in sample_idx]
    labels = [all_labels[i] for i in sample_idx]

    from src.models.baseline import BaselineClassifier
    clf = BaselineClassifier.load(str(BASELINE_PATH))
    results = clf.predict(texts)
    preds   = [LABEL_TO_INT[r.label] for r in results]

    acc       = accuracy_score(labels, preds)
    macro_f1  = f1_score(labels, preds, average="macro")
    cm        = confusion_matrix(labels, preds)

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(4)); ax.set_yticks(range(4))
    ax.set_xticklabels(LABEL_NAMES, rotation=45, ha="right", fontsize=10)
    ax.set_yticklabels(LABEL_NAMES, fontsize=10)
    ax.set_xlabel("Predicted", fontsize=11)
    ax.set_ylabel("Actual", fontsize=11)
    ax.set_title(
        f"Baseline: TF-IDF + LogReg\n"
        f"Accuracy: {acc:.4f} | Macro-F1: {macro_f1:.4f}",
        fontsize=11
    )
    for i in range(4):
        for j in range(4):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max()/2 else "black",
                    fontsize=9)
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(PLOTS / "03_baseline_confusion_matrix.png",
                bbox_inches="tight", dpi=150)
    plt.close()
    print(f"  Saved: 03_baseline_confusion_matrix.png "
          f"(acc={acc:.4f}, f1={macro_f1:.4f})")
else:
    print("  SKIPPED: baseline artefact not found at artefacts/baseline_pipeline.joblib")
    print("  Run the notebook first to generate the artefact.")

# ── Plots 4 and 5: require DistilBERT artefact ──
DB_PATH = ROOT / "artefacts" / "distilbert_finsight.pt"
if not DB_PATH.exists():
    print("\nPlots 4 and 5 require artefacts/distilbert_finsight.pt")
    print("Run the notebook on Colab, download notebooks/plots/,")
    print("place it in finsight/notebooks/plots/, then re-run this script.")
else:
    print("\nDistilBERT artefact found — generating plots 4 and 5...")
    import torch
    from sklearn.metrics import confusion_matrix, f1_score, accuracy_score
    from src.models.distilbert import FinSightClassifier

    db_clf = FinSightClassifier.load(str(DB_PATH))

    test_idx = list(range(min(500, len(all_texts))))
    texts  = [all_texts[i] for i in test_idx]
    labels = [all_labels[i] for i in test_idx]

    # Reload baseline for the side-by-side
    if BASELINE_PATH.exists():
        from src.models.baseline import BaselineClassifier
        base_clf   = BaselineClassifier.load(str(BASELINE_PATH))
        base_res   = base_clf.predict(texts)
        base_preds = [LABEL_TO_INT[r.label] for r in base_res]
        cm_base    = confusion_matrix(labels, base_preds)
        acc_base   = accuracy_score(labels, base_preds)
        f1_base    = f1_score(labels, base_preds, average="macro")
    else:
        cm_base = None

    db_res   = db_clf.predict_batch(texts)
    db_preds = [LABEL_TO_INT[r.label] for r in db_res]
    cm_db    = confusion_matrix(labels, db_preds)
    acc_db   = accuracy_score(labels, db_preds)
    f1_db    = f1_score(labels, db_preds, average="macro")

    # Plot 4: side-by-side confusion matrices
    if cm_base is not None:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
        for ax, cm, title in [
            (ax1, cm_base,
             f"TF-IDF + LogReg\nAcc: {acc_base:.4f} | F1: {f1_base:.4f}"),
            (ax2, cm_db,
             f"DistilBERT\nAcc: {acc_db:.4f} | F1: {f1_db:.4f}"),
        ]:
            im = ax.imshow(cm, cmap="Blues")
            ax.set_xticks(range(4)); ax.set_yticks(range(4))
            ax.set_xticklabels(LABEL_NAMES, rotation=45,
                               ha="right", fontsize=9)
            ax.set_yticklabels(LABEL_NAMES, fontsize=9)
            ax.set_xlabel("Predicted", fontsize=10)
            ax.set_ylabel("Actual", fontsize=10)
            ax.set_title(title, fontsize=10)
            for i in range(4):
                for j in range(4):
                    ax.text(j, i, str(cm[i, j]),
                            ha="center", va="center",
                            color="white" if cm[i, j] > cm.max()/2
                            else "black", fontsize=8)
            plt.colorbar(im, ax=ax)
        plt.suptitle("Confusion Matrices: Baseline vs DistilBERT",
                     fontsize=12, fontweight="bold")
        plt.tight_layout()
        plt.savefig(PLOTS / "04_confusion_matrices_comparison.png",
                    bbox_inches="tight", dpi=150)
        plt.close()
        print("  Saved: 04_confusion_matrices_comparison.png")

    # Plot 5: attention visualisation
    from transformers import AutoTokenizer, AutoModel
    _tok = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    _vis = AutoModel.from_pretrained(
        "distilbert-base-uncased",
        attn_implementation="eager"
    )
    _vis.eval()

    sample_text = texts[0]
    enc = _tok(sample_text, return_tensors="pt",
               max_length=32, truncation=True)
    with torch.no_grad():
        out = _vis(**enc, output_attentions=True)

    attn   = out.attentions[-1][0].mean(0)[0].numpy()
    tokens = _tok.convert_ids_to_tokens(enc["input_ids"][0])

    fig, ax = plt.subplots(figsize=(max(10, len(tokens)*0.75), 2.2))
    im = ax.imshow(attn[np.newaxis, :], aspect="auto",
                   cmap="Blues", vmin=0)
    ax.set_xticks(range(len(tokens)))
    ax.set_xticklabels(tokens, rotation=45, ha="right", fontsize=9)
    ax.set_yticks([])
    ax.set_title(
        "Token-level attention weights — last layer, "
        "mean over all heads, from [CLS]\n"
        f"Article: {sample_text[:80]}...",
        fontsize=9
    )
    plt.colorbar(im, ax=ax, fraction=0.02, pad=0.04)
    plt.tight_layout()
    plt.savefig(PLOTS / "05_attention_visualisation.png",
                bbox_inches="tight", dpi=150)
    plt.close()
    del _vis
    print("  Saved: 05_attention_visualisation.png")

print("\nDone. Plots saved to notebooks/plots/")
print("Files generated:")
for f in sorted(PLOTS.glob("*.png")):
    print(f"  {f.name}")
