"""Apply all HuffPost dataset swap changes to the notebook."""
import json
from pathlib import Path

NB = Path("notebooks/WM9B7_finsight.ipynb")
with open(NB, encoding="utf-8") as f:
    nb = json.load(f)


def cell_src(idx):
    return "".join(nb["cells"][idx]["source"])


def set_src(idx, new_src):
    # store as a list of lines ending in \n (last without)
    lines = new_src.splitlines(keepends=True)
    if lines and not lines[-1].endswith("\n"):
        pass  # already fine
    elif lines and lines[-1].endswith("\n"):
        lines[-1] = lines[-1].rstrip("\n")
    nb["cells"][idx]["source"] = lines
    nb["cells"][idx]["outputs"] = []
    nb["cells"][idx]["execution_count"] = None


# ── CHANGE 1: LABEL_NAMES / LABEL_TO_INT / SELECTED_CATEGORIES in cell 3 ──
old_labels = (
    'LABEL_NAMES = ["World", "Sports", "Business", "Sci/Tech"]\n'
    'LABEL_TO_INT = {"World": 0, "Sports": 1, "Business": 2, "Sci/Tech": 3}'
)
new_labels = (
    'SELECTED_CATEGORIES = ["POLITICS", "BUSINESS", "ENTERTAINMENT", "WELLNESS"]\n'
    'LABEL_NAMES = ["Politics", "Business", "Entertainment", "Wellness"]\n'
    'LABEL_TO_INT = {\n'
    '    "POLITICS": 0, "BUSINESS": 1,\n'
    '    "ENTERTAINMENT": 2, "WELLNESS": 3\n'
    '}\n'
    '\n'
    '# Plots output directory\n'
    'PLOTS = Path("notebooks/plots")\n'
    'PLOTS.mkdir(parents=True, exist_ok=True)'
)
src3 = cell_src(3)
assert old_labels in src3, f"Change 1 target not found in cell 3"
src3 = src3.replace(old_labels, new_labels, 1)
set_src(3, src3)
print("Change 1 DONE: LABEL_NAMES / LABEL_TO_INT / SELECTED_CATEGORIES")

# ── CHANGE 2: Replace EDA cell (cell 4) ──
new_eda = '''\
# ── Cell 4: Dataset loading and EDA ──
import random
from collections import Counter

print("Loading HuffPost News Category Dataset (CC BY 4.0)...")
print("Source: https://www.kaggle.com/datasets/rmisra/news-category-dataset")
print("Citation: Misra, R. (2022). arXiv:2209.11429\\n")

raw_ds = load_dataset("heegyu/news-category-dataset", split="train")
print(f"Full dataset: {len(raw_ds):,} rows | {len(set(raw_ds['category']))} categories")

# ── Select and balance 4 classes ──
N_PER_CLASS = 5_000
random.seed(SEED)

per_class = {cat: [] for cat in SELECTED_CATEGORIES}
for item in raw_ds:
    cat = item["category"]
    if cat in per_class and len(per_class[cat]) < N_PER_CLASS:
        per_class[cat].append(item)

# Check all classes have enough samples
for cat, items in per_class.items():
    assert len(items) == N_PER_CLASS, \\
        f"{cat} only has {len(items)} samples (need {N_PER_CLASS})"

all_items = [item for items in per_class.values() for item in items]
random.shuffle(all_items)

# Input = headline + short_description concatenated
all_texts  = [
    (item["headline"] + " " + item["short_description"]).strip()
    for item in all_items
]
all_labels = [LABEL_TO_INT[item["category"]] for item in all_items]

print(f"\\nBalanced subset: {len(all_texts):,} samples")
print(f"Classes: {SELECTED_CATEGORIES}")

# ── Class distribution table ──
counts = Counter(all_labels)
dist_df = pd.DataFrame({
    "Class":  LABEL_NAMES,
    "Count":  [counts[i] for i in range(4)],
    "Pct %":  [round(counts[i] / len(all_labels) * 100, 1) for i in range(4)],
})
print("\\nClass distribution:")
print(dist_df.to_string(index=False))

# ── Plot 1: Class distribution bar chart ──
fig, ax = plt.subplots(figsize=(8, 4))
ax.bar(LABEL_NAMES, [counts[i] for i in range(4)],
       color=["#4C72B0", "#DD8452", "#55A868", "#C44E52"], alpha=0.8)
ax.set_xlabel("Class", fontsize=11)
ax.set_ylabel("Article count", fontsize=11)
ax.set_title("HuffPost News Category Distribution (balanced 4-class subset)",
             fontsize=11)
ax.set_ylim(0, N_PER_CLASS * 1.15)
for i, v in enumerate([counts[j] for j in range(4)]):
    ax.text(i, v + 50, str(v), ha="center", va="bottom", fontsize=10)
plt.tight_layout()
plt.savefig(PLOTS / "01_class_distribution.png",
            bbox_inches="tight", dpi=150)
plt.show()

# ── Plot 2: Article length boxplot ──
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
colours = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]
for patch, colour in zip(bp["boxes"], colours):
    patch.set_facecolor(colour)
    patch.set_alpha(0.6)
ax.set_xlabel("Class", fontsize=11)
ax.set_ylabel("Article length (words)", fontsize=11)
ax.set_title("Article Length Distribution by Class", fontsize=11)
plt.tight_layout()
plt.savefig(PLOTS / "02_article_length_distribution.png",
            bbox_inches="tight", dpi=150)
plt.show()

# ── 3 sample articles per class ──
print("\\n--- Sample articles (truncated to 120 chars) ---")
for label_idx, label_name in enumerate(LABEL_NAMES):
    samples = [t for t, l in zip(all_texts, all_labels) if l == label_idx][:3]
    print(f"\\n[{label_name}]")
    for i, s in enumerate(samples, 1):
        print(f"  {i}. {s[:120]}...")'''

set_src(4, new_eda)
print("Change 2 DONE: EDA cell (cell 4) replaced")

# ── CHANGE 3: Replace preprocessing cell (cell 5) ──
new_preproc = '''\
# ── Cell 5: Preprocessing ──
from src.preprocessing.pipeline import TextCleaner

cleaner = TextCleaner()

# Demonstrate clean() on 3 raw articles
print("=== TextCleaner.clean() demonstration ===\\n")
raw_samples = all_texts[:3]
for i, raw in enumerate(raw_samples, 1):
    cleaned = cleaner.clean(raw)
    print(f"--- Article {i} [{LABEL_NAMES[all_labels[i-1]]}] ---")
    print(f"  BEFORE: {raw[:120]!r}")
    print(f"  AFTER:  {cleaned[:120]!r}")
    print()

# Clean all texts
print(f"Cleaning {len(all_texts):,} articles...")
cleaned_texts = [cleaner.clean(t) for t in all_texts]

# Create splits
x_train, x_val, x_test, y_train, y_val, y_test = cleaner.create_splits(
    cleaned_texts, all_labels,
    test_size=0.2, val_size=0.1, random_state=SEED,
)

print(f"\\nSplit sizes:")
print(f"  train={len(x_train):,} | val={len(x_val):,} | test={len(x_test):,}")

def _class_dist(labels):
    c = Counter(labels)
    return " | ".join(f"{LABEL_NAMES[i]}: {c.get(i, 0)}" for i in range(4))

print("\\nClass distribution per split:")
print(f"  Train : {_class_dist(y_train)}")
print(f"  Val   : {_class_dist(y_val)}")
print(f"  Test  : {_class_dist(y_test)}")'''

set_src(5, new_preproc)
print("Change 3 DONE: preprocessing cell (cell 5) replaced")

# ── CHANGE 4: README markdown cell (cell 1) — dataset line ──
src1 = cell_src(1)
old_dataset_line = "Dataset loads automatically via HuggingFace `datasets`."
new_dataset_line = (
    "Dataset: HuffPost News Category Dataset (CC BY 4.0, Misra 2022).  \n"
    "Loads automatically: `load_dataset(\"heegyu/news-category-dataset\")`  \n"
    "Kaggle: https://www.kaggle.com/datasets/rmisra/news-category-dataset"
)
assert old_dataset_line in src1, "Change 4 target not found in cell 1"
src1 = src1.replace(old_dataset_line, new_dataset_line, 1)
set_src(1, src1)
print("Change 4 DONE: README markdown cell dataset line")

# ── CHANGE 5: Problem framing markdown (cell 2) ──
src2 = cell_src(2)
src2 = src2.replace(
    "AG News",
    "HuffPost News Category Dataset (Misra, 2022)"
)
set_src(2, src2)
print("Change 5 DONE: problem framing cell AG News references")

# ── CHANGE 5b: Why Baseline Has a Ceiling markdown (cell 7) ──
src7 = cell_src(7)
src7 = src7.replace(
    "AG News",
    "HuffPost News Category Dataset (Misra, 2022)"
)
set_src(7, src7)
print("Change 5b DONE: cell 7 markdown AG News references")

# ── CHANGE 6: References cell (cell 12) — Zhang → Misra ──
src12 = cell_src(12)
old_zhang = (
    "6. Zhang, X., Zhao, J., & LeCun, Y. (2015). Character-level convolutional "
    "networks for text classification. *NeurIPS 2015*. https://arxiv.org/abs/1509.01626"
)
new_misra = (
    "6. Misra, R. (2022). News Category Dataset. *arXiv:2209.11429*. "
    "https://www.kaggle.com/datasets/rmisra/news-category-dataset"
)
if old_zhang in src12:
    src12 = src12.replace(old_zhang, new_misra, 1)
    print("Change 6 DONE: Zhang → Misra in references cell")
else:
    # Try partial match
    import re
    src12 = re.sub(
        r"6\.\s+Zhang.*?(?=\n\d+\.|\Z)",
        new_misra,
        src12,
        flags=re.DOTALL
    )
    print("Change 6 DONE (regex): Zhang → Misra in references cell")
set_src(12, src12)

# ── CHANGE 7: "AG News test set" in evaluation commentary (cell 9) ──
src9 = cell_src(9)
src9 = src9.replace(
    "AG News test set",
    "HuffPost News Category Dataset test set (Misra, 2022)"
)
set_src(9, src9)
print("Change 7 DONE: evaluation commentary AG News test set")

# Write notebook
with open(NB, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
    f.write("\n")

print("\nNotebook written.")
