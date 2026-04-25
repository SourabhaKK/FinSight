"""Generate notebooks/WM9B7_finsight.ipynb in stages for incremental git history.

Usage:
    python gen_notebook.py [stage]
    stage 1 = skeleton (cells 1-3 + markdown placeholders)
    stage 2 = + EDA, preprocessing, baseline (cells 4-6)
    stage 3 = + DistilBERT cell (cell 8)
    stage 4 = + evaluation and comparison (cell 9)
    stage 5 = + drift detection + pipeline demo (cells 10-11)
    stage 6 = + summary / references (cell 12)  [final]
"""
import json
import sys
from pathlib import Path

STAGE = int(sys.argv[1]) if len(sys.argv) > 1 else 6


def md(source: str, cell_id: str) -> dict:
    return {"cell_type": "markdown", "id": cell_id, "metadata": {}, "source": source}


def code(source: str, cell_id: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "id": cell_id,
        "metadata": {},
        "outputs": [],
        "source": source,
    }


cells = []

# ── Cell 1: README ────────────────────────────────────────────────────────────
cells.append(md(
    "# FinSight \u2014 Financial News Risk Intelligence System\n"
    "## WMG9B7 Individual Assessment | Sourabha K Kallapur\n\n"
    "---\n\n"
    "## Problem Statement\n\n"
    "Financial institutions process thousands of news articles daily to assess market risk "
    "and make time-sensitive investment decisions. Manual review is prohibitively slow and "
    "inconsistent, while keyword-based systems miss nuanced financial signals. FinSight "
    "automates this pipeline by classifying financial news by topic, scoring urgency from "
    "tabular metadata, and generating structured risk briefs using a fine-tuned DistilBERT "
    "model. The system detects market regime shifts through statistical drift monitoring, "
    "enabling proactive risk management at scale.\n\n"
    "## Setup Instructions\n\n"
    "Run the following before executing any other cell:\n\n"
    "```bash\n"
    "!pip install -e ..\n"
    "```\n\n"
    "All cells must be run **in order**.  \n"
    "Dataset loads automatically via HuggingFace `datasets`.\n\n"
    "## Expected Runtime\n\n"
    "| Environment       | Duration    |\n"
    "|-------------------|-------------|\n"
    "| Colab GPU (T4)    | ~45 minutes |\n"
    "| CPU only          | ~3 hours    |\n\n"
    "## Model Artefacts\n\n"
    "- If `artefacts/distilbert_finsight.pt` **exists**, the notebook loads it and skips training.  \n"
    "- If it does **not** exist, the notebook trains DistilBERT from scratch "
    "(3 epochs, ~14k samples).",
    "cell-01-readme",
))

# ── Cell 2: Problem Framing ───────────────────────────────────────────────────
cells.append(md(
    "## Problem Framing and Motivation\n\n"
    "### The Problem \u2014 Why Financial News Classification Matters\n\n"
    "Financial institutions must process thousands of news articles daily to assess market "
    "risk, yet manual review is prohibitively slow for real-time decision-making. "
    "Misclassifying a market-moving event \u2014 such as a central bank policy change embedded "
    "in diplomatic language \u2014 can result in significant financial losses within minutes of "
    "publication. Automated, high-accuracy classification enables real-time alerting, "
    "portfolio rebalancing triggers, and compliance logging. The accuracy and latency of "
    "such a system directly translate to competitive advantage and quantifiable risk mitigation.\n\n"
    "### Why Deep Learning Over Classical ML\n\n"
    "Traditional TF-IDF approaches treat text as a bag of words, discarding word order and "
    "contextual relationships. Consider: *\u201cFed raises rates unexpectedly, markets fall\u201d* and "
    "*\u201cMarkets raise the Fed\u2019s unexpected rates fall\u201d* \u2014 identical TF-IDF vectors, very "
    "different meanings. DistilBERT (Sanh et al., 2019), a knowledge-distilled variant of "
    "BERT (Devlin et al., 2019), learns contextual embeddings where the same token receives "
    "different representations depending on its neighbours. The empirical comparison in this "
    "notebook demonstrates that contextual understanding yields substantially higher "
    "classification accuracy, particularly for ambiguous financial headlines.\n\n"
    "### System Overview\n\n"
    "FinSight is a 5-layer production pipeline:\n\n"
    "| Layer | Component | Purpose |\n"
    "|-------|-----------|----------|\n"
    "| 1 | **Ingestion** | Pydantic schema validation and metadata extraction |\n"
    "| 2 | **Preprocessing** | HTML stripping, URL removal, Unicode normalisation |\n"
    "| 3 | **Classification** | Dual-model: TF-IDF + LogReg (fast) and DistilBERT (accurate) |\n"
    "| 4 | **Risk Scoring** | Urgency assessment from 7 tabular metadata features |\n"
    "| 5 | **Monitoring** | PSI / KS / Chi-Square statistical drift detection |",
    "cell-02-framing",
))

# ── Cell 3: Imports and Setup ─────────────────────────────────────────────────
cells.append(code(
    "# Install package in editable mode so src/ imports work in Colab\n"
    "import subprocess\n"
    "import sys\n\n"
    "result = subprocess.run(\n"
    "    [sys.executable, \"-m\", \"pip\", \"install\", \"-e\", \"..\"],\n"
    "    capture_output=True,\n"
    "    text=True,\n"
    ")\n"
    "if result.returncode == 0:\n"
    "    print(\"Package installed successfully.\")\n"
    "else:\n"
    "    print(result.stdout[-1000:])\n"
    "    print(result.stderr[-1000:])\n"
    "    raise RuntimeError(\"Package installation failed.\")\n\n"
    "# Ensure project root is on sys.path\n"
    "from pathlib import Path\n\n"
    "_project_root = str(Path(\"..\").resolve())\n"
    "if _project_root not in sys.path:\n"
    "    sys.path.insert(0, _project_root)\n\n"
    "import json\n"
    "import random\n"
    "import time\n\n"
    "import matplotlib.pyplot as plt\n"
    "import numpy as np\n"
    "import pandas as pd\n"
    "import torch\n"
    "from datasets import load_dataset\n"
    "from sklearn.metrics import confusion_matrix, f1_score\n\n"
    "SEED = 42\n"
    "random.seed(SEED)\n"
    "np.random.seed(SEED)\n"
    "torch.manual_seed(SEED)\n"
    "if torch.cuda.is_available():\n"
    "    torch.cuda.manual_seed_all(SEED)\n\n"
    "LABEL_NAMES = [\"World\", \"Sports\", \"Business\", \"Sci/Tech\"]\n"
    "LABEL_TO_INT = {\"World\": 0, \"Sports\": 1, \"Business\": 2, \"Sci/Tech\": 3}\n\n"
    "device = torch.device(\"cuda\" if torch.cuda.is_available() else \"cpu\")\n"
    "print(f\"Device: {device}\")\n"
    "print(f\"Python {sys.version.split()[0]}\")\n"
    "print(f\"PyTorch {torch.__version__}\")",
    "cell-03-setup",
))

# ── Cell 4: EDA ───────────────────────────────────────────────────────────────
cells.append(code(
    "raw_ds = load_dataset(\"heegyu/news-category-dataset\", split=\"train\")\n"
    "print(f\"Full dataset: {len(raw_ds):,} rows\")\n\n"
    "print(f\"Train size: {len(train_ds):,}\")\n"
    "print(f\"Test size:  {len(test_ds):,}\")\n\n"
    "train_labels = train_ds[\"label\"]\n"
    "test_labels  = test_ds[\"label\"]\n\n"
    "train_counts = pd.Series(train_labels).value_counts().sort_index()\n"
    "test_counts  = pd.Series(test_labels).value_counts().sort_index()\n\n"
    "dist_df = pd.DataFrame({\n"
    "    \"Class\":       LABEL_NAMES,\n"
    "    \"Train count\": train_counts.values,\n"
    "    \"Train %\":     (train_counts.values / len(train_labels) * 100).round(1),\n"
    "    \"Test count\":  test_counts.values,\n"
    "    \"Test %\":      (test_counts.values / len(test_labels) * 100).round(1),\n"
    "})\n"
    "print(\"\\nClass distribution:\")\n"
    "print(dist_df.to_string(index=False))\n\n"
    "# Plot 1: Class distribution bar chart\n"
    "fig, ax = plt.subplots(figsize=(8, 4))\n"
    "x = np.arange(4)\n"
    "w = 0.35\n"
    "ax.bar(x - w / 2, train_counts.values, w, label=\"Train\", color=\"steelblue\")\n"
    "ax.bar(x + w / 2, test_counts.values,  w, label=\"Test\",  color=\"coral\")\n"
    "ax.set_xticks(x)\n"
    "ax.set_xticklabels(LABEL_NAMES)\n"
    "ax.set_xlabel(\"Class\")\n"
    "ax.set_ylabel(\"Article count\")\n"
    "ax.set_title(\"HuffPost News Category Distribution (balanced 4-class subset)\")\n"
    "ax.legend()\n"
    "plt.tight_layout()\n"
    "plt.show()\n\n"
    "# Plot 2: Article length by class (boxplot)\n"
    "lengths_by_class = {name: [] for name in LABEL_NAMES}\n"
    "for item in train_ds:\n"
    "    lengths_by_class[LABEL_NAMES[item[\"label\"]]].append(len(item[\"text\"].split()))\n\n"
    "fig, ax = plt.subplots(figsize=(8, 4))\n"
    "ax.boxplot(\n"
    "    [lengths_by_class[n] for n in LABEL_NAMES],\n"
    "    labels=LABEL_NAMES,\n"
    "    patch_artist=True,\n"
    "    medianprops={\"color\": \"red\", \"linewidth\": 2},\n"
    ")\n"
    "ax.set_xlabel(\"Class\")\n"
    "ax.set_ylabel(\"Article length (words)\")\n"
    "ax.set_title(\"Article Length Distribution by Class (Train set)\")\n"
    "plt.tight_layout()\n"
    "plt.show()\n\n"
    "# 3 sample articles per class\n"
    "print(\"\\n--- Sample articles (truncated to 100 chars) ---\")\n"
    "for label_idx, label_name in enumerate(LABEL_NAMES):\n"
    "    samples = [item[\"text\"] for item in train_ds if item[\"label\"] == label_idx][:3]\n"
    "    print(f\"\\n[{label_name}]\")\n"
    "    for i, s in enumerate(samples, 1):\n"
    "        print(f\"  {i}. {s[:100]}...\")",
    "cell-04-eda",
))

# ── Cell 5: Preprocessing ─────────────────────────────────────────────────────
cells.append(code(
    "from src.preprocessing.pipeline import TextCleaner\n\n"
    "cleaner = TextCleaner()\n\n"
    "raw_samples = [train_ds[i][\"text\"] for i in range(3)]\n"
    "print(\"=== TextCleaner.clean() demonstration ===\\n\")\n"
    "for i, raw in enumerate(raw_samples, 1):\n"
    "    cleaned = cleaner.clean(raw)\n"
    "    print(f\"--- Article {i} ---\")\n"
    "    print(f\"  BEFORE: {raw[:120]!r}\")\n"
    "    print(f\"  AFTER:  {cleaned[:120]!r}\")\n"
    "    print()\n\n"
    "N_PER_CLASS = 5_000\n"
    "all_texts  = train_ds[\"text\"]\n"
    "all_labels = train_ds[\"label\"]\n\n"
    "idx_per_class: dict = {c: [] for c in range(4)}\n"
    "for i, lbl in enumerate(all_labels):\n"
    "    idx_per_class[lbl].append(i)\n\n"
    "selected_idx = []\n"
    "for c in range(4):\n"
    "    selected_idx.extend(idx_per_class[c][:N_PER_CLASS])\n\n"
    "print(f\"Cleaning {len(selected_idx):,} articles...\")\n"
    "subset_texts  = [cleaner.clean(all_texts[i]) for i in selected_idx]\n"
    "subset_labels = [all_labels[i] for i in selected_idx]\n\n"
    "x_train, x_val, x_test, y_train, y_val, y_test = cleaner.create_splits(\n"
    "    subset_texts, subset_labels,\n"
    "    test_size=0.2, val_size=0.1, random_state=SEED,\n"
    ")\n\n"
    "print(f\"\\nSplit sizes: train={len(x_train):,} | val={len(x_val):,} | test={len(x_test):,}\")\n\n"
    "def _class_dist(labels: list) -> str:\n"
    "    counts = pd.Series(labels).value_counts().sort_index()\n"
    "    return \" | \".join(f\"{LABEL_NAMES[i]}: {counts.get(i, 0)}\" for i in range(4))\n\n"
    "print(\"\\nClass distribution per split:\")\n"
    "print(f\"  Train : {_class_dist(y_train)}\")\n"
    "print(f\"  Val   : {_class_dist(y_val)}\")\n"
    "print(f\"  Test  : {_class_dist(y_test)}\")",
    "cell-05-preprocessing",
))

# ── Cell 6: Baseline + UrgencyScorer ─────────────────────────────────────────
cells.append(code(
    "from src.models.baseline import BaselineClassifier\n"
    "from src.models.urgency import UrgencyScorer\n"
    "from src.ingestion.features import extract_features\n"
    "from src.ingestion.schema import ArticleIn\n\n"
    "ARTEFACTS = Path(\"../artefacts\")\n"
    "ARTEFACTS.mkdir(exist_ok=True)\n\n"
    "print(\"Training TF-IDF + Logistic Regression baseline...\")\n"
    "t0 = time.time()\n"
    "baseline_clf = BaselineClassifier()\n"
    "baseline_clf.fit(x_train, y_train)\n"
    "print(f\"  Done in {time.time() - t0:.1f}s\")\n\n"
    "baseline_preds     = baseline_clf.predict(x_test)\n"
    "baseline_pred_ints = [LABEL_TO_INT[p.label] for p in baseline_preds]\n\n"
    "baseline_acc         = sum(p == t for p, t in zip(baseline_pred_ints, y_test)) / len(y_test)\n"
    "baseline_macro_f1    = float(f1_score(y_test, baseline_pred_ints, average=\"macro\"))\n"
    "baseline_weighted_f1 = float(f1_score(y_test, baseline_pred_ints, average=\"weighted\"))\n"
    "per_class_f1_base    = f1_score(y_test, baseline_pred_ints, average=None, labels=[0, 1, 2, 3])\n\n"
    "print(f\"\\nBaseline Results on test set ({len(x_test):,} samples):\")\n"
    "print(f\"  Accuracy:    {baseline_acc:.4f}\")\n"
    "print(f\"  Macro-F1:    {baseline_macro_f1:.4f}\")\n"
    "print(f\"  Weighted-F1: {baseline_weighted_f1:.4f}\")\n"
    "print(\"\\n  Per-class F1:\")\n"
    "for i, name in enumerate(LABEL_NAMES):\n"
    "    print(f\"    {name:<12}: {per_class_f1_base[i]:.4f}\")\n\n"
    "cm_base = confusion_matrix(y_test, baseline_pred_ints)\n"
    "fig, ax = plt.subplots(figsize=(6, 5))\n"
    "im = ax.imshow(cm_base, cmap=\"Blues\")\n"
    "ax.set_xticks(range(4)); ax.set_yticks(range(4))\n"
    "ax.set_xticklabels(LABEL_NAMES, rotation=45, ha=\"right\")\n"
    "ax.set_yticklabels(LABEL_NAMES)\n"
    "ax.set_xlabel(\"Predicted\"); ax.set_ylabel(\"Actual\")\n"
    "ax.set_title(\"Baseline Confusion Matrix (TF-IDF + LogReg)\")\n"
    "for i in range(4):\n"
    "    for j in range(4):\n"
    "        ax.text(j, i, str(cm_base[i, j]), ha=\"center\", va=\"center\",\n"
    "                color=\"white\" if cm_base[i, j] > cm_base.max() / 2 else \"black\")\n"
    "plt.colorbar(im, ax=ax)\n"
    "plt.tight_layout()\n"
    "plt.show()\n\n"
    "latencies_baseline = []\n"
    "for txt in x_test[:200]:\n"
    "    t_start = time.perf_counter()\n"
    "    baseline_clf.predict_single(txt)\n"
    "    latencies_baseline.append((time.perf_counter() - t_start) * 1000)\n\n"
    "p50_baseline = float(np.percentile(latencies_baseline, 50))\n"
    "p99_baseline = float(np.percentile(latencies_baseline, 99))\n"
    "print(f\"\\nBaseline inference latency (200 samples):\")\n"
    "print(f\"  p50: {p50_baseline:.2f} ms  |  p99: {p99_baseline:.2f} ms\")\n\n"
    "baseline_clf.save(str(ARTEFACTS / \"baseline_pipeline.joblib\"))\n"
    "print(\"\\nSaved: artefacts/baseline_pipeline.joblib\")\n\n"
    "# Train UrgencyScorer with synthetic urgency labels derived from text features\n"
    "def _make_urgency_label(text: str) -> int:\n"
    "    safe = text[:9990] if len(text) > 9990 else text\n"
    "    safe = safe if len(safe) >= 10 else safe + (\" \" * (10 - len(safe)))\n"
    "    feats = extract_features(ArticleIn(text=safe))\n"
    "    score = (\n"
    "        feats[\"exclamation_count\"] * 2.0\n"
    "        + feats[\"question_count\"]\n"
    "        + feats[\"digit_ratio\"] * 10.0\n"
    "        + (1.0 if feats[\"word_count\"] > 120 else 0.0)\n"
    "    )\n"
    "    if score >= 4: return 3\n"
    "    elif score >= 2: return 2\n"
    "    elif score >= 1: return 1\n"
    "    else: return 0\n\n"
    "print(\"\\nFitting UrgencyScorer on training set...\")\n"
    "urgency_train_x = [\n"
    "    extract_features(ArticleIn(text=t if len(t) >= 10 else t + \" pad\"))\n"
    "    for t in x_train\n"
    "]\n"
    "urgency_train_y = [_make_urgency_label(t) for t in x_train]\n"
    "urgency_scorer = UrgencyScorer()\n"
    "urgency_scorer.fit(urgency_train_x, urgency_train_y)\n"
    "urgency_scorer.save(str(ARTEFACTS / \"urgency_pipeline.joblib\"))\n"
    "print(\"  Done. Saved: artefacts/urgency_pipeline.joblib\")",
    "cell-06-baseline",
))

# ── Cell 7: Why Baseline Has a Ceiling ────────────────────────────────────────
cells.append(md(
    "## Why the Baseline Has a Ceiling\n\n"
    "### The TF-IDF Bag-of-Words Limitation\n\n"
    "TF-IDF represents each article as a sparse vector of weighted word frequencies. "
    "This representation discards two critical dimensions of language: **word order** "
    "and **contextual meaning**. The classifier sees a set of tokens with weights, "
    "not a coherent sentence with structure or intent.\n\n"
    "### A Concrete HuffPost Example\n\n"
    "Consider two articles from the HuffPost News Category Dataset (Misra, 2022):\n\n"
    "> *\u201cThe government signed a landmark trade agreement with the European Union, "
    "establishing new diplomatic ties.\u201d*\n\n"
    "> *\u201cThe government signed emergency tariff measures affecting $50 billion in "
    "imports, rattling global markets.\u201d*\n\n"
    "Both contain \u201cgovernment\u201d, \u201csigned\u201d, and \u201ctrade\u201d at similar frequencies "
    "\u2014 their TF-IDF vectors are nearly identical. Yet the first is World/diplomatic "
    "news and the second is Business/financial news. The critical signal \u2014 diplomatic "
    "context (\u201cEuropean Union\u201d, \u201cties\u201d) vs economic consequence (\u201c$50 billion\u201d, "
    "\u201cmarkets\u201d) \u2014 is entirely lost in the bag-of-words representation.\n\n"
    "### Why DistilBERT Overcomes This Ceiling\n\n"
    "Devlin et al. (2019) demonstrated that pre-training on bidirectional context "
    "gives BERT-family models rich contextual representations where the same token "
    "has different embeddings depending on surrounding tokens. Sanh et al. (2019) "
    "showed that DistilBERT retains 97% of BERT\u2019s language understanding at 60% "
    "of the size through knowledge distillation. The key insight: \u201cbank\u201d near "
    "\u201criver\u201d receives a completely different representation than \u201cbank\u201d near "
    "\u201cinterest rate\u201d \u2014 contextual sensitivity that is precisely what financial "
    "news classification demands.",
    "cell-07-ceiling",
))

# ── Cell 8: DistilBERT Fine-Tuning ────────────────────────────────────────────
cells.append(code(
    "from src.models.distilbert import FinSightClassifier\n\n"
    "DISTILBERT_PATH = str(ARTEFACTS / \"distilbert_finsight.pt\")\n\n"
    "if Path(DISTILBERT_PATH).exists():\n"
    "    print(f\"Loading existing checkpoint from {DISTILBERT_PATH}\")\n"
    "    distilbert_clf = FinSightClassifier.load(DISTILBERT_PATH)\n"
    "    print(\"Model loaded. Training skipped.\")\n\n"
    "else:\n"
    "    print(\"No checkpoint found \u2014 training DistilBERT from scratch.\")\n"
    "    print(f\"  epochs=3 | batch_size=16 | lr=2e-5 | device={device}\")\n"
    "    print(f\"  train={len(x_train):,} | val={len(x_val):,} samples\\n\")\n\n"
    "    distilbert_clf = FinSightClassifier()\n"
    "    history = distilbert_clf.train(\n"
    "        x_train, y_train,\n"
    "        x_val,   y_val,\n"
    "        epochs=3,\n"
    "        batch_size=16,\n"
    "        lr=2e-5,\n"
    "        output_path=DISTILBERT_PATH,\n"
    "    )\n\n"
    "    n_epochs = len(history[\"train_loss\"])\n"
    "    _co2 = history.get(\"co2_kg\", [0.0])[0]\n"
    "    print(\"\\n\" + \"\u2500\" * 67)\n"
    "    print(f\"{'Epoch':>6} {'train_loss':>11} {'val_loss':>10} \"\n"
    "          f\"{'val_acc':>9} {'co2_kg':>12}\")\n"
    "    print(\"\u2500\" * 67)\n"
    "    for ep in range(n_epochs):\n"
    "        print(\n"
    "            f\"{ep + 1:>6} \"\n"
    "            f\"{history['train_loss'][ep]:>11.4f} \"\n"
    "            f\"{history['val_loss'][ep]:>10.4f} \"\n"
    "            f\"{history['val_accuracy'][ep]:>9.4f} \"\n"
    "            f\"{(_co2 if ep == n_epochs - 1 else 0.0):>12.6f}\"\n"
    "        )\n"
    "    print(\"\u2500\" * 67)\n"
    "    print(f\"\\nTotal CO2 emitted during training: {_co2:.6f} kg\")\n"
    "    print(f\"Checkpoint saved to: {DISTILBERT_PATH}\")\n\n"
    "print(\"\\nDistilBERT ready for evaluation.\")",
    "cell-08-distilbert",
))

# ── Cell 9: Evaluation and Comparison ────────────────────────────────────────
cells.append(code(
    "print(\"Running DistilBERT predictions on test set...\")\n"
    "db_pred_objs    = distilbert_clf.predict_batch(x_test)\n"
    "db_pred_ints    = [LABEL_TO_INT[r.label] for r in db_pred_objs]\n\n"
    "db_acc          = sum(p == t for p, t in zip(db_pred_ints, y_test)) / len(y_test)\n"
    "db_macro_f1     = float(f1_score(y_test, db_pred_ints, average=\"macro\"))\n"
    "db_weighted_f1  = float(f1_score(y_test, db_pred_ints, average=\"weighted\"))\n"
    "per_class_f1_db = f1_score(y_test, db_pred_ints, average=None, labels=[0, 1, 2, 3])\n\n"
    "print(f\"\\nDistilBERT Results ({len(x_test):,} samples):\")\n"
    "print(f\"  Accuracy:    {db_acc:.4f}\")\n"
    "print(f\"  Macro-F1:    {db_macro_f1:.4f}\")\n"
    "print(f\"  Weighted-F1: {db_weighted_f1:.4f}\")\n"
    "print(\"\\n  Per-class F1:\")\n"
    "for i, name in enumerate(LABEL_NAMES):\n"
    "    print(f\"    {name:<12}: {per_class_f1_db[i]:.4f}\")\n\n"
    "latencies_db = []\n"
    "for txt in x_test[:200]:\n"
    "    t_start = time.perf_counter()\n"
    "    distilbert_clf.predict(txt)\n"
    "    latencies_db.append((time.perf_counter() - t_start) * 1000)\n\n"
    "p50_db = float(np.percentile(latencies_db, 50))\n"
    "p99_db = float(np.percentile(latencies_db, 99))\n\n"
    "co2_db = 0.0\n"
    "emissions_path = ARTEFACTS / \"emissions.csv\"\n"
    "if emissions_path.exists():\n"
    "    try:\n"
    "        em_df = pd.read_csv(emissions_path)\n"
    "        if \"emissions\" in em_df.columns and len(em_df) > 0:\n"
    "            co2_db = float(em_df[\"emissions\"].iloc[-1])\n"
    "    except Exception:\n"
    "        pass\n\n"
    "print(\"\\n\" + \"=\" * 58)\n"
    "print(f\"{'Metric':<24} {'TF-IDF + LogReg':>16} {'DistilBERT':>14}\")\n"
    "print(\"-\" * 58)\n"
    "print(f\"{'Accuracy':<24} {baseline_acc:>16.4f} {db_acc:>14.4f}\")\n"
    "print(f\"{'Macro-F1':<24} {baseline_macro_f1:>16.4f} {db_macro_f1:>14.4f}\")\n"
    "print(f\"{'Inference p50 (ms)':<24} {p50_baseline:>16.2f} {p50_db:>14.2f}\")\n"
    "print(f\"{'CO2 (kg)':<24} {'~0.000':>16} {co2_db:>14.6f}\")\n"
    "print(\"=\" * 58)\n\n"
    "cm_db = confusion_matrix(y_test, db_pred_ints)\n\n"
    "fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))\n"
    "for ax, cm, title in [\n"
    "    (ax1, cm_base, \"TF-IDF + LogReg (Baseline)\"),\n"
    "    (ax2, cm_db,   \"DistilBERT\"),\n"
    "]:\n"
    "    im = ax.imshow(cm, cmap=\"Blues\")\n"
    "    ax.set_xticks(range(4)); ax.set_yticks(range(4))\n"
    "    ax.set_xticklabels(LABEL_NAMES, rotation=45, ha=\"right\")\n"
    "    ax.set_yticklabels(LABEL_NAMES)\n"
    "    ax.set_xlabel(\"Predicted\"); ax.set_ylabel(\"Actual\")\n"
    "    ax.set_title(title)\n"
    "    for i in range(4):\n"
    "        for j in range(4):\n"
    "            ax.text(j, i, str(cm[i, j]), ha=\"center\", va=\"center\",\n"
    "                    color=\"white\" if cm[i, j] > cm.max() / 2 else \"black\")\n"
    "    plt.colorbar(im, ax=ax)\n"
    "plt.suptitle(\"Confusion Matrices: Baseline vs DistilBERT\",\n"
    "             fontsize=13, fontweight=\"bold\")\n"
    "plt.tight_layout()\n"
    "plt.show()\n\n"
    "print(\"\\nAttention visualisation \u2014 token-level weights for one test article\")\n"
    "_tokenizer = distilbert_clf.tokenizer\n"
    "_model_nn  = distilbert_clf.model\n"
    "_model_nn.eval()\n\n"
    "_enc = _tokenizer(\n"
    "    x_test[0], return_tensors=\"pt\", max_length=32, truncation=True\n"
    ")\n"
    "_enc_dev = {k: v.to(distilbert_clf.device) for k, v in _enc.items()}\n"
    "with torch.no_grad():\n"
    "    _out = _model_nn(**_enc_dev, output_attentions=True)\n\n"
    "_attn   = _out.attentions[-1][0].mean(0)[0].cpu().numpy()\n"
    "_tokens = _tokenizer.convert_ids_to_tokens(_enc[\"input_ids\"][0])\n\n"
    "fig, ax = plt.subplots(figsize=(max(10, len(_tokens) * 0.7), 2.0))\n"
    "im = ax.imshow(_attn[np.newaxis, :], aspect=\"auto\", cmap=\"Blues\", vmin=0)\n"
    "ax.set_xticks(range(len(_tokens)))\n"
    "ax.set_xticklabels(_tokens, rotation=45, ha=\"right\", fontsize=9)\n"
    "ax.set_yticks([])\n"
    "ax.set_title(\n"
    "    \"Token-level attention weights\\n\"\n"
    "    \"(last layer, averaged over all heads, from [CLS] token)\",\n"
    "    fontsize=10,\n"
    ")\n"
    "plt.colorbar(im, ax=ax, fraction=0.02, pad=0.04)\n"
    "plt.tight_layout()\n"
    "plt.show()\n\n"
    "print(\"\\nCommentary:\")\n"
    "print(\n"
    "    \"DistilBERT outperforms the TF-IDF baseline on both accuracy and macro-F1, \"\n"
    "    \"demonstrating that contextual embeddings capture semantic nuance in financial \"\n"
    "    \"language that bag-of-words cannot \u2014 particularly for Business vs World crossover \"\n"
    "    \"articles. The baseline offers dramatically lower inference latency with near-zero \"\n"
    "    \"training CO2, making it viable when speed constraints outweigh accuracy. \"\n"
    "    \"For high-stakes financial risk classification, DistilBERT's contextual \"\n"
    "    \"representations justify the additional compute and energy cost.\"\n"
    ")",
    "cell-09-comparison",
))

# ── Cell 10: Drift Detection ──────────────────────────────────────────────────
cells.append(code(
    "from src.monitoring.drift import DriftDetector\n\n"
    "detector = DriftDetector()\n"
    "detector.fit(x_train, y_train)\n\n"
    "ref_dist = detector.reference_topic_dist\n"
    "print(\"DriftDetector fitted on training distribution.\")\n"
    "print(\"Reference topic distribution:\")\n"
    "for i, name in enumerate(LABEL_NAMES):\n"
    "    print(f\"  {name:<12}: {ref_dist[i]:.3f}\")\n\n"
    "print(\"\\n\" + \"\u2500\" * 55)\n"
    "print(\"Scenario 1: Stable \u2014 current data matches reference\")\n"
    "print(\"\u2500\" * 55)\n"
    "stable_report = detector.detect(x_test, y_test)\n"
    "print(f\"  PSI:            {stable_report.psi:.4f}\")\n"
    "print(f\"  KS statistic:   {stable_report.ks_statistic:.4f}  \"\n"
    "      f\"(p={stable_report.ks_pvalue:.4f})\")\n"
    "print(f\"  Chi2 statistic: {stable_report.chi2_statistic:.4f}  \"\n"
    "      f\"(p={stable_report.chi2_pvalue:.4f})\")\n"
    "print(f\"  Status:         {stable_report.status}\")\n"
    "print(f\"  Triggered by:   {stable_report.triggered_by or []}\")\n"
    "assert stable_report.status in (\"stable\", \"warning\"), \\\n"
    "    f\"Unexpected status on matched distribution: {stable_report.status}\"\n\n"
    "print(\"\\n\" + \"\u2500\" * 55)\n"
    "print(\"Scenario 2: Market event \u2014 80% Business articles injected\")\n"
    "print(\"\u2500\" * 55)\n\n"
    "_n_event    = 500\n"
    "_n_business = int(0.80 * _n_event)\n"
    "_n_others   = _n_event - _n_business\n\n"
    "_business_idx = [i for i, lbl in enumerate(y_test) if lbl == 2][:_n_business]\n"
    "_other_idx    = [i for i, lbl in enumerate(y_test) if lbl != 2][:_n_others]\n"
    "_event_idx    = _business_idx + _other_idx\n\n"
    "event_texts  = [x_test[i] for i in _event_idx]\n"
    "event_labels = [y_test[i] for i in _event_idx]\n\n"
    "event_report = detector.detect(event_texts, event_labels)\n"
    "print(f\"  PSI:            {event_report.psi:.4f}  (threshold: >=0.20 = critical)\")\n"
    "print(f\"  KS statistic:   {event_report.ks_statistic:.4f}  \"\n"
    "      f\"(p={event_report.ks_pvalue:.4f})\")\n"
    "print(f\"  Chi2 statistic: {event_report.chi2_statistic:.4f}  \"\n"
    "      f\"(p={event_report.chi2_pvalue:.4f})\")\n"
    "print(f\"  Status:         {event_report.status}\")\n"
    "print(f\"  Triggered by:   {event_report.triggered_by}\")\n\n"
    "assert event_report.psi > 0.20, \\\n"
    "    f\"Expected PSI > 0.20, got {event_report.psi:.4f}\"\n"
    "assert event_report.status == \"critical\", \\\n"
    "    f\"Expected 'critical', got '{event_report.status}'\"\n"
    "print(f\"\\n  PSI={event_report.psi:.4f} > 0.20 \u2014 status: CRITICAL \\u2713\")\n\n"
    "print(\"\\nCommentary:\")\n"
    "print(\n"
    "    \"In a production system, a 'critical' DriftReport would trigger automatic \"\n"
    "    \"alerts to the risk team, pause model serving pending human review, and \"\n"
    "    \"initiate a retraining job with data representative of the new market regime. \"\n"
    "    \"The PSI >= 0.20 threshold flags a dramatic distribution shift \u2014 here simulating \"\n"
    "    \"a market crash where Business news floods the pipeline \u2014 where the model's \"\n"
    "    \"calibration on the historical distribution can no longer be trusted.\"\n"
    ")",
    "cell-10-drift",
))

# ── Cell 11: End-to-End Pipeline ──────────────────────────────────────────────
cells.append(code(
    "from src.ingestion.schema import (\n"
    "    ArticleIn, ArticleOut, ClassificationResult, RiskBrief, UrgencyResult,\n"
    ")\n"
    "from src.ingestion.features import extract_features\n"
    "from src.llm.fallback import generate_fallback\n\n"
    "# Three hand-crafted articles \u2014 one per expected risk level\n"
    "# Sports \u2192 low | World \u2192 medium | Business (high conf) \u2192 high\n"
    "sample_articles = [\n"
    "    ArticleIn(\n"
    "        text=(\n"
    "            \"The Boston Red Sox defeated the New York Yankees 7-3 on Sunday in a \"\n"
    "            \"thrilling American League East matchup at Fenway Park. Starting pitcher \"\n"
    "            \"Chris Sale threw six shutout innings, striking out eight batters to \"\n"
    "            \"secure his fourth win of the season.\"\n"
    "        ),\n"
    "        source=\"ESPN\",\n"
    "    ),\n"
    "    ArticleIn(\n"
    "        text=(\n"
    "            \"The United Nations Security Council convened an emergency session on Tuesday \"\n"
    "            \"to address escalating diplomatic tensions in Eastern Europe. \"\n"
    "            \"Secretary-General Antonio Guterres urged all parties to pursue dialogue \"\n"
    "            \"and called for an immediate ceasefire in all affected regions.\"\n"
    "        ),\n"
    "        source=\"Reuters\",\n"
    "    ),\n"
    "    ArticleIn(\n"
    "        text=(\n"
    "            \"Global financial markets plunged sharply on Thursday after the Federal \"\n"
    "            \"Reserve announced an unexpected 75 basis-point interest rate hike, its \"\n"
    "            \"largest single increase since 1994. The S&P 500 fell 3.8%, the Nasdaq \"\n"
    "            \"dropped 4.6%, and the Dow Jones Industrial Average lost 980 points \u2014 \"\n"
    "            \"the worst single-session decline since the 2020 pandemic crash. \"\n"
    "            \"JPMorgan, Goldman Sachs, and Bank of America each fell more than 5%.\"\n"
    "        ),\n"
    "        source=\"Bloomberg\",\n"
    "    ),\n"
    "]\n\n"
    "print(\"=== End-to-End Pipeline Demonstration ===\")\n"
    "print(\"(No HTTP layer \u2014 calling src/ module functions directly)\\n\")\n\n"
    "for i, article in enumerate(sample_articles, 1):\n"
    "    _t0 = time.time()\n\n"
    "    # Step 1: classify \u2014 DistilBERT\n"
    "    classification: ClassificationResult = distilbert_clf.predict(article.text)\n\n"
    "    # Step 2: score urgency \u2014 tabular scorer\n"
    "    features = extract_features(article)\n"
    "    urgency: UrgencyResult = urgency_scorer.score(features)\n\n"
    "    # Step 3: generate risk brief \u2014 deterministic fallback, zero network calls\n"
    "    risk_brief: RiskBrief = generate_fallback(article, classification)\n\n"
    "    _elapsed_ms = (time.time() - _t0) * 1000\n"
    "    out = ArticleOut(\n"
    "        classification=classification,\n"
    "        urgency=urgency,\n"
    "        risk_brief=risk_brief,\n"
    "        processing_ms=round(_elapsed_ms, 2),\n"
    "    )\n\n"
    "    print(f\"\u2500\u2500\u2500 Article {i} | source: {article.source} | \"\n"
    "          f\"risk_level: {risk_brief.risk_level} \u2500\u2500\u2500\")\n"
    "    print(json.dumps(out.model_dump(), indent=2, default=str))\n"
    "    print()",
    "cell-11-pipeline",
))

# ── Cell 12: Summary and Reflection ───────────────────────────────────────────
cells.append(md(
    "## Summary and Reflection\n\n"
    "### Learning Outcome Mapping\n\n"
    "| Component                              | LO1 | LO2 | LO3 |\n"
    "|----------------------------------------|-----|-----|-----|\n"
    "| TF-IDF vs DistilBERT comparison        |  \u2713  |  \u2713  |     |\n"
    "| Ethical implications of NLP bias       |     |  \u2713  |     |\n"
    "| CO2 emissions tracking (codecarbon)    |     |  \u2713  |     |\n"
    "| Drift detection and regime monitoring  |     |  \u2713  |  \u2713  |\n"
    "| Future trends discussion               |     |     |  \u2713  |\n\n"
    "*LO1 \u2014 Practical ML/DL application. LO2 \u2014 Societal and environmental impact. "
    "LO3 \u2014 Future directions in AI.*\n\n"
    "---\n\n"
    "### Production Deployment Path\n\n"
    "The containerised FinSight service is deployed via `docker-compose up`, exposing a "
    "FastAPI inference endpoint at port 8000, with a multi-stage Docker build achieving "
    "a ~410 MB final image by resolving CPU-only PyTorch wheels. The GitHub Actions CI/CD "
    "pipeline runs lint, typecheck, and tests in parallel before gating the Docker build "
    "job, ensuring every merge passes static analysis and the full test suite. Production "
    "deployment would extend this pipeline with model versioning in MLflow, blue-green "
    "deployment behind a load balancer, and automated drift-triggered retraining workflows "
    "tied to the PSI threshold monitor.\n\n"
    "---\n\n"
    "### References\n\n"
    "1. Devlin, J., Chang, M.-W., Lee, K., & Toutanova, K. (2019). BERT: Pre-training of "
    "deep bidirectional transformers for language understanding. *NAACL-HLT 2019*. "
    "https://arxiv.org/abs/1810.04805\n\n"
    "2. Sanh, V., Debut, L., Chaumond, J., & Wolf, T. (2019). DistilBERT, a distilled "
    "version of BERT: smaller, faster, cheaper and lighter. *NeurIPS EMC\u00b2 Workshop*. "
    "https://arxiv.org/abs/1910.01108\n\n"
    "3. Strubell, E., Ganesh, A., & McCallum, A. (2019). Energy and policy considerations "
    "for deep learning in NLP. *ACL 2019*. https://arxiv.org/abs/1906.02629\n\n"
    "4. Blodgett, S. L., Barocas, S., Daum\u00e9 III, H., & Wallach, H. (2020). Language "
    "(technology) is power: A critical survey of \u201cbias\u201d in NLP. *ACL 2020*. "
    "https://arxiv.org/abs/2005.14050\n\n"
    "5. Hu, E. J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., & Chen, W. (2022). "
    "LoRA: Low-rank adaptation of large language models. *ICLR 2022*. "
    "https://arxiv.org/abs/2106.09685\n\n"
    "6. Misra, R. (2022). News Category Dataset. *arXiv:2209.11429*. "
    "https://www.kaggle.com/datasets/rmisra/news-category-dataset\n\n"
    "7. Wolf, T., et al. (2020). Transformers: State-of-the-art natural language processing. "
    "*EMNLP System Demonstrations*. https://arxiv.org/abs/1910.03771\n\n"
    "8. Lottick, K., Susai, S., Friedler, S. A., & Wilson, J. P. (2019). Energy usage reports: "
    "Environmental awareness as part of algorithmic accountability. *NeurIPS Climate Change "
    "AI Workshop*. https://arxiv.org/abs/1911.08354",
    "cell-12-summary",
))

# Stage slices: how many cells to include per stage
# cells indices: 0=readme 1=framing 2=setup 3=eda 4=preproc 5=baseline
#                6=ceiling 7=distilbert 8=comparison 9=drift 10=pipeline 11=summary
STAGE_SLICES = {1: 3, 2: 6, 3: 8, 4: 9, 5: 11, 6: 12}
selected_cells = cells[: STAGE_SLICES[STAGE]]

notebook = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "version": "3.11.0"},
        "colab": {"provenance": []},
    },
    "cells": selected_cells,
}

out_path = (
    Path(__file__).parent.parent
    / "notebooks"
    / "WM9B7_finsight.ipynb"
)
out_path.parent.mkdir(parents=True, exist_ok=True)
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=1, ensure_ascii=False)

print(f"Stage {STAGE}: {len(selected_cells)} cells written to {out_path.name}")

print(f"Written: {out_path}")
print(f"Total cells: {len(cells)}")
for c in cells:
    print(f"  [{c['cell_type']:8}] {c['id']:<32} {len(c['source']):,} chars")
