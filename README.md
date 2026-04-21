# FinSight — Financial News Risk Intelligence System

A multi-signal AI pipeline that classifies financial news by topic, scores urgency from article metadata, and generates structured risk briefs using a fine-tuned DistilBERT model with statistical drift monitoring. Submitted as the individual assessment for WMG9B7 — Artificial Intelligence and Deep Learning.

---

## Assessment details

| | |
|---|---|
| **Module** | WMG9B7 — Artificial Intelligence and Deep Learning |
| **Student** | Sourabha K Kallapur |
| **Institution** | WMG, University of Warwick |
| **Submission** | Individual Assessment (70% weighting) |
| **Deadline** | Monday 27 April 2026 |

---

## Submission files

- `notebooks/WM9B7_finsight.ipynb` — the main submission notebook
- `notebooks/WM9B7_reflection_draft.md` — the critical reflection report

---

## Running the notebook

### Google Colab (recommended)

1. Open [https://colab.research.google.com](https://colab.research.google.com)
2. **File → Open notebook → GitHub** tab
3. Enter `https://github.com/SourabhaKK/finsight`
4. Select `notebooks/WM9B7_finsight.ipynb`
5. **Runtime → Change runtime type → T4 GPU**
6. Run **Cell 1** (Colab setup — clones the repo and installs dependencies)
7. **Runtime → Run all remaining cells**

Expected runtime: ~45 minutes on T4 GPU.

### Local (Python 3.11+ required, GPU recommended)

```bash
git clone https://github.com/SourabhaKK/finsight.git
cd finsight
pip install uv && uv sync
jupyter notebook notebooks/WM9B7_finsight.ipynb
```

**Note:** Cell 1 is the Colab setup cell. When running locally, skip Cell 1 — the repo is already cloned and dependencies are installed via `uv sync`.

---

## Dataset

AG News corpus — loads automatically via HuggingFace `datasets`. No manual download required.

Citation: Zhang, X., Zhao, J., & LeCun, Y. (2015). Character-level convolutional networks for text classification. *NeurIPS*.

---

## Model artefacts

The notebook checks for a saved DistilBERT checkpoint at `artefacts/distilbert_finsight.pt` on startup.

- **If found:** loads the checkpoint and skips training (~2 minutes total)
- **If not found:** trains DistilBERT from scratch (~45 minutes on T4 GPU)

The baseline model (TF-IDF + Logistic Regression) is always trained fresh in the notebook and completes in under 10 seconds.

---

## Project structure

```
finsight/
├── notebooks/              — submission notebook and reflection report
├── src/
│   ├── ingestion/          — Pydantic schema validation and feature extraction
│   ├── preprocessing/      — text cleaning and leakage-safe splits
│   ├── models/             — baseline classifier, DistilBERT, urgency scorer
│   ├── llm/                — provider-agnostic risk brief generator
│   └── monitoring/         — PSI / KS / Chi-Square drift detection
├── tests/                  — 110 pytest cases across all modules
├── artefacts/              — saved model checkpoints (gitignored except logs)
└── scripts/                — standalone training script
```

---

## Test suite

```bash
uv sync --dev
pytest tests/ -m "not slow and not benchmark" -q
```

110 tests across 10 modules. All pass on Python 3.11.

---

## Key results

| Metric | TF-IDF + LogReg | DistilBERT |
|---|---|---|
| Accuracy | 0.8985 | 0.9273 |
| Macro-F1 | 0.8980 | 0.9269 |
| Inference p50 | 1.54 ms | 11.64 ms (GPU) |
| Training CO2 (kg) | ~0.000001 | 0.007699 |

Test set: AG News (4,000 samples). DistilBERT trained on 14,000 samples, 3 epochs, T4 GPU.

---

## Learning outcomes addressed

- **LO1 (ML vs DL):** Notebook Cells 6–9 compare TF-IDF+LogReg and DistilBERT empirically and structurally; reflection Section 2 provides the theoretical analysis of why contextual embeddings outperform bag-of-words on financial text.

- **LO2 (Implications):** Reflection Section 3 covers design decisions (provider-agnostic LLM layer, three-tier fault tolerance, early stopping), algorithmic bias from corpus composition, LLM hallucination risk, and CO2 emissions; codecarbon tracks training emissions in Cell 8.

- **LO3 (Emerging trends):** Reflection Section 4 discusses LoRA/PEFT, retrieval-augmented generation, and agentic retraining in the context of this system's drift detection and deployment architecture.

---

## References

Bai, Y., Jones, A., Ndousse, K., Askell, A., Chen, A., DasSarma, N., & Kaplan, J. (2022). Training a helpful and harmless assistant with reinforcement learning from human feedback. *arXiv preprint arXiv:2204.05862*.

Blodgett, S. L., Barocas, S., Daumé III, H., & Wallach, H. (2020). Language (technology) is power: A critical survey of "bias" in NLP. In *Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics* (pp. 5454–5476).

Devlin, J., Chang, M.-W., Lee, K., & Toutanova, K. (2019). BERT: Pre-training of deep bidirectional transformers for language understanding. In *Proceedings of NAACL-HLT 2019* (pp. 4171–4186).

Hu, E. J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., & Chen, W. (2022). LoRA: Low-rank adaptation of large language models. In *Proceedings of ICLR 2022*.

Lewis, P., Perez, E., Piktus, A., Petroni, F., Karpukhin, V., Goyal, N., & Kiela, D. (2020). Retrieval-augmented generation for knowledge-intensive NLP tasks. In *Advances in Neural Information Processing Systems 33* (pp. 9459–9474).

Mikolov, T., Chen, K., Corrado, G., & Dean, J. (2013). Efficient estimation of word representations in vector space. *arXiv preprint arXiv:1301.3781*.

Sanh, V., Debut, L., Chaumond, J., & Wolf, T. (2019). DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter. *arXiv preprint arXiv:1910.01108*.

Strubell, E., Ganesh, A., & McCallum, A. (2019). Energy and policy considerations for deep learning in NLP. In *Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics* (pp. 3645–3650).

Zhang, X., Zhao, J., & LeCun, Y. (2015). Character-level convolutional networks for text classification. In *Advances in Neural Information Processing Systems 28* (pp. 649–657).
