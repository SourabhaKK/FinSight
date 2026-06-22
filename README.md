# FinSight

![Python 3.11](https://img.shields.io/badge/python-3.11-blue) ![FastAPI](https://img.shields.io/badge/FastAPI-0.111-green) ![DistilBERT](https://img.shields.io/badge/DistilBERT-fine--tuned-orange) ![Docker](https://img.shields.io/badge/docker-multi--stage-blue) ![CI](https://img.shields.io/badge/CI-passing-brightgreen)

Financial news arrives faster than any team can read it. FinSight is a production-grade pipeline that classifies news articles by topic, scores urgency from article metadata, and generates structured risk briefs using a provider-agnostic LLM layer — all behind a FastAPI service with Pydantic-validated I/O. It pairs a TF-IDF baseline against a fine-tuned DistilBERT model for direct empirical comparison, and monitors live inference traffic for distribution shift using PSI, KS, and Chi-Square tests with CLI exit codes suitable for CI/CD alerting.

---

## Why this exists

Analysts at banks, asset managers, and compliance desks face thousands of news articles a day, and manual triage doesn't scale. Keyword-based filtering fails on context — "rate cut boosts market" and "rate cut fails market" share vocabulary but mean opposite things — so the system needs a model that can read for meaning, not just match tokens. FinSight exists to turn that unfiltered article stream into structured, ranked signal: topic, urgency, and a plain-language risk brief, generated automatically and fast enough to sit in a live pipeline. It's built as a system would be in production — typed contracts, fault-tolerant LLM calls, drift monitoring on live traffic — not as a one-off notebook experiment.

---

## Production decisions

- **Provider-agnostic LLM layer instead of hardcoding a single vendor.** Gemini, Groq, and Ollama are interchangeable behind one client interface, selected by an env var. LLM providers change pricing, rate limits, and availability on short notice; a hard dependency on one vendor is an outage and a renegotiation waiting to happen. The abstraction cost is small and paid once.
- **TF-IDF baseline kept in production alongside DistilBERT, not retired after DistilBERT shipped.** The baseline is ~10x faster at inference and has no GPU dependency, so it's the right choice under latency or cost pressure, while DistilBERT is the right choice when accuracy matters more than latency. Keeping both, with a shared evaluation harness, makes that tradeoff a routing decision instead of an irreversible bet on one model.
- **PSI thresholds set at 0.1 / 0.2, not arbitrary round numbers.** These are the de facto industry thresholds used in credit risk and fraud monitoring for population stability index: below 0.1 is treated as no meaningful shift, 0.1–0.2 warrants investigation, above 0.2 indicates a distribution change that likely requires retraining. Reusing an established convention means the thresholds are defensible without needing a bespoke calibration study before the system has production traffic to calibrate against.
- **Three-tier fault tolerance instead of a single retry loop.** LLM calls fail in distinct ways that need distinct responses: transient errors warrant a short exponential backoff, rate-limit errors warrant a longer backoff because retrying immediately makes the problem worse, and exhausted retries warrant a deterministic fallback rather than an exception bubbling up to the caller. Collapsing these into one retry policy would either under-react to rate limits or leave the caller with no response at all.
- **Pydantic validation at every layer boundary (request schema, internal feature shape, exception handler), not just at the API edge.** Each layer fails differently — malformed client input, a feature extractor returning an unexpected shape, an unhandled exception inside model inference — and a single validation point at the edge wouldn't catch the other two. The cost is a few extra schema definitions; the payoff is that failures are typed and structured instead of raw stack traces leaking through the API.
- **Drift detection emits CLI exit codes instead of just logging.** Exit codes (0/1/2 for stable/warning/critical) let the drift check be wired directly into CI/CD or a cron job without writing a log-parsing layer — the monitoring step fails the pipeline the same way a failed test would, which is the integration point that actually gets acted on.

---

## Features

- Dual-model NLP classification — TF-IDF + LogReg baseline and fine-tuned DistilBERT running side by side with a shared evaluation harness
- Provider-agnostic LLM risk brief generation — swap between Gemini, Groq, and Ollama with a single environment variable, no code changes
- Three-tier fault tolerance with exponential backoff and a deterministic fallback that produces valid output with zero network calls
- FastAPI inference service with `/classify`, `/score`, and `/analyze` endpoints, Pydantic v2 schema validation, and lifespan-based model loading
- Statistical drift detection — PSI, KS test, and Chi-Square — with `stable` / `warning` / `critical` status and CLI exit codes for pipeline integration
- Retrieval-augmented generation over a real SEC EDGAR / FOMC document corpus — `/analyze/rag` answers questions with citations grounded in retrieved source excerpts, scored with RAGAS
- 110+ pytest cases across multiple modules, all passing without a GPU
- Multi-stage Docker build and GitHub Actions CI/CD pipeline (lint → typecheck → test → docker build)

---

## Architecture

| Layer | Module | What it does |
|---|---|---|
| 1 | `ingestion` | Pydantic schema validation, tabular metadata extraction |
| 2 | `preprocessing` | Text cleaning, leakage-safe train / val / test splits |
| 3 | `models` | TF-IDF + LogReg baseline, DistilBERT fine-tuned, tabular urgency scorer |
| 4 | `api` | FastAPI — `/classify`, `/score`, `/analyze`, `/analyze/rag`, `/health`, `/ready` |
| 5 | `monitoring` | PSI / KS / Chi-Square drift detection, CLI alerts with exit codes |
| 6 | `rag` | Chunking, embeddings, Qdrant vector store, retrieval, grounded generation |

---

## Quick start

```bash
# Clone and install
git clone https://github.com/SourabhaKK/finsight.git
cd finsight
pip install uv && uv sync
```

```bash
# Copy env template and add your API key
cp .env.example .env
# Set LLM_PROVIDER=gemini and add GEMINI_API_KEY
# Get a free key at https://aistudio.google.com/app/apikey
```

```bash
# Run the API
uvicorn src.api.main:app --reload

# Or with Docker
docker-compose up
```

---

## API endpoints

| Method | Endpoint | Description | Response |
|---|---|---|---|
| POST | `/classify` | Topic classification (4-class) | `ClassificationResult` |
| POST | `/score` | Urgency scoring from article metadata | `UrgencyResult` |
| POST | `/analyze` | Full pipeline — classify + score + LLM risk brief | `ArticleOut` |
| POST | `/analyze/rag` | Document-grounded Q&A over the SEC/FOMC corpus | `GroundedBrief` |
| GET | `/health` | Health check | `{"status": "ok"}` |
| GET | `/ready` | Models loaded check | `{"models_loaded": bool}` |

**Example:**

```bash
curl -X POST http://localhost:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{"text": "Federal Reserve raises rates by 75 basis points in largest single hike since 1994, S&P 500 falls 3.8%", "source": "Bloomberg"}'
```

```json
{
  "classification": {"label": "Business", "confidence": 0.97, "model": "distilbert"},
  "urgency": {"score": 0.84, "level": "high", "features_used": ["word_count", "digit_ratio", ...]},
  "risk_brief": {
    "summary": "Unexpected 75bps rate hike signals aggressive tightening cycle; equity markets pricing recession risk.",
    "risk_level": "high",
    "key_entities": ["Federal Reserve", "S&P 500"],
    "recommended_action": "Review fixed-income and equity exposure; flag for senior analyst review.",
    "generated_by": "llm"
  },
  "processing_ms": 312.4
}
```

---

## Model performance

Results on HuffPost News Category Dataset test set (Misra, 2022) — 4,000 samples across Politics, Business, Entertainment, and Wellness. DistilBERT fine-tuned on 16,000 samples, 3 epochs on T4 GPU.

| Metric | TF-IDF + LogReg | DistilBERT (fine-tuned) |
|---|---|---|
| Accuracy | 0.8985 | 0.9235 |
| Macro-F1 | 0.8980 | 0.9237 |
| Inference p50 | 1.54 ms | 11.64 ms (GPU) |
| Training CO2 (kg) | ~0.000001 | 0.002187 |

Use the TF-IDF baseline for latency-critical applications. Use DistilBERT for accuracy-critical batch workflows where the 150 ms CPU inference budget is acceptable.

---

## Experiment tracking

Experiments are tracked with MLflow. To view the experiment UI after a training run:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Then open http://localhost:5000 to compare runs across:

- Hyperparameters (epochs, batch size, learning rate)
- Per-epoch training and validation metrics
- Final test accuracy, macro-F1, and per-class F1
- CO2 emissions per run
- Model artefacts

The `finsight` experiment contains two run types:

- `distilbert-finetune` — full fine-tuning run (~4 min on GPU)
- `tfidf-logreg-baseline` — baseline run (<5 seconds on CPU)

---

## RAG pipeline

Retrieval-augmented generation over a small, real corpus of SEC EDGAR filing excerpts and Federal Reserve FOMC statements, so `/analyze/rag` can answer questions with citations grounded in actual source text rather than the LLM's own training data.

**Architecture:**

```
data/corpus/*.txt  →  chunking (recursive splitter, 500/50 tokens)
                   →  embeddings (all-MiniLM-L6-v2, 384-dim, CPU)
                   →  Qdrant (cosine similarity)
                   →  retrieval (dense top-k)
                   →  grounded generation (existing provider-agnostic LLM client)
                   →  GroundedBrief (answer + cited sources + confidence)
```

`src/rag/` holds the pipeline: `chunking.py`, `embeddings.py`, `vectorstore.py` (Qdrant wrapper), `retriever.py`, `generator.py`, `schema.py`.

**Endpoint:**

```bash
curl -X POST http://localhost:8000/analyze/rag \
  -H "Content-Type: application/json" \
  -d '{"query": "What did the FOMC decide about the federal funds rate in its June 2026 statement?"}'
```

```json
{
  "answer": "The Committee decided to maintain the target range for the federal funds rate at 3-1/2 to 3-3/4 percent [fomc_statement_20260617.txt].",
  "sources": [
    {"filename": "fomc_statement_20260617.txt", "excerpt": "The Committee decided to maintain the target range...", "relevance_score": 0.87}
  ],
  "confidence": 0.9
}
```

**How to run:**

```bash
# 1. Start Qdrant
docker-compose up -d qdrant

# 2. Fetch the corpus (live from SEC EDGAR + Federal Reserve, no API key)
python scripts/fetch_corpus.py

# 3. Chunk, embed, and load it into Qdrant
python scripts/ingest_corpus.py

# 4. Run the API
uvicorn src.api.main:app --reload
```

**RAGAS evaluation:**

```bash
python scripts/evaluate_rag.py --mode dense
```

Runs a 10-question hand-written eval set (each question verifiable against the actual fetched corpus) through retrieval + generation, then scores the results with [RAGAS](https://github.com/explodinggradients/ragas):

- **faithfulness** — does the answer only state things supported by the retrieved context?
- **answer_relevancy** — does the answer actually address the question asked?
- **context_precision** — of the retrieved chunks, how many were relevant?
- **context_recall** — did retrieval surface the chunks actually needed to answer?

Results are saved to `artefacts/ragas_eval_results.json`.

**Document corpus:** 22 real documents fetched live (no fallback excerpts needed) — 12 SEC EDGAR 10-K/10-Q risk-factor excerpts for Apple, Microsoft, JPMorgan Chase, and Tesla (via the stable `data.sec.gov/submissions` API), and 10 FOMC statement excerpts from the Federal Reserve's public press release archive. Chosen for recognisable, verifiable, regularly-updated financial content with zero auth required. See `data/corpus/metadata.json` for per-document source URLs and dates.

### Retrieval ablation

A second retrieval mode, `dense_reranked`, is implemented in `src/rag/retriever.py`: dense search retrieves the top 15 candidates, then a cross-encoder (`cross-encoder/ms-marco-MiniLM-L-6-v2`) reranks them down to the top 5. It is not wired into the API — `/analyze/rag` always uses `mode="dense"` — but `scripts/evaluate_rag.py --mode dense_reranked` reuses the same eval harness to compare it against dense-only retrieval.

**Status: code complete, not yet run.** The reranking logic is unit-tested and verified in isolation (`tests/test_rag.py::test_dense_reranked_mode_reorders_by_cross_encoder_score`), but producing the actual comparison table requires running `scripts/evaluate_rag.py` twice against a live Qdrant instance with real LLM calls — which wasn't run as part of this change, since live execution of the evaluation scripts was explicitly out of scope for this session.

```bash
python scripts/evaluate_rag.py --mode dense
python scripts/evaluate_rag.py --mode dense_reranked
```

Results are generated at runtime — run the two commands above after ingesting the corpus to reproduce. The comparison table is saved to artefacts/retrieval_ablation_results.json. The dense_reranked configuration retrieves top-15 candidates via dense search then reranks using a cross-encoder (cross-encoder/ms-marco-MiniLM-L-6-v2), selecting the final top-5 by rerank score. Expected behaviour: reranking improves context_precision at the cost of ~50ms additional latency per query.

---

## LLM providers

| Provider | Model | Free tier | Structured output |
|---|---|---|---|
| Gemini | `gemini-2.0-flash` | 1M tokens/day | Native `response_schema` |
| Groq | `llama-3.3-70b-versatile` | 500k tokens/day | `json_object` mode |
| Ollama | `llama3.2:3b` / `phi4` | Unlimited (local) | `format=json` |

Switch providers with one env var — no code changes:

```bash
LLM_PROVIDER=groq   # or gemini, ollama
```

---

## Development

```bash
# Install with dev dependencies
uv sync --dev

# Run tests (excludes slow GPU tests)
pytest tests/ -m "not slow and not benchmark" -q

# Lint and type check
ruff check src/ tests/
mypy src/ --ignore-missing-imports

# Train DistilBERT (GPU required — use Colab or local GPU)
python scripts/train_distilbert.py --quick   # smoke test
python scripts/train_distilbert.py           # full training
```

---

## Project structure

```
finsight/
├── src/
│   ├── ingestion/      # Pydantic schemas, feature extraction
│   ├── preprocessing/  # TextCleaner, leakage-safe splits
│   ├── models/         # baseline, distilbert, urgency scorer
│   ├── llm/            # client abstraction, providers, fallback
│   ├── rag/            # chunking, embeddings, Qdrant store, retriever, generator
│   ├── api/            # FastAPI app, routes, middleware
│   └── monitoring/     # drift detection, CLI alerts
├── tests/              # 110+ pytest cases
├── notebooks/          # exploratory notebook
├── scripts/            # training, corpus fetch/ingest, RAGAS evaluation
├── data/corpus/        # SEC EDGAR + FOMC document corpus
├── Dockerfile
└── docker-compose.yml  # finsight + qdrant services
```

---

## Environment variables

| Variable | Description | Default |
|---|---|---|
| `LLM_PROVIDER` | Active LLM backend | `gemini` |
| `GEMINI_API_KEY` | Gemini API key | — |
| `GROQ_API_KEY` | Groq API key | — |
| `OLLAMA_BASE_URL` | Ollama server URL | `http://localhost:11434` |
| `OLLAMA_MODEL` | Ollama model name | `llama3.2:3b` |
| `DISTILBERT_MODEL_PATH` | Path to `.pt` artefact | `artefacts/distilbert_finsight.pt` |
| `BASELINE_MODEL_PATH` | Path to joblib artefact | `artefacts/baseline_pipeline.joblib` |
| `URGENCY_MODEL_PATH` | Path to joblib artefact | `artefacts/urgency_pipeline.joblib` |
| `QDRANT_URL` | Qdrant vector store URL | `http://localhost:6333` |
| `RAG_COLLECTION_NAME` | Qdrant collection name for the RAG corpus | `finsight_corpus` |

---

## Dataset

HuffPost News Category Dataset (CC BY 4.0) — 209,527 articles across 42 categories published 2012–2022. FinSight uses a balanced 4-class subset of 20,000 samples:

| Class | Label | Samples |
|---|---|---|
| POLITICS | 0 | 5,000 |
| BUSINESS | 1 | 5,000 |
| ENTERTAINMENT | 2 | 5,000 |
| WELLNESS | 3 | 5,000 |

Input text = `headline + " " + short_description`. Loads automatically via HuggingFace `datasets`:

```python
from datasets import load_dataset
ds = load_dataset("heegyu/news-category-dataset", split="train")
```

Citation: Misra, R. (2022). News Category Dataset. arXiv:2209.11429. https://www.kaggle.com/datasets/rmisra/news-category-dataset

---

## License

MIT

---

## Acknowledgements

- DistilBERT — Sanh et al. (2019), [arXiv:1910.01108](https://arxiv.org/abs/1910.01108)
- HuffPost News Category Dataset — Misra, R. (2022), arXiv:2209.11429
- CO2 tracking — [codecarbon](https://github.com/mlco2/codecarbon)
