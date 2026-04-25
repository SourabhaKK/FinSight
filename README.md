# FinSight

![Python 3.11](https://img.shields.io/badge/python-3.11-blue) ![FastAPI](https://img.shields.io/badge/FastAPI-0.111-green) ![DistilBERT](https://img.shields.io/badge/DistilBERT-fine--tuned-orange) ![Docker](https://img.shields.io/badge/docker-multi--stage-blue) ![CI](https://img.shields.io/badge/CI-passing-brightgreen)

Financial news arrives faster than any team can read it. FinSight is a production-grade pipeline that classifies news articles by topic, scores urgency from article metadata, and generates structured risk briefs using a provider-agnostic LLM layer — all behind a FastAPI service with Pydantic-validated I/O. It pairs a TF-IDF baseline against a fine-tuned DistilBERT model for direct empirical comparison, and monitors live inference traffic for distribution shift using PSI, KS, and Chi-Square tests with CLI exit codes suitable for CI/CD alerting.

---

## Features

- Dual-model NLP classification — TF-IDF + LogReg baseline and fine-tuned DistilBERT running side by side with a shared evaluation harness
- Provider-agnostic LLM risk brief generation — swap between Gemini, Groq, and Ollama with a single environment variable, no code changes
- Three-tier fault tolerance with exponential backoff and a deterministic fallback that produces valid output with zero network calls
- FastAPI inference service with `/classify`, `/score`, and `/analyze` endpoints, Pydantic v2 schema validation, and lifespan-based model loading
- Statistical drift detection — PSI, KS test, and Chi-Square — with `stable` / `warning` / `critical` status and CLI exit codes for pipeline integration
- 110 pytest cases across 10 modules, all passing without a GPU
- Multi-stage Docker build and GitHub Actions CI/CD pipeline (lint → typecheck → test → docker build)

---

## Architecture

| Layer | Module | What it does |
|---|---|---|
| 1 | `ingestion` | Pydantic schema validation, tabular metadata extraction |
| 2 | `preprocessing` | Text cleaning, leakage-safe train / val / test splits |
| 3 | `models` | TF-IDF + LogReg baseline, DistilBERT fine-tuned, tabular urgency scorer |
| 4 | `api` | FastAPI — `/classify`, `/score`, `/analyze`, `/health`, `/ready` |
| 5 | `monitoring` | PSI / KS / Chi-Square drift detection, CLI alerts with exit codes |

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
| Accuracy | 0.8985 | 0.9273 |
| Macro-F1 | 0.8980 | 0.9269 |
| Inference p50 | 1.54 ms | 11.64 ms (GPU) |
| Training CO2 (kg) | ~0.000001 | 0.007699 |

Use the TF-IDF baseline for latency-critical applications. Use DistilBERT for accuracy-critical batch workflows where the 150 ms CPU inference budget is acceptable.

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
│   ├── api/            # FastAPI app, routes, middleware
│   └── monitoring/     # drift detection, CLI alerts
├── tests/              # 110 pytest cases
├── notebooks/          # exploratory notebook
├── scripts/            # training scripts
├── Dockerfile
└── docker-compose.yml
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
