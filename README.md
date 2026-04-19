# FinSight — Financial News Risk Intelligence System

A production-grade multi-signal AI pipeline that classifies financial news, scores urgency, and generates structured risk briefs via a provider-agnostic LLM with statistical drift monitoring.

## Architecture

```
ArticleIn (text)
     │
     ├──► TF-IDF + Logistic Regression (baseline)   ─┐
     │                                                ├──► ClassificationResult
     └──► DistilBERT fine-tuned (primary)            ─┘
                                                      │
     ├──► Tabular Metadata Scorer ────────────────────► UrgencyResult
     │
     └──► LLM Generator (Gemini / Groq / Ollama) ───► RiskBrief
                    │
                    └── Fallback (zero-network) ──────► RiskBrief

Drift Monitor: PSI + KS + Chi-Square on rolling inference windows
```

## Quick start

```bash
git clone https://github.com/SourabhaKK/FinSight.git && cd FinSight
cp .env.example .env        # fill in LLM_PROVIDER and the relevant API key
docker-compose up
```

## API endpoints

| Method | Path       | Description                              | Response schema           |
|--------|------------|------------------------------------------|---------------------------|
| POST   | /classify  | Topic classification (4-class)           | ClassificationResult      |
| POST   | /score     | Urgency scoring from article metadata    | UrgencyResult             |
| POST   | /analyze   | Full pipeline: classify + score + brief  | ArticleOut                |
| GET    | /health    | Liveness probe                           | `{"status": "ok"}`        |
| GET    | /ready     | Readiness probe (models loaded)          | `{"models_loaded": bool}` |

All POST routes accept `{"text": "...", "source": "...", "published_at": "..."}`.

## Model performance

| Model      | Dataset       | Accuracy | Macro F1 | ROC-AUC |
|------------|---------------|----------|----------|---------|
| Baseline   | AG News test  | ~92%     | ~0.92    | >0.98   |
| DistilBERT | AG News test  | ~95%     | ~0.95    | >0.99   |

Classes: World · Sports · Business · Sci/Tech

## Development setup

```bash
pip install uv
uv sync --dev

# Run tests (excludes slow GPU tests)
uv run pytest tests/ -m "not slow and not benchmark" -q

# Lint
uv run ruff check src/ tests/

# Type check
uv run mypy src/ --ignore-missing-imports
```

## Project structure

```
finsight/
├── src/
│   ├── config.py               # pydantic-settings Settings
│   ├── ingestion/              # Pydantic schemas + feature extraction
│   ├── preprocessing/          # TextCleaner + leakage-safe split
│   ├── models/                 # baseline, distilbert, urgency
│   ├── llm/                    # LLM clients + fault-tolerant generator
│   ├── api/                    # FastAPI app, routes, middleware
│   └── monitoring/             # PSI/KS/Chi-Square drift detection + CLI
├── tests/                      # 160+ pytest cases, no GPU required
├── artefacts/                  # Saved model files (gitignored, volume-mounted)
├── scripts/train_distilbert.py # Full training script
├── notebooks/                  # Assignment submission notebook
├── Dockerfile                  # Multi-stage build
├── docker-compose.yml
└── .github/workflows/ci.yml    # Lint + typecheck + test + docker-build
```
