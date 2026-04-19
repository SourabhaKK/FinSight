# FinSight — Project Specification for Claude Code

## Project identity

- **Name:** FinSight — Intelligent Financial News Risk Intelligence System
- **Language:** Python 3.11
- **Package manager:** uv (not pip, not poetry)
- **Test runner:** pytest
- **Formatter:** ruff
- **Type checker:** mypy (strict mode)
- **LLM provider:** Provider-agnostic — resolved from LLM_PROVIDER env var

---

## What this project does

A production-grade multi-signal AI pipeline that:
1. Ingests financial news article text via a validated Pydantic schema
2. Classifies topic (4-class: World, Sports, Business, Sci/Tech) using a dual-model NLP architecture — TF-IDF + Logistic Regression baseline AND DistilBERT fine-tuned
3. Scores urgency using a tabular metadata scorer (article length, source credibility, publication timing)
4. Generates a structured RiskBrief via a provider-agnostic LLM client with 3-tier fault tolerance
5. Serves all outputs through a multi-endpoint FastAPI inference service
6. Monitors input distribution for market regime shifts via PSI / KS / Chi-Square statistical tests

---

## Absolute constraints — never violate these

- All sklearn transformers (TfidfVectorizer, StandardScaler etc.) must be fit ONLY on training data. Fitting on val or test data is label leakage and is a critical bug.
- Every public function and method must have a type annotation.
- Every module in src/ must have a corresponding test file in tests/ with a minimum of 10 test cases.
- No Jupyter-style code in src/. src/ is pure Python modules only.
- The assignment notebook imports from src/ — it does not re-implement logic inline.
- Pydantic v2 syntax only. Do not use v1 validators or @validator decorators.
- FastAPI lifespan pattern for model loading — do not use deprecated @app.on_event("startup").
- Docker multi-stage build: builder stage installs dependencies, final stage copies artefacts only.
- Temperature = 0.0 on all LLM calls for determinism.

---

## Repository structure — create exactly this

```
finsight/
├── src/
│   ├── __init__.py
│   ├── config.py                  # pydantic-settings Settings class
│   ├── ingestion/
│   │   ├── __init__.py
│   │   ├── schema.py              # Pydantic models: ArticleIn, ClassificationResult,
│   │   │                          # UrgencyResult, RiskBrief, ArticleOut
│   │   └── features.py            # Tabular metadata extractor
│   ├── preprocessing/
│   │   ├── __init__.py
│   │   └── pipeline.py            # TextCleaner + leakage-safe split
│   ├── models/
│   │   ├── __init__.py
│   │   ├── baseline.py            # TF-IDF + Logistic Regression classifier
│   │   ├── distilbert.py          # DistilBERT fine-tuning + inference
│   │   └── urgency.py             # Tabular metadata → urgency score
│   ├── llm/
│   │   ├── __init__.py
│   │   ├── client.py              # Abstract LLMClient base + factory function
│   │   ├── gemini.py              # Gemini 2.0 Flash implementation
│   │   ├── groq_client.py         # Groq + Llama 3.3 70B implementation
│   │   ├── ollama_client.py       # Ollama local implementation
│   │   ├── generator.py           # Orchestrator with fault tolerance
│   │   └── fallback.py            # Deterministic fallback, zero network
│   ├── api/
│   │   ├── __init__.py
│   │   ├── main.py                # FastAPI app + lifespan context manager
│   │   ├── routes.py              # /classify, /score, /analyze, /health, /ready
│   │   └── middleware.py          # Global exception handler + request logging
│   └── monitoring/
│       ├── __init__.py
│       ├── drift.py               # PSI / KS / Chi-Square drift detection engine
│       └── alerts.py              # Alert levels + CLI entrypoint with exit codes
├── tests/
│   ├── conftest.py                # Shared fixtures and mock factories
│   ├── test_schema.py
│   ├── test_features.py
│   ├── test_preprocessing.py
│   ├── test_baseline.py
│   ├── test_distilbert.py
│   ├── test_urgency.py
│   ├── test_generator.py
│   ├── test_fallback.py
│   ├── test_routes.py
│   └── test_drift.py
├── notebooks/
│   └── WM9B7_finsight.ipynb       # Assignment submission notebook
├── scripts/
│   └── train_distilbert.py        # Standalone training script
├── artefacts/                     # Saved model files — gitignored except .gitkeep
│   └── .gitkeep
├── data/                          # AG News subsets — gitignored except .gitkeep
│   └── .gitkeep
├── Dockerfile
├── docker-compose.yml
├── .github/
│   └── workflows/
│       └── ci.yml
├── pyproject.toml
├── .env
├── .env.example
└── README.md
```

---

## Dataset

AG News corpus. Load using HuggingFace datasets:

```python
from datasets import load_dataset
ds = load_dataset("ag_news")
# Classes: 0=World, 1=Sports, 2=Business, 3=Sci/Tech
# Train: 120,000 samples | Test: 7,600 samples
```

For development use 5,000 train samples. For notebook training use 20,000. For full training via scripts/train_distilbert.py use all 120,000.

---

## Environment variables

All loaded via pydantic-settings Settings class in src/config.py.

```bash
# LLM provider — choose one: gemini | groq | ollama
LLM_PROVIDER=gemini

# Gemini (free — https://aistudio.google.com/app/apikey)
GEMINI_API_KEY=
GEMINI_MODEL=gemini-2.0-flash

# Groq (free — https://console.groq.com)
GROQ_API_KEY=
GROQ_MODEL=llama-3.3-70b-versatile

# Ollama (local — no key needed)
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3.2:3b

# Model artefact paths
DISTILBERT_MODEL_PATH=artefacts/distilbert_finsight.pt
BASELINE_MODEL_PATH=artefacts/baseline_pipeline.joblib
URGENCY_MODEL_PATH=artefacts/urgency_pipeline.joblib

# API settings
LOG_LEVEL=INFO
MAX_TEXT_LENGTH=10000
MIN_TEXT_LENGTH=10
```

---

## Pydantic schemas — exact field names, do not rename

```python
from pydantic import BaseModel, Field
from typing import Literal
from datetime import datetime

class ArticleIn(BaseModel):
    text: str = Field(min_length=10, max_length=10000)
    source: str = "unknown"
    published_at: datetime | None = None

class ClassificationResult(BaseModel):
    label: Literal["World", "Sports", "Business", "Sci/Tech"]
    confidence: float = Field(ge=0.0, le=1.0)
    model: Literal["distilbert", "baseline"]

class UrgencyResult(BaseModel):
    score: float = Field(ge=0.0, le=1.0)
    level: Literal["low", "medium", "high", "critical"]
    features_used: list[str]

class RiskBrief(BaseModel):
    summary: str
    risk_level: Literal["low", "medium", "high", "critical"]
    key_entities: list[str] = Field(max_length=5)
    recommended_action: str
    generated_by: Literal["llm", "fallback"]

class ArticleOut(BaseModel):
    classification: ClassificationResult
    urgency: UrgencyResult
    risk_brief: RiskBrief
    processing_ms: float
```

---

## FastAPI routes — exact signatures

```
POST /classify   → ClassificationResult
POST /score      → UrgencyResult
POST /analyze    → ArticleOut    (full pipeline: classify + score + risk brief)
GET  /health     → {"status": "ok"}
GET  /ready      → {"status": "ready", "models_loaded": bool}
```

All POST routes accept `ArticleIn` as the request body. `/analyze` orchestrates all three sub-systems and returns `ArticleOut` with `processing_ms` populated.

---

## LLM client architecture

### Abstract base (src/llm/client.py)

```python
from abc import ABC, abstractmethod

class LLMClient(ABC):
    @abstractmethod
    async def generate_risk_brief(
        self,
        article_text: str,
        classification_label: str
    ) -> dict: ...

def get_llm_client(provider: str) -> LLMClient:
    match provider:
        case "gemini": return GeminiClient()
        case "groq":   return GroqClient()
        case "ollama": return OllamaClient()
        case _: raise ValueError(f"Unknown provider: {provider}")
```

### Gemini (src/llm/gemini.py)

- SDK: `google-generativeai`
- Use `response_mime_type="application/json"` and `response_schema=RiskBrief`
- Pass the Pydantic class directly to response_schema — SDK infers JSON schema
- Temperature = 0.0

### Groq (src/llm/groq_client.py)

- SDK: `groq` (AsyncGroq client)
- Use `response_format={"type": "json_object"}`
- System prompt must include explicit JSON schema instruction
- Parse with `json.loads` then `RiskBrief(**data)`

### Ollama (src/llm/ollama_client.py)

- HTTP: `httpx.AsyncClient` → `POST http://localhost:11434/api/generate`
- Payload: `{"model": ..., "format": "json", "prompt": ..., "stream": false}`
- Parse `response["response"]` with `json.loads` then validate
- If Ollama is unreachable, raise `ConnectionError` immediately — do not retry, triggers fallback tier

---

## Fault tolerance pattern (src/llm/generator.py)

```
Tier 1: 3 retry attempts on any exception — exponential backoff 2^n seconds (2, 4, 8)
Tier 2: RateLimitError / HTTP 429 → 5 retry attempts — exponential backoff 2^n seconds
Tier 3: Any persistent failure after retries → call fallback.generate_fallback()
         fallback.py must produce a valid RiskBrief with zero network calls
```

---

## Drift detection specification

```python
@dataclass
class DriftReport:
    psi: float                              # Population Stability Index on topic distribution
    ks_statistic: float                     # KS test statistic on article length distribution
    ks_pvalue: float                        # KS test p-value
    chi2_statistic: float                   # Chi-Square statistic on source categories
    chi2_pvalue: float                      # Chi-Square p-value
    status: Literal["stable", "warning", "critical"]
    triggered_by: list[str]                 # Which tests exceeded thresholds
```

Thresholds:
- PSI < 0.1 → stable | 0.1–0.2 → warning | >= 0.2 → critical
- KS p-value < 0.05 → warning
- Chi-Square p-value < 0.05 → warning
- Any critical threshold → overall status = "critical"
- Any warning threshold with no critical → overall status = "warning"

CLI exit codes: 0 = stable, 1 = warning, 2 = critical

---

## Testing requirements per module

| Module | Minimum cases | Must include |
|---|---|---|
| test_schema.py | 15 | Valid input, missing fields, field coercion, min/max length guards |
| test_features.py | 12 | All 7 features extracted, correct types, edge cases (empty text) |
| test_preprocessing.py | 12 | Leakage test: assert vectoriser not fit on val/test; split ratios |
| test_baseline.py | 15 | Train/predict cycle; ROC-AUC > 0.85; save/load round-trip |
| test_distilbert.py | 12 | Mock HuggingFace in unit tests; artefact load; valid label + confidence |
| test_urgency.py | 10 | Fit/score cycle; valid UrgencyResult; level thresholds correct |
| test_generator.py | 15 | All 3 LLM clients mocked; retry logic; fallback triggered correctly |
| test_fallback.py | 10 | Passes with network disabled; valid RiskBrief; generated_by = "fallback" |
| test_routes.py | 20 | 200/422/500; dependency injection override; /ready false when models None |
| test_drift.py | 15 | Stable on identical; critical on shifted; save/load; CLI exit codes |

---

## pyproject.toml — exact dependency versions

```toml
[project]
name = "finsight"
version = "0.1.0"
description = "Financial News Risk Intelligence System"
requires-python = ">=3.11"
dependencies = [
    "fastapi>=0.111",
    "uvicorn[standard]>=0.29",
    "pydantic>=2.7",
    "pydantic-settings>=2.2",
    "google-generativeai>=0.8",
    "groq>=0.9",
    "httpx>=0.27",
    "transformers>=4.40",
    "torch>=2.2",
    "datasets>=2.19",
    "scikit-learn>=1.4",
    "joblib>=1.4",
    "scipy>=1.13",
    "numpy>=1.26",
    "pandas>=2.2",
    "codecarbon>=2.3",
]

[dependency-groups]
dev = [
    "pytest>=8",
    "pytest-asyncio>=0.23",
    "pytest-mock>=3.14",
    "ruff>=0.4",
    "mypy>=1.10",
]

[tool.pytest.ini_options]
asyncio_mode = "auto"
markers = [
    "slow: marks tests as slow (deselect with '-m not slow')",
    "benchmark: marks tests as latency benchmarks",
]

[tool.ruff]
line-length = 88
select = ["E", "F", "I", "N", "W"]

[tool.mypy]
strict = true
ignore_missing_imports = true
```

---

## Critical engineering decisions

1. **Leakage is the most common student bug.** Every time you create a sklearn Pipeline or transformer, verify it is only fitted inside the training split. Add an explicit leakage test that asserts the vectoriser's `vocabulary_` attribute is None before `fit()` is called.

2. **Mock HuggingFace in unit tests.** DistilBERT is 250MB. Loading it in every test run takes 10–30 seconds and makes the test suite unusable. Use `unittest.mock.patch` to replace the model loader in unit tests. Only load the real model in tests tagged `@pytest.mark.slow`.

3. **FastAPI lifespan, not startup events.** Use `@asynccontextmanager` lifespan pattern. Models are stored in `app.state`. If an artefact is missing, log a warning and set that model to `None` — do not crash.

4. **Ollama for development.** Set `LLM_PROVIDER=ollama` during development to avoid any rate limits. Switch to `gemini` for final notebook run.

5. **codecarbon wraps the training loop only.** Do not wrap the entire script — you want emissions from training, not from data loading.
