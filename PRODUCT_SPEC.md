# Product Specification — FinSight

**Financial News Risk Intelligence System**

| | |
|---|---|
| Version | 1.0 |
| Status | Final |
| Owner | Sourabha K Kallapur |

## 1. Executive Summary

FinSight is a production-grade multi-signal pipeline that classifies, scores, and summarizes financial news articles in real time. It combines a dual-model NLP classification architecture (TF-IDF baseline + fine-tuned DistilBERT), a provider-agnostic LLM layer for structured risk-brief generation, a FastAPI inference service, and a statistical drift-monitoring engine for detecting distribution shift in live traffic.

- **Dataset:** HuffPost News Category Dataset (CC BY 4.0) — balanced 4-class subset of 20,000 samples (Politics, Business, Entertainment, Wellness)
- **Core DL model:** DistilBERT-base-uncased, fine-tuned for sequence classification
- **LLM provider:** Gemini 2.0 Flash (default), provider-agnostic — swappable to Groq or Ollama via env var
- **Deployment:** FastAPI + Docker + GitHub Actions CI/CD

## 2. Problem Statement

### 2.1 The real-world problem

Financial institutions — banks, asset managers, hedge funds, compliance teams — consume thousands of news articles daily. Manual classification of article topic, urgency, and risk relevance is slow, inconsistent, and unscalable. Keyword-based systems fail on context: "rate cut boosts market" and "rate cut fails market" are opposite signals that share identical vocabulary.

A system that classifies financial news at sub-second latency, scores urgency from article metadata, and generates a plain-language risk brief lets analysts triage information flow at scale. Bloomberg, Reuters, and most tier-1 investment banks operate internal versions of this kind of system.

### 2.2 Why deep learning is required

Classical NLP (TF-IDF + Logistic Regression) treats each word as an independent feature — it cannot model word order, negation, or context. The sentence "the central bank unexpectedly held rates" contains no individually alarming tokens, but "unexpectedly" in proximity to "held" is the signal. DistilBERT's contextual attention captures this. The project makes this argument empirically by training both models on the same dataset and reporting side-by-side metrics.

### 2.3 Scope exclusions

- Real-time market data ingestion (stocks API) — out of scope for this version
- User authentication and multi-tenancy — out of scope
- Paid API services — all LLM calls use free-tier providers

## 3. System Architecture

| Layer | Name | Key components |
|---|---|---|
| L1 | Data ingestion | Pydantic schema validation, feature extraction, stratified splits |
| L2 | Dual-model NLP | TF-IDF + LogReg baseline, fine-tuned DistilBERT |
| L3 | LLM structured output | Provider-agnostic client, RiskBrief schema, fault tolerance |
| L4 | Inference service | FastAPI, 3 endpoints, 3-layer validation, lifespan, Docker, CI/CD |
| L5 | Drift monitoring | PSI, KS test, Chi-Square, CLI alerting with exit codes |

### 3.1 Data flow

A raw article enters via `POST /analyze` and is validated against `ArticleIn` (Pydantic v2). The text passes through `TextCleaner` (normalization). DistilBERT classifies topic → `ClassificationResult`. The feature extractor derives metadata → `UrgencyResult`. The LLM generator synthesizes a `RiskBrief`. All three outputs merge into `ArticleOut` with processing latency. Independently, `DriftDetector` monitors incoming article distributions against a reference baseline and emits alerts when statistical thresholds are exceeded.

## 4. Functional Requirements

### 4.1 Data layer (L1)
- Accepts article text between 10 and 10,000 characters
- `ArticleIn` schema rejects malformed input with HTTP 422 and a structured error body
- Feature extractor produces 7 metadata features: `word_count`, `avg_word_length`, `digit_ratio`, `uppercase_ratio`, `exclamation_count`, `question_count`, `text_length`
- Train/val/test splits use stratified sampling; vectorizer fitted exclusively on the training partition
- Leakage test asserts vectorizer `vocabulary_` is absent before `fit()` is called

### 4.2 Dual-model NLP (L2)
- Baseline (TF-IDF + LogReg) trained on the HuffPost subset
- DistilBERT fine-tuned with early stopping
- Both models save to joblib / PyTorch artefacts and restore via `load()` classmethods
- `ClassificationResult` returns label (4-class literal), confidence (0–1), and model identifier
- `codecarbon` `EmissionsTracker` wraps the DistilBERT training loop; CO2 emissions logged to `artefacts/emissions.csv`

### 4.3 LLM structured output (L3)
- Three LLM providers supported: Gemini 2.0 Flash (primary), Groq/Llama 3.3 70B (secondary), Ollama (local/offline)
- Active provider resolved from `LLM_PROVIDER` environment variable
- All LLM calls use `temperature=0.0` for determinism
- Output validated against the `RiskBrief` Pydantic schema before returning
- Fault tolerance: Tier 1 (3 retries, exponential backoff), Tier 2 (rate-limit: 5 retries), Tier 3 (deterministic fallback with zero network calls)
- Fallback generates a valid `RiskBrief` from classification label and confidence — no external calls

### 4.4 Inference service (L4)
- `POST /classify` returns `ClassificationResult`; uses DistilBERT if loaded, falls back to baseline
- `POST /score` returns `UrgencyResult` from the tabular metadata scorer
- `POST /analyze` orchestrates classify → score → risk brief and returns `ArticleOut` with `processing_ms`
- `GET /health` always returns HTTP 200
- `GET /ready` returns `models_loaded: true` only when all three models are in `app.state`
- Global exception handler returns `{"error": str, "status_code": 500}` on unhandled exceptions
- Baseline `/classify` p50 latency target < 50ms, p99 < 200ms

### 4.5 Drift monitoring (L5)
- `DriftDetector.fit()` stores reference topic distribution and article length distribution
- `DriftDetector.detect()` computes PSI, KS statistic, and Chi-Square p-value on the current batch
- Status thresholds: PSI < 0.1 stable, 0.1–0.2 warning, ≥ 0.2 critical; KS p < 0.05 warning; Chi-Square p < 0.05 warning
- CLI exits 0 (stable), 1 (warning), 2 (critical) — integrates into CI/CD without code changes

## 5. Non-Functional Requirements

| Requirement | Target | Rationale |
|---|---|---|
| Baseline inference latency (p50) | < 50ms | Sub-second triage at scale |
| DistilBERT inference latency (p50) | < 500ms | Acceptable for batch analysis; not real-time trading |
| Test suite runtime (unit only) | < 60s | CI/CD viability without GPU |
| Docker image size | < 2GB | Model artefact mounted as volume, not baked in |
| LLM provider cost | $0 at default tier | Free-tier providers, swappable if budget allows |
| CO2 per training run | Logged | Environmental cost visibility |
| Code coverage (`src/`) | > 80% | Production-grade quality signal |
| Pydantic validation | All inputs | 3-layer validation: schema, shape, exception handler |

## 6. Dataset Specification

**Source:** HuffPost News Category Dataset (CC BY 4.0), via HuggingFace `datasets` (`heegyu/news-category-dataset`). 209,527 articles across 42 categories, published 2012–2022. FinSight uses a balanced 4-class subset of 20,000 samples.

| Class | Label | Samples |
|---|---|---|
| Politics | 0 | 5,000 |
| Business | 1 | 5,000 |
| Entertainment | 2 | 5,000 |
| Wellness | 3 | 5,000 |

Input text = `headline + " " + short_description`.

## 7. Model Specifications

### 7.1 Baseline — TF-IDF + Logistic Regression

| Parameter | Value |
|---|---|
| Vectorizer | `TfidfVectorizer(max_features=10000, ngram_range=(1,2), sublinear_tf=True)` |
| Classifier | `LogisticRegression(max_iter=1000, C=1.0, multi_class='multinomial', solver='lbfgs')` |
| Pipeline | `sklearn.pipeline.Pipeline` — vectorizer fit inside pipeline only |
| Artefact format | joblib dump |

### 7.2 Deep learning — DistilBERT fine-tuned

| Parameter | Value |
|---|---|
| Base model | `distilbert-base-uncased` |
| Parameters | 66M (40% smaller than BERT-base) |
| Max sequence length | 128 tokens |
| Learning rate | 2e-5 with linear warmup (10% of steps) |
| Optimizer | AdamW |
| Epochs | 3 with early stopping (patience=2) |
| Artefact format | `torch.save` (state_dict + tokenizer) |

### 7.3 Urgency scorer — tabular metadata

| Parameter | Value |
|---|---|
| Input features | 7 metadata features from `features.extract_features()` |
| Pipeline | `StandardScaler` + `LogisticRegression(max_iter=1000)` |
| Output | `UrgencyResult`: `score` (0–1), `level` (low/medium/high/critical) |
| Artefact format | joblib dump |

## 8. LLM Layer Specification

### 8.1 Provider comparison

| Provider | Model | Free limit | Structured output |
|---|---|---|---|
| Gemini | `gemini-2.0-flash` | 1M tokens/day, 15 RPM | Native `response_schema` |
| Groq | `llama-3.3-70b-versatile` | 500k tokens/day | `json_object` mode |
| Ollama | `llama3.2:3b` / `phi4` | Unlimited (local) | `format='json'` |

### 8.2 RiskBrief output schema

All LLM providers are instructed to return JSON conforming to the `RiskBrief` Pydantic model:

- `summary: str` — 2–3 sentence risk assessment
- `risk_level: 'low' | 'medium' | 'high' | 'critical'`
- `key_entities: list[str]` — maximum 5 named entities from the article
- `recommended_action: str` — one-sentence action recommendation
- `generated_by: 'llm' | 'fallback'` — provenance tracking

### 8.3 Fault tolerance sequence

1. Attempt LLM call. On success, validate output and return.
2. On any exception: wait `2^n` seconds, retry up to 3 times (Tier 1).
3. If the exception is a rate-limit error (HTTP 429): wait `2^n * 2` seconds, retry up to 5 times (Tier 2).
4. If all retries are exhausted: call `generate_fallback()`, which produces a deterministic `RiskBrief` from classification label and confidence with zero network calls (Tier 3).

The caller always receives a valid `RiskBrief`; exceptions are never propagated from the generator.

## 9. API Contract

| Endpoint | Method | Request body | Response |
|---|---|---|---|
| `/classify` | POST | `ArticleIn` | `ClassificationResult` |
| `/score` | POST | `ArticleIn` | `UrgencyResult` |
| `/analyze` | POST | `ArticleIn` | `ArticleOut` (full pipeline) |
| `/health` | GET | — | `{"status": "ok"}` |
| `/ready` | GET | — | `{"status": "ready", "models_loaded": bool}` |

All endpoints return HTTP 422 on validation failure (Pydantic error format), HTTP 500 on unhandled exceptions, HTTP 503 if a required model is not loaded.

## 10. Testing Strategy

| Module | Key assertions |
|---|---|
| `test_schema.py` | Validation errors, field coercion, literal constraints |
| `test_features.py` | All 7 features returned as float, edge cases |
| `test_preprocessing.py` | Leakage test, split ratios, text cleaning steps |
| `test_baseline.py` | ROC-AUC threshold, save/load round-trip, leakage guard |
| `test_distilbert.py` | Artefact load, valid output schema |
| `test_urgency.py` | Valid `UrgencyResult`, level thresholds, save/load |
| `test_generator.py` | All 3 providers mocked, retry logic, fallback triggered |
| `test_fallback.py` | Zero network, valid `RiskBrief`, `generated_by='fallback'` |
| `test_routes.py` | 200/422/500, dependency injection override, latency benchmark |
| `test_drift.py` | Stable/warning/critical scenarios, CLI exit codes |

**Test markers:** no marker = unit tests, run in CI, complete in < 60s on CPU; `@pytest.mark.slow` = integration tests requiring model artefacts, local only; `@pytest.mark.benchmark` = latency benchmarks, local only.

## 11. Risks and Mitigations

| Risk | Mitigation |
|---|---|
| DistilBERT training fails or runs too long | GPU-backed training with a `--quick` smoke-test flag before full runs |
| LLM provider unavailable or quota hit | Provider-agnostic architecture — switch to Groq or Ollama via env var |
| Dataset unavailable at runtime | Dataset loads via HuggingFace `datasets`; artefacts cached in `artefacts/` |

## 12. References

- Sanh, V., Debut, L., Chaumond, J., & Wolf, T. (2019). DistilBERT, a distilled version of BERT. arXiv:1910.01108.
- Devlin, J., Chang, M. W., Lee, K., & Toutanova, K. (2019). BERT: Pre-training of deep bidirectional transformers for language understanding. NAACL-HLT 2019.
- Strubell, E., Ganesh, A., & McCallum, A. (2019). Energy and policy considerations for deep learning in NLP. ACL 2019.
- Misra, R. (2022). News Category Dataset. arXiv:2209.11429.
