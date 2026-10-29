# Architecture

Drafted from the code and README. Items marked "inferred, please check" were not confirmed by running the system.

## Request flow (`POST /analyze`)

1. `src/api` validates the request against `ArticleIn` (Pydantic v2); bad input returns HTTP 422.
2. `src/preprocessing` cleans the text.
3. `src/models` classifies the topic. DistilBERT is used when loaded, otherwise the TF-IDF baseline (inferred, please check; the spec says `/classify` falls back this way).
4. `src/ingestion` extracts metadata features and the urgency scorer returns a score and level.
5. `src/llm` generates a structured risk brief. Failures retry with backoff, then fall back to a deterministic brief with no network calls.
6. The three results merge into `ArticleOut` with `processing_ms`.

## RAG flow (`POST /analyze/rag`)

`data/corpus/*.txt` is chunked, embedded (`all-MiniLM-L6-v2`), stored in Qdrant, retrieved by dense search, and passed to the same LLM client for a grounded answer with cited sources (`GroundedBrief`). A reranked mode exists in `src/rag/retriever.py` but is not wired into the API.

## Monitoring

`src/monitoring` compares live feature distributions to a reference using PSI, KS and Chi-Square, and exits with 0, 1 or 2 for stable, warning or critical so it can run in CI or cron.

## Deployment

Multi-stage `Dockerfile`; `docker-compose.yml` runs the API and Qdrant. CI (`.github/workflows/ci.yml`) runs lint, typecheck and test, then a Docker build.

## Diagram

<!-- TODO: add an architecture diagram -->
