# FinSight

Financial news risk-intelligence service: classifies articles by topic (TF-IDF baseline and fine-tuned DistilBERT), scores urgency from metadata, generates an LLM risk brief, answers document-grounded questions over an SEC/FOMC corpus (RAG), and monitors input drift. Served with FastAPI.

## Structure

```
src/ingestion/      Pydantic schemas, feature extraction
src/preprocessing/  text cleaning, leakage-safe splits
src/models/         baseline, distilbert, urgency scorer
src/llm/            provider-agnostic client (Gemini, Groq, Ollama, OpenAI), fallback
src/rag/            chunking, embeddings, Qdrant store, retriever, generator
src/api/            FastAPI app, routes, middleware
src/monitoring/     PSI / KS / Chi-Square drift, CLI alerts
tests/              pytest suite (one file per module)
scripts/            training, corpus fetch/ingest, RAGAS evaluation, check.py
data/corpus/        SEC EDGAR + FOMC documents
artefacts/          model files (distilbert_finsight.pt is stored in Git LFS)
```

## Run and test

- Install: `uv sync --dev` (Python 3.11 or 3.12; the project is capped below 3.13).
- Run the API: `uvicorn src.api.main:app --reload`, or `docker-compose up` (also starts Qdrant).
- All CI checks locally: `python scripts/check.py` (lint, typecheck, test).
- Individually: `uv run ruff check src/ tests/`, `uv run mypy src/ --ignore-missing-imports`, `uv run pytest tests/ -m "not slow and not benchmark and not integration"`.
- Config comes from environment variables; copy `.env.example` to `.env`. Never commit `.env`.

## Conventions (inferred, please check)

- Ruff line length 88, rules E, F, I, N, W. Mypy strict.
- Pydantic v2 validation at every layer boundary.
- Tests are written alongside features; keep the suite green without a GPU. Mark slow, benchmark and integration tests with the existing pytest markers.
- Commit messages use conventional prefixes (`feat:`, `fix:`, `chore:`).
- Log significant design decisions in `docs/DECISIONS.md`.

## Standing rules for every session

1. Read the issue first: `gh issue view <N>`.
2. Ask me the "Predict first" question from the issue and wait for my answer before implementing.
3. Explain what, why and how as you work, and ask me at decision points.
4. Work on a branch, never on `main`.
5. Open a PR whose description includes `Closes #N`.
6. Run `python scripts/check.py` before opening the PR.
7. Do not merge unless the change is size S, the checks are green, and I type "approve merge".
8. Never touch CI workflows, dependency files (`pyproject.toml`, `uv.lock`) or secrets handling without asking.
9. Never print, copy or commit a secret value.
10. End commit messages with the AI-assistance trailer from my configuration.
