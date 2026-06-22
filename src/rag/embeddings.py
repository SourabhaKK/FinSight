"""Sentence-transformer embeddings for the RAG pipeline.

Uses all-MiniLM-L6-v2 — small (~80MB), fast, runs on CPU, no API key.
Keeps the embedding step free-tier-friendly, consistent with the existing
LLM provider pattern (src/llm/) where Ollama gives a fully local option.
"""

from __future__ import annotations

from functools import lru_cache

_MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384


@lru_cache(maxsize=1)
def _get_model() -> object:
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(_MODEL_NAME)


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed a batch of texts, returning one normalised vector per text."""
    if not texts:
        return []
    model = _get_model()
    embeddings = model.encode(  # type: ignore[attr-defined]
        texts, convert_to_numpy=True, normalize_embeddings=True
    )
    return embeddings.tolist()  # type: ignore[no-any-return]


def embed_query(query: str) -> list[float]:
    """Embed a single query string."""
    return embed_texts([query])[0]
