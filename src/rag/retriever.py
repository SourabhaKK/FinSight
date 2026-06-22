"""Retrieval over the Qdrant vector store, with a pluggable retrieval mode.

`mode="dense"` is the default and only mode exposed via the API
(POST /analyze/rag always uses dense retrieval). `mode="dense_reranked"`
is an additive evaluation-only mode for the retrieval ablation in
scripts/evaluate_rag.py — it is not wired into the API.
"""

from __future__ import annotations

from functools import lru_cache

import torch

from src.rag.embeddings import embed_query
from src.rag.schema import RetrievedChunk
from src.rag.vectorstore import QdrantVectorStore

DEFAULT_COLLECTION = "finsight_corpus"
_RERANK_CANDIDATE_K = 15
_CROSS_ENCODER_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"


@lru_cache(maxsize=1)
def _get_cross_encoder() -> object:
    from sentence_transformers import CrossEncoder

    return CrossEncoder(_CROSS_ENCODER_NAME)


def _rerank(
    query: str, candidates: list[RetrievedChunk], top_k: int
) -> list[RetrievedChunk]:
    if not candidates:
        return []
    encoder = _get_cross_encoder()
    pairs = [(query, c.text) for c in candidates]
    # Sigmoid the raw logits into (0, 1) so the cross-encoder score shares
    # the same [-1, 1] scale as the dense cosine-similarity score.
    scores = encoder.predict(  # type: ignore[attr-defined]
        pairs, activation_fn=torch.nn.Sigmoid()
    )
    reranked = sorted(
        zip(candidates, scores, strict=True), key=lambda pair: pair[1], reverse=True
    )
    return [
        chunk.model_copy(update={"score": float(score)})
        for chunk, score in reranked[:top_k]
    ]


def retrieve(
    query: str,
    top_k: int = 5,
    mode: str = "dense",
    store: QdrantVectorStore | None = None,
    collection_name: str = DEFAULT_COLLECTION,
) -> list[RetrievedChunk]:
    """Embed the query, search Qdrant, and return ranked chunks.

    mode="dense": vector search only, returns top_k results.
    mode="dense_reranked": vector search for the top _RERANK_CANDIDATE_K,
        then cross-encoder rerank down to top_k.
    """
    if store is None:
        raise ValueError("store must be provided (no implicit global client)")

    query_embedding = embed_query(query)

    if mode == "dense":
        return store.search(collection_name, query_embedding, top_k=top_k)
    if mode == "dense_reranked":
        candidates = store.search(
            collection_name, query_embedding, top_k=_RERANK_CANDIDATE_K
        )
        return _rerank(query, candidates, top_k=top_k)
    raise ValueError(f"Unknown retrieval mode: {mode!r}")
