"""Grounded answer generation over retrieved chunks.

Reuses the existing provider-agnostic LLM client (src/llm/client.py) via its
generic `complete()` method — no new LLM client is created here.
"""

from __future__ import annotations

import json
import logging

from src.config import settings
from src.llm.client import get_llm_client
from src.rag.schema import GroundedBrief, RetrievedChunk, SourceCitation

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = (
    "You are a financial analyst assistant. Answer the user's question "
    "using ONLY the provided source excerpts — do not use outside "
    "knowledge. For every claim in your answer, cite which source it "
    "came from by filename. If the sources don't contain enough "
    "information to answer, say so explicitly.\n\n"
    "Return ONLY a JSON object with this exact shape, no markdown, no "
    'extra text:\n{"answer": "<your answer, with inline [filename] '
    'citations>", "confidence": <float 0.0-1.0>}'
)


def _build_user_prompt(query: str, retrieved_chunks: list[RetrievedChunk]) -> str:
    context_blocks = []
    for chunk in retrieved_chunks:
        context_blocks.append(
            f"[{chunk.filename}] ({chunk.document_type}, {chunk.date}):\n{chunk.text}"
        )
    context = "\n\n---\n\n".join(context_blocks)
    return f"Sources:\n\n{context}\n\nQuestion: {query}"


def _parse_llm_response(raw: str) -> tuple[str, float]:
    try:
        data = json.loads(raw)
        return str(data["answer"]), float(data["confidence"])
    except (json.JSONDecodeError, KeyError, TypeError, ValueError):
        logger.warning(
            "Could not parse grounded-generation response as JSON; using raw text"
        )
        return raw.strip(), 0.3


async def generate_grounded_brief(
    query: str, retrieved_chunks: list[RetrievedChunk]
) -> GroundedBrief:
    sources = [
        SourceCitation(
            filename=chunk.filename,
            excerpt=chunk.text[:300],
            relevance_score=chunk.score,
        )
        for chunk in retrieved_chunks
    ]

    if not retrieved_chunks:
        return GroundedBrief(
            answer="No relevant sources were found in the corpus for this query.",
            sources=[],
            confidence=0.0,
        )

    client = get_llm_client(settings.llm_provider)
    user_prompt = _build_user_prompt(query, retrieved_chunks)
    raw_response = await client.complete(_SYSTEM_PROMPT, user_prompt)
    answer, confidence = _parse_llm_response(raw_response)

    return GroundedBrief(answer=answer, sources=sources, confidence=confidence)
