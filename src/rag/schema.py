from __future__ import annotations

from pydantic import BaseModel, Field


class Chunk(BaseModel):
    """A single chunk of a source document, with provenance metadata."""

    text: str
    filename: str
    document_type: str
    date: str
    chunk_index: int


_COSINE_TOLERANCE = 1e-3  # float32 cosine similarity can slightly overshoot [-1, 1]


class RetrievedChunk(Chunk):
    """A Chunk returned from vector search, with its similarity score."""

    score: float = Field(ge=-1.0 - _COSINE_TOLERANCE, le=1.0 + _COSINE_TOLERANCE)


class RAGQuery(BaseModel):
    query: str = Field(min_length=10, max_length=2000)


class SourceCitation(BaseModel):
    filename: str
    excerpt: str
    relevance_score: float = Field(
        ge=-1.0 - _COSINE_TOLERANCE, le=1.0 + _COSINE_TOLERANCE
    )


class GroundedBrief(BaseModel):
    answer: str
    sources: list[SourceCitation]
    confidence: float = Field(ge=0.0, le=1.0)
