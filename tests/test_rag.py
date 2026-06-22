"""Tests for the RAG pipeline — chunking, embeddings, vector store,
retrieval, generation, and the /analyze/rag endpoint.

All tests run without Docker/Qdrant except the ones marked
@pytest.mark.integration, which hit a real local Qdrant instance and are
deselected by default (mirrors the slow/benchmark marker convention
already used in this repo).
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from qdrant_client import QdrantClient

from src.api.main import app
from src.rag.chunking import chunk_document, chunk_text
from src.rag.generator import generate_grounded_brief
from src.rag.retriever import retrieve
from src.rag.schema import Chunk, GroundedBrief, RetrievedChunk
from src.rag.vectorstore import QdrantVectorStore


def _chunk(text: str, filename: str, chunk_index: int = 0) -> Chunk:
    return Chunk(
        text=text,
        filename=filename,
        document_type="X",
        date="2025",
        chunk_index=chunk_index,
    )


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------


def test_chunk_text_produces_expected_chunk_count() -> None:
    text = " ".join(f"word{i}" for i in range(1200))
    chunks = chunk_text(text, chunk_size_tokens=500, overlap_tokens=50)
    assert len(chunks) == 3
    assert len(chunks[0].split()) == 500
    assert len(chunks[-1].split()) == 300


def test_chunk_text_overlap_is_present_between_consecutive_chunks() -> None:
    text = " ".join(f"word{i}" for i in range(1000))
    chunks = chunk_text(text, chunk_size_tokens=500, overlap_tokens=50)
    last_words_of_first = chunks[0].split()[-50:]
    first_words_of_second = chunks[1].split()[:50]
    assert last_words_of_first == first_words_of_second


def test_chunk_text_empty_string_returns_no_chunks() -> None:
    assert chunk_text("", chunk_size_tokens=500, overlap_tokens=50) == []


def test_chunk_document_preserves_source_metadata() -> None:
    text = " ".join(f"word{i}" for i in range(50))
    chunks = chunk_document(
        text, filename="doc.txt", document_type="10-K", date="2025-01-01"
    )
    assert len(chunks) == 1
    chunk = chunks[0]
    assert isinstance(chunk, Chunk)
    assert chunk.filename == "doc.txt"
    assert chunk.document_type == "10-K"
    assert chunk.date == "2025-01-01"
    assert chunk.chunk_index == 0


def test_chunk_document_chunk_index_increments() -> None:
    text = " ".join(f"word{i}" for i in range(1200))
    chunks = chunk_document(
        text,
        filename="doc.txt",
        document_type="10-K",
        date="2025-01-01",
        chunk_size_tokens=500,
        overlap_tokens=50,
    )
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))


# ---------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------


def test_embed_texts_produces_correct_dimension() -> None:
    from src.rag.embeddings import EMBEDDING_DIM, embed_texts

    vectors = embed_texts(["hello world", "goodbye world"])
    assert len(vectors) == 2
    assert all(len(v) == EMBEDDING_DIM for v in vectors)


def test_embed_texts_empty_list_returns_empty_list() -> None:
    from src.rag.embeddings import embed_texts

    assert embed_texts([]) == []


def test_embed_query_returns_single_vector() -> None:
    from src.rag.embeddings import EMBEDDING_DIM, embed_query

    vector = embed_query("a financial risk question")
    assert len(vector) == EMBEDDING_DIM


# ---------------------------------------------------------------------------
# Vector store — Qdrant round-trip (in-memory, no Docker required)
# ---------------------------------------------------------------------------


@pytest.fixture
def memory_store() -> QdrantVectorStore:
    client = QdrantClient(location=":memory:", check_compatibility=False)
    return QdrantVectorStore(client, embedding_dim=4)


def test_upsert_and_search_round_trip(memory_store: QdrantVectorStore) -> None:
    collection = "test_collection"
    memory_store.create_collection(collection)

    chunks = [
        _chunk("alpha", "a.txt"),
        _chunk("beta", "b.txt"),
    ]
    embeddings = [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]
    memory_store.upsert_chunks(collection, chunks, embeddings)

    results = memory_store.search(collection, [1.0, 0.0, 0.0, 0.0], top_k=2)
    assert len(results) == 2
    assert results[0].filename == "a.txt"
    assert isinstance(results[0], RetrievedChunk)


def test_search_results_ranked_by_relevance_descending(
    memory_store: QdrantVectorStore,
) -> None:
    collection = "test_collection_ranked"
    memory_store.create_collection(collection)

    chunks = [
        _chunk("close match", "close.txt"),
        _chunk("far match", "far.txt"),
    ]
    embeddings = [[0.9, 0.1, 0.0, 0.0], [0.0, 0.0, 0.1, 0.9]]
    memory_store.upsert_chunks(collection, chunks, embeddings)

    results = memory_store.search(collection, [1.0, 0.0, 0.0, 0.0], top_k=2)
    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)
    assert results[0].filename == "close.txt"


def test_upsert_length_mismatch_raises_value_error(
    memory_store: QdrantVectorStore,
) -> None:
    memory_store.create_collection("test_mismatch")
    chunks = [
        _chunk("alpha", "a.txt")
    ]
    with pytest.raises(ValueError):
        memory_store.upsert_chunks("test_mismatch", chunks, [[1.0], [2.0]])


# ---------------------------------------------------------------------------
# Retriever
# ---------------------------------------------------------------------------


def test_retrieve_requires_explicit_store() -> None:
    with pytest.raises(ValueError):
        retrieve("a query that is long enough", store=None)


def test_retrieve_unknown_mode_raises_value_error(
    memory_store: QdrantVectorStore,
) -> None:
    memory_store.create_collection("test_mode")
    with (
        patch("src.rag.retriever.embed_query", return_value=[0.0, 0.0, 0.0, 0.0]),
        pytest.raises(ValueError),
    ):
        retrieve(
            "a query",
            mode="nonexistent",
            store=memory_store,
            collection_name="test_mode",
        )


def test_retrieve_dense_mode_returns_ranked_results(
    memory_store: QdrantVectorStore,
) -> None:
    collection = "test_retrieve_dense"
    memory_store.create_collection(collection)
    chunks = [
        _chunk("close", "close.txt"),
        _chunk("far", "far.txt"),
    ]
    memory_store.upsert_chunks(
        collection, chunks, [[0.9, 0.1, 0.0, 0.0], [0.0, 0.0, 0.1, 0.9]]
    )

    with patch(
        "src.rag.retriever.embed_query", return_value=[1.0, 0.0, 0.0, 0.0]
    ):
        results = retrieve(
            "a query long enough", top_k=2, mode="dense", store=memory_store,
            collection_name=collection,
        )

    assert len(results) == 2
    assert results[0].filename == "close.txt"
    assert results[0].score >= results[1].score


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_generate_grounded_brief_returns_valid_schema() -> None:
    chunks = [
        RetrievedChunk(
            text="The Fed maintained the target rate.",
            filename="fomc.txt",
            document_type="FOMC Statement",
            date="2026",
            chunk_index=0,
            score=0.9,
        )
    ]
    mock_client = AsyncMock()
    mock_client.complete.return_value = (
        '{"answer": "The Fed maintained rates [fomc.txt].", "confidence": 0.8}'
    )
    with patch("src.rag.generator.get_llm_client", return_value=mock_client):
        brief = await generate_grounded_brief("What did the Fed decide?", chunks)

    assert isinstance(brief, GroundedBrief)
    assert "fomc.txt" in brief.answer
    assert brief.confidence == 0.8
    assert brief.sources[0].filename == "fomc.txt"


@pytest.mark.asyncio
async def test_generate_grounded_brief_no_chunks_returns_zero_confidence() -> None:
    brief = await generate_grounded_brief("anything", [])
    assert brief.sources == []
    assert brief.confidence == 0.0


@pytest.mark.asyncio
async def test_generate_grounded_brief_falls_back_on_unparsable_llm_output() -> None:
    chunks = [
        RetrievedChunk(
            text="text", filename="f.txt", document_type="X", date="2025",
            chunk_index=0, score=0.5,
        )
    ]
    mock_client = AsyncMock()
    mock_client.complete.return_value = "not valid json at all"
    with patch("src.rag.generator.get_llm_client", return_value=mock_client):
        brief = await generate_grounded_brief("a question", chunks)

    assert brief.answer == "not valid json at all"
    assert brief.confidence == 0.3


# ---------------------------------------------------------------------------
# /analyze/rag endpoint
# ---------------------------------------------------------------------------


def _mock_rag_store() -> MagicMock:
    m = MagicMock()
    m.search.return_value = [
        RetrievedChunk(
            text="The Fed maintained rates.",
            filename="fomc.txt",
            document_type="FOMC Statement",
            date="2026",
            chunk_index=0,
            score=0.9,
        )
    ]
    return m


@pytest.fixture
def rag_client() -> TestClient:  # type: ignore[misc]
    with TestClient(app) as client:
        app.state.rag_store = _mock_rag_store()
        with patch(
            "src.rag.retriever.embed_query", return_value=[0.1, 0.2, 0.3, 0.4]
        ):
            yield client  # type: ignore[misc]


@pytest.fixture
def no_rag_store_client() -> TestClient:  # type: ignore[misc]
    with TestClient(app) as client:
        app.state.rag_store = None
        yield client  # type: ignore[misc]


def test_analyze_rag_valid_query_returns_200(rag_client: TestClient) -> None:
    mock_brief = GroundedBrief(
        answer="Rates held steady.", sources=[], confidence=0.7
    )
    with patch(
        "src.api.routes.generate_grounded_brief",
        new=AsyncMock(return_value=mock_brief),
    ):
        response = rag_client.post(
            "/analyze/rag",
            json={"query": "What did the Fed decide about interest rates?"},
        )
    assert response.status_code == 200
    body = response.json()
    assert "answer" in body
    assert "sources" in body
    assert "confidence" in body


def test_analyze_rag_invalid_query_returns_422(rag_client: TestClient) -> None:
    response = rag_client.post("/analyze/rag", json={"query": "short"})
    assert response.status_code == 422


def test_analyze_rag_missing_body_returns_422(rag_client: TestClient) -> None:
    assert rag_client.post("/analyze/rag").status_code == 422


def test_analyze_rag_no_store_returns_503(no_rag_store_client: TestClient) -> None:
    response = no_rag_store_client.post(
        "/analyze/rag", json={"query": "What did the Fed decide?"}
    )
    assert response.status_code == 503


# ---------------------------------------------------------------------------
# Reranking (Task 6b) — cross-encoder mode
# ---------------------------------------------------------------------------


def test_dense_reranked_mode_reorders_by_cross_encoder_score(
    memory_store: QdrantVectorStore,
) -> None:
    collection = "test_rerank"
    memory_store.create_collection(collection)
    chunks = [
        _chunk("irrelevant filler text", "irrelevant.txt"),
        _chunk("Apple's risk factors discuss supply chain disruption.", "relevant.txt"),
    ]
    memory_store.upsert_chunks(
        collection, chunks, [[0.5, 0.5, 0.0, 0.0], [0.5, 0.5, 0.0, 0.0]]
    )

    fake_cross_encoder = MagicMock()
    fake_cross_encoder.predict.side_effect = lambda pairs, **kwargs: [
        0.95 if "supply chain" in pair[1] else 0.05 for pair in pairs
    ]

    with (
        patch("src.rag.retriever.embed_query", return_value=[0.5, 0.5, 0.0, 0.0]),
        patch("src.rag.retriever._get_cross_encoder", return_value=fake_cross_encoder),
    ):
        results = retrieve(
            "What does Apple say about supply chain risk?",
            top_k=2,
            mode="dense_reranked",
            store=memory_store,
            collection_name=collection,
        )

    assert results[0].filename == "relevant.txt"
    assert results[0].score == 0.95


# ---------------------------------------------------------------------------
# Integration — requires a live Qdrant at localhost:6333 (skipped by default)
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_live_qdrant_connection_and_round_trip() -> None:
    client = QdrantClient(url="http://localhost:6333", timeout=5)
    store = QdrantVectorStore(client, embedding_dim=4)
    store.create_collection("finsight_test_integration")
    chunks = [
        _chunk("alpha", "a.txt")
    ]
    store.upsert_chunks("finsight_test_integration", chunks, [[1.0, 0.0, 0.0, 0.0]])
    results = store.search("finsight_test_integration", [1.0, 0.0, 0.0, 0.0], top_k=1)
    assert results[0].filename == "a.txt"
    client.delete_collection("finsight_test_integration")
