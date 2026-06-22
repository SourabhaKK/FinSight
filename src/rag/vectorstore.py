"""Thin wrapper around qdrant-client for the RAG pipeline's vector store."""

from __future__ import annotations

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from src.rag.schema import Chunk, RetrievedChunk


class QdrantVectorStore:
    def __init__(self, client: QdrantClient, embedding_dim: int) -> None:
        self._client = client
        self._embedding_dim = embedding_dim

    def create_collection(self, collection_name: str) -> None:
        if self._client.collection_exists(collection_name):
            self._client.delete_collection(collection_name)
        self._client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=self._embedding_dim, distance=Distance.COSINE
            ),
        )

    def upsert_chunks(
        self,
        collection_name: str,
        chunks: list[Chunk],
        embeddings: list[list[float]],
    ) -> None:
        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings must have the same length")
        points = [
            PointStruct(
                id=i,
                vector=embeddings[i],
                payload=chunks[i].model_dump(),
            )
            for i in range(len(chunks))
        ]
        self._client.upsert(collection_name=collection_name, points=points)

    def search(
        self,
        collection_name: str,
        query_embedding: list[float],
        top_k: int = 5,
    ) -> list[RetrievedChunk]:
        results = self._client.query_points(
            collection_name=collection_name,
            query=query_embedding,
            limit=top_k,
        ).points
        return [
            RetrievedChunk(**point.payload, score=point.score)  # type: ignore[arg-type]
            for point in results
        ]
