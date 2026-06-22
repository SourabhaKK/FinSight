"""Chunk, embed, and upsert the document corpus into Qdrant.

Run once (or whenever data/corpus/ changes) to populate the vector store:

    docker-compose up -d qdrant
    python scripts/ingest_corpus.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from qdrant_client import QdrantClient

from src.rag.chunking import chunk_document
from src.rag.embeddings import EMBEDDING_DIM, embed_texts
from src.rag.schema import Chunk
from src.rag.vectorstore import QdrantVectorStore

CORPUS_DIR = Path(__file__).parent.parent / "data" / "corpus"
QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "finsight_corpus"


def load_chunks() -> list[Chunk]:
    metadata = json.loads((CORPUS_DIR / "metadata.json").read_text(encoding="utf-8"))
    all_chunks: list[Chunk] = []
    for doc in metadata:
        text = (CORPUS_DIR / doc["filename"]).read_text(encoding="utf-8")
        all_chunks.extend(
            chunk_document(
                text=text,
                filename=doc["filename"],
                document_type=doc["document_type"],
                date=doc["date"],
            )
        )
    return all_chunks


def main() -> None:
    print(f"Loading documents from {CORPUS_DIR}...")
    chunks = load_chunks()
    n_docs = len(json.loads((CORPUS_DIR / "metadata.json").read_text(encoding="utf-8")))
    print(f"  {n_docs} documents -> {len(chunks)} chunks")

    print("Embedding chunks (all-MiniLM-L6-v2)...")
    embeddings = embed_texts([c.text for c in chunks])
    print(f"  embedding dimension: {len(embeddings[0]) if embeddings else 0}")

    print(f"Connecting to Qdrant at {QDRANT_URL}...")
    client = QdrantClient(url=QDRANT_URL)
    store = QdrantVectorStore(client, embedding_dim=EMBEDDING_DIM)
    store.create_collection(COLLECTION_NAME)
    store.upsert_chunks(COLLECTION_NAME, chunks, embeddings)

    print("\n=== Ingestion summary ===")
    print(f"Documents:         {n_docs}")
    print(f"Chunks:            {len(chunks)}")
    print(f"Embedding dim:     {EMBEDDING_DIM}")
    print(f"Collection:        {COLLECTION_NAME}")


if __name__ == "__main__":
    main()
