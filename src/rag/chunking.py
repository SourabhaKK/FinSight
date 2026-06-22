"""Recursive character text splitter for the RAG corpus.

Implemented directly (no langchain dependency) since chunking is a small,
well-understood algorithm and sentence-transformers already covers the
embedding step needed for Task 2 — adding a framework just for splitting
text would be disproportionate to what this needs.

"Tokens" here means whitespace-delimited words, which is a close enough
proxy for chunk-sizing purposes without pulling in a tokenizer dependency.
"""

from __future__ import annotations

from src.rag.schema import Chunk

_SEPARATORS = ["\n\n", "\n", ". ", " "]


def _split_on_separator(text: str, separator: str) -> list[str]:
    if separator == "":
        return list(text)
    parts = text.split(separator)
    # Re-append the separator to all but the last part so we don't lose
    # sentence/paragraph boundaries when reassembling chunks.
    return [p + separator for p in parts[:-1]] + [parts[-1]]


def _recursive_split(text: str, separators: list[str]) -> list[str]:
    if not text:
        return []
    if not separators:
        return [text]

    separator, rest = separators[0], separators[1:]
    pieces = _split_on_separator(text, separator)

    # Word count is the chunking unit, so anything under ~600 words doesn't
    # need further splitting even if it's one "piece" from this separator.
    final_pieces: list[str] = []
    for piece in pieces:
        if len(piece.split()) > 600 and rest:
            final_pieces.extend(_recursive_split(piece, rest))
        else:
            final_pieces.append(piece)
    return final_pieces


def chunk_text(
    text: str,
    chunk_size_tokens: int = 500,
    overlap_tokens: int = 50,
) -> list[str]:
    """Split text into overlapping chunks of approximately chunk_size_tokens
    words, preferring to break on paragraph/sentence boundaries."""
    pieces = _recursive_split(text, _SEPARATORS)
    words: list[str] = []
    for piece in pieces:
        words.extend(piece.split())

    if not words:
        return []

    chunks: list[str] = []
    start = 0
    step = max(chunk_size_tokens - overlap_tokens, 1)
    while start < len(words):
        chunk_words = words[start : start + chunk_size_tokens]
        chunks.append(" ".join(chunk_words))
        if start + chunk_size_tokens >= len(words):
            break
        start += step
    return chunks


def chunk_document(
    text: str,
    filename: str,
    document_type: str,
    date: str,
    chunk_size_tokens: int = 500,
    overlap_tokens: int = 50,
) -> list[Chunk]:
    """Chunk a document's text and attach source metadata to each chunk."""
    raw_chunks = chunk_text(text, chunk_size_tokens, overlap_tokens)
    return [
        Chunk(
            text=raw_chunk,
            filename=filename,
            document_type=document_type,
            date=date,
            chunk_index=i,
        )
        for i, raw_chunk in enumerate(raw_chunks)
    ]
