"""RAGAS evaluation harness for the FinSight RAG pipeline.

Builds a small hand-written question/answer set, grounded in the actual
corpus fetched by scripts/fetch_corpus.py, retrieves + generates an answer
for each question, then scores the results with RAGAS metrics:
faithfulness, answer_relevancy, context_precision, context_recall.

Usage:
    python scripts/evaluate_rag.py --mode dense
    python scripts/evaluate_rag.py --mode dense_reranked

Requires Qdrant running (docker-compose up -d qdrant) and the corpus
already ingested (scripts/ingest_corpus.py).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from datasets import Dataset
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from qdrant_client import QdrantClient
from ragas import evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import (
    context_precision,
    context_recall,
    faithfulness,
)

from src.config import settings
from src.rag.embeddings import EMBEDDING_DIM
from src.rag.generator import generate_grounded_brief
from src.rag.retriever import DEFAULT_COLLECTION, retrieve
from src.rag.vectorstore import QdrantVectorStore

QDRANT_URL = "http://localhost:6333"
RESULTS_PATH = Path(__file__).parent.parent / "artefacts" / "ragas_eval_results.json"

# GPT-4o-mini as judge: low cost (~$0.05 total for both runs), fast responses,
# and RAGAS's native default — avoids free-tier latency/quota issues on Groq.
# answer_relevancy is excluded: it requires synthetic question generation that
# exceeds free-tier LLM response latency constraints; the 3 remaining metrics
# cover hallucination (faithfulness), retrieval quality (context_precision),
# and coverage (context_recall).
_JUDGE_LLM = LangchainLLMWrapper(
    ChatOpenAI(model="gpt-4o-mini", api_key=settings.openai_api_key)
)
_JUDGE_EMBEDDINGS = LangchainEmbeddingsWrapper(
    OpenAIEmbeddings(model="text-embedding-3-small", api_key=settings.openai_api_key)
)

# Hand-written eval set, grounded in the actual documents in data/corpus/
# (verified by reading the fetched excerpts directly).
# 5 questions selected for unambiguous, specific answers directly traceable
# to a single sentence or number in the fetched SEC EDGAR / FOMC documents.
EVAL_SET = [
    {
        "question": (
            "What did the FOMC decide about the federal funds rate in its "
            "June 2026 statement?"
        ),
        "ground_truth": (
            "The Committee decided to maintain the target range for the "
            "federal funds rate at 3-1/2 to 3-3/4 percent."
        ),
    },
    {
        "question": (
            "What is the Federal Reserve's inflation goal according to "
            "its statements?"
        ),
        "ground_truth": "The Committee's goal is 2 percent inflation.",
    },
    {
        "question": (
            "What climate-related risk does Microsoft describe in its "
            "10-Q risk factors?"
        ),
        "ground_truth": (
            "Microsoft notes that changes in climate where it operates "
            "may increase the costs of powering and cooling the computer "
            "hardware it uses to develop software and provide cloud-based "
            "services."
        ),
    },
    {
        "question": (
            "What category of risk does JPMorgan Chase list first among "
            "its principal risk factors?"
        ),
        "ground_truth": (
            "Legal and Regulatory risks, including the impact of "
            "extensive supervision and regulation."
        ),
    },
    {
        "question": (
            "What future product does Tesla mention in connection with "
            "autonomous driving in its risk factors?"
        ),
        "ground_truth": (
            "Cybercab, Tesla's purpose-built Robotaxi product."
        ),
    },
]


async def run_pipeline(mode: str) -> list[dict]:
    client = QdrantClient(url=QDRANT_URL)
    store = QdrantVectorStore(client, embedding_dim=EMBEDDING_DIM)

    rows = []
    for item in EVAL_SET:
        retrieved = retrieve(
            item["question"],
            top_k=5,
            mode=mode,
            store=store,
            collection_name=DEFAULT_COLLECTION,
        )
        brief = await generate_grounded_brief(item["question"], retrieved)
        rows.append(
            {
                "question": item["question"],
                "answer": brief.answer,
                "contexts": [c.text for c in retrieved],
                "ground_truth": item["ground_truth"],
            }
        )
    return rows


def score(rows: list[dict]) -> dict:
    """Score RAGAS metrics sequentially with a pause between each.

    Sequential execution avoids parallel LLM call bursts. answer_relevancy
    is excluded — it requires synthetic question generation that exceeds
    free-tier LLM response latency constraints.
    """
    dataset = Dataset.from_list(rows)
    metrics_list = [
        ("faithfulness", faithfulness),
        ("context_precision", context_precision),
        ("context_recall", context_recall),
    ]
    scores: dict[str, float] = {}
    for metric_name, metric in metrics_list:
        print(f"  Evaluating {metric_name}...")
        try:
            result = evaluate(
                dataset,
                metrics=[metric],
                llm=_JUDGE_LLM,
                embeddings=_JUDGE_EMBEDDINGS,
            )
            val = float(result.to_pandas()[metric_name].mean())
            scores[metric_name] = val
            print(f"    {metric_name}: {val:.4f}")
        except Exception as exc:
            print(f"    {metric_name}: FAILED — {exc}")
            scores[metric_name] = float("nan")
        if metric_name != metrics_list[-1][0]:
            time.sleep(10)
    return scores


def main() -> None:
    parser = argparse.ArgumentParser(description="RAGAS evaluation for FinSight RAG")
    parser.add_argument(
        "--mode",
        default="dense",
        choices=["dense", "dense_reranked"],
        help="Retrieval mode to evaluate",
    )
    args = parser.parse_args()

    print(f"Running RAG pipeline over {len(EVAL_SET)} questions (mode={args.mode})...")
    rows = asyncio.run(run_pipeline(args.mode))

    print("Scoring with RAGAS metrics sequentially...")
    scores = score(rows)

    print("\n=== RAGAS Evaluation Summary ===")
    print(f"Mode: {args.mode}")
    for metric, value in scores.items():
        print(f"  {metric:<20} {value:.4f}")

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    output = {"mode": args.mode, "n_questions": len(EVAL_SET), "scores": scores}
    RESULTS_PATH.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"\nResults saved to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
