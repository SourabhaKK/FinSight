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
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from datasets import Dataset
from qdrant_client import QdrantClient
from ragas import evaluate
from ragas.metrics import (
    answer_relevancy,
    context_precision,
    context_recall,
    faithfulness,
)

from src.rag.embeddings import EMBEDDING_DIM
from src.rag.generator import generate_grounded_brief
from src.rag.retriever import DEFAULT_COLLECTION, retrieve
from src.rag.vectorstore import QdrantVectorStore

QDRANT_URL = "http://localhost:6333"
RESULTS_PATH = Path(__file__).parent.parent / "artefacts" / "ragas_eval_results.json"

# Hand-written eval set, grounded in the actual documents in data/corpus/
# (verified by reading the fetched excerpts directly).
EVAL_SET = [
    {
        "question": "What did the FOMC decide about the federal funds rate in its June 2026 statement?",
        "ground_truth": (
            "The Committee decided to maintain the target range for the "
            "federal funds rate at 3-1/2 to 3-3/4 percent."
        ),
    },
    {
        "question": "According to the June 2026 FOMC statement, what is contributing to elevated inflation?",
        "ground_truth": (
            "Inflation remains elevated relative to the Committee's 2 "
            "percent goal, in part reflecting supply shocks that have "
            "driven price increases in certain sectors, including energy."
        ),
    },
    {
        "question": "What is the Federal Reserve's inflation goal according to its statements?",
        "ground_truth": "The Committee's goal is 2 percent inflation.",
    },
    {
        "question": "What does Apple's 10-K say about the nature of its risk factors disclosure?",
        "ground_truth": (
            "Apple states the risk factors are not exhaustive and should "
            "not be considered a complete statement of all potential "
            "risks the Company faces or may face in the future."
        ),
    },
    {
        "question": "What climate-related risk does Microsoft describe in its 10-Q risk factors?",
        "ground_truth": (
            "Microsoft notes that changes in climate where it operates "
            "may increase the costs of powering and cooling the computer "
            "hardware it uses to develop software and provide cloud-based "
            "services."
        ),
    },
    {
        "question": "What category of risk does JPMorgan Chase list first among its principal risk factors?",
        "ground_truth": (
            "Legal and Regulatory risks, including the impact of "
            "extensive supervision and regulation."
        ),
    },
    {
        "question": "What does Tesla's 10-K say about risks related to growing its business?",
        "ground_truth": (
            "Tesla may experience issues or delays in developing, "
            "launching, and ramping production of its products, services, "
            "and features, or may be unable to control manufacturing "
            "costs."
        ),
    },
    {
        "question": "What future product does Tesla mention in connection with autonomous driving in its risk factors?",
        "ground_truth": (
            "Cybercab, Tesla's purpose-built Robotaxi product."
        ),
    },
    {
        "question": "Does Apple's 10-K say it can accurately predict all the risks it describes?",
        "ground_truth": (
            "No — Apple states it may not be able to accurately predict, "
            "control, or mitigate the risks described."
        ),
    },
    {
        "question": "What role does international/global business play in Microsoft's described risk exposure?",
        "ground_truth": (
            "Microsoft's global business — customers, employees, and "
            "infrastructure located worldwide, with significant revenue "
            "from international sales — exposes it to operational, "
            "economic, and geopolitical risks."
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
    dataset = Dataset.from_list(rows)
    result = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
    )
    return result.to_pandas().mean(numeric_only=True).to_dict()  # type: ignore[no-any-return]


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

    print("Scoring with RAGAS (faithfulness, answer_relevancy, "
          "context_precision, context_recall)...")
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
