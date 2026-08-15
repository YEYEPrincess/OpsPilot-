from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from time import perf_counter
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evaluation.retrieval_metrics import (  # noqa: E402
    hit_rate_at_k,
    recall_at_k,
    reciprocal_rank_at_k,
)
from retrieval.bm25_store import BM25Index  # noqa: E402
from retrieval.embeddings import create_embedding_provider  # noqa: E402
from retrieval.hybrid_search import reciprocal_rank_fusion  # noqa: E402
from retrieval.qdrant_store import QdrantVectorStore  # noqa: E402


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def p95(values: list[float]) -> float:
    """返回简单的 nearest-rank P95。"""
    ordered = sorted(values)
    index = max(0, int(0.95 * len(ordered) + 0.9999) - 1)
    return ordered[index]


def summarize_latency(values: list[float]) -> dict[str, float]:
    return {
        "mean_ms": round(statistics.mean(values), 3),
        "p50_ms": round(statistics.median(values), 3),
        "p95_ms": round(p95(values), 3),
    }


def summarize_metrics(
    rows: list[dict[str, Any]],
    method: str,
    cutoffs: list[int],
) -> dict[str, dict[str, float]]:
    summary: dict[str, dict[str, float]] = {}
    for k in cutoffs:
        recall_values: list[float] = []
        hit_values: list[float] = []
        mrr_values: list[float] = []

        for row in rows:
            retrieved = row["retrieved_ids"][method]
            gold = row["gold_ids"]
            recall_values.append(recall_at_k(gold, retrieved, k))
            hit_values.append(hit_rate_at_k(gold, retrieved, k))
            mrr_values.append(reciprocal_rank_at_k(gold, retrieved, k))

        summary[str(k)] = {
            "recall_at_k": round(statistics.mean(recall_values), 4),
            "hit_rate_at_k": round(statistics.mean(hit_values), 4),
            "mrr_at_k": round(statistics.mean(mrr_values), 4),
        }
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-k", type=int, default=20)
    parser.add_argument("--rrf-k", type=int, default=60)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "data/eval/day8_retrieval_comparison.json",
    )
    args = parser.parse_args()

    eval_path = ROOT / "data/eval/day6_eval.jsonl"
    bm25_path = ROOT / "data/index/bm25/opspilot_bm25_v1.json.gz"
    records = [
        row
        for row in load_jsonl(eval_path)
        if row.get("answerability") == "answerable"
    ]

    provider = create_embedding_provider(
        provider="hash", model_name="", dimension=384, device="cpu"
    )
    dense_store = QdrantVectorStore(
        ROOT / "data/index/qdrant",
        "opspilot_chunks_v1",
        provider.dimension,
    )
    bm25_store = BM25Index.load(bm25_path)

    rows: list[dict[str, Any]] = []
    try:
        for record in records:
            question = record["question"]
            gold_ids = [item["chunk_id"] for item in record["gold_evidence"]]

            start = perf_counter()
            query_vector = provider.encode([question])[0]
            dense_hits = dense_store.search(
                query_vector,
                limit=args.candidate_k,
                query_text="",  # 保持 Day 7 dense-only 定义
            )
            dense_ms = (perf_counter() - start) * 1000

            start = perf_counter()
            bm25_hits = bm25_store.search(question, limit=args.candidate_k)
            bm25_ms = (perf_counter() - start) * 1000

            start = perf_counter()
            hybrid_hits = reciprocal_rank_fusion(
                dense_hits,
                bm25_hits,
                limit=args.candidate_k,
                rrf_k=args.rrf_k,
            )
            fusion_ms = (perf_counter() - start) * 1000

            rows.append(
                {
                    "id": record["id"],
                    "question": question,
                    "gold_ids": gold_ids,
                    "retrieved_ids": {
                        "dense": [hit.payload["chunk_id"] for hit in dense_hits],
                        "bm25": [hit.payload["chunk_id"] for hit in bm25_hits],
                        "hybrid": [hit.payload["chunk_id"] for hit in hybrid_hits],
                    },
                    "latency_ms": {
                        "dense": dense_ms,
                        "bm25": bm25_ms,
                        # 当前是顺序调用，因此端到端延迟使用求和。
                        "hybrid_sequential": dense_ms + bm25_ms + fusion_ms,
                        # 如果以后真正并行，理论近似为较慢分支+融合时间。
                        "hybrid_parallel_estimate": max(dense_ms, bm25_ms) + fusion_ms,
                    },
                }
            )
    finally:
        dense_store.close()

    cutoffs = [1, 3, 5, 10]
    methods = ["dense", "bm25", "hybrid"]
    result = {
        "config": {
            "candidate_k": args.candidate_k,
            "rrf_k": args.rrf_k,
            "evaluated_questions": len(rows),
        },
        "metrics": {
            method: summarize_metrics(rows, method, cutoffs)
            for method in methods
        },
        "latency": {
            name: summarize_latency(
                [row["latency_ms"][name] for row in rows]
            )
            for name in [
                "dense",
                "bm25",
                "hybrid_sequential",
                "hybrid_parallel_estimate",
            ]
        },
        "per_query": rows,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({k: result[k] for k in ["config", "metrics", "latency"]},
                     ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())