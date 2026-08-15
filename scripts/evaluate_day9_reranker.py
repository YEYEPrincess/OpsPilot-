"""对 Day 8 固定候选运行 Reranker A/B 实验。"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evaluation.retrieval_metrics import (  # noqa: E402
    hit_rate_at_k,
    recall_at_k,
    reciprocal_rank_at_k,
)
from retrieval.reranker import CrossEncoderReranker  # noqa: E402


@dataclass(frozen=True)
class CachedCandidate:
    score: float
    payload: dict[str, Any]


def mean_metric(rows: list[dict[str, Any]], key: str) -> float:
    return round(statistics.mean(row[key] for row in rows), 4)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-k", type=int, default=20)
    parser.add_argument("--final-k", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=0)
    parser.add_argument(
        "--model",
        default="BAAI/bge-reranker-v2-m3",
    )
    parser.add_argument(
        "--day8-result",
        type=Path,
        default=ROOT / "data/eval/day8_retrieval_comparison.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "data/eval/day9_reranker_comparison.json",
    )
    args = parser.parse_args()

    chunks = [
        json.loads(line)
        for line in (ROOT / "data/processed/chunks_section.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line
    ]
    chunk_by_id = {chunk["chunk_id"]: chunk for chunk in chunks}
    day8 = json.loads(args.day8_result.read_text(encoding="utf-8"))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size = args.batch_size or (8 if device == "cuda" else 2)
    reranker = CrossEncoderReranker(
        args.model,
        device=device,
        batch_size=batch_size,
        max_length=512,
    )

    # Warm-up：不纳入正式延迟。
    first = day8["per_query"][0]
    warm_ids = first["retrieved_ids"]["hybrid"][:2]
    warm_candidates = [
        CachedCandidate(score=0.0, payload=chunk_by_id[chunk_id])
        for chunk_id in warm_ids
    ]
    reranker.rerank(first["question"], warm_candidates, top_n=2)

    rows: list[dict[str, Any]] = []
    for item in day8["per_query"]:
        candidate_ids = item["retrieved_ids"]["hybrid"][: args.candidate_k]
        candidates = [
            # 缓存文件没有保存完整RRF分数；本实验只需要原始顺序。
            CachedCandidate(score=1.0 / rank, payload=chunk_by_id[chunk_id])
            for rank, chunk_id in enumerate(candidate_ids, start=1)
        ]

        start = perf_counter()
        reranked = reranker.rerank(
            item["question"], candidates, top_n=args.final_k
        )
        rerank_ms = (perf_counter() - start) * 1000

        baseline_ids = candidate_ids[: args.final_k]
        reranked_ids = [hit.payload["chunk_id"] for hit in reranked]
        gold_ids = item["gold_ids"]

        rows.append(
            {
                "id": item["id"],
                "baseline_ids": baseline_ids,
                "reranked_ids": reranked_ids,
                "candidate_recall": recall_at_k(
                    gold_ids, candidate_ids, args.candidate_k
                ),
                "baseline_hit": hit_rate_at_k(
                    gold_ids, baseline_ids, args.final_k
                ),
                "reranked_hit": hit_rate_at_k(
                    gold_ids, reranked_ids, args.final_k
                ),
                "baseline_mrr": reciprocal_rank_at_k(
                    gold_ids, baseline_ids, args.final_k
                ),
                "reranked_mrr": reciprocal_rank_at_k(
                    gold_ids, reranked_ids, args.final_k
                ),
                "rerank_ms": rerank_ms,
            }
        )

    result = {
        "config": {
            "model": args.model,
            "device": device,
            "batch_size": batch_size,
            "candidate_k": args.candidate_k,
            "final_k": args.final_k,
        },
        "metrics": {
            "candidate_recall": mean_metric(rows, "candidate_recall"),
            "baseline_hit_at_final_k": mean_metric(rows, "baseline_hit"),
            "reranked_hit_at_final_k": mean_metric(rows, "reranked_hit"),
            "baseline_mrr_at_final_k": mean_metric(rows, "baseline_mrr"),
            "reranked_mrr_at_final_k": mean_metric(rows, "reranked_mrr"),
        },
        "latency": {
            "mean_ms": round(statistics.mean(row["rerank_ms"] for row in rows), 3),
            "p50_ms": round(statistics.median(row["rerank_ms"] for row in rows), 3),
            "max_ms": round(max(row["rerank_ms"] for row in rows), 3),
        },
        "per_query": rows,
    }
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({k: result[k] for k in ["config", "metrics", "latency"]},
                     ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
