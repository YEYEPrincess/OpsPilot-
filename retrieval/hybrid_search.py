"""使用 RRF 融合 Dense 与 BM25 结果。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from retrieval.bm25_store import BM25Hit
from retrieval.qdrant_store import SearchHit


@dataclass(frozen=True)
class HybridHit:
    score: float
    payload: dict[str, Any]
    dense_rank: int | None
    bm25_rank: int | None


def reciprocal_rank_fusion(
    dense_hits: list[SearchHit],
    bm25_hits: list[BM25Hit],
    limit: int = 10,
    rrf_k: int = 60,
    dense_weight: float = 1.0,
    bm25_weight: float = 1.0,
) -> list[HybridHit]:
    """按 Chunk ID 合并两路排名。"""
    if rrf_k <= 0:
        raise ValueError("rrf_k must be positive")

    merged: dict[str, dict[str, Any]] = {}

    for rank, hit in enumerate(dense_hits, start=1):
        chunk_id = str(hit.payload["chunk_id"])
        item = merged.setdefault(
            chunk_id,
            {
                "score": 0.0,
                "payload": hit.payload,
                "dense_rank": None,
                "bm25_rank": None,
            },
        )
        item["score"] += dense_weight / (rrf_k + rank)
        item["dense_rank"] = rank

    for rank, hit in enumerate(bm25_hits, start=1):
        chunk_id = str(hit.payload["chunk_id"])
        item = merged.setdefault(
            chunk_id,
            {
                "score": 0.0,
                "payload": hit.payload,
                "dense_rank": None,
                "bm25_rank": None,
            },
        )
        item["score"] += bm25_weight / (rrf_k + rank)
        item["bm25_rank"] = rank

    ranked = sorted(
        merged.values(),
        key=lambda item: item["score"],
        reverse=True,
    )[:limit]

    return [
        HybridHit(
            score=float(item["score"]),
            payload=item["payload"],
            dense_rank=item["dense_rank"],
            bm25_rank=item["bm25_rank"],
        )
        for item in ranked
    ]