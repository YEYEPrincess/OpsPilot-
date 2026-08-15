"""Cross-Encoder Reranker。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np
from sentence_transformers import CrossEncoder


class Candidate(Protocol):
    score: float
    payload: dict[str, Any]


@dataclass(frozen=True)
class RerankedHit:
    reranker_score: float
    retrieval_score: float
    original_rank: int
    payload: dict[str, Any]


def rank_by_scores(
    candidates: list[Candidate], scores: list[float]
) -> list[RerankedHit]:
    """把模型分数与候选绑定并按分数降序排列。"""
    if len(candidates) != len(scores):
        raise ValueError("Candidate and score counts differ")

    ranked = [
        RerankedHit(
            reranker_score=float(score),
            retrieval_score=float(candidate.score),
            original_rank=rank,
            payload=candidate.payload,
        )
        for rank, (candidate, score) in enumerate(
            zip(candidates, scores, strict=True), start=1
        )
    ]
    return sorted(ranked, key=lambda hit: hit.reranker_score, reverse=True)


class CrossEncoderReranker:
    def __init__(
        self,
        model_name: str,
        device: str,
        batch_size: int = 8,
        max_length: int = 512,
    ) -> None:
        self.model_name = model_name
        self.batch_size = batch_size
        self.model = CrossEncoder(
            model_name,
            device=device,
            max_length=max_length,
        )

    def rerank(
        self,
        query: str,
        candidates: list[Candidate],
        top_n: int = 5,
    ) -> list[RerankedHit]:
        if not candidates or top_n <= 0:
            return []

        pairs = [
            (query, str(candidate.payload.get("text", "")))
            for candidate in candidates
        ]
        raw_scores = self.model.predict(
            pairs,
            batch_size=self.batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        scores = np.asarray(raw_scores).reshape(-1).astype(float).tolist()
        return rank_by_scores(candidates, scores)[:top_n]