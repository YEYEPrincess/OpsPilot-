"""Query-level retrieval metrics used by the Day 7 baseline.

The functions are intentionally small and dependency-free so that the metric
definitions can be unit-tested independently from Qdrant and embedding code.
"""

from __future__ import annotations

from collections.abc import Iterable


def _ranked_ids(retrieved_ids: Iterable[str], k: int) -> list[str]:
    """Return the first *k* unique IDs, preserving retrieval order."""
    if k <= 0:
        return []
    result: list[str] = []
    seen: set[str] = set()
    for chunk_id in retrieved_ids:
        chunk_id = str(chunk_id)
        if chunk_id in seen:
            continue
        seen.add(chunk_id)
        result.append(chunk_id)
        if len(result) == k:
            break
    return result


def recall_at_k(gold_ids: Iterable[str], retrieved_ids: Iterable[str], k: int) -> float:
    """Return the fraction of gold evidence IDs found in the first *k* hits."""
    gold = {str(chunk_id) for chunk_id in gold_ids}
    if not gold:
        return 0.0
    retrieved = set(_ranked_ids(retrieved_ids, k))
    return len(gold & retrieved) / len(gold)


def hit_rate_at_k(gold_ids: Iterable[str], retrieved_ids: Iterable[str], k: int) -> float:
    """Return 1 when at least one gold evidence ID appears in the first *k* hits."""
    gold = {str(chunk_id) for chunk_id in gold_ids}
    if not gold:
        return 0.0
    return float(bool(gold & set(_ranked_ids(retrieved_ids, k))))


def reciprocal_rank_at_k(
    gold_ids: Iterable[str], retrieved_ids: Iterable[str], k: int
) -> float:
    """Return reciprocal rank of the first relevant hit, or zero when absent."""
    gold = {str(chunk_id) for chunk_id in gold_ids}
    if not gold:
        return 0.0
    for rank, chunk_id in enumerate(_ranked_ids(retrieved_ids, k), 1):
        if chunk_id in gold:
            return 1.0 / rank
    return 0.0

