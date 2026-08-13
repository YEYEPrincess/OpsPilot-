"""检索指标。

这个文件只负责计算数学指标，不连接Qdrant，也不生成Embedding。
把指标和数据库分开，方便我们单独测试公式是否正确。
"""

from __future__ import annotations

from collections.abc import Iterable


def top_k_unique(retrieved_ids: Iterable[str], k: int) -> list[str]:
    """取前k个不重复的Chunk ID，并保持原来的排序。

    为什么需要去重：
    如果检索结果意外重复返回同一个Chunk，不能把它算成两条证据。
    """
    if k <= 0:
        return []

    results: list[str] = []
    seen: set[str] = set()

    for chunk_id in retrieved_ids:
        chunk_id = str(chunk_id)

        if chunk_id in seen:
            continue

        seen.add(chunk_id)
        results.append(chunk_id)

        if len(results) == k:
            break

    return results


def recall_at_k(
    gold_ids: Iterable[str],
    retrieved_ids: Iterable[str],
    k: int,
) -> float:
    """计算Recall@k：找回的gold证据比例。"""
    gold = {str(chunk_id) for chunk_id in gold_ids}

    # 没有gold evidence的问题不能计算检索召回率。
    if not gold:
        return 0.0

    retrieved = set(top_k_unique(retrieved_ids, k))
    hit_count = len(gold & retrieved)

    return hit_count / len(gold)


def hit_rate_at_k(
    gold_ids: Iterable[str],
    retrieved_ids: Iterable[str],
    k: int,
) -> float:
    """Top-k至少命中一个gold evidence时返回1，否则返回0。"""
    gold = {str(chunk_id) for chunk_id in gold_ids}

    if not gold:
        return 0.0

    retrieved = set(top_k_unique(retrieved_ids, k))

    return float(bool(gold & retrieved))


def reciprocal_rank_at_k(
    gold_ids: Iterable[str],
    retrieved_ids: Iterable[str],
    k: int,
) -> float:
    """计算第一条正确证据的倒数排名。"""
    gold = {str(chunk_id) for chunk_id in gold_ids}

    if not gold:
        return 0.0

    for rank, chunk_id in enumerate(top_k_unique(retrieved_ids, k), start=1):
        if chunk_id in gold:
            return 1.0 / rank

    return 0.0