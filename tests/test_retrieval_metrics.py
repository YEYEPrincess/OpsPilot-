"""检索指标的单元测试。

测试使用人工构造的小例子，不依赖Qdrant和真实知识库。
"""

from evaluation.retrieval_metrics import (
    hit_rate_at_k,
    recall_at_k,
    reciprocal_rank_at_k,
)


def test_recall_at_k_with_multiple_gold_chunks() -> None:
    """两个gold中，Top-2只找回一个，因此Recall为0.5。"""
    gold = ["chunk-a", "chunk-b"]
    retrieved = ["chunk-x", "chunk-a", "chunk-b"]

    assert recall_at_k(gold, retrieved, 2) == 0.5
    assert recall_at_k(gold, retrieved, 3) == 1.0


def test_hit_rate_at_k() -> None:
    """Top-2存在一个gold，因此Hit为1。"""
    gold = ["chunk-a"]
    retrieved = ["chunk-x", "chunk-a"]

    assert hit_rate_at_k(gold, retrieved, 1) == 0.0
    assert hit_rate_at_k(gold, retrieved, 2) == 1.0


def test_reciprocal_rank_at_k() -> None:
    """第一条正确证据排第2，因此RR为0.5。"""
    gold = ["chunk-a"]
    retrieved = ["chunk-x", "chunk-a", "chunk-y"]

    assert reciprocal_rank_at_k(gold, retrieved, 3) == 0.5


def test_empty_gold_is_not_a_hit() -> None:
    """不可回答问题没有gold evidence，不能算成检索命中。"""
    assert recall_at_k([], ["chunk-a"], 5) == 0.0
    assert hit_rate_at_k([], ["chunk-a"], 5) == 0.0
    assert reciprocal_rank_at_k([], ["chunk-a"], 5) == 0.0