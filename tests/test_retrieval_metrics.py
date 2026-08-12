from evaluation.retrieval_metrics import hit_rate_at_k, recall_at_k, reciprocal_rank_at_k


def test_recall_at_k_handles_multiple_gold_chunks() -> None:
    assert recall_at_k(["a", "b"], ["x", "a", "b"], 2) == 0.5
    assert recall_at_k(["a", "b"], ["x", "a", "b"], 3) == 1.0


def test_hit_rate_and_mrr_use_first_relevant_rank() -> None:
    retrieved = ["noise", "gold", "other"]
    assert hit_rate_at_k(["gold"], retrieved, 2) == 1.0
    assert reciprocal_rank_at_k(["gold"], retrieved, 3) == 0.5


def test_empty_gold_is_not_counted_as_a_hit() -> None:
    assert recall_at_k([], ["a"], 5) == 0.0
    assert hit_rate_at_k([], ["a"], 5) == 0.0
    assert reciprocal_rank_at_k([], ["a"], 5) == 0.0
