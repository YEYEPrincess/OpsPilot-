from retrieval.bm25_store import BM25Hit, tokenize
from retrieval.hybrid_search import reciprocal_rank_fusion
from retrieval.qdrant_store import SearchHit


def payload(chunk_id: str) -> dict[str, str]:
    return {"chunk_id": chunk_id, "text": chunk_id}


def test_tokenizer_preserves_technical_identifier() -> None:
    assert "torch.cuda.is_available" in tokenize("Use torch.cuda.is_available now")


def test_rrf_rewards_results_found_by_both_retrievers() -> None:
    dense = [
        SearchHit(score=0.8, payload=payload("dense-only")),
        SearchHit(score=0.7, payload=payload("shared")),#shared表示融合
    ]
    bm25 = [
        BM25Hit(score=9.0, payload=payload("bm25-only")),
        BM25Hit(score=8.0, payload=payload("shared")),
    ]

    fused = reciprocal_rank_fusion(dense, bm25, limit=3, rrf_k=60)  #执行RRF融合
# → 取融合后的第一名
# → 检查第一名的Chunk ID
# → 必须等于shared
    assert fused[0].payload["chunk_id"] == "shared"