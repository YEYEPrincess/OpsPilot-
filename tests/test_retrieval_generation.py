import json

from generation.model_client import MockGenerationClient
from generation.prompt import build_prompt
from generation.rag_pipeline import RAGPipeline
from retrieval.embeddings import HashEmbeddingProvider
from retrieval.qdrant_store import QdrantVectorStore


def test_hash_embedding_is_normalized_and_deterministic():
    provider = HashEmbeddingProvider(dimension=32)
    first = provider.encode(["CUDA out of memory"])[0]
    second = provider.encode(["CUDA out of memory"])[0]
    assert first == second
    assert len(first) == 32
    assert abs(sum(value * value for value in first) - 1) < 1e-6


def test_qdrant_store_round_trip(tmp_path):
    store = QdrantVectorStore(tmp_path / "qdrant", "test", 32)
    store.recreate()
    provider = HashEmbeddingProvider(dimension=32)
    payload = {
        "chunk_id": "chunk-1",
        "text": "CUDA memory troubleshooting",
        "source_url": "https://example.test/docs",
        "section_path": ["GPU"],
    }
    store.upsert(provider.encode([payload["text"]]), [payload])
    hits = store.search(provider.encode(["CUDA memory"])[0], limit=1)
    store.close()
    assert len(hits) == 1
    assert hits[0].payload["chunk_id"] == "chunk-1"


def test_prompt_and_mock_rag_are_structured(tmp_path):
    store = QdrantVectorStore(tmp_path / "qdrant", "test", 32)
    store.recreate()
    provider = HashEmbeddingProvider(dimension=32)
    payload = {
        "chunk_id": "chunk-1",
        "text": "Check the CUDA driver and PyTorch compatibility.",
        "source_url": "https://example.test/docs",
        "section_path": ["CUDA"],
    }
    store.upsert(provider.encode([payload["text"]]), [payload])
    prompt = build_prompt("Why is CUDA unavailable?", [{**payload, "score": 0.9}])
    assert "S1" in prompt and "citations" in prompt
    result = RAGPipeline(provider, store, MockGenerationClient(), 1).answer(
        "Why is CUDA unavailable?"
    )
    store.close()
    json.dumps(result, ensure_ascii=False)
    assert result["status"] == "ok"
    assert result["sources"][0]["chunk_id"] == "chunk-1"


def test_invalid_generation_json_degrades_to_evidence(tmp_path):
    class BadClient:
        def generate(self, prompt, evidence):
            del prompt, evidence
            return "not-json"

    store = QdrantVectorStore(tmp_path / "qdrant", "test", 32)
    store.recreate()
    provider = HashEmbeddingProvider(dimension=32)
    payload = {"chunk_id": "chunk-1", "text": "CUDA driver", "source_url": "https://example.test"}
    store.upsert(provider.encode([payload["text"]]), [payload])
    result = RAGPipeline(provider, store, BadClient(), 1).answer("CUDA driver")
    store.close()
    assert result["status"] == "degraded"
    assert result["sources"][0]["chunk_id"] == "chunk-1"
