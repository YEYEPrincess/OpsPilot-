"""Public-demo limits and real local HTTP wiring, without external model calls."""
import asyncio
import sqlite3
from concurrent.futures import ThreadPoolExecutor

import httpx
import pytest

from deploy.community_cloud import runtime
from generation.async_client import ModelChunk, ModelError


def test_quota_is_atomic_and_survives_new_instance(tmp_path):
    path = tmp_path / "quota.sqlite3"
    quota = runtime.GenerationQuota(path, daily=3, per_minute=10, concurrent=10,
                                    clock=lambda: 1000)

    def attempt(_):
        try:
            quota.start()
        except ModelError as exc:
            return exc.code
        quota.finish()
        return "allowed"

    with ThreadPoolExecutor(max_workers=8) as workers:
        results = list(workers.map(attempt, range(20)))
    assert results.count("allowed") == 3
    assert results.count("PUBLIC_DAILY_LIMIT") == 17
    restarted = runtime.GenerationQuota(path, daily=3, clock=lambda: 1000)
    with pytest.raises(ModelError, match="PUBLIC_DAILY_LIMIT"):
        restarted.start()
    next_day = runtime.GenerationQuota(path, daily=3, clock=lambda: 86401)
    next_day.start()
    next_day.finish()


def test_busy_rejection_does_not_consume_allowance(tmp_path):
    quota = runtime.GenerationQuota(tmp_path / "quota.sqlite3", concurrent=1)
    quota.start()
    with pytest.raises(ModelError, match="PUBLIC_BUSY"):
        quota.start()
    quota.finish()
    with sqlite3.connect(quota.path) as db:
        assert db.execute("SELECT n FROM calls WHERE kind='day'").fetchone()[0] == 1


def test_cancelled_or_failed_generation_releases_slot(tmp_path):
    class Model:
        async def stream(self, _):
            yield ModelChunk(text="partial")
            raise RuntimeError("upstream failed")

    async def run():
        quota = runtime.GenerationQuota(tmp_path / "quota.sqlite3", concurrent=1)
        model = runtime.BudgetedModel(Model(), quota)
        stream = model.stream([])
        await anext(stream)
        assert quota.active == 1
        await stream.aclose()
        assert quota.active == 0
        with pytest.raises(RuntimeError, match="upstream failed"):
            async for _ in model.stream([]):
                pass
        assert quota.active == 0
    asyncio.run(run())


def test_full_corpus_http_stream_and_pre_generation_limit(tmp_path, monkeypatch):
    calls = []

    class Model:
        def __init__(self, settings):
            assert settings.generation_retries == 0
            assert settings.retrieval_mode == "baseline"
            assert settings.llm_api_key.get_secret_value() == "test-key"

        async def stream(self, messages):
            calls.append(messages)
            yield ModelChunk(text="使用 docker logs 查看日志 [S1]。", usage={"total_tokens": 10})

        async def health(self):
            return True

        async def aclose(self):
            pass

    monkeypatch.setattr(runtime, "AsyncModelClient", Model)
    monkeypatch.setattr(runtime.tempfile, "gettempdir", lambda: str(tmp_path))
    server = runtime.CloudRuntime("test-key", "test-model")
    server.quota.per_minute = 1
    try:
        with httpx.Client(base_url=server.url, trust_env=False, timeout=30) as client:
            assert client.get("/health/ready").status_code == 200
            assert server.service.retriever.store.count() == 1042
            first = client.post("/v1/query/stream", json={
                "question": "docker logs", "top_k": 3,
            })
            assert "event: token" in first.text and "event: done" in first.text
            assert '"citation_valid": true' in first.text
            second = client.post("/v1/query", json={"question": "docker logs", "top_k": 3})
            assert second.json()["cache_hit"] is True
            limited = client.post("/v1/query", json={
                "question": "docker logs", "top_k": 3, "use_cache": False,
            }).json()
            assert limited["error_code"] == "PUBLIC_MINUTE_LIMIT"
            assert limited["status"] == "refuse"
            assert len(calls) == 1
            clarified = client.post("/v1/query", json={"question": "CUDA 有问题怎么处理？"})
            assert clarified.json()["status"] == "clarify"
            assert len(calls) == 1
    finally:
        server.close()
    assert not server.thread.is_alive()
