"""One lightweight local API per Streamlit app; no GPU or external database."""
from __future__ import annotations

import atexit
import gzip
import hashlib
import json
import socket
import sqlite3
import tempfile
import threading
import time
from contextlib import aclosing
from pathlib import Path

import httpx
import uvicorn

from app.main import create_app
from app.model_service import ModelQueryService
from core.model_settings import ModelSettings
from generation.async_client import AsyncModelClient, ModelError
from retrieval.embeddings import HashEmbeddingProvider
from retrieval.qdrant_store import QdrantVectorStore
from scripts.seed_qdrant_service import seed

BUNDLE = Path(__file__).resolve().parent


class GenerationQuota:
    """Count attempted generations globally, including failures, before billing.

    SQLite survives a Python restart on the same filesystem. Platform rebuilds
    may erase it; this is a demo limiter, not a provider-side billing ceiling.
    """

    def __init__(self, path: Path, *, daily=50, per_minute=4, concurrent=2, clock=time.time):
        self.path = path
        self.daily, self.per_minute, self.concurrent = daily, per_minute, concurrent
        self.clock = clock
        self.lock = threading.Lock()
        self.active = 0
        path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(path) as db:
            db.execute("CREATE TABLE IF NOT EXISTS calls "
                       "(kind TEXT, bucket INTEGER, n INTEGER, PRIMARY KEY(kind, bucket))")

    def start(self):
        with self.lock:
            if self.active >= self.concurrent:
                raise ModelError("PUBLIC_BUSY")
            now = int(self.clock())
            with sqlite3.connect(self.path, timeout=5) as db:
                db.execute("BEGIN IMMEDIATE")
                for kind, bucket, limit in [
                    ("day", now // 86400, self.daily),
                    ("minute", now // 60, self.per_minute),
                ]:
                    row = db.execute("SELECT n FROM calls WHERE kind=? AND bucket=?",
                                     (kind, bucket)).fetchone()
                    if row and row[0] >= limit:
                        raise ModelError("PUBLIC_DAILY_LIMIT" if kind == "day"
                                         else "PUBLIC_MINUTE_LIMIT")
                for kind, bucket in [("day", now // 86400), ("minute", now // 60)]:
                    db.execute("INSERT INTO calls VALUES (?, ?, 1) "
                               "ON CONFLICT(kind, bucket) DO UPDATE SET n=n+1", (kind, bucket))
                db.execute("DELETE FROM calls WHERE (kind='day' AND bucket<?) "
                           "OR (kind='minute' AND bucket<?)",
                           (now // 86400 - 7, now // 60 - 120))
            self.active += 1

    def finish(self):
        with self.lock:
            self.active -= 1


class BudgetedModel:
    def __init__(self, client, quota: GenerationQuota):
        self.client, self.quota = client, quota

    async def stream(self, messages):
        self.quota.start()
        try:
            async with aclosing(self.client.stream(messages)) as chunks:
                async for chunk in chunks:
                    yield chunk
        finally:
            self.quota.finish()

    async def health(self):
        return await self.client.health()

    async def aclose(self):
        await self.client.aclose()


def prepare_index(directory: Path) -> Path:
    """Build from the shipped corpus; no runtime document/model download."""
    directory.mkdir(parents=True, exist_ok=True)
    metadata = json.loads((BUNDLE / "knowledge_manifest.json").read_text(encoding="utf-8"))
    raw = gzip.decompress((BUNDLE / "knowledge.jsonl.gz").read_bytes())
    if hashlib.sha256(raw).hexdigest() != metadata["sha256"]:
        raise ValueError("Knowledge bundle checksum mismatch")
    chunks = [json.loads(line) for line in raw.decode("utf-8").splitlines() if line.strip()]
    if len(chunks) != metadata["chunks"]:
        raise ValueError("Knowledge bundle count mismatch")
    index_path = directory / ("index-" + metadata["sha256"][:12])
    collection = "opspilot_public_v1"
    encoder = HashEmbeddingProvider(metadata["dimension"])
    store = QdrantVectorStore(index_path, collection, encoder.dimension)
    try:
        seed(store, chunks, encoder)
    finally:
        store.close()
    manifest = directory / "vector_index.json"
    manifest.write_text(json.dumps({
        "provider": "hash-v1", "dimension": encoder.dimension,
        "collection": collection, "qdrant_path": str(index_path.resolve()),
        "points": len(chunks),
    }), encoding="utf-8")
    return manifest


class CloudRuntime:
    def __init__(self, api_key: str, model: str):
        if not api_key.strip() or not model.strip():
            raise ValueError("Configure LLM_API_KEY and LLM_MODEL in platform secrets")
        directory = Path(tempfile.gettempdir()) / "opspilot-public-demo"
        manifest = prepare_index(directory)
        # Explicit values and _env_file=None prevent accidental use of a local .env.
        settings = ModelSettings(
            _env_file=None, opspilot_mode="model", llm_provider="openai-compatible",
            llm_base_url="https://api.deepseek.com", llm_api_key=api_key, llm_model=model,
            llm_thinking="disabled", llm_stream_usage=True,
            generation_max_tokens=512, generation_retries=0,
            generation_timeout_seconds=45, request_timeout_seconds=55,
            retrieval_mode="baseline", clarification_policy="context_v1",
            grounding_min_score=0.35, vector_manifest=manifest, qdrant_remote_url="",
            answer_cache_max_entries=64, answer_cache_ttl_seconds=300,
            event_log_path=None, log_json_stdout=False,
        )
        self.quota = GenerationQuota(directory / "generation_quota.sqlite3")
        self.service = ModelQueryService(settings, model=BudgetedModel(
            AsyncModelClient(settings), self.quota))
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.bind(("127.0.0.1", 0))
        self.url = f"http://127.0.0.1:{self.socket.getsockname()[1]}"
        application = create_app(settings=settings, query_service=self.service)
        # The generic factory does not own injected services. Close ours on shutdown.
        from contextlib import asynccontextmanager

        @asynccontextmanager
        async def lifespan(_app):
            try:
                yield
            finally:
                await self.service.aclose()

        application.router.lifespan_context = lifespan
        self.server = uvicorn.Server(uvicorn.Config(
            application, log_level="warning", access_log=False,
        ))
        self.thread = threading.Thread(
            target=self.server.run, kwargs={"sockets": [self.socket]}, daemon=True,
        )
        self.thread.start()
        atexit.register(self.close)
        for _ in range(100):
            if not self.thread.is_alive():
                break
            try:
                with httpx.Client(trust_env=False, timeout=0.5) as client:
                    if client.get(self.url + "/health/live").status_code == 200:
                        return
            except httpx.HTTPError:
                pass
            time.sleep(0.1)
        self.close()
        raise RuntimeError("Local API startup failed")

    def close(self):
        self.server.should_exit = True
        self.thread.join(timeout=5)
        self.socket.close()
