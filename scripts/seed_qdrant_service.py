"""Seed a NEW remote hash collection; refuse to overwrite a different dataset."""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from qdrant_client.models import Distance  # noqa: E402

from retrieval.embeddings import HashEmbeddingProvider  # noqa: E402
from retrieval.qdrant_store import QdrantVectorStore  # noqa: E402


def seed(store, chunks, encoder):
    if not chunks:
        raise ValueError("Chunk input is empty")
    ids = [str(row["chunk_id"]) for row in chunks]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate chunk IDs")
    # Verify existing content before any write; retries/restarts are idempotent.
    store.ensure_collection()
    info = store.client.get_collection(store.collection)
    vector_config = info.config.params.vectors
    if vector_config.size != encoder.dimension or vector_config.distance != Distance.COSINE:
        raise ValueError("Existing collection vector configuration differs")
    expected = {
        str(row["chunk_id"]): hashlib.sha256(
            json.dumps(row, sort_keys=True, ensure_ascii=False).encode()
        ).hexdigest()
        for row in chunks
    }
    offset = None
    while True:
        rows, offset = store.client.scroll(
            collection_name=store.collection, limit=100, offset=offset, with_payload=True
        )
        for row in rows:
            payload = row.payload or {}
            digest = hashlib.sha256(
                json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()
            ).hexdigest()
            if expected.get(str(payload.get("chunk_id"))) != digest:
                raise ValueError("Collection has different data; choose a new collection")
        if offset is None:
            break
    for start in range(0, len(chunks), 32):
        batch = chunks[start:start + 32]
        store.upsert(encoder.encode([row["text"] for row in batch]), batch)
    if store.count() != len(chunks):
        raise ValueError("Indexed count does not match input")
    return store.count()


def main():
    manifest = json.loads(Path(os.environ["VECTOR_MANIFEST"]).read_text(encoding="utf-8"))
    if manifest["provider"] != "hash-v1":
        raise ValueError("Only hash-v1 manifests are supported")
    chunks = [
        json.loads(line) for line in Path(os.environ["CHUNKS_PATH"]).read_text(
            encoding="utf-8"
        ).splitlines() if line.strip()
    ]
    encoder = HashEmbeddingProvider(dimension=int(manifest["dimension"]))
    store = QdrantVectorStore(
        None, manifest["collection"], encoder.dimension,
        url=os.environ["QDRANT_REMOTE_URL"], api_key=os.getenv("QDRANT_API_KEY"),
    )
    try:
        deadline = time.monotonic() + 60
        while True:
            try:
                store.client.get_collections()
                break
            except Exception:
                if time.monotonic() >= deadline:
                    raise RuntimeError("QDRANT_NOT_READY") from None
                time.sleep(1)
        print(json.dumps({"indexed_points": seed(store, chunks, encoder)}))
    finally:
        store.close()


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        # No HTTP exception body or credential is written to container logs.
        print(json.dumps({"seed_failed": type(exc).__name__}), file=sys.stderr)
        raise SystemExit(1) from None
