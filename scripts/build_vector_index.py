"""Build a versioned Qdrant collection from section-aware Chunk JSONL."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

# Allow direct execution with `python scripts/<file>.py`.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from retrieval.embeddings import create_embedding_provider, vector_metadata  # noqa: E402
from retrieval.qdrant_store import QdrantVectorStore  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chunks", type=Path, default=ROOT / "data/processed/chunks_section.jsonl")
    parser.add_argument("--qdrant-path", type=Path, default=ROOT / "data/index/qdrant")
    parser.add_argument("--collection", default="opspilot_chunks_v1")
    parser.add_argument("--provider", choices=["hash", "sentence-transformers"], default="hash")
    parser.add_argument("--model", default="")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dimension", type=int, default=384)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--recreate", action="store_true")
    parser.add_argument("--manifest", type=Path, default=ROOT / "data/manifest/vector_index.json")
    return parser.parse_args()


def load_chunks(path: Path) -> list[dict[str, Any]]:
    """Read one JSON object per line and reject empty or duplicate Chunk ids."""
    chunks = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    ids = [str(chunk["chunk_id"]) for chunk in chunks]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate chunk_id found")
    return chunks


def main() -> int:
    args = parse_args()
    chunks = load_chunks(args.chunks)
    provider = create_embedding_provider(args.provider, args.model, args.dimension, args.device)
    store = QdrantVectorStore(args.qdrant_path, args.collection, provider.dimension)
    try:
        if args.recreate:
            store.recreate()
        else:
            store.ensure_collection()
        for start in range(0, len(chunks), args.batch_size):
            batch = chunks[start : start + args.batch_size]
            vectors = provider.encode([chunk["text"] for chunk in batch], args.batch_size)
            store.upsert(vectors, batch)
            print(f"indexed {min(start + len(batch), len(chunks))}/{len(chunks)}")
        manifest = {
            "generated_at": datetime.now(UTC).isoformat(),
            "index_version": "v1",
            "collection": args.collection,
            "qdrant_path": str(args.qdrant_path.relative_to(ROOT)),
            "source_chunks": str(args.chunks.relative_to(ROOT)),
            "points": store.count(),
            "batch_size": args.batch_size,
            **vector_metadata(provider),
        }
    finally:
        store.close()
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
