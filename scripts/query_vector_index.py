"""Query the Day 4 Qdrant index and print source-aware Top-k results."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Allow direct execution with `python scripts/<file>.py`.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from retrieval.embeddings import create_embedding_provider  # noqa: E402
from retrieval.qdrant_store import QdrantVectorStore  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("question")
    parser.add_argument("--qdrant-path", type=Path, default=ROOT / "data/index/qdrant")
    parser.add_argument("--collection", default="opspilot_chunks_v1")
    parser.add_argument("--provider", choices=["hash", "sentence-transformers"], default="hash")
    parser.add_argument("--model", default="")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dimension", type=int, default=384)
    parser.add_argument("--top-k", type=int, default=5)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    provider = create_embedding_provider(args.provider, args.model, args.dimension, args.device)
    store = QdrantVectorStore(args.qdrant_path, args.collection, provider.dimension)
    try:
        hits = store.search(
            provider.encode([args.question])[0], args.top_k, query_text=args.question
        )
    finally:
        store.close()
    print(
        json.dumps(
            [{"score": hit.score, **hit.payload} for hit in hits],
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
