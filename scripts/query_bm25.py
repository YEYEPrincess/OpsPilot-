"""在命令行中测试 BM25 关键词检索。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from retrieval.bm25_store import BM25Index  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("question")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--index",
        type=Path,
        default=ROOT / "data/index/bm25/opspilot_bm25_v1.json.gz",
    )
    args = parser.parse_args()

    index = BM25Index.load(args.index)
    hits = index.search(args.question, limit=args.top_k)

    print(
        json.dumps(
            [
                {
                    "score": hit.score,
                    "chunk_id": hit.payload["chunk_id"],
                    "product": hit.payload.get("product"),
                    "section_path": hit.payload.get("section_path"),
                    "text_preview": hit.payload.get("text", "")[:300],
                }
                for hit in hits
            ],
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())