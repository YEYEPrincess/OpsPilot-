"""从 section Chunk 建立 BM25 索引。"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from retrieval.bm25_store import BM25Index  # noqa: E402


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--chunks",
        type=Path,
        default=ROOT / "data/processed/chunks_section.jsonl",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "data/index/bm25/opspilot_bm25_v1.json.gz",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=ROOT / "data/manifest/bm25_index.json",
    )
    args = parser.parse_args()

    chunks = [
        json.loads(line)
        for line in args.chunks.read_text(encoding="utf-8").splitlines()
        if line
    ]
    index = BM25Index.from_chunks(chunks)
    index.save(args.output)

    manifest = {
        "generated_at": datetime.now(UTC).isoformat(),
        "index_version": "bm25-v1",
        "source_chunks": str(args.chunks.relative_to(ROOT)),
        "source_sha256": sha256_file(args.chunks),
        "documents": len(chunks),
        "index_path": str(args.output.relative_to(ROOT)),
        "index_sha256": sha256_file(args.output),
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())