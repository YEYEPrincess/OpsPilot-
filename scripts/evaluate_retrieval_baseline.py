"""Evaluate dense retrieval against Day 6 gold evidence annotations.

The baseline deliberately calls Qdrant without the optional lexical tie-breaker
(``query_text=""``), so the reported result isolates the configured embedding
and vector index.  It evaluates only ``answerable`` records because the other
records intentionally have no gold evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evaluation.retrieval_metrics import (  # noqa: E402
    hit_rate_at_k,
    recall_at_k,
    reciprocal_rank_at_k,
)
from retrieval.embeddings import create_embedding_provider, vector_metadata  # noqa: E402
from retrieval.qdrant_store import QdrantVectorStore  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval", type=Path, default=ROOT / "data/eval/day6_eval.jsonl")
    parser.add_argument("--chunks", type=Path, default=ROOT / "data/processed/chunks_section.jsonl")
    parser.add_argument("--gold-chunks", type=Path, default=None,
                        help="Chunk file containing annotation IDs; defaults to --chunks")
    parser.add_argument("--qdrant-path", type=Path, default=ROOT / "data/index/qdrant")
    parser.add_argument("--collection", default="opspilot_chunks_v1")
    parser.add_argument("--provider", choices=["hash", "sentence-transformers"], default="hash")
    parser.add_argument("--model", default="")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dimension", type=int, default=384)
    parser.add_argument("--top-k", default="1,3,5,10", help="Comma-separated cutoffs to report")
    parser.add_argument("--analysis-k", type=int, default=5)
    parser.add_argument("--vector-manifest", type=Path, default=None)
    parser.add_argument(
        "--results",
        type=Path,
        default=ROOT / "data/eval/day7_retrieval_baseline.json",
    )
    parser.add_argument(
        "--errors",
        type=Path,
        default=ROOT / "data/eval/day7_retrieval_errors.jsonl",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "data/eval/day7_experiment_config.json",
    )
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_revision() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def repo_relative(path: Path) -> str:
    resolved = path if path.is_absolute() else ROOT / path
    return str(resolved.resolve().relative_to(ROOT))

def file_fingerprint(path: Path) -> dict[str, Any]:
    resolved = path if path.is_absolute() else ROOT / path
    return {
        "path": str(resolved.relative_to(ROOT)) if resolved.is_relative_to(ROOT) else str(resolved),
        "bytes": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
    }


def chunk_summary(chunks: list[dict[str, Any]]) -> dict[str, Any]:
    lengths = [int(chunk.get("char_count") or len(chunk.get("text", ""))) for chunk in chunks]
    return {
        "strategy": chunks[0].get("strategy") if chunks else "unknown",
        "count": len(chunks),
        "min_chars": min(lengths) if lengths else 0,
        "max_chars": max(lengths) if lengths else 0,
        "mean_chars": round(sum(lengths) / len(lengths), 2) if lengths else 0,
    }



def build_gold_mapping(
    gold_chunks: list[dict[str, Any]], query_chunks: list[dict[str, Any]]
) -> dict[str, list[str]]:
    """Map annotation IDs into the query index ID space."""
    query_by_id = {str(chunk["chunk_id"]): chunk for chunk in query_chunks}
    mapping: dict[str, list[str]] = {}
    for gold in gold_chunks:
        gold_id = str(gold["chunk_id"])
        if gold_id in query_by_id:
            mapping[gold_id] = [gold_id]
            continue
        target_path = gold.get("section_path") or []
        mapping[gold_id] = [
            str(chunk["chunk_id"])
            for chunk in query_chunks
            if chunk.get("document_id") == gold.get("document_id")
            and target_path in (chunk.get("section_paths") or [])
        ]
    return mapping
def aggregate(rows: list[dict[str, Any]], k: int) -> dict[str, float]:
    return {
        "recall_at_k": round(sum(row["recall_at_k"][str(k)] for row in rows) / len(rows), 4),
        "hit_rate_at_k": round(sum(row["hit_rate_at_k"][str(k)] for row in rows) / len(rows), 4),
        "mrr_at_k": round(sum(row["mrr_at_k"][str(k)] for row in rows) / len(rows), 4),
    }


def main() -> int:
    args = parse_args()
    cutoffs = sorted({int(value.strip()) for value in args.top_k.split(",") if value.strip()})
    if not cutoffs or any(value <= 0 for value in cutoffs):
        raise ValueError("--top-k must contain positive integers")
    if args.analysis_k not in cutoffs:
        cutoffs.append(args.analysis_k)
        cutoffs.sort()
    max_k = max(cutoffs)
    qdrant_path = args.qdrant_path if args.qdrant_path.is_absolute() else ROOT / args.qdrant_path
    qdrant_path = qdrant_path.resolve()

    records = load_jsonl(args.eval)
    chunks = load_jsonl(args.chunks)
    gold_chunks = load_jsonl(args.gold_chunks or args.chunks)
    gold_mapping = build_gold_mapping(gold_chunks, chunks)
    answerable = [record for record in records if record.get("answerability") == "answerable"]
    provider = create_embedding_provider(args.provider, args.model, args.dimension, args.device)
    store = QdrantVectorStore(qdrant_path, args.collection, provider.dimension)
    rows: list[dict[str, Any]] = []
    try:
        for record in answerable:
            gold_ids = [
                mapped_id
                for item in record.get("gold_evidence", [])
                for mapped_id in gold_mapping.get(str(item["chunk_id"]), [])
            ]
            vector = provider.encode([record["question"]])[0]
            # Empty query_text disables the optional lexical rerank in QdrantVectorStore.
            hits = store.search(vector, limit=max_k, query_text="")
            retrieved_ids = [str(hit.payload.get("chunk_id")) for hit in hits]
            row = {
                "id": record["id"],
                "question": record["question"],
                "domain": record["domain"],
                "split": record["split"],
                "gold_ids": gold_ids,
                "retrieved_ids": retrieved_ids,
                "retrieved_scores": [round(hit.score, 6) for hit in hits],
                "recall_at_k": {
                    str(k): recall_at_k(gold_ids, retrieved_ids, k) for k in cutoffs
                },
                "hit_rate_at_k": {
                    str(k): hit_rate_at_k(gold_ids, retrieved_ids, k) for k in cutoffs
                },
                "mrr_at_k": {
                    str(k): reciprocal_rank_at_k(gold_ids, retrieved_ids, k) for k in cutoffs
                },
            }
            rows.append(row)
    finally:
        store.close()

    by_split = {
        split: {
            str(k): aggregate([row for row in rows if row["split"] == split], k)
            for k in cutoffs
        }
        for split in sorted({row["split"] for row in rows})
    }
    metrics = {str(k): aggregate(rows, k) for k in cutoffs}
    analysis_k = args.analysis_k
    errors: list[dict[str, Any]] = []
    chunk_by_id = {str(chunk["chunk_id"]): chunk for chunk in chunks}
    for row in rows:
        first_relevant_rank = next(
            (rank for rank, chunk_id in enumerate(row["retrieved_ids"], 1)
             if chunk_id in row["gold_ids"]),
            None,
        )
        if first_relevant_rank is None or first_relevant_rank > analysis_k:
            errors.append(
                {
                    "id": row["id"],
                    "split": row["split"],
                    "domain": row["domain"],
                    "question": row["question"],
                    "gold_ids": row["gold_ids"],
                    "retrieved_ids_at_k": row["retrieved_ids"][:analysis_k],
                    "first_relevant_rank": first_relevant_rank,
                    "retrieved_products_at_k": [
                        chunk_by_id[chunk_id].get("product")
                        for chunk_id in row["retrieved_ids"][:analysis_k]
                        if chunk_id in chunk_by_id
                    ],
                    "error_type": (
                        "no_gold_in_top_k"
                        if first_relevant_rank is None
                        else "late_gold_rank"
                    ),
                }
            )

    manifest_path = args.vector_manifest
    if manifest_path is None and args.collection == "opspilot_chunks_v1":
        manifest_path = ROOT / "data/manifest/vector_index.json"
    storage_path = qdrant_path / "collection" / args.collection / "storage.sqlite"
    config = {
        "experiment": "day7-retrieval-baseline",
        "generated_at": datetime.now(UTC).isoformat(),
        "git_revision": git_revision(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "qdrant_client": importlib.metadata.version("qdrant-client"),
        "retrieval_mode": "dense_vector_only",
        "provider": vector_metadata(provider),
        "collection": args.collection,
        "qdrant_path": str(qdrant_path.relative_to(ROOT)),
        "top_k": cutoffs,
        "analysis_k": analysis_k,
        "data": {
            "eval": file_fingerprint(args.eval),
            "chunks": file_fingerprint(args.chunks),
            "gold_chunks": file_fingerprint(args.gold_chunks or args.chunks),
            "vector_manifest": (
                file_fingerprint(manifest_path)
                if manifest_path and manifest_path.exists()
                else None
            ),
            "qdrant_storage": file_fingerprint(storage_path) if storage_path.exists() else None,
            "chunk_summary": chunk_summary(chunks),
            "gold_mapping": {
                "mapped": sum(bool(value) for value in gold_mapping.values()),
                "total": len(gold_mapping),
                "strategy": "same-id or document_id + exact section_path",
            },
        },
        "excluded_records": {
            "total": len(records) - len(answerable),
            "reason": "Only answerable records have gold evidence for retrieval metrics.",
        },
    }
    result = {
        "status": "ok",
        "experiment": "day7-retrieval-baseline",
        "records_total": len(records),
        "records_evaluated": len(rows),
        "metrics": metrics,
        "by_split": by_split,
        "error_summary": {
            "analysis_k": analysis_k,
            "errors": len(errors),
            "no_gold_in_top_k": sum(item["error_type"] == "no_gold_in_top_k" for item in errors),
            "late_gold_rank": sum(item["error_type"] == "late_gold_rank" for item in errors),
        },
        "config_path": repo_relative(args.config),
        "errors_path": repo_relative(args.errors),
    }

    for path, payload in ((args.config, config), (args.results, result)):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    args.errors.parent.mkdir(parents=True, exist_ok=True)
    args.errors.write_text(
        "".join(json.dumps(error, ensure_ascii=False) + "\n" for error in errors), encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
