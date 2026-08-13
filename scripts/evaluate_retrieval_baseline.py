"""运行OpsPilot检索基线评测。

输入：
1. Day 6评测问题；
2. gold evidence；
3. 已构建的Qdrant索引。

输出：
1. Recall@k、Hit Rate@k、MRR@k；
2. Top-5错误案例；
3. 实验配置和输入文件指纹。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

# 项目根目录
ROOT = Path(__file__).resolve().parents[1]

# 允许直接运行python scripts/evaluate_retrieval_baseline.py
sys.path.insert(0, str(ROOT))

from evaluation.retrieval_metrics import (  # noqa: E402
    hit_rate_at_k,
    recall_at_k,
    reciprocal_rank_at_k,
)
from retrieval.embeddings import create_embedding_provider  # noqa: E402
from retrieval.qdrant_store import QdrantVectorStore  # noqa: E402


def parse_args() -> argparse.Namespace:
    """定义命令行参数。"""
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--eval",
        type=Path,
        default=ROOT / "data/eval/day6_eval.jsonl",
        help="Day 6评测集",
    )
    parser.add_argument(
        "--chunks",
        type=Path,
        default=ROOT / "data/processed/chunks_section.jsonl",
        help="建立当前索引所使用的Chunk文件",
    )
    parser.add_argument(
        "--qdrant-path",
        type=Path,
        default=ROOT / "data/index/qdrant",
        help="本地Qdrant存储路径",
    )
    parser.add_argument(
        "--collection",
        default="opspilot_chunks_v1",
        help="Qdrant集合名称",
    )
    parser.add_argument(
        "--provider",
        choices=["hash", "sentence-transformers"],
        default="hash",
        help="必须与建库时使用的Embedding Provider一致",
    )
    parser.add_argument("--model", default="")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dimension", type=int, default=384)

    parser.add_argument(
        "--top-k",
        default="1,3,5,10",
        help="要统计的多个Top-k，用逗号分隔",
    )
    parser.add_argument(
        "--analysis-k",
        type=int,
        default=5,
        help="错误分析使用的k",
    )

    parser.add_argument(
        "--result",
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


def resolve_path(path: Path) -> Path:
    """把相对路径转换为项目中的绝对路径。"""
    if path.is_absolute():
        return path.resolve()

    return (ROOT / path).resolve()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    """读取JSONL文件。"""
    records: list[dict[str, Any]] = []

    with path.open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue

            value = json.loads(line)

            if not isinstance(value, dict):
                raise ValueError(
                    f"{path}:{line_number}不是JSON对象"
                )

            records.append(value)

    return records


def sha256_file(path: Path) -> str:
    """计算文件SHA-256，用于确认实验输入是否改变。"""
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)

    return digest.hexdigest()


def file_fingerprint(path: Path) -> dict[str, Any]:
    """记录文件路径、大小和SHA-256。"""
    return {
        "path": str(path.relative_to(ROOT)),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def git_revision() -> str:
    """记录当前Git commit。"""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def average(values: list[float]) -> float:
    """计算平均值并保留4位小数。"""
    if not values:
        return 0.0

    return round(sum(values) / len(values), 4)


def aggregate_metrics(
    rows: list[dict[str, Any]],
    cutoffs: list[int],
) -> dict[str, dict[str, float]]:
    """把单题指标聚合成整个评测集的平均指标。"""
    return {
        str(k): {
            "recall_at_k": average(
                [row["recall_at_k"][str(k)] for row in rows]
            ),
            "hit_rate_at_k": average(
                [row["hit_rate_at_k"][str(k)] for row in rows]
            ),
            "mrr_at_k": average(
                [row["mrr_at_k"][str(k)] for row in rows]
            ),
        }
        for k in cutoffs
    }


def main() -> int:
    args = parse_args()

    eval_path = resolve_path(args.eval)
    chunks_path = resolve_path(args.chunks)
    qdrant_path = resolve_path(args.qdrant_path)
    result_path = resolve_path(args.result)
    errors_path = resolve_path(args.errors)
    config_path = resolve_path(args.config)

    # 把"1,3,5,10"转换成整数列表。
    cutoffs = sorted({
        int(value.strip())
        for value in args.top_k.split(",")
        if value.strip()
    })

    if not cutoffs or any(k <= 0 for k in cutoffs):
        raise ValueError("--top-k必须是正整数，例如1,3,5,10")

    # 必须至少检索到最大的k。
    max_k = max(max(cutoffs), args.analysis_k)

    eval_records = load_jsonl(eval_path)
    chunks = load_jsonl(chunks_path)

    # 只有answerable问题有完整gold evidence。
    answerable_records = [
        record
        for record in eval_records
        if record.get("answerability") == "answerable"
    ]

    if not answerable_records:
        raise ValueError("评测集中没有answerable记录")

    # Embedding配置必须和构建向量索引时一致。
    provider = create_embedding_provider(
        provider=args.provider,
        model_name=args.model,
        dimension=args.dimension,
        device=args.device,
    )

    store = QdrantVectorStore(
        path=qdrant_path,
        collection=args.collection,
        vector_size=provider.dimension,
    )

    rows: list[dict[str, Any]] = []

    try:
        for record in answerable_records:
            question = str(record["question"])

            gold_ids = [
                str(evidence["chunk_id"])
                for evidence in record["gold_evidence"]
            ]

            # 将问题编码成查询向量。
            query_vector = provider.encode([question])[0]

            # query_text=""表示关闭项目中的轻量词法重排，
            # 只评估Embedding + 向量检索能力。
            hits = store.search(
                vector=query_vector,
                limit=max_k,
                query_text="",
            )

            retrieved_ids = [
                str(hit.payload["chunk_id"])
                for hit in hits
            ]

            row = {
                "id": record["id"],
                "question": question,
                "domain": record["domain"],
                "split": record["split"],
                "gold_ids": gold_ids,
                "retrieved_ids": retrieved_ids,
                "retrieved_scores": [
                    round(hit.score, 6)
                    for hit in hits
                ],
                "recall_at_k": {
                    str(k): recall_at_k(
                        gold_ids,
                        retrieved_ids,
                        k,
                    )
                    for k in cutoffs
                },
                "hit_rate_at_k": {
                    str(k): hit_rate_at_k(
                        gold_ids,
                        retrieved_ids,
                        k,
                    )
                    for k in cutoffs
                },
                "mrr_at_k": {
                    str(k): reciprocal_rank_at_k(
                        gold_ids,
                        retrieved_ids,
                        k,
                    )
                    for k in cutoffs
                },
            }

            rows.append(row)

    finally:
        store.close()

    # 总体指标
    metrics = aggregate_metrics(rows, cutoffs)

    # 分别查看development、validation、test。
    by_split = {
        split: aggregate_metrics(
            [row for row in rows if row["split"] == split],
            cutoffs,
        )
        for split in sorted({row["split"] for row in rows})
    }

    errors: list[dict[str, Any]] = []

    for row in rows:
        gold = set(row["gold_ids"])

        first_relevant_rank = next(
            (
                rank
                for rank, chunk_id in enumerate(
                    row["retrieved_ids"],
                    start=1,
                )
                if chunk_id in gold
            ),
            None,
        )

        # Top-5没有正确证据，就进入错误分析文件。
        if (
            first_relevant_rank is None
            or first_relevant_rank > args.analysis_k
        ):
            errors.append({
                "id": row["id"],
                "question": row["question"],
                "domain": row["domain"],
                "split": row["split"],
                "gold_ids": row["gold_ids"],
                "retrieved_ids_at_k": (
                    row["retrieved_ids"][:args.analysis_k]
                ),
                "first_relevant_rank": first_relevant_rank,
                "error_type": (
                    "no_gold_in_retrieved_results"
                    if first_relevant_rank is None
                    else "late_gold_rank"
                ),
            })

    result = {
        "status": "ok",
        "records_total": len(eval_records),
        "records_evaluated": len(rows),
        "metrics": metrics,
        "by_split": by_split,
        "error_summary": {
            "analysis_k": args.analysis_k,
            "errors": len(errors),
            "no_gold": sum(
                error["error_type"]
                == "no_gold_in_retrieved_results"
                for error in errors
            ),
            "late_gold_rank": sum(
                error["error_type"] == "late_gold_rank"
                for error in errors
            ),
        },
        # 保留单题结果，方便后续排查。
        "per_query": rows,
    }

    config = {
        "experiment": "day7-retrieval-baseline",
        "generated_at": datetime.now(UTC).isoformat(),
        "git_revision": git_revision(),
        "python": platform.python_version(),
        "retrieval_mode": "dense_vector_only",
        "embedding": {
            "provider": provider.name,
            "dimension": provider.dimension,
            "model": args.model or None,
            "device": args.device,
        },
        "qdrant": {
            "path": str(qdrant_path.relative_to(ROOT)),
            "collection": args.collection,
        },
        "top_k": cutoffs,
        "analysis_k": args.analysis_k,
        "data": {
            "eval": file_fingerprint(eval_path),
            "chunks": file_fingerprint(chunks_path),
            "chunk_count": len(chunks),
            "chunk_strategy": (
                chunks[0].get("strategy")
                if chunks
                else "unknown"
            ),
        },
        "excluded_records": {
            "count": len(eval_records) - len(rows),
            "reason": (
                "clarification_required和unanswerable"
                "没有完整gold evidence"
            ),
        },
    }

    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    errors_path.write_text(
        "".join(
            json.dumps(error, ensure_ascii=False) + "\n"
            for error in errors
        ),
        encoding="utf-8",
    )

    config_path.write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(json.dumps(
        {
            "status": result["status"],
            "records_total": result["records_total"],
            "records_evaluated": result["records_evaluated"],
            "metrics": result["metrics"],
            "error_summary": result["error_summary"],
        },
        ensure_ascii=False,
        indent=2,
    ))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())