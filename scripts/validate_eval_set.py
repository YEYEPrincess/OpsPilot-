"""Validate Day 6 evaluation JSONL schema, distributions, and evidence IDs."""
# #这个脚本检查：
# JSONL 格式；
# 必需字段；
# 枚举值；
# ID 是否重复；
# 问题是否重复；
# Chunk ID 是否真实存在；
# 可回答问题是否有 gold evidence；
# 不可回答问题是否错误标注了证据；
# 所有标注是否经过审核；
# 总题数是否达到 50。
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

VALID_CASE_TYPES = {"simple", "complex", "ambiguous", "unanswerable"}
VALID_ANSWERABILITY = {
    "answerable",
    "clarification_required",
    "unanswerable",
}
VALID_SAFETY = {"normal", "caution", "unsafe"}
VALID_SPLITS = {"development", "validation", "test"}
VALID_RELEVANCE = {"direct", "supporting"}

REQUIRED_FIELDS = {
    "id",
    "question",
    "domain",
    "case_type",
    "answerability",
    "expected_products",
    "gold_evidence",
    "gold_points",
    "must_not_claim",
    "required_clarifications",
    "safety",
    "split",
    "kb_version",
    "annotation",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--eval",
        type=Path,
        default=Path("data/eval/day6_eval.jsonl"),
    )
    parser.add_argument(
        "--chunks",
        type=Path,
        default=Path("data/processed/chunks_section.jsonl"),
    )
    parser.add_argument(
        "--min-count",
        type=int,
        default=50,
    )
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []

    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"{path}:{line_number} is invalid JSON: {exc}"
                ) from exc

            if not isinstance(record, dict):
                raise ValueError(f"{path}:{line_number} must be a JSON object")

            records.append(record)

    return records


def normalize_question(question: str) -> str:
    return re.sub(r"\s+", " ", question.strip().lower())


def load_chunk_ids(path: Path) -> set[str]:
    chunk_ids: set[str] = set()

    if not path.exists():
        raise FileNotFoundError(
            f"Chunk file not found: {path}. Run scripts/parse_docs.py first."
        )

    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                chunk = json.loads(line)
                chunk_ids.add(str(chunk["chunk_id"]))

    return chunk_ids


def validate_record(
    record: dict[str, Any],
    index: int,
    chunk_ids: set[str],
) -> list[str]:
    errors: list[str] = []
    prefix = f"record[{index}]"

    missing = REQUIRED_FIELDS - set(record)
    if missing:
        errors.append(f"{prefix} missing fields: {sorted(missing)}")

    for field in ("id", "question", "domain", "kb_version"):
        if not isinstance(record.get(field), str) or not record[field].strip():
            errors.append(f"{prefix}.{field} must be a non-empty string")

    if record.get("case_type") not in VALID_CASE_TYPES:
        errors.append(f"{prefix}.case_type is invalid")

    if record.get("answerability") not in VALID_ANSWERABILITY:
        errors.append(f"{prefix}.answerability is invalid")

    if record.get("safety") not in VALID_SAFETY:
        errors.append(f"{prefix}.safety is invalid")

    if record.get("split") not in VALID_SPLITS:
        errors.append(f"{prefix}.split is invalid")

    for field in (
        "expected_products",
        "gold_points",
        "must_not_claim",
        "required_clarifications",
    ):
        if not isinstance(record.get(field), list):
            errors.append(f"{prefix}.{field} must be a list")

    evidence = record.get("gold_evidence")
    if not isinstance(evidence, list):
        errors.append(f"{prefix}.gold_evidence must be a list")
    else:
        if len(evidence) > 3:
            errors.append(f"{prefix}.gold_evidence has more than 3 items")

        for evidence_index, item in enumerate(evidence):
            if not isinstance(item, dict):
                errors.append(
                    f"{prefix}.gold_evidence[{evidence_index}] must be an object"
                )
                continue

            chunk_id = item.get("chunk_id")
            relevance = item.get("relevance")

            if chunk_id not in chunk_ids:
                errors.append(
                    f"{prefix}.gold_evidence[{evidence_index}] "
                    f"references unknown chunk_id={chunk_id}"
                )

            if relevance not in VALID_RELEVANCE:
                errors.append(
                    f"{prefix}.gold_evidence[{evidence_index}] "
                    "has invalid relevance"
                )

            if not isinstance(item.get("reason"), str) or not item["reason"].strip():
                errors.append(
                    f"{prefix}.gold_evidence[{evidence_index}] "
                    "requires a reason"
                )

    answerability = record.get("answerability")
    if answerability == "answerable":
        if not record.get("gold_evidence"):
            errors.append(f"{prefix} answerable record requires gold_evidence")

        if not record.get("gold_points"):
            errors.append(f"{prefix} answerable record requires gold_points")

    if answerability == "unanswerable" and record.get("gold_evidence"):
        errors.append(
            f"{prefix} unanswerable record should not have gold_evidence"
        )

    annotation = record.get("annotation")
    if not isinstance(annotation, dict):
        errors.append(f"{prefix}.annotation must be an object")
    else:
        if annotation.get("status") != "reviewed":
            errors.append(f"{prefix} annotation.status must be reviewed")

    return errors


def main() -> int:
    args = parse_args()
    records = load_jsonl(args.eval)

    if len(records) < args.min_count:
        raise ValueError(
            f"Evaluation set has {len(records)} records; "
            f"minimum is {args.min_count}"
        )

    chunk_ids = load_chunk_ids(args.chunks)

    errors: list[str] = []
    ids: list[str] = []
    normalized_questions: list[str] = []

    for index, record in enumerate(records, start=1):
        errors.extend(validate_record(record, index, chunk_ids))
        ids.append(str(record.get("id", "")))
        normalized_questions.append(
            normalize_question(str(record.get("question", "")))
        )

    duplicate_ids = [
        item for item, count in Counter(ids).items() if count > 1
    ]
    duplicate_questions = [
        item
        for item, count in Counter(normalized_questions).items()
        if count > 1
    ]

    if duplicate_ids:
        errors.append(f"duplicate ids: {duplicate_ids}")

    if duplicate_questions:
        errors.append(
            f"duplicate normalized questions: {duplicate_questions}"
        )

    distribution = {
        "domain": dict(Counter(str(record["domain"]) for record in records)),
        "case_type": dict(
            Counter(str(record["case_type"]) for record in records)
        ),
        "answerability": dict(
            Counter(str(record["answerability"]) for record in records)
        ),
        "split": dict(Counter(str(record["split"]) for record in records)),
    }

    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        return 1

    print(json.dumps({
        "status": "ok",
        "count": len(records),
        "distribution": distribution,
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())