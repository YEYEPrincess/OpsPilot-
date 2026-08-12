"""Cross-check Day 6 gold-evidence annotations.

The checker deliberately separates structural checks (which can be automated)
from semantic checks (which still need a human second pass).  It produces a
small JSON report that can be committed with the evaluation set.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number} is not a JSON object")
            records.append(value)
    return records


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval", type=Path, default=Path("data/eval/day6_eval.jsonl"))
    parser.add_argument(
        "--chunks", type=Path, default=Path("data/processed/chunks_section.jsonl")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("data/eval/day6_evidence_review.json")
    )
    args = parser.parse_args()

    eval_records = load_jsonl(args.eval)
    chunks = load_jsonl(args.chunks)
    chunk_by_id = {item.get("chunk_id"): item for item in chunks}

    missing_chunk_ids: list[dict[str, Any]] = []
    product_mismatches: list[dict[str, Any]] = []
    duplicate_evidence_refs: list[str] = []
    answerable_without_evidence: list[str] = []
    unanswerable_with_evidence: list[str] = []
    invalid_evidence_shape: list[str] = []
    unreviewed_records: list[str] = []
    evidence_items = 0
    relevance_counts: Counter[str] = Counter()

    for record in eval_records:
        record_id = str(record.get("id", "<missing-id>"))
        evidence = record.get("gold_evidence") or []
        answerability = record.get("answerability")

        if not isinstance(evidence, list):
            invalid_evidence_shape.append(record_id)
            evidence = []

        refs: list[str] = []
        for item in evidence:
            if not isinstance(item, dict) or not item.get("chunk_id"):
                invalid_evidence_shape.append(record_id)
                continue
            chunk_id = str(item["chunk_id"])
            refs.append(chunk_id)
            evidence_items += 1
            relevance = item.get("relevance")
            if relevance:
                relevance_counts[str(relevance)] += 1
            if chunk_id not in chunk_by_id:
                missing_chunk_ids.append({"id": record_id, "chunk_id": chunk_id})

        if len(refs) != len(set(refs)):
            duplicate_evidence_refs.append(record_id)

        if answerability == "answerable" and not evidence:
            answerable_without_evidence.append(record_id)
        if answerability == "unanswerable" and evidence:
            unanswerable_with_evidence.append(record_id)

        annotation = record.get("annotation") or {}
        if annotation.get("status") != "reviewed":
            unreviewed_records.append(record_id)

        expected_products = set(record.get("expected_products") or [])
        actual_products = {
            str(chunk_by_id[chunk_id].get("product"))
            for chunk_id in refs
            if chunk_id in chunk_by_id and chunk_by_id[chunk_id].get("product")
        }
        # Product overlap is a useful sanity check, but not a semantic proof.
        if answerability == "answerable" and expected_products and not (
            expected_products & actual_products
        ):
            product_mismatches.append(
                {
                    "id": record_id,
                    "expected_products": sorted(expected_products),
                    "evidence_products": sorted(actual_products),
                }
            )

    structural_errors = (
        missing_chunk_ids
        + [{"id": item} for item in duplicate_evidence_refs]
        + [{"id": item} for item in answerable_without_evidence]
        + [{"id": item} for item in unanswerable_with_evidence]
        + [{"id": item} for item in invalid_evidence_shape]
        + [{"id": item} for item in unreviewed_records]
        + product_mismatches
    )

    report = {
        "status": "passed" if not structural_errors else "failed",
        "scope": "Day 6 evidence annotation cross-check",
        "records_checked": len(eval_records),
        "answerable_checked": sum(
            record.get("answerability") == "answerable" for record in eval_records
        ),
        "evidence_items_checked": evidence_items,
        "relevance_counts": dict(sorted(relevance_counts.items())),
        "checks": {
            "chunk_ids_exist": not missing_chunk_ids,
            "expected_product_overlaps_evidence": not product_mismatches,
            "no_duplicate_evidence_refs": not duplicate_evidence_refs,
            "answerable_has_evidence": not answerable_without_evidence,
            "unanswerable_has_no_evidence": not unanswerable_with_evidence,
            "annotation_status_reviewed": not unreviewed_records,
            "evidence_shape_valid": not invalid_evidence_shape,
        },
        "errors": {
            "missing_chunk_ids": missing_chunk_ids,
            "product_mismatches": product_mismatches,
            "duplicate_evidence_refs": duplicate_evidence_refs,
            "answerable_without_evidence": answerable_without_evidence,
            "unanswerable_with_evidence": unanswerable_with_evidence,
            "invalid_evidence_shape": invalid_evidence_shape,
            "unreviewed_records": unreviewed_records,
        },
        "semantic_review_note": (
            "结构检查通过不等于答案一定正确；仍需按问题、gold_points 与 evidence.text "
            "逐条人工确认结论覆盖和引用边界。"
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    args.output.write_text(serialized, encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
