"""Evaluate naive always-answer behaviour against the Day 10 safety guard."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from generation.grounding import AnswerabilityPolicy  # noqa: E402


def metrics(
    rows: list[dict[str, Any]], predictions: list[str]
) -> dict[str, float]:
    expected = [row["expected_action"] for row in rows]
    correct = sum(a == b for a, b in zip(expected, predictions, strict=True))
    risky = [i for i, label in enumerate(expected) if label != "answer"]
    unsafe = sum(predictions[i] == "answer" for i in risky)
    answerable = [i for i, label in enumerate(expected) if label == "answer"]
    covered = sum(predictions[i] == "answer" for i in answerable)
    return {
        "action_accuracy": round(correct / len(rows), 4),
        "unsafe_answer_rate": round(unsafe / len(risky), 4),
        "answerable_coverage": round(covered / len(answerable), 4),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cases",
        type=Path,
        default=ROOT / "data/eval/day10_grounding_cases.jsonl",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "data/eval/day10_grounding_results.json",
    )
    parser.add_argument("--threshold", type=float, default=0.35)
    args = parser.parse_args()

    rows = [
        json.loads(line)
        for line in args.cases.read_text(encoding="utf-8").splitlines()
        if line
    ]
    policy = AnswerabilityPolicy(min_top_score=args.threshold)
    guarded: list[str] = []
    baseline: list[str] = []
    details: list[dict[str, Any]] = []
    for row in rows:
        evidence = [
            {"score": score, "citation_id": f"S{index}"}
            for index, score in enumerate(row["scores"], start=1)
        ]
        # Baseline imitates a common unsafe RAG: answer whenever retrieval
        # returned anything, regardless of confidence or question completeness.
        baseline_action = "answer" if evidence else "refuse"
        decision = policy.decide(row["question"], evidence)
        baseline.append(baseline_action)
        guarded.append(decision.action.value)
        details.append(
            {
                "id": row["id"],
                "expected": row["expected_action"],
                "baseline": baseline_action,
                "guarded": decision.action.value,
                "reason": decision.reason,
                "top_score": decision.top_score,
            }
        )

    result = {
        "experiment": "day10_answerability_guard",
        "case_count": len(rows),
        "distribution": dict(Counter(row["expected_action"] for row in rows)),
        "config": {"min_top_score": args.threshold},
        "metrics": {
            "naive_baseline": metrics(rows, baseline),
            "guarded_policy": metrics(rows, guarded),
        },
        "limitations": [
            "This is a controlled calibration set, not production traffic.",
            "Citation structure validation does not prove semantic entailment.",
        ],
        "per_case": details,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({k: result[k] for k in ("config", "metrics")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
