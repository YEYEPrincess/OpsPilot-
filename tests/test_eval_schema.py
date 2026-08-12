import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_day6_eval_file_exists():
    path = ROOT / "data/eval/day6_eval.jsonl"
    assert path.exists()

    records = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    assert len(records) >= 50
    assert len({record["id"] for record in records}) == len(records)


def test_day6_eval_contains_all_case_types():
    path = ROOT / "data/eval/day6_eval.jsonl"

    records = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    case_types = {record["case_type"] for record in records}

    assert {"simple", "complex", "ambiguous", "unanswerable"} <= case_types