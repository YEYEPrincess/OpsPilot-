"""Generate structured API logs and a Day 13 metrics summary."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.main import create_app  # noqa: E402
from core.observability import load_jsonl, summarize_log_records  # noqa: E402

QUESTIONS = [
    "Docker 容器退出后如何查看日志？",
    "PyTorch 如何确认当前 CUDA 可用？",
    "FastAPI 的存活检查与就绪检查有什么区别？",
]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--requests", type=int, default=36)
    parser.add_argument(
        "--log",
        type=Path,
        default=ROOT / "data/eval/day13_sample_logs.jsonl",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "data/eval/day13_metrics_summary.json",
    )
    args = parser.parse_args()
    if args.log.exists():
        args.log.unlink()
    client = TestClient(create_app(log_path=args.log))
    successful = 0
    for index in range(args.requests):
        response = client.post(
            "/v1/query",
            json={"question": QUESTIONS[index % len(QUESTIONS)], "top_k": 5},
            headers={"X-Request-ID": f"req_observe_{index:03d}"},
        )
        successful += int(response.status_code == 200)
    records = load_jsonl(args.log)
    summary = summarize_log_records(records)
    stage_means = summary["stage_mean_ms"]
    bottleneck = max(stage_means, key=stage_means.get) if stage_means else "unknown"
    result = {
        "experiment": "day13_observability_demo",
        "successful_requests": successful,
        "metrics": summary,
        "largest_mean_stage": bottleneck,
        "privacy_check": {
            "raw_questions_logged": any(
                question in args.log.read_text(encoding="utf-8")
                for question in QUESTIONS
            ),
            "question_fingerprint_present": any(
                "question_sha256" in row for row in records
            ),
        },
    }
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
