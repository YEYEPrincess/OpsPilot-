"""Measure SSE time-to-first-event and complete response latency."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.main import create_app  # noqa: E402
from core.observability import percentile  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--requests", type=int, default=20)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "data/eval/day12_streaming_results.json",
    )
    args = parser.parse_args()
    client = TestClient(create_app())
    first_event_ms: list[float] = []
    total_ms: list[float] = []
    complete = 0
    for _ in range(args.requests):
        started = time.perf_counter()
        first_seen = False
        body = ""
        with client.stream(
            "POST",
            "/v1/query/stream",
            json={"question": "PyTorch 如何确认 CUDA 可用？"},
        ) as response:
            for line in response.iter_lines():
                if line and not first_seen:
                    first_event_ms.append((time.perf_counter() - started) * 1000)
                    first_seen = True
                body += line + "\n"
        total_ms.append((time.perf_counter() - started) * 1000)
        complete += int("event: done" in body)
    feedback = client.post(
        "/v1/feedback",
        json={
            "request_id": "req_stream_benchmark",
            "rating": "up",
            "category": "correct",
        },
    )
    result = {
        "experiment": "day12_sse_in_process",
        "requests": args.requests,
        "complete_rate": round(complete / args.requests, 4),
        "first_event_ms": {
            "mean": round(statistics.mean(first_event_ms), 3),
            "p50": round(percentile(first_event_ms, 0.5), 3),
            "p95": round(percentile(first_event_ms, 0.95), 3),
        },
        "total_ms": {
            "mean": round(statistics.mean(total_ms), 3),
            "p50": round(percentile(total_ms, 0.5), 3),
            "p95": round(percentile(total_ms, 0.95), 3),
        },
        "feedback_endpoint_ok": feedback.status_code == 200,
        "note": "TestClient may buffer chunks; verify true TTFT through Uvicorn before production.",
    }
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
