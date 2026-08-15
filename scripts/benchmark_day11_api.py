"""Run a repeatable in-process FastAPI contract and latency benchmark."""

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
    parser.add_argument("--requests", type=int, default=30)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "data/eval/day11_api_results.json",
    )
    args = parser.parse_args()
    client = TestClient(create_app())
    question = "Docker 容器退出后如何查看日志？"
    latency: list[float] = []
    statuses: list[int] = []
    for index in range(args.requests):
        started = time.perf_counter()
        response = client.post(
            "/v1/query",
            json={"question": question, "top_k": 5},
            headers={"X-Request-ID": f"req_bench_{index:03d}"},
        )
        latency.append((time.perf_counter() - started) * 1000)
        statuses.append(response.status_code)
    result = {
        "experiment": "day11_fastapi_in_process",
        "requests": args.requests,
        "success_rate": round(sum(code == 200 for code in statuses) / len(statuses), 4),
        "latency_ms": {
            "mean": round(statistics.mean(latency), 3),
            "p50": round(percentile(latency, 0.5), 3),
            "p95": round(percentile(latency, 0.95), 3),
            "max": round(max(latency), 3),
        },
        "contract_checks": {
            "openapi": client.get("/openapi.json").status_code == 200,
            "liveness": client.get("/health/live").status_code == 200,
            "readiness": client.get("/health/ready").status_code == 200,
            "validation_422": client.post(
                "/v1/query", json={"question": "x", "top_k": 100}
            ).status_code
            == 422,
        },
        "note": "In-process TestClient excludes network and reverse-proxy latency.",
    }
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
