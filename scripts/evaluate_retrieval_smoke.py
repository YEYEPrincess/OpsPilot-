"""Summarize the ten-question retrieval smoke output with a transparent proxy metric."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


EXPECTED_PRODUCTS = {
    "docker": {"docker"},
    "gpu": {"nvidia-container-toolkit", "pytorch", "vllm"},
    "cuda": {"pytorch", "nvidia-container-toolkit", "vllm"},
    "vllm": {"vllm"},
    "model": {"vllm", "pytorch"},
}


def main() -> int:
    input_path = ROOT / "data/eval/rag_smoke_results.jsonl"
    output_path = ROOT / "data/eval/retrieval_smoke_summary.json"
    rows = [
        json.loads(line)
        for line in input_path.read_text(encoding="utf-8").splitlines()
        if line
    ]
    evaluated = []
    for row in rows:
        products = {source.get("product") for source in row.get("sources", [])}
        expected = EXPECTED_PRODUCTS.get(row.get("category"), set())
        evaluated.append(
            {
                "seed_id": row.get("seed_id"),
                "category": row.get("category"),
                "top_k": len(row.get("sources", [])),
                "products": sorted(products),
                "category_product_hit": bool(products & expected),
                "status": row.get("status"),
            }
        )
    summary = {
        "metric": "category_product_hit@k",
        "note": "Proxy smoke metric, not human-judged answer correctness.",
        "questions": len(evaluated),
        "status_ok": sum(row["status"] == "ok" for row in evaluated),
        "category_product_hits": sum(row["category_product_hit"] for row in evaluated),
        "hit_rate": round(
            sum(row["category_product_hit"] for row in evaluated) / len(evaluated), 4
        )
        if evaluated
        else 0,
        "details": evaluated,
    }
    output_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
