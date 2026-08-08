"""Run ten seed questions through retrieval + generation and write JSONL results."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Allow direct execution with `python scripts/<file>.py`.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from generation.model_client import MockGenerationClient, OpenAICompatibleClient  # noqa: E402
from generation.rag_pipeline import RAGPipeline  # noqa: E402

# Allow direct execution with `python scripts/<file>.py`.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from retrieval.embeddings import create_embedding_provider  # noqa: E402
from retrieval.qdrant_store import QdrantVectorStore  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", type=Path, default=ROOT / "data/eval/seed_questions.jsonl")
    parser.add_argument("--output", type=Path, default=ROOT / "data/eval/rag_smoke_results.jsonl")
    parser.add_argument("--qdrant-path", type=Path, default=ROOT / "data/index/qdrant")
    parser.add_argument("--collection", default="opspilot_chunks_v1")
    parser.add_argument("--provider", choices=["mock", "openai-compatible"], default="mock")
    parser.add_argument("--llm-base-url", default="")
    parser.add_argument("--llm-api-key", default="")
    parser.add_argument("--llm-model", default="")
    parser.add_argument("--top-k", type=int, default=5)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    questions = [
        json.loads(line) for line in args.questions.read_text(encoding="utf-8").splitlines() if line
    ][:10]
    embedder = create_embedding_provider("hash")
    store = QdrantVectorStore(args.qdrant_path, args.collection, embedder.dimension)
    if args.provider == "mock":
        generator = MockGenerationClient()
    else:
        generator = OpenAICompatibleClient(args.llm_base_url, args.llm_api_key, args.llm_model)
    pipeline = RAGPipeline(embedder, store, generator, args.top_k)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        with args.output.open("w", encoding="utf-8", newline="\n") as handle:
            for item in questions:
                result = pipeline.answer(item["question"])
                result["seed_id"] = item["id"]
                result["category"] = item["category"]
                handle.write(json.dumps(result, ensure_ascii=False) + "\n")
                print(
                    f"{item['id']} status={result['status']} citations={len(result['citations'])}"
                )
    finally:
        store.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
