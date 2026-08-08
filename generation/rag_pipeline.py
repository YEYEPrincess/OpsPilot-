"""Retrieve evidence, call generation, validate JSON and expose citations."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from generation.model_client import GenerationClient
from generation.prompt import PROMPT_VERSION, build_prompt
from retrieval.embeddings import EmbeddingProvider
from retrieval.qdrant_store import QdrantVectorStore

REQUIRED_FIELDS = {"answer", "possible_causes", "steps", "risks", "citations", "clarification"}


@dataclass
class RAGPipeline:
    """Basic RAG orchestration with a safe structured-output fallback."""

    embedder: EmbeddingProvider
    store: QdrantVectorStore
    generator: GenerationClient
    top_k: int = 5

    def _fallback(self, question: str, hits: list[dict[str, Any]], error: str) -> dict[str, Any]:
        return {
            "status": "degraded",
            "question": question,
            "answer": "生成模型输出不可用，已返回检索到的证据供人工判断。",
            "possible_causes": [],
            "steps": [],
            "risks": [error],
            "citations": [f"S{index}" for index in range(1, len(hits) + 1)],
            "clarification": "请检查模型服务状态，或根据下方原文人工确认。",
            "sources": hits,
            "prompt_version": PROMPT_VERSION,
        }

    def answer(self, question: str) -> dict[str, Any]:
        """Run retrieval and generation, preserving source text in the response."""
        query_vector = self.embedder.encode([question])[0]
        search_hits = self.store.search(query_vector, self.top_k, query_text=question)
        hits = [{"score": hit.score, **hit.payload} for hit in search_hits]
        prompt = build_prompt(question, hits)
        try:
            raw = self.generator.generate(prompt, hits)
            result = json.loads(raw)
            if not isinstance(result, dict) or not REQUIRED_FIELDS.issubset(result):
                raise ValueError("structured output is missing required fields")
        except (json.JSONDecodeError, TypeError, ValueError, RuntimeError) as exc:
            return self._fallback(question, hits, f"模型输出解析失败：{exc}")
        result.update(
            {
                "status": "ok",
                "question": question,
                "sources": hits,
                "prompt_version": PROMPT_VERSION,
            }
        )
        return result
