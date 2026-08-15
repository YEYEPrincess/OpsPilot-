"""Testable application services behind the FastAPI transport layer."""

from __future__ import annotations

import asyncio
import hashlib
import re
import time
import uuid
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any

from generation.grounding import AnswerabilityPolicy, AnswerAction, guarded_response

TOKEN_RE = re.compile(r"[A-Za-z0-9_./:-]+|[\u4e00-\u9fff]")

DEFAULT_DOCUMENTS = [
    {
        "document_id": "demo-docker",
        "title": "Docker container diagnostics",
        "source_url": "https://docs.docker.com/reference/cli/docker/container/logs/",
        "product": "docker",
        "content": (
            "Docker 容器退出后，先使用 docker ps -a 查看状态，"
            "再使用 docker logs 查看退出前日志。"
        ),
    },
    {
        "document_id": "demo-pytorch",
        "title": "PyTorch CUDA availability",
        "source_url": "https://pytorch.org/docs/stable/generated/torch.cuda.is_available.html",
        "product": "pytorch",
        "content": "torch.cuda.is_available 返回布尔值，用于判断当前 PyTorch 环境是否可使用 CUDA。",
    },
    {
        "document_id": "demo-fastapi",
        "title": "FastAPI health checks",
        "source_url": "https://fastapi.tiangolo.com/",
        "product": "fastapi",
        "content": "存活检查判断进程是否运行；就绪检查判断模型和索引是否已经能够接收流量。",
    },
]


@dataclass
class DocumentRegistry:
    """In-memory metadata registry; production can replace it with a database."""

    documents: dict[str, dict[str, Any]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.documents:
            self.documents = {
                row["document_id"]: deepcopy(row) for row in DEFAULT_DOCUMENTS
            }

    def create(self, title: str, source_url: str, product: str, content: str) -> dict[str, Any]:
        digest = hashlib.sha256(content.encode("utf-8")).hexdigest()
        document_id = f"doc-{digest[:12]}"
        record = {
            "document_id": document_id,
            "title": title,
            "source_url": source_url,
            "product": product,
            "content": content,
            "content_sha256": digest,
        }
        self.documents[document_id] = record
        return deepcopy(record)

    def list(self) -> list[dict[str, Any]]:
        return [deepcopy(row) for row in self.documents.values()]

    def delete(self, document_id: str) -> bool:
        return self.documents.pop(document_id, None) is not None


@dataclass
class DemoQueryService:
    """CPU-only reference engine used for API, streaming and monitoring tests.

    Its interface can later wrap the real Hybrid + Reranker + LLM pipeline.
    Keeping infrastructure tests model-free makes CI fast and deterministic.
    """

    registry: DocumentRegistry
    policy: AnswerabilityPolicy = field(
        default_factory=lambda: AnswerabilityPolicy(min_top_score=0.35)
    )
    cache: dict[str, dict[str, Any]] = field(default_factory=dict)

    async def answer(self, question: str, top_k: int = 5) -> dict[str, Any]:
        started = time.perf_counter()
        key = hashlib.sha256(f"{question}|{top_k}".encode()).hexdigest()
        if key in self.cache:
            result = deepcopy(self.cache[key])
            result["cache_hit"] = True
            cache_ms = (time.perf_counter() - started) * 1000
            # Never reuse cold-path timings for a cache hit. Otherwise metrics
            # falsely attribute work that did not happen to this request.
            result["timings_ms"] = {
                "cache_lookup": round(cache_ms, 3),
                "total": round(cache_ms, 3),
            }
            return result

        stage_started = time.perf_counter()
        sources = self._retrieve(question, top_k)
        retrieval_ms = (time.perf_counter() - stage_started) * 1000
        await asyncio.sleep(0)

        stage_started = time.perf_counter()
        # The demo uses lexical score as a deterministic stand-in. In production
        # this stage is bge-reranker-v2-m3 over Hybrid Top-20 candidates.
        sources.sort(key=lambda item: item["score"], reverse=True)
        rerank_ms = (time.perf_counter() - stage_started) * 1000

        decision = self.policy.decide(question, sources)
        stage_started = time.perf_counter()
        if decision.action != AnswerAction.ANSWER:
            result = guarded_response(question, sources, self.policy)
        else:
            top = sources[0]
            answer = (
                f"建议先依据官方文档执行：{top['text']} "
                f"[{top['citation_id']}]"
            )
            result = {
                "status": "answer",
                "answer": answer,
                "clarification": "",
                "citations": [top["citation_id"]],
                "sources": sources,
            }
        generation_ms = (time.perf_counter() - stage_started) * 1000
        result.update(
            {
                "cache_hit": False,
                "token_total": max(1, (len(question) + len(result["answer"])) // 2),
                "timings_ms": {
                    "retrieval": round(retrieval_ms, 3),
                    "rerank": round(rerank_ms, 3),
                    "generation": round(generation_ms, 3),
                    "total": round((time.perf_counter() - started) * 1000, 3),
                },
            }
        )
        self.cache[key] = deepcopy(result)
        return result

    def _retrieve(self, question: str, top_k: int) -> list[dict[str, Any]]:
        query_terms = set(TOKEN_RE.findall(question.lower()))
        hits: list[dict[str, Any]] = []
        for row in self.registry.documents.values():
            text = f"{row['title']} {row['product']} {row['content']}".lower()
            doc_terms = set(TOKEN_RE.findall(text))
            overlap = len(query_terms & doc_terms) / max(1, len(query_terms))
            if overlap <= 0:
                continue
            hits.append(
                {
                    "citation_id": "",
                    "title": row["title"],
                    "source_url": row["source_url"],
                    "score": round(overlap, 4),
                    "text": row["content"],
                }
            )
        hits.sort(key=lambda item: item["score"], reverse=True)
        for index, hit in enumerate(hits[:top_k], start=1):
            hit["citation_id"] = f"S{index}"
        return hits[:top_k]


def new_request_id() -> str:
    return f"req_{uuid.uuid4().hex}"
