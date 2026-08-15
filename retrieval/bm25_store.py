"""轻量、可持久化的 BM25 检索模块。"""

from __future__ import annotations

import gzip
import heapq
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rank_bm25 import BM25Okapi

# 保留技术标识符；中文暂时按单字切分。
# 索引和查询必须使用完全相同的 tokenizer。
TOKEN_RE = re.compile(r"[A-Za-z0-9_./:=+\-]+|[\u4e00-\u9fff]")


def tokenize(text: str) -> list[str]:
    """把文本转换为 BM25 使用的 token 列表。"""
    return TOKEN_RE.findall(text.lower())


@dataclass(frozen=True)
class BM25Hit:
    """统一表示一条 BM25 搜索结果。"""

    score: float
    payload: dict[str, Any]


class BM25Index:
    """从 Chunk 构建、保存、加载和查询 BM25 索引。"""

    schema_version = "bm25-v1"

    def __init__(self, documents: list[dict[str, Any]]) -> None:
        if not documents:
            raise ValueError("BM25 documents cannot be empty")

        self.documents = documents
        tokenized_corpus = [document["tokens"] for document in documents]
        self.model = BM25Okapi(tokenized_corpus)

    @classmethod
    def from_chunks(cls, chunks: list[dict[str, Any]]) -> BM25Index:
        """把 section title 和正文一起加入 BM25 文档。"""
        documents: list[dict[str, Any]] = []

        for chunk in chunks:
            section_title = " > ".join(chunk.get("section_path") or [])
            searchable_text = f"{section_title}\n{chunk.get('text', '')}"

            documents.append(
                {
                    "tokens": tokenize(searchable_text),
                    # 保存完整 payload，之后无需再次查 Chunk 文件。
                    "payload": chunk,
                }
            )

        return cls(documents)

    def save(self, path: Path) -> None:
        """使用 gzip JSON 保存，避免不安全的 pickle。"""
        path.parent.mkdir(parents=True, exist_ok=True)
        artifact = {
            "schema_version": self.schema_version,
            "documents": self.documents,
        }

        with gzip.open(path, "wt", encoding="utf-8") as handle:
            json.dump(artifact, handle, ensure_ascii=False)

    @classmethod
    def load(cls, path: Path) -> BM25Index:
        """加载 token 化语料并重新计算 BM25 统计量。"""
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            artifact = json.load(handle)

        if artifact.get("schema_version") != cls.schema_version:
            raise ValueError("Unsupported BM25 index schema")

        return cls(artifact["documents"])

    def search(self, query: str, limit: int = 10) -> list[BM25Hit]:
        """返回 BM25 分数最高的结果。"""
        query_tokens = tokenize(query)
        if not query_tokens or limit <= 0:
            return []

        scores = self.model.get_scores(query_tokens)
        top_indices = heapq.nlargest(
            min(limit, len(scores)),
            range(len(scores)),
            key=lambda index: float(scores[index]),
        )

        # 分数不大于 0 表示没有有效的词法匹配，不返回噪声结果。
        return [
            BM25Hit(
                score=float(scores[index]),
                payload=self.documents[index]["payload"],
            )
            for index in top_indices
            if float(scores[index]) > 0
        ]
