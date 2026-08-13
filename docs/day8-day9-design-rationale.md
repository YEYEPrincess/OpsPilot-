# Day 8–9：BM25、混合检索与 Reranker 学习指导

> 使用方式：本文件是“你自己实现、运行和记录”的实验手册。文中的代码不会自动出现在项目中；请按照步骤逐个创建文件，每完成一步就运行相应检查。不要一次性复制全部代码后再排错。

## 1. 两天的总体目标

Day 7 已经得到纯向量检索基线：问题经过 `hash-v1` 编码后，Qdrant 按余弦相似度返回 Top-k。这个基线证明检索链路能够运行，也暴露了两个问题：

1. `hash-v1` 对中文问题和英文技术文档的语义对应能力弱；
2. 只靠向量分数，精确命令、错误码、类名和版本号可能排得不够靠前。

Day 8 增加 BM25 关键词检索，并用 RRF 把 Dense 与 BM25 的结果融合。Day 9 再用 Cross-Encoder Reranker 对少量候选逐对打分。

```text
                         ┌─ Dense / Qdrant ─┐
用户问题 ──预处理────────┤                   ├─ RRF融合 ─ Top-20候选
                         └─ BM25关键词检索 ──┘                  │
                                                              ▼
                                              Cross-Encoder Reranker
                                                              │
                                                              ▼
                                                        最终Top-5证据
```

这三层承担不同职责：

| 层级 | 首要目标 | 特点 |
|---|---|---|
| Dense / BM25 召回 | 尽量不要漏掉正确证据 | 快，可以搜索全部 Chunk |
| RRF 融合 | 合并不同检索器的优势 | 不需要让两种原始分数处于同一量纲 |
| Reranker 精排 | 把最相关证据排到前面 | 更准但更慢，只处理少量候选 |

## 2. 实验纪律：一次只改变一个变量

比较 Dense、BM25、Hybrid、Hybrid + Reranker 时，必须固定：

- 同一份 `day6_eval.jsonl`；
- 同一份 `chunks_section.jsonl`；
- 同一套 gold evidence；
- 同一组 answerable 问题；
- 同样的 Recall@k、Hit Rate@k、MRR@k 定义；
- 同一台机器或至少记录设备差异。

否则指标变化可能来自数据、Chunk 或机器，而不是新检索方法。

## 3. 先安装依赖

在 PowerShell 中进入项目目录：

```powershell
cd D:\Documents\大模型项目\opspilot
```

Day 8 安装 BM25：

```powershell
uv add rank-bm25
```

Day 9 安装 Cross-Encoder 支持：

```powershell
uv add sentence-transformers
```

为什么使用 `uv add`：它会同时更新 `pyproject.toml` 和 `uv.lock`。`pyproject.toml` 表达允许的依赖范围，`uv.lock` 记录这次实际解析到的精确版本，有利于复现实验。

为什么 Day 8 选择 `rank-bm25`：当前只有约一千个 Chunk，它的接口简单，适合理解 BM25 原理。没有选择 Elasticsearch/OpenSearch，是因为后者需要额外服务、JVM、索引配置和运维，四小时内会把重点从检索原理转移到基础设施。生产数据达到几十万或百万 Chunk、需要持久化倒排索引和并发过滤时，再考虑 Elasticsearch/OpenSearch。

---

# Day 8：BM25 与混合检索

## 4. BM25 到底在算什么

BM25 是词法检索算法。它不把文本编码成神经网络向量，而是根据“查询词是否出现在文档中”计算相关性。简化公式为：

```text
BM25(q, d) = 对查询中每个词t求和：

IDF(t) × TF(t,d)的饱和函数 × 文档长度修正
```

三个核心思想：

1. **IDF**：越少见的词越重要。`docker` 可能出现在大量文档中，而 `CUDA_VISIBLE_DEVICES` 很少出现，后者区分力更强；
2. **词频饱和**：一个词从出现 0 次变成 1 次很重要，从 50 次变成 51 次并不会继续大幅加分；
3. **长度归一化**：长文档自然包含更多词，不能仅因为更长就获得不公平优势。

BM25 擅长：

- `torch.cuda.is_available`、`docker logs` 等命令；
- `HTTP 429`、`CUDA OOM` 等错误标识；
- `v0.6.3`、`CUDA 12.4` 等版本；
- 类名、配置键和参数名。

BM25 不擅长：

- 同义表达，例如“显存不足”和“GPU out of memory”；
- 中英文跨语言匹配；
- 拼写变化和没有共同词的改写。

因此它适合补充 Dense Retrieval，而不是完全替代 Dense Retrieval。

## 5. 第一步：实现 BM25 索引模块（约 45 分钟）

请创建：

```text
retrieval/bm25_store.py
```

代码：

```python
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
    def from_chunks(cls, chunks: list[dict[str, Any]]) -> "BM25Index":
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
    def load(cls, path: Path) -> "BM25Index":
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
```

为什么标题和正文一起索引：用户可能问“Qdrant collection 的向量维度”，关键词只出现在章节标题。只索引正文会损失这类信号。

为什么不使用 pickle：pickle 加载时可以执行任意 Python 对象，不适合加载来源不可信的索引文件。JSON gzip 更透明、安全、可检查。代价是加载时需要重新构造 BM25 统计量；对一千个 Chunk 可以接受。

## 6. 第二步：建立 BM25 索引（约 15 分钟）

请创建：

```text
scripts/build_bm25_index.py
```

```python
"""从 section Chunk 建立 BM25 索引。"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from retrieval.bm25_store import BM25Index  # noqa: E402


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--chunks",
        type=Path,
        default=ROOT / "data/processed/chunks_section.jsonl",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "data/index/bm25/opspilot_bm25_v1.json.gz",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=ROOT / "data/manifest/bm25_index.json",
    )
    args = parser.parse_args()

    chunks = [
        json.loads(line)
        for line in args.chunks.read_text(encoding="utf-8").splitlines()
        if line
    ]
    index = BM25Index.from_chunks(chunks)
    index.save(args.output)

    manifest = {
        "generated_at": datetime.now(UTC).isoformat(),
        "index_version": "bm25-v1",
        "source_chunks": str(args.chunks.relative_to(ROOT)),
        "source_sha256": sha256_file(args.chunks),
        "documents": len(chunks),
        "index_path": str(args.output.relative_to(ROOT)),
        "index_sha256": sha256_file(args.output),
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

运行：

```powershell
.\.venv\Scripts\python.exe scripts\build_bm25_index.py
```

验收：`documents` 应与 section Chunk 数量一致；manifest 中保存输入和索引 SHA-256，这样能确认之后的实验使用同一份数据。

## 7. 第三步：实现单独的关键词查询（约 30 分钟）

请创建：

```text
scripts/query_bm25.py
```

```python
"""在命令行中测试 BM25 关键词检索。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from retrieval.bm25_store import BM25Index  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("question")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--index",
        type=Path,
        default=ROOT / "data/index/bm25/opspilot_bm25_v1.json.gz",
    )
    args = parser.parse_args()

    index = BM25Index.load(args.index)
    hits = index.search(args.question, limit=args.top_k)

    print(
        json.dumps(
            [
                {
                    "score": hit.score,
                    "chunk_id": hit.payload["chunk_id"],
                    "product": hit.payload.get("product"),
                    "section_path": hit.payload.get("section_path"),
                    "text_preview": hit.payload.get("text", "")[:300],
                }
                for hit in hits
            ],
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

运行两个问题对比：

```powershell
.\.venv\Scripts\python.exe scripts\query_bm25.py `
  "torch.cuda.is_available" --top-k 5

.\.venv\Scripts\python.exe scripts\query_bm25.py `
  "显存不够怎么办" --top-k 5
```

预期：精确 API 问题较容易命中；纯中文同义表达可能很差。这不是程序错误，而是 BM25 的跨语言局限。

## 8. 第四步：实现 RRF 融合（约 45 分钟）

### 8.1 为什么选择 RRF，不先选择加权分数融合

Dense 余弦分数可能在 `0.1～0.8`，BM25 分数可能在 `0～20`。直接相加没有意义。加权融合必须先做 min-max、z-score 或其他校准，而且校准结果会随查询改变。

RRF（Reciprocal Rank Fusion）只使用排名：

```text
RRF分数(chunk) = Σ weight / (rrf_k + rank)
```

如果一个 Chunk 在 Dense 排第 2、BM25 排第 5：

```text
1 / (60 + 2) + 1 / (60 + 5)
```

它不要求两种分数在同一量纲，所以适合作为第一版混合检索。`rrf_k=60` 是平滑常数：值越大，前几名与后几名的差距越缓和。不要把它与检索 `top_k` 混淆。

请创建：

```text
retrieval/hybrid_search.py
```

```python
"""使用 RRF 融合 Dense 与 BM25 结果。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from retrieval.bm25_store import BM25Hit
from retrieval.qdrant_store import SearchHit


@dataclass(frozen=True)
class HybridHit:
    score: float
    payload: dict[str, Any]
    dense_rank: int | None
    bm25_rank: int | None


def reciprocal_rank_fusion(
    dense_hits: list[SearchHit],
    bm25_hits: list[BM25Hit],
    limit: int = 10,
    rrf_k: int = 60,
    dense_weight: float = 1.0,
    bm25_weight: float = 1.0,
) -> list[HybridHit]:
    """按 Chunk ID 合并两路排名。"""
    if rrf_k <= 0:
        raise ValueError("rrf_k must be positive")

    merged: dict[str, dict[str, Any]] = {}

    for rank, hit in enumerate(dense_hits, start=1):
        chunk_id = str(hit.payload["chunk_id"])
        item = merged.setdefault(
            chunk_id,
            {
                "score": 0.0,
                "payload": hit.payload,
                "dense_rank": None,
                "bm25_rank": None,
            },
        )
        item["score"] += dense_weight / (rrf_k + rank)
        item["dense_rank"] = rank

    for rank, hit in enumerate(bm25_hits, start=1):
        chunk_id = str(hit.payload["chunk_id"])
        item = merged.setdefault(
            chunk_id,
            {
                "score": 0.0,
                "payload": hit.payload,
                "dense_rank": None,
                "bm25_rank": None,
            },
        )
        item["score"] += bm25_weight / (rrf_k + rank)
        item["bm25_rank"] = rank

    ranked = sorted(
        merged.values(),
        key=lambda item: item["score"],
        reverse=True,
    )[:limit]

    return [
        HybridHit(
            score=float(item["score"]),
            payload=item["payload"],
            dense_rank=item["dense_rank"],
            bm25_rank=item["bm25_rank"],
        )
        for item in ranked
    ]
```

优点：实现简单、对分数尺度不敏感、容易解释。缺点：丢弃了原始分数的置信度；两个检索器中“第 1 名比第 2 名强很多”的信息不会被利用。后续数据足够时，可以在 validation 集上校准加权融合，但不能在 test 集反复调权重。

## 9. 第五步：比较 Dense、BM25 与 Hybrid（约 60 分钟）

请创建 `scripts/evaluate_day8_hybrid.py`。为避免复制 Day 7 的全部评测代码，这个脚本复用已有指标函数。

```python
"""在同一评测集上比较 Dense、BM25 和 RRF Hybrid。"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from time import perf_counter
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evaluation.retrieval_metrics import (  # noqa: E402
    hit_rate_at_k,
    recall_at_k,
    reciprocal_rank_at_k,
)
from retrieval.bm25_store import BM25Index  # noqa: E402
from retrieval.embeddings import create_embedding_provider  # noqa: E402
from retrieval.hybrid_search import reciprocal_rank_fusion  # noqa: E402
from retrieval.qdrant_store import QdrantVectorStore  # noqa: E402


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def p95(values: list[float]) -> float:
    """返回简单的 nearest-rank P95。"""
    ordered = sorted(values)
    index = max(0, int(0.95 * len(ordered) + 0.9999) - 1)
    return ordered[index]


def summarize_latency(values: list[float]) -> dict[str, float]:
    return {
        "mean_ms": round(statistics.mean(values), 3),
        "p50_ms": round(statistics.median(values), 3),
        "p95_ms": round(p95(values), 3),
    }


def summarize_metrics(
    rows: list[dict[str, Any]],
    method: str,
    cutoffs: list[int],
) -> dict[str, dict[str, float]]:
    summary: dict[str, dict[str, float]] = {}
    for k in cutoffs:
        recall_values: list[float] = []
        hit_values: list[float] = []
        mrr_values: list[float] = []

        for row in rows:
            retrieved = row["retrieved_ids"][method]
            gold = row["gold_ids"]
            recall_values.append(recall_at_k(gold, retrieved, k))
            hit_values.append(hit_rate_at_k(gold, retrieved, k))
            mrr_values.append(reciprocal_rank_at_k(gold, retrieved, k))

        summary[str(k)] = {
            "recall_at_k": round(statistics.mean(recall_values), 4),
            "hit_rate_at_k": round(statistics.mean(hit_values), 4),
            "mrr_at_k": round(statistics.mean(mrr_values), 4),
        }
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-k", type=int, default=20)
    parser.add_argument("--rrf-k", type=int, default=60)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "data/eval/day8_retrieval_comparison.json",
    )
    args = parser.parse_args()

    eval_path = ROOT / "data/eval/day6_eval.jsonl"
    bm25_path = ROOT / "data/index/bm25/opspilot_bm25_v1.json.gz"
    records = [
        row
        for row in load_jsonl(eval_path)
        if row.get("answerability") == "answerable"
    ]

    provider = create_embedding_provider(
        provider="hash", model_name="", dimension=384, device="cpu"
    )
    dense_store = QdrantVectorStore(
        ROOT / "data/index/qdrant",
        "opspilot_chunks_v1",
        provider.dimension,
    )
    bm25_store = BM25Index.load(bm25_path)

    rows: list[dict[str, Any]] = []
    try:
        for record in records:
            question = record["question"]
            gold_ids = [item["chunk_id"] for item in record["gold_evidence"]]

            start = perf_counter()
            query_vector = provider.encode([question])[0]
            dense_hits = dense_store.search(
                query_vector,
                limit=args.candidate_k,
                query_text="",  # 保持 Day 7 dense-only 定义
            )
            dense_ms = (perf_counter() - start) * 1000

            start = perf_counter()
            bm25_hits = bm25_store.search(question, limit=args.candidate_k)
            bm25_ms = (perf_counter() - start) * 1000

            start = perf_counter()
            hybrid_hits = reciprocal_rank_fusion(
                dense_hits,
                bm25_hits,
                limit=args.candidate_k,
                rrf_k=args.rrf_k,
            )
            fusion_ms = (perf_counter() - start) * 1000

            rows.append(
                {
                    "id": record["id"],
                    "question": question,
                    "gold_ids": gold_ids,
                    "retrieved_ids": {
                        "dense": [hit.payload["chunk_id"] for hit in dense_hits],
                        "bm25": [hit.payload["chunk_id"] for hit in bm25_hits],
                        "hybrid": [hit.payload["chunk_id"] for hit in hybrid_hits],
                    },
                    "latency_ms": {
                        "dense": dense_ms,
                        "bm25": bm25_ms,
                        # 当前是顺序调用，因此端到端延迟使用求和。
                        "hybrid_sequential": dense_ms + bm25_ms + fusion_ms,
                        # 如果以后真正并行，理论近似为较慢分支+融合时间。
                        "hybrid_parallel_estimate": max(dense_ms, bm25_ms) + fusion_ms,
                    },
                }
            )
    finally:
        dense_store.close()

    cutoffs = [1, 3, 5, 10]
    methods = ["dense", "bm25", "hybrid"]
    result = {
        "config": {
            "candidate_k": args.candidate_k,
            "rrf_k": args.rrf_k,
            "evaluated_questions": len(rows),
        },
        "metrics": {
            method: summarize_metrics(rows, method, cutoffs)
            for method in methods
        },
        "latency": {
            name: summarize_latency(
                [row["latency_ms"][name] for row in rows]
            )
            for name in [
                "dense",
                "bm25",
                "hybrid_sequential",
                "hybrid_parallel_estimate",
            ]
        },
        "per_query": rows,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({k: result[k] for k in ["config", "metrics", "latency"]},
                     ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

运行：

```powershell
.\.venv\Scripts\python.exe scripts\evaluate_day8_hybrid.py `
  --candidate-k 20 --rrf-k 60
```

分析时填写：

| 方法 | Recall@5 | Hit Rate@5 | MRR@5 | P50延迟 | P95延迟 |
|---|---:|---:|---:|---:|---:|
| Dense | 待填写 | 待填写 | 待填写 | 待填写 | 待填写 |
| BM25 | 待填写 | 待填写 | 待填写 | 待填写 | 待填写 |
| Hybrid RRF | 待填写 | 待填写 | 待填写 | 待填写 | 待填写 |

不要预设 Hybrid 一定最好。中文问题与英文文档没有共同词时，BM25 可能贡献很少；这本身就是有效实验结论。

## 10. Day 8 测试与验收

请创建 `tests/test_hybrid_retrieval.py`：

```python
from retrieval.bm25_store import BM25Hit, tokenize
from retrieval.hybrid_search import reciprocal_rank_fusion
from retrieval.qdrant_store import SearchHit


def payload(chunk_id: str) -> dict[str, str]:
    return {"chunk_id": chunk_id, "text": chunk_id}


def test_tokenizer_preserves_technical_identifier() -> None:
    assert "torch.cuda.is_available" in tokenize("Use torch.cuda.is_available now")


def test_rrf_rewards_results_found_by_both_retrievers() -> None:
    dense = [
        SearchHit(score=0.8, payload=payload("dense-only")),
        SearchHit(score=0.7, payload=payload("shared")),
    ]
    bm25 = [
        BM25Hit(score=9.0, payload=payload("bm25-only")),
        BM25Hit(score=8.0, payload=payload("shared")),
    ]

    fused = reciprocal_rank_fusion(dense, bm25, limit=3, rrf_k=60)
    assert fused[0].payload["chunk_id"] == "shared"
```

运行：

```powershell
.\.venv\Scripts\python.exe -m ruff check `
  retrieval\bm25_store.py retrieval\hybrid_search.py `
  scripts\build_bm25_index.py scripts\query_bm25.py `
  scripts\evaluate_day8_hybrid.py tests\test_hybrid_retrieval.py

.\.venv\Scripts\python.exe -m pytest -q
```

---

# Day 9：Reranker 重排序

## 11. Reranker 与 Embedding 的根本区别

Dense Retriever 通常是 Bi-Encoder：问题和文档分别编码，文档向量可以预先存入 Qdrant。查询时只需编码问题并做近邻搜索，所以速度快。

Cross-Encoder Reranker 把 `(问题, 候选Chunk)` 同时送进 Transformer。问题 token 可以直接关注文档 token，相关性判断更细，但每个候选都需要一次前向计算，不能预先缓存成单独文档向量。

```text
Bi-Encoder：encode(query) 与 encode(document) → 余弦相似度

Cross-Encoder：[query tokens; document tokens] → Transformer → relevance score
```

因此典型流程是“先召回，再精排”：检索器从全部 Chunk 中取 Top-20/50，Reranker 只处理这些候选。官方 Sentence Transformers 文档也采用 Retrieve & Re-Rank 两阶段结构。

## 12. 候选数量为什么选 20，最终为什么选 5

本地学习默认：

```text
Hybrid candidate_k = 20
Reranker final_k = 5
batch_size = GPU 8 / CPU 2
max_length = 512 tokens
```

候选太少：正确证据没有进入候选，Reranker 再强也无法创造新证据。候选太多：延迟和显存近似随 `(query, document)` 对数增加。

最关键的上限关系：

```text
Reranker后的Hit@5 ≤ 候选集合的Hit@20上限
```

Reranker主要改善 Top-5 的 MRR 和 Hit Rate；它不应该改变候选 Top-20 的 Recall，因为只是重新排序同一批候选。

## 13. 模型选择

推荐主实验：

```text
BAAI/bge-reranker-v2-m3
```

选择理由：项目问题主要是中文、文档主要是英文，需要多语言相关性判断；模型卡将它定位为 multilingual reranker。它约 0.6B 参数，质量和部署成本都高于小型 MiniLM。

资源不足时的替代：

- `BAAI/bge-reranker-base`：中英文、相对轻量；
- `cross-encoder/ms-marco-MiniLM-L6-v2`：速度快，适合验证代码，但主要针对英文检索，不适合作为本项目最终多语言结论。

为什么不直接使用大语言模型 Prompt 打分：LLM 重排更灵活，但延迟、成本、输出稳定性和批处理难度更高。Day 9 先用输出单个 relevance score 的专用 Cross-Encoder，实验变量更清晰。

## 14. 第一步：实现 Reranker（约 60 分钟）

请创建：

```text
retrieval/reranker.py
```

```python
"""Cross-Encoder Reranker。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np
from sentence_transformers import CrossEncoder


class Candidate(Protocol):
    score: float
    payload: dict[str, Any]


@dataclass(frozen=True)
class RerankedHit:
    reranker_score: float
    retrieval_score: float
    original_rank: int
    payload: dict[str, Any]


def rank_by_scores(
    candidates: list[Candidate], scores: list[float]
) -> list[RerankedHit]:
    """把模型分数与候选绑定并按分数降序排列。"""
    if len(candidates) != len(scores):
        raise ValueError("Candidate and score counts differ")

    ranked = [
        RerankedHit(
            reranker_score=float(score),
            retrieval_score=float(candidate.score),
            original_rank=rank,
            payload=candidate.payload,
        )
        for rank, (candidate, score) in enumerate(
            zip(candidates, scores, strict=True), start=1
        )
    ]
    return sorted(ranked, key=lambda hit: hit.reranker_score, reverse=True)


class CrossEncoderReranker:
    def __init__(
        self,
        model_name: str,
        device: str,
        batch_size: int = 8,
        max_length: int = 512,
    ) -> None:
        self.model_name = model_name
        self.batch_size = batch_size
        self.model = CrossEncoder(
            model_name,
            device=device,
            max_length=max_length,
        )

    def rerank(
        self,
        query: str,
        candidates: list[Candidate],
        top_n: int = 5,
    ) -> list[RerankedHit]:
        if not candidates or top_n <= 0:
            return []

        pairs = [
            (query, str(candidate.payload.get("text", "")))
            for candidate in candidates
        ]
        raw_scores = self.model.predict(
            pairs,
            batch_size=self.batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        scores = np.asarray(raw_scores).reshape(-1).astype(float).tolist()
        return rank_by_scores(candidates, scores)[:top_n]
```

为什么使用 raw logits 而不强制 sigmoid：重排序只关心相对顺序，sigmoid 是单调函数，不会改变排名。需要把分数展示为概率时才考虑归一化，但 reranker score 通常不应直接解释成真实概率。

## 15. 第二步：先做单问题冒烟测试（约 20 分钟）

创建 `scripts/query_reranker.py` 时，可以复用 Day 8 的 Hybrid 候选。核心调用如下：

```python
import torch

from retrieval.reranker import CrossEncoderReranker

device = "cuda" if torch.cuda.is_available() else "cpu"
batch_size = 8 if device == "cuda" else 2

reranker = CrossEncoderReranker(
    model_name="BAAI/bge-reranker-v2-m3",
    device=device,
    batch_size=batch_size,
    max_length=512,
)

# hybrid_hits 是 Day 8 reciprocal_rank_fusion 返回的 Top-20。
reranked = reranker.rerank(question, hybrid_hits[:20], top_n=5)

for hit in reranked:
    print(
        hit.reranker_score,
        hit.original_rank,
        hit.payload["chunk_id"],
    )
```

第一次运行需要下载模型，不能把下载耗时算进稳定推理延迟。先加载模型并做一次 warm-up，再正式计时。

## 16. 第三步：运行有无 Reranker 的严格对照（约 60 分钟）

Day 8 的 `day8_retrieval_comparison.json` 已保存每个问题的 Hybrid 候选 ID。Day 9 应固定这批候选，避免重新检索导致候选变化。请创建：

```text
scripts/evaluate_day9_reranker.py
```

```python
"""对 Day 8 固定候选运行 Reranker A/B 实验。"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evaluation.retrieval_metrics import (  # noqa: E402
    hit_rate_at_k,
    recall_at_k,
    reciprocal_rank_at_k,
)
from retrieval.reranker import CrossEncoderReranker  # noqa: E402


@dataclass(frozen=True)
class CachedCandidate:
    score: float
    payload: dict[str, Any]


def mean_metric(rows: list[dict[str, Any]], key: str) -> float:
    return round(statistics.mean(row[key] for row in rows), 4)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-k", type=int, default=20)
    parser.add_argument("--final-k", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=0)
    parser.add_argument(
        "--model",
        default="BAAI/bge-reranker-v2-m3",
    )
    parser.add_argument(
        "--day8-result",
        type=Path,
        default=ROOT / "data/eval/day8_retrieval_comparison.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "data/eval/day9_reranker_comparison.json",
    )
    args = parser.parse_args()

    chunks = [
        json.loads(line)
        for line in (ROOT / "data/processed/chunks_section.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line
    ]
    chunk_by_id = {chunk["chunk_id"]: chunk for chunk in chunks}
    day8 = json.loads(args.day8_result.read_text(encoding="utf-8"))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size = args.batch_size or (8 if device == "cuda" else 2)
    reranker = CrossEncoderReranker(
        args.model,
        device=device,
        batch_size=batch_size,
        max_length=512,
    )

    # Warm-up：不纳入正式延迟。
    first = day8["per_query"][0]
    warm_ids = first["retrieved_ids"]["hybrid"][:2]
    warm_candidates = [
        CachedCandidate(score=0.0, payload=chunk_by_id[chunk_id])
        for chunk_id in warm_ids
    ]
    reranker.rerank(first["question"], warm_candidates, top_n=2)

    rows: list[dict[str, Any]] = []
    for item in day8["per_query"]:
        candidate_ids = item["retrieved_ids"]["hybrid"][: args.candidate_k]
        candidates = [
            # 缓存文件没有保存完整RRF分数；本实验只需要原始顺序。
            CachedCandidate(score=1.0 / rank, payload=chunk_by_id[chunk_id])
            for rank, chunk_id in enumerate(candidate_ids, start=1)
        ]

        start = perf_counter()
        reranked = reranker.rerank(
            item["question"], candidates, top_n=args.final_k
        )
        rerank_ms = (perf_counter() - start) * 1000

        baseline_ids = candidate_ids[: args.final_k]
        reranked_ids = [hit.payload["chunk_id"] for hit in reranked]
        gold_ids = item["gold_ids"]

        rows.append(
            {
                "id": item["id"],
                "baseline_ids": baseline_ids,
                "reranked_ids": reranked_ids,
                "candidate_recall": recall_at_k(
                    gold_ids, candidate_ids, args.candidate_k
                ),
                "baseline_hit": hit_rate_at_k(
                    gold_ids, baseline_ids, args.final_k
                ),
                "reranked_hit": hit_rate_at_k(
                    gold_ids, reranked_ids, args.final_k
                ),
                "baseline_mrr": reciprocal_rank_at_k(
                    gold_ids, baseline_ids, args.final_k
                ),
                "reranked_mrr": reciprocal_rank_at_k(
                    gold_ids, reranked_ids, args.final_k
                ),
                "rerank_ms": rerank_ms,
            }
        )

    result = {
        "config": {
            "model": args.model,
            "device": device,
            "batch_size": batch_size,
            "candidate_k": args.candidate_k,
            "final_k": args.final_k,
        },
        "metrics": {
            "candidate_recall": mean_metric(rows, "candidate_recall"),
            "baseline_hit_at_final_k": mean_metric(rows, "baseline_hit"),
            "reranked_hit_at_final_k": mean_metric(rows, "reranked_hit"),
            "baseline_mrr_at_final_k": mean_metric(rows, "baseline_mrr"),
            "reranked_mrr_at_final_k": mean_metric(rows, "reranked_mrr"),
        },
        "latency": {
            "mean_ms": round(statistics.mean(row["rerank_ms"] for row in rows), 3),
            "p50_ms": round(statistics.median(row["rerank_ms"] for row in rows), 3),
            "max_ms": round(max(row["rerank_ms"] for row in rows), 3),
        },
        "per_query": rows,
    }
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({k: result[k] for k in ["config", "metrics", "latency"]},
                     ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

运行：

```powershell
.\.venv\Scripts\python.exe scripts\evaluate_day9_reranker.py `
  --candidate-k 20 --final-k 5
```

如果 CPU 太慢，先用小模型验证代码：

```powershell
.\.venv\Scripts\python.exe scripts\evaluate_day9_reranker.py `
  --model cross-encoder/ms-marco-MiniLM-L6-v2 `
  --candidate-k 10 --final-k 5 --batch-size 2
```

小模型结果只能作为代码冒烟测试，不能代替多语言主实验结论。

## 17. Day 9 测试

向 `tests/test_hybrid_retrieval.py` 增加不下载模型的排序测试：

```python
from dataclasses import dataclass

from retrieval.reranker import rank_by_scores


@dataclass
class FakeCandidate:
    score: float
    payload: dict[str, str]


def test_rank_by_scores_uses_reranker_score() -> None:
    candidates = [
        FakeCandidate(0.9, {"chunk_id": "old-first"}),
        FakeCandidate(0.1, {"chunk_id": "new-first"}),
    ]
    reranked = rank_by_scores(candidates, [0.2, 0.8])
    assert reranked[0].payload["chunk_id"] == "new-first"
    assert reranked[0].original_rank == 2
```

不在单元测试中下载真实模型，因为这会让测试依赖网络、耗时长并产生缓存。真实模型加载放在冒烟测试和对照实验中。

## 18. 如何分析质量—延迟权衡

填写：

| 方法 | Candidate k | Final k | Hit@5 | MRR@5 | P50 | P95/Max | 设备 |
|---|---:|---:|---:|---:|---:|---:|---|
| Hybrid | 20 | 5 | 待填写 | 待填写 | 待填写 | 待填写 | CPU/GPU |
| Hybrid + Reranker | 20 | 5 | 待填写 | 待填写 | 待填写 | 待填写 | CPU/GPU |

如果 MRR 明显提高而 Recall@20 不变，这是正常结果：Reranker 改善排序，没有扩大候选集合。如果 candidate Recall@20 本身很低，应先改进 BM25/Dense 召回，而不是继续堆更大的 Reranker。

推荐做一个小型参数实验：

```text
candidate_k ∈ {10, 20, 50}
batch_size  ∈ {2, 4, 8}
final_k = 5
```

只用 development 调试、validation 选参数，test 最后报告一次。不要根据 test 反复调整候选数。

---

# 部署思考

## 19. Dense 与 BM25 是否并行调用

线上建议并行，因为两者互不依赖：

```text
请求到达
 ├─ 分支A：Query Embedding → Qdrant
 └─ 分支B：Tokenize → BM25
             ↓
         两路完成后RRF
```

顺序延迟约为：

```text
T_dense + T_bm25 + T_fusion
```

并行延迟约为：

```text
max(T_dense, T_bm25) + T_fusion
```

但 Day 8 先写顺序版本，因为更容易调试、复现实验。确认正确后再用线程池或异步任务实现真实并行；文档中的 `hybrid_parallel_estimate` 只是估计，不应冒充实测并行延迟。

## 20. 一个检索器失败时怎么办

推荐 fail-soft：

| 状态 | 行为 |
|---|---|
| Dense成功、BM25成功 | RRF融合 |
| Dense失败、BM25成功 | 降级为BM25 |
| Dense成功、BM25失败 | 降级为Dense |
| 两者都失败 | 返回503，不调用生成模型猜答案 |

响应和日志应包含：

```json
{
  "retrieval_mode": "bm25_fallback",
  "degraded": true,
  "failed_component": "dense",
  "request_id": "..."
}
```

降级不等于悄悄忽略错误。需要记录失败率、超时率和降级率，否则线上质量下降很难追踪。

## 21. Reranker 是否会成为瓶颈

很可能。每个请求有 20 个候选，就需要构造 20 个 `(query, passage)` 对。并发 10 个请求时瞬间可能产生 200 对。

部署措施：

- 模型启动时加载并 warm-up，不要每次请求重新加载；
- 使用有上限的队列和 semaphore 控制并发；
- 将不同请求的 pair 合并成动态 batch；
- 设置最大 candidate_k 和最大输入长度；
- 记录排队时间、模型推理时间和端到端时间；
- GPU OOM 或超时时回退到 RRF 顺序；
- 队列已满时限流，而不是无限堆积造成雪崩。

Reranker 降级原则：它只负责改善排序，因此失败时使用 Hybrid 原排序仍能提供服务；不能因为可选精排失败就让整个问答系统完全不可用。

## 22. 两天的时间安排

### Day 8（4小时）

| 时间 | 任务 | 交付 |
|---|---|---|
| 0:00–1:00 | 实现 tokenizer、BM25Index、构建脚本 | BM25 索引与 manifest |
| 1:00–1:30 | 实现 query_bm25 并抽查 | 关键词查询结果 |
| 1:30–2:30 | 实现 RRF 和单元测试 | Hybrid 模块 |
| 2:30–3:30 | 跑 Dense/BM25/Hybrid 对照 | 指标与 per-query 结果 |
| 3:30–4:00 | 分析指标和延迟 | 实验表与结论 |

### Day 9（4小时）

| 时间 | 任务 | 交付 |
|---|---|---|
| 0:00–0:30 | 选择 candidate_k=20、final_k=5 | 参数假设 |
| 0:30–1:30 | 实现 CrossEncoderReranker | Reranker 模块 |
| 1:30–2:30 | 冒烟、warm-up、调整 batch | 可运行推理配置 |
| 2:30–3:30 | 有无 Reranker A/B | 指标与延迟 |
| 3:30–4:00 | 质量—延迟分析 | 参数选择说明 |

## 23. 最终验收清单

Day 8：

- [ ] BM25 索引数量与 Chunk 数量一致；
- [ ] 精确命令查询能返回合理文档；
- [ ] RRF 单元测试通过；
- [ ] Dense、BM25、Hybrid 使用同一评测集；
- [ ] 同时记录质量指标与延迟；
- [ ] 不把并行延迟估计写成实测值。

Day 9：

- [ ] Reranker 只重排固定候选；
- [ ] 正式计时前完成 warm-up；
- [ ] 记录模型名、设备、batch、candidate_k、final_k；
- [ ] 比较 Hybrid 与 Hybrid + Reranker；
- [ ] 验证 candidate Recall 没有被重排改变；
- [ ] 记录失败时回退策略。

## 24. 完成后提交

先检查，不要使用 `git add .`：

```powershell
git status
```

Day 8 建议提交：

```powershell
git add pyproject.toml uv.lock
git add retrieval\bm25_store.py retrieval\hybrid_search.py
git add scripts\build_bm25_index.py scripts\query_bm25.py
git add scripts\evaluate_day8_hybrid.py tests\test_hybrid_retrieval.py
git add data\manifest\bm25_index.json
git add data\eval\day8_retrieval_comparison.json
git add docs\day8-day9-design-rationale.md
git commit -m "feat(retrieval): add bm25 and hybrid search"
git push origin main
```

Day 9 建议提交：

```powershell
git add pyproject.toml uv.lock
git add retrieval\reranker.py scripts\evaluate_day9_reranker.py
git add tests\test_hybrid_retrieval.py
git add data\eval\day9_reranker_comparison.json
git add docs\day8-day9-design-rationale.md
git commit -m "feat(retrieval): add cross-encoder reranking"
git push origin main
```

本地 BM25/Qdrant 索引通常体积较大且在 `.gitignore` 中；仓库提交构建脚本、manifest 和校验值即可。团队部署时应把索引产物放入版本化对象存储。

## 25. 参考资料

- Rank-BM25 官方仓库：https://github.com/dorianbrown/rank_bm25
- Sentence Transformers Retrieve & Re-Rank：https://www.sbert.net/examples/sentence_transformer/applications/retrieve_rerank/README.html
- BAAI BGE Reranker 模型卡：https://huggingface.co/BAAI/bge-reranker-v2-m3
