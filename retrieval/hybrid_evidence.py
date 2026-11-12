"""Real Dense/BM25/RRF/BGE retrieval; BGE logits are retained separately."""
from __future__ import annotations
import asyncio
import json
import math
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any
from core.model_settings import ROOT
from retrieval.embeddings import HashEmbeddingProvider
from retrieval.qdrant_store import QdrantVectorStore
from retrieval.bm25_store import BM25Index
from retrieval.hybrid_search import reciprocal_rank_fusion


def sigmoid(logit: float) -> float:
    if logit >= 0:
        return 1.0 / (1.0 + math.exp(-logit))
    z = math.exp(logit)
    return z / (1.0 + z)


class HybridEvidenceRetriever:
    def __init__(self, settings: Any) -> None:
        self.settings = settings
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="hybrid")
        self.store = None
        self.last_trace: dict[str, Any] = {}

    def _path(self, value: str | Path) -> Path:
        p = Path(value)
        return p if p.is_absolute() else ROOT / p

    def _open(self) -> None:
        if self.store is not None:
            return
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        torch.set_num_threads(self.settings.reranker_threads)
        manifest = json.loads(self._path(self.settings.vector_manifest).read_text(encoding="utf-8"))
        if manifest["provider"] != "hash-v1":
            raise ValueError("Hybrid experiment requires the recorded hash-v1 index")
        self.encoder = HashEmbeddingProvider(int(manifest["dimension"]))
        self.store = QdrantVectorStore(
            ROOT / manifest["qdrant_path"].replace("\\", "/"), manifest["collection"],
            self.encoder.dimension, url=self.settings.qdrant_remote_url,
            api_key=self.settings.qdrant_api_key.get_secret_value(),
        )
        self.bm25 = BM25Index.load(self._path(self.settings.bm25_index_path))
        self.tokenizer = AutoTokenizer.from_pretrained(self.settings.reranker_model, local_files_only=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.settings.reranker_model, local_files_only=True,
        ).to(self.settings.reranker_device).eval()
        self.parameter_count = sum(p.numel() for p in self.model.parameters())
        self.parameter_bytes = sum(p.numel() * p.element_size() for p in self.model.parameters())
        if getattr(self.settings, "reranker_quantize", False):
            self.model = torch.quantization.quantize_dynamic(self.model, {torch.nn.Linear}, dtype=torch.qint8)
        self.torch = torch

    def _search(self, question: str, top_k: int) -> list[dict[str, Any]]:
        self._open()
        started = time.perf_counter()
        stage = {}
        t = time.perf_counter()
        dense = self.store.search(self.encoder.encode([question])[0], self.settings.retrieval_candidate_k, query_text="")
        stage["dense"] = (time.perf_counter() - t) * 1000
        t = time.perf_counter()
        bm25 = self.bm25.search(question, self.settings.retrieval_candidate_k)
        stage["bm25"] = (time.perf_counter() - t) * 1000
        t = time.perf_counter()
        fused = reciprocal_rank_fusion(dense, bm25, limit=self.settings.retrieval_candidate_k, rrf_k=self.settings.rrf_k)
        stage["rrf"] = (time.perf_counter() - t) * 1000
        t = time.perf_counter()
        logits = []
        lengths = []
        batch_size = self.settings.reranker_batch_size
        for start in range(0, len(fused), batch_size):
            batch = fused[start:start + batch_size]
            pairs = [(question, str(x.payload.get("text", ""))) for x in batch]
            lengths.extend(len(ids) for ids in self.tokenizer(pairs, truncation=False)["input_ids"])
            inputs = self.tokenizer(pairs, padding=True, truncation=True,
                                    max_length=self.settings.reranker_max_length, return_tensors="pt")
            inputs = {key: value.to(self.settings.reranker_device) for key, value in inputs.items()}
            with self.torch.inference_mode():
                logits.extend(self.model(**inputs).logits.view(-1).float().cpu().tolist())
        stage["rerank"] = (time.perf_counter() - t) * 1000
        order = sorted(range(len(fused)), key=lambda index: logits[index], reverse=True)
        evidence = []
        for index in order[:top_k]:
            hit = fused[index]
            evidence.append({
                "citation_id": f"S{len(evidence)+1}",
                "chunk_id": hit.payload["chunk_id"],
                "title": " > ".join(hit.payload.get("section_path") or []),
                "source_url": hit.payload.get("source_url", ""),
                "text": hit.payload.get("text", ""),
                "score": sigmoid(logits[index]),
                "score_kind": "sigmoid_bge_logit_uncalibrated",
                "reranker_logit": logits[index], "rrf_score": hit.score,
                "dense_rank": hit.dense_rank, "bm25_rank": hit.bm25_rank,
                "retrieval_timings_ms": stage,
            })
        self.last_trace = {
            "question": question, "dense_ids": [x.payload["chunk_id"] for x in dense],
            "bm25_ids": [x.payload["chunk_id"] for x in bm25],
            "fused": [{"chunk_id":x.payload["chunk_id"], "rrf_score":x.score,
                       "bge_logit":logits[i], "bge_score":sigmoid(logits[i]),
                       "input_tokens_before_truncation":lengths[i]}
                      for i,x in enumerate(fused)],
            "evidence": evidence, "timings_ms": stage,
            "retrieval_total_ms": (time.perf_counter()-started)*1000,
            "parameter_count": self.parameter_count, "parameter_bytes": self.parameter_bytes,
            "quantization": "dynamic-int8-linear" if getattr(self.settings,"reranker_quantize",False) else "fp32",
        }
        return evidence

    async def retrieve(self, question: str, top_k: int) -> list[dict[str, Any]]:
        return await asyncio.get_running_loop().run_in_executor(self.executor, self._search, question, top_k)

    async def health(self) -> bool:
        def check():
            self._open()
            return self.store.count() > 0
        try:
            return await asyncio.get_running_loop().run_in_executor(self.executor, check)
        except Exception:
            return False

    async def aclose(self) -> None:
        def close():
            if self.store:
                self.store.close()
                self.store = None
        await asyncio.get_running_loop().run_in_executor(self.executor, close)
        self.executor.shutdown(wait=False)