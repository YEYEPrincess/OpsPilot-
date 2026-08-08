"""Small Qdrant adapter for local development and remote deployment."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams


@dataclass
class SearchHit:
    """A normalized search result independent of Qdrant response details."""

    score: float
    payload: dict[str, Any]


def stable_point_id(chunk_id: str) -> int:
    """Map a chunk id to a stable positive integer accepted by Qdrant."""
    return (
        int.from_bytes(hashlib.sha256(chunk_id.encode("utf-8")).digest()[:8], "big")
        & 0x7FFF_FFFF_FFFF_FFFF
    )


class QdrantVectorStore:
    """Create a collection, upsert vectors, and execute cosine Top-k queries."""

    def __init__(self, path: str | Path, collection: str, vector_size: int) -> None:
        self.path = str(path)
        self.collection = collection
        self.vector_size = vector_size
        self.client = QdrantClient(path=self.path)

    def recreate(self) -> None:
        """Drop and recreate the collection so an index build is reproducible."""
        if self.client.collection_exists(self.collection):
            self.client.delete_collection(self.collection)
        self.client.create_collection(
            collection_name=self.collection,
            vectors_config=VectorParams(size=self.vector_size, distance=Distance.COSINE),
        )

    def ensure_collection(self) -> None:
        """Create the collection only when it does not exist."""
        if not self.client.collection_exists(self.collection):
            self.client.create_collection(
                collection_name=self.collection,
                vectors_config=VectorParams(size=self.vector_size, distance=Distance.COSINE),
            )

    def upsert(self, vectors: list[list[float]], payloads: list[dict[str, Any]]) -> None:
        """Write a batch of vectors and payloads to Qdrant."""
        points = [
            PointStruct(
                id=stable_point_id(str(payload["chunk_id"])), vector=vector, payload=payload
            )
            for vector, payload in zip(vectors, payloads, strict=True)
        ]
        self.client.upsert(collection_name=self.collection, points=points, wait=True)

    def search(self, vector: list[float], limit: int = 5, query_text: str = "") -> list[SearchHit]:
        """Return Top-k matches, optionally applying a transparent lexical tie-breaker."""
        candidate_limit = max(limit * 10, 100) if query_text else limit
        response = self.client.query_points(
            collection_name=self.collection,
            query=vector,
            limit=candidate_limit,
            with_payload=True,
        )
        hits = [
            SearchHit(score=float(point.score), payload=dict(point.payload or {}))
            for point in response.points
        ]
        if not query_text:
            return hits[:limit]
        query_terms = set(re.findall(r"[A-Za-z0-9_./:-]+|[\u4e00-\u9fff]", query_text.lower()))
        if not query_terms:
            return hits[:limit]
        for hit in hits:
            doc_terms = set(
                re.findall(
                    r"[A-Za-z0-9_./:-]+|[\u4e00-\u9fff]",
                    hit.payload.get("text", "").lower(),
                )
            )
            lexical = len(query_terms & doc_terms) / len(query_terms)
            hit.payload["vector_score"] = hit.score
            hit.payload["lexical_overlap"] = lexical
            hit.score = 0.35 * hit.score + 0.65 * lexical
        return sorted(hits, key=lambda item: item.score, reverse=True)[:limit]

    def count(self) -> int:
        """Return the number of indexed points."""
        return int(self.client.count(collection_name=self.collection, exact=True).count)

    def close(self) -> None:
        """Close local Qdrant resources."""
        self.client.close()
