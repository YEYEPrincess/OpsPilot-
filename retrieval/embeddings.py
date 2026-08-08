"""Embedding providers used by the Day 4 indexing and retrieval pipeline.

The default hash provider is intentionally deterministic and dependency-light so
the complete project can run offline.  A SentenceTransformers or
OpenAI-compatible provider can be selected in a GPU/API deployment.
"""

from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass
from typing import Any, Protocol


class EmbeddingProvider(Protocol):
    """Minimal interface shared by local and remote embedding implementations."""

    dimension: int
    name: str

    def encode(self, texts: list[str], batch_size: int = 32) -> list[list[float]]:
        """Encode a batch of texts into normalized dense vectors."""


TOKEN_RE = re.compile(r"[A-Za-z0-9_./:-]+|[\u4e00-\u9fff]")


@dataclass
class HashEmbeddingProvider:
    """Deterministic hashed word/character baseline with cosine-compatible vectors."""

    dimension: int = 384
    name: str = "hash-v1"

    def _encode_one(self, text: str) -> list[float]:
        vector = [0.0] * self.dimension
        tokens = TOKEN_RE.findall(text.lower())
        # Word features provide lexical matching; character trigrams help with
        # technical identifiers and minor spelling/format variations.
        features = tokens + [text.lower()[i : i + 3] for i in range(max(0, len(text) - 2))]
        for feature in features:
            digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
            index = int.from_bytes(digest[:4], "big") % self.dimension
            sign = 1.0 if digest[4] & 1 else -1.0
            vector[index] += sign
        norm = math.sqrt(sum(value * value for value in vector)) or 1.0
        return [value / norm for value in vector]

    def encode(self, texts: list[str], batch_size: int = 32) -> list[list[float]]:
        del batch_size
        return [self._encode_one(text) for text in texts]


class SentenceTransformerProvider:
    """Optional real semantic embedding provider, loaded only when selected."""

    name = "sentence-transformers"

    def __init__(self, model_name: str, device: str = "cpu") -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover - optional deployment path
            raise RuntimeError(
                "Install sentence-transformers to use the semantic local provider."
            ) from exc
        self.model = SentenceTransformer(model_name, device=device)
        self.dimension = int(self.model.get_sentence_embedding_dimension())
        self.name = f"sentence-transformers:{model_name}"

    def encode(self, texts: list[str], batch_size: int = 32) -> list[list[float]]:
        vectors = self.model.encode(
            texts, batch_size=batch_size, normalize_embeddings=True, show_progress_bar=False
        )
        return vectors.tolist()


def create_embedding_provider(
    provider: str = "hash", model_name: str = "", dimension: int = 384, device: str = "cpu"
) -> EmbeddingProvider:
    """Create an embedding backend from CLI/configuration values."""
    if provider == "hash":
        return HashEmbeddingProvider(dimension=dimension)
    if provider == "sentence-transformers":
        if not model_name:
            raise ValueError("--model is required for sentence-transformers provider")
        return SentenceTransformerProvider(model_name, device=device)
    raise ValueError(f"Unsupported embedding provider: {provider}")


def vector_metadata(provider: EmbeddingProvider) -> dict[str, Any]:
    """Return the metadata needed to reproduce an index."""
    return {"provider": provider.name, "dimension": provider.dimension}
