"""Structured logs, redaction and lightweight in-process metrics."""

from __future__ import annotations

import hashlib
import json
import re
import statistics
import threading
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

SECRET_KEYS = {
    "authorization",
    "api_key",
    "apikey",
    "password",
    "secret",
    "token",
}
EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
BEARER_RE = re.compile(r"(?i)bearer\s+[A-Za-z0-9._~+/-]+")


def redact(value: Any, key: str = "") -> Any:
    """Recursively remove secrets and obvious personal identifiers."""
    if key.lower() in SECRET_KEYS:
        return "[REDACTED]"
    if isinstance(value, dict):
        return {str(k): redact(v, str(k)) for k, v in value.items()}
    if isinstance(value, list):
        return [redact(item) for item in value]
    if isinstance(value, str):
        return BEARER_RE.sub(
            "Bearer [REDACTED]", EMAIL_RE.sub("[EMAIL]", value)
        )
    return value


def question_fingerprint(question: str) -> dict[str, Any]:
    """Log stable metadata instead of raw user text."""
    return {
        "question_sha256": hashlib.sha256(
            question.encode("utf-8")
        ).hexdigest(),
        "question_chars": len(question),
    }


@dataclass
class JsonEventLogger:
    """Append one JSON object per line so logs remain machine-readable."""

    path: Path | None = None
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def emit(self, event: str, **fields: Any) -> dict[str, Any]:
        record = {
            "timestamp": datetime.now(UTC).isoformat(),
            "event": event,
            **redact(fields),
        }
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self._lock, self.path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        return record


def percentile(values: list[float], quantile: float) -> float:
    """Nearest-rank percentile; deterministic and dependency-free."""
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, int(len(ordered) * quantile + 0.999) - 1))
    return ordered[index]


def summarize_log_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate query logs into latency, reliability and stage views."""
    completed = [row for row in records if row.get("event") == "query_completed"]
    total_ms = [float(row.get("latency_ms", 0.0)) for row in completed]
    stages: dict[str, list[float]] = defaultdict(list)
    errors: Counter[str] = Counter()
    cache_hits = 0
    token_total = 0
    for row in completed:
        for name, value in row.get("stage_ms", {}).items():
            if name == "total":
                continue
            stages[str(name)].append(float(value))
        if row.get("error_type"):
            errors[str(row["error_type"])] += 1
        cache_hits += int(bool(row.get("cache_hit")))
        token_total += int(row.get("token_total", 0))
    count = len(completed)
    return {
        "request_count": count,
        "latency_ms": {
            "mean": round(statistics.mean(total_ms), 3) if total_ms else 0.0,
            "p50": round(percentile(total_ms, 0.50), 3),
            "p95": round(percentile(total_ms, 0.95), 3),
            "max": round(max(total_ms), 3) if total_ms else 0.0,
        },
        "stage_mean_ms": {
            name: round(statistics.mean(values), 3)
            for name, values in sorted(stages.items())
        },
        "error_rate": round(sum(errors.values()) / count, 4) if count else 0.0,
        "errors": dict(errors),
        "cache_hit_rate": round(cache_hits / count, 4) if count else 0.0,
        "token_total": token_total,
    }


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Load a JSONL log while ignoring blank lines."""
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]
