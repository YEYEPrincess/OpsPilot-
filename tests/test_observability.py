"""Tests for Day 13 privacy and metrics behaviour."""

from core.observability import redact, summarize_log_records


def test_redact_removes_nested_secrets_and_email() -> None:
    clean = redact(
        {
            "api_key": "sk-secret",
            "nested": {
                "authorization": "Bearer abc.def",
                "message": "contact user@example.com",
            },
        }
    )
    assert clean["api_key"] == "[REDACTED]"
    assert clean["nested"]["authorization"] == "[REDACTED]"
    assert clean["nested"]["message"] == "contact [EMAIL]"


def test_metrics_show_stage_latency_and_failures() -> None:
    rows = [
        {
            "event": "query_completed",
            "latency_ms": 10,
            "stage_ms": {"retrieval": 2, "rerank": 5, "generation": 3},
            "cache_hit": False,
            "token_total": 20,
            "error_type": "",
        },
        {
            "event": "query_completed",
            "latency_ms": 20,
            "stage_ms": {"retrieval": 4, "rerank": 10, "generation": 6},
            "cache_hit": True,
            "token_total": 30,
            "error_type": "timeout",
        },
    ]
    summary = summarize_log_records(rows)
    assert summary["request_count"] == 2
    assert summary["latency_ms"]["p50"] == 10
    assert summary["stage_mean_ms"]["rerank"] == 7.5
    assert summary["cache_hit_rate"] == 0.5
    assert summary["error_rate"] == 0.5
    assert summary["token_total"] == 50
