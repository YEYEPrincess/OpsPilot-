"""Integration tests for Day 11 and Day 12 HTTP contracts."""

from fastapi.testclient import TestClient

from app.main import create_app


def test_liveness_and_readiness_are_different() -> None:
    client = TestClient(create_app(ready=False))
    assert client.get("/health/live").status_code == 200
    response = client.get("/health/ready")
    assert response.status_code == 503
    assert response.json()["error_code"] == "SERVICE_NOT_READY"


def test_query_has_request_id_sources_and_timings() -> None:
    client = TestClient(create_app())
    response = client.post(
        "/v1/query",
        json={"question": "Docker 容器退出后如何查看日志？", "top_k": 5},
        headers={"X-Request-ID": "req_test_query"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["request_id"] == "req_test_query"
    assert body["status"] == "answer"
    assert body["citations"] == ["S1"]
    assert body["sources"]
    assert "retrieval" in body["timings_ms"]


def test_pydantic_rejects_invalid_top_k() -> None:
    response = TestClient(create_app()).post(
        "/v1/query", json={"question": "valid question", "top_k": 100}
    )
    assert response.status_code == 422
    assert response.json()["error_code"] == "INVALID_REQUEST"


def test_document_create_list_delete() -> None:
    client = TestClient(create_app())
    response = client.post(
        "/v1/documents",
        json={
            "title": "Local runbook",
            "source_url": "https://example.com/runbook",
            "product": "demo",
            "content": "This is a sufficiently long troubleshooting runbook body.",
        },
    )
    assert response.status_code == 200
    document_id = response.json()["document_id"]
    assert any(
        row["document_id"] == document_id
        for row in client.get("/v1/documents").json()
    )
    assert client.delete(f"/v1/documents/{document_id}").json()["deleted"]
    assert client.delete(f"/v1/documents/{document_id}").status_code == 404


def test_stream_and_feedback_contracts() -> None:
    client = TestClient(create_app())
    with client.stream(
        "POST",
        "/v1/query/stream",
        json={"question": "PyTorch 如何确认 CUDA 可用？"},
    ) as response:
        text = "".join(response.iter_text())
    assert "event: meta" in text
    assert "event: token" in text
    assert "event: sources" in text
    assert "event: done" in text
    feedback = client.post(
        "/v1/feedback",
        json={
            "request_id": "req_feedback_123",
            "rating": "up",
            "category": "correct",
        },
    )
    assert feedback.status_code == 200
    assert feedback.json()["accepted"]
