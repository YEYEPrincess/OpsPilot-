"""Small HTTP client shared by Streamlit and integration tests."""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import httpx


@dataclass
class OpsPilotAPIClient:
    base_url: str = "http://127.0.0.1:8000"
    timeout_seconds: float = 60.0

    def query(self, question: str, top_k: int = 5) -> dict[str, Any]:
        response = httpx.post(
            f"{self.base_url}/v1/query",
            json={"question": question, "top_k": top_k},
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()
        return response.json()

    def stream_query(
        self, question: str, top_k: int = 5
    ) -> Iterator[dict[str, Any]]:
        """Parse the event/data pairs defined by Server-Sent Events."""
        with httpx.stream(
            "POST",
            f"{self.base_url}/v1/query/stream",
            json={"question": question, "top_k": top_k},
            timeout=self.timeout_seconds,
        ) as response:
            response.raise_for_status()
            event_name = "message"
            for line in response.iter_lines():
                if line.startswith("event: "):
                    event_name = line[7:]
                elif line.startswith("data: "):
                    yield {
                        "event": event_name,
                        "data": json.loads(line[6:]),
                    }

    def send_feedback(
        self,
        request_id: str,
        rating: str,
        category: str = "other",
        comment: str = "",
    ) -> dict[str, Any]:
        response = httpx.post(
            f"{self.base_url}/v1/feedback",
            json={
                "request_id": request_id,
                "rating": rating,
                "category": category,
                "comment": comment,
            },
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()
        return response.json()
