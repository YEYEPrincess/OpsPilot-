"""Generation clients with a deterministic offline fallback and retry handling."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Protocol

import httpx


class GenerationClient(Protocol):
    """Interface implemented by mock and OpenAI-compatible generation backends."""

    def generate(self, prompt: str, evidence: list[dict[str, Any]]) -> str:
        """Return model text, expected to be a JSON object."""


@dataclass
class MockGenerationClient:
    """Extractive fallback that makes the complete demo work without a model server."""

    name: str = "mock-extractive-v1"

    def generate(self, prompt: str, evidence: list[dict[str, Any]]) -> str:
        del prompt
        if not evidence:
            answer = "证据不足，无法基于当前知识库给出可靠判断。"
            citations: list[str] = []
        else:
            answer = "根据检索到的文档证据，优先检查：" + evidence[0].get("text", "")[:500]
            citations = [f"S{index}" for index in range(1, min(3, len(evidence)) + 1)]
        return json.dumps(
            {
                "answer": answer,
                "possible_causes": [],
                "steps": ["先核对引用文档中的版本、配置和运行环境，再进行变更。"],
                "risks": ["不要直接在生产环境执行未验证的删除或覆盖操作。"],
                "citations": citations,
                "clarification": "" if evidence else "请提供完整错误日志、版本和运行命令。",
            },
            ensure_ascii=False,
        )


@dataclass
class OpenAICompatibleClient:
    """Call vLLM, Ollama or a hosted OpenAI-compatible chat endpoint."""

    base_url: str
    api_key: str
    model: str
    timeout_seconds: float = 60.0
    retries: int = 2

    def generate(self, prompt: str, evidence: list[dict[str, Any]]) -> str:
        del evidence
        url = self.base_url.rstrip("/") + "/v1/chat/completions"
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
            "response_format": {"type": "json_object"},
        }
        last_error: Exception | None = None
        for attempt in range(self.retries + 1):
            try:
                response = httpx.post(
                    url, headers=headers, json=payload, timeout=self.timeout_seconds
                )
                response.raise_for_status()
                body = response.json()
                return str(body["choices"][0]["message"]["content"])
            except (httpx.HTTPError, KeyError, IndexError, ValueError) as exc:
                last_error = exc
                if attempt < self.retries:
                    continue
        raise RuntimeError(f"Generation request failed after retries: {last_error}") from last_error
