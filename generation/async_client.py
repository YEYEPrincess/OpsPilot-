"""Async Chat Completions SSE and Ollama NDJSON adapters.

No automatic replay once answer text has been delivered. Transport errors expose
stable codes, never an upstream response body, URL, credential or user prompt.
"""
from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from contextlib import aclosing
from dataclasses import dataclass, field
from typing import Any

import httpx

from core.model_settings import ModelSettings


class ModelError(RuntimeError):
    def __init__(self, code: str, *, retryable: bool = False) -> None:
        super().__init__(code)
        self.code = code
        self.retryable = retryable


@dataclass
class ModelChunk:
    text: str = ""
    usage: dict[str, int] = field(default_factory=dict)


def compatible_root(base_url: str) -> str:
    """Accept server root or versioned API root without doubling /v1."""
    root = base_url.rstrip("/")
    if root.endswith("/chat/completions"):
        return root.removesuffix("/chat/completions")
    if root.endswith("/v1"):
        return root
    return root + "/v1"


async def sse_data(response: httpx.Response) -> AsyncIterator[str]:
    """Parse SSE frames, including comments, CRLF and multi-line data."""
    data: list[str] = []
    async for line in response.aiter_lines():
        if line == "":
            if data:
                yield "\n".join(data)
                data = []
        elif line.startswith("data:"):
            data.append(line[5:].removeprefix(" "))
        elif line == "data":
            data.append("")
    if data:
        yield "\n".join(data)


class AsyncModelClient:
    def __init__(
        self, settings: ModelSettings, *, client: httpx.AsyncClient | None = None
    ) -> None:
        self.settings = settings
        self._owns_client = client is None
        headers = {}
        key = settings.llm_api_key.get_secret_value()
        if key:
            headers["Authorization"] = f"Bearer {key}"
        self.client = client or httpx.AsyncClient(
            headers=headers,
            timeout=httpx.Timeout(settings.generation_timeout_seconds),
            limits=httpx.Limits(max_connections=20, max_keepalive_connections=10),
        )
        self.headers = headers
        root = settings.llm_base_url.rstrip("/")
        if settings.llm_provider == "ollama":
            root = root.removesuffix("/api/chat").removesuffix("/api")
            self.endpoint = root + "/api/chat"
            self.health_url = root + "/api/tags"
        else:
            root = compatible_root(root)
            self.endpoint = root + "/chat/completions"
            self.health_url = root + "/models"

    async def aclose(self) -> None:
        if self._owns_client:
            await self.client.aclose()

    @staticmethod
    def _check_status(response: httpx.Response) -> None:
        if response.is_error:
            code = response.status_code
            raise ModelError(
                f"MODEL_HTTP_{code}", retryable=code in {408, 429, 500, 502, 503, 504}
            )

    async def health(self) -> bool:
        """Check connectivity/auth and model listing; does not run inference."""
        try:
            async with asyncio.timeout(min(self.settings.generation_timeout_seconds, 5)):
                response = await self.client.get(self.health_url, headers=self.headers)
                self._check_status(response)
                data = response.json()
                if self.settings.llm_provider == "ollama":
                    names = {item["name"] for item in data["models"]}
                    name = self.settings.llm_model
                    return name in names or (":" not in name and name + ":latest" in names)
                return any(item["id"] == self.settings.llm_model for item in data["data"])
        except (httpx.HTTPError, TimeoutError, ModelError, ValueError, KeyError, TypeError):
            return False

    async def generate(self, messages: list[dict[str, str]]) -> tuple[str, dict[str, int]]:
        parts: list[str] = []
        usage: dict[str, int] = {}
        async with aclosing(self.stream(messages)) as stream:
            async for chunk in stream:
                parts.append(chunk.text)
                usage.update(chunk.usage)
        return "".join(parts), usage

    async def stream(self, messages: list[dict[str, str]]) -> AsyncIterator[ModelChunk]:
        emitted = False
        chars = 0
        payload: dict[str, Any] = {
            "model": self.settings.llm_model, "messages": messages, "stream": True,
        }
        if self.settings.generation_max_tokens is not None:
            if self.settings.llm_provider == "ollama":
                payload["options"] = {"num_predict": self.settings.generation_max_tokens}
            else:
                payload["max_tokens"] = self.settings.generation_max_tokens
        if self.settings.llm_thinking is not None:
            if self.settings.llm_provider == "ollama":
                payload["think"] = self.settings.llm_thinking == "enabled"
            else:
                payload["thinking"] = {"type": self.settings.llm_thinking}
        if self.settings.llm_stream_usage and self.settings.llm_provider != "ollama":
            payload["stream_options"] = {"include_usage": True}
        # No forced JSON mode/temperature: compatible backends differ in capabilities.
        for attempt in range(self.settings.generation_retries + 1):
            try:
                async with asyncio.timeout(self.settings.generation_timeout_seconds):
                    async with self.client.stream(
                        "POST", self.endpoint, json=payload, headers=self.headers
                    ) as response:
                        self._check_status(response)
                        async with aclosing(self._decode(response)) as decoded:
                            async for chunk in decoded:
                                chars += len(chunk.text)
                                if chars > self.settings.generation_max_output_chars:
                                    raise ModelError("MODEL_OUTPUT_TOO_LARGE")
                                if chunk.text:
                                    emitted = True
                                yield chunk
                if not emitted:
                    raise ModelError("MODEL_EMPTY_OUTPUT")
                return
            except ModelError as exc:
                error = exc
            except (TimeoutError, httpx.TimeoutException):
                error = ModelError("MODEL_TIMEOUT", retryable=True)
            except httpx.TransportError:
                error = ModelError("MODEL_CONNECTION_ERROR", retryable=True)
            except (ValueError, KeyError, IndexError, TypeError, AttributeError):
                error = ModelError("MODEL_INVALID_RESPONSE")
            if emitted or not error.retryable or attempt >= self.settings.generation_retries:
                raise error from None
            await asyncio.sleep(self.settings.generation_backoff_seconds * (2 ** attempt))

    async def _decode(self, response: httpx.Response) -> AsyncIterator[ModelChunk]:
        if self.settings.llm_provider == "ollama":
            async for line in response.aiter_lines():
                if not line.strip():
                    continue
                body = json.loads(line)
                if "error" in body:
                    raise ModelError("MODEL_UPSTREAM_ERROR")
                text = body.get("message", {}).get("content", "")
                if not isinstance(text, str):
                    raise ModelError("MODEL_INVALID_RESPONSE")
                usage = {}
                if body.get("done"):
                    for source, target in (
                        ("prompt_eval_count", "prompt_tokens"),
                        ("eval_count", "completion_tokens"),
                    ):
                        if isinstance(body.get(source), int):
                            usage[target] = body[source]
                    if len(usage) == 2:
                        usage["total_tokens"] = sum(usage.values())
                if text or usage:
                    yield ModelChunk(text, usage)
                if body.get("done"):
                    if body.get("done_reason") not in {None, "stop"}:
                        raise ModelError("MODEL_INCOMPLETE_OUTPUT")
                    return
        else:
            async with aclosing(sse_data(response)) as frames:
                async for data in frames:
                    if data.strip() == "[DONE]":
                        return
                    body = json.loads(data)
                    if "error" in body:
                        raise ModelError("MODEL_UPSTREAM_ERROR")
                    usage = {
                        key: value for key, value in (body.get("usage") or {}).items()
                        if key in {"prompt_tokens", "completion_tokens", "total_tokens"}
                        and isinstance(value, int)
                    }
                    choices = body.get("choices", [])
                    text = ""
                    if choices:
                        choice = choices[0]
                        if choice.get("finish_reason") not in {None, "stop"}:
                            raise ModelError("MODEL_INCOMPLETE_OUTPUT")
                        delta = choice.get("delta", {})
                        if delta.get("refusal"):
                            raise ModelError("MODEL_REFUSAL")
                        text = delta.get("content") or ""
                        if not isinstance(text, str):
                            raise ModelError("MODEL_INVALID_RESPONSE")
                    if text or usage:
                        yield ModelChunk(text, usage)
        # EOF without a terminal marker must not silently turn a partial answer into success.
        raise ModelError("MODEL_TRUNCATED_STREAM")
