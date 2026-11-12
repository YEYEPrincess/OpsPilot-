"""Validated settings; live inference is opt-in and does not select a model for the user."""
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parents[1]


class ModelSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=ROOT / ".env", env_file_encoding="utf-8", extra="ignore"
    )
    opspilot_mode: Literal["demo", "model"] = "demo"
    llm_provider: Literal["openai-compatible", "vllm", "ollama"] = "openai-compatible"
    llm_base_url: str = ""
    llm_api_key: SecretStr = SecretStr("")
    llm_model: str = ""
    generation_timeout_seconds: float = Field(default=30, gt=0)
    generation_retries: int = Field(default=2, ge=0, le=5)
    generation_backoff_seconds: float = Field(default=0.25, ge=0, le=10)
    generation_max_tokens: int | None = Field(default=None, ge=1, le=384000)
    # Optional vendor extension; set only for backends documented to support it.
    llm_thinking: Literal["enabled", "disabled"] | None = None
    llm_stream_usage: bool = False
    generation_max_output_chars: int = Field(default=32000, ge=100, le=200000)
    request_timeout_seconds: float = Field(default=60, gt=0)
    answer_cache_max_entries: int = Field(default=128, ge=0, le=10000)
    answer_cache_ttl_seconds: float = Field(default=300, gt=0, le=86400)
    event_log_path: Path | None = None
    log_json_stdout: bool = False
    grounding_min_score: float = 0.35
    clarification_policy: Literal["legacy", "context_v1"] = "legacy"
    retrieval_mode: Literal["baseline", "hybrid_rerank"] = "baseline"
    bm25_index_path: Path = ROOT / "data/index/bm25/opspilot_bm25_v1.json.gz"
    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    reranker_device: str = "cpu"
    reranker_quantize: bool = False
    reranker_threads: int = Field(default=2, ge=1, le=32)
    reranker_batch_size: int = Field(default=4, ge=1, le=64)
    reranker_max_length: int = Field(default=512, ge=32, le=8192)
    retrieval_candidate_k: int = Field(default=20, ge=5, le=100)
    rrf_k: int = Field(default=60, ge=1)
    qdrant_remote_url: str = ""
    qdrant_api_key: SecretStr = SecretStr("")
    vector_manifest: Path = ROOT / "data/manifest/vector_index.json"

    @field_validator("llm_provider", mode="before")
    @classmethod
    def normalize_provider(cls, value: str) -> str:
        return value.replace("_", "-")

    @model_validator(mode="after")
    def validate_live(self) -> "ModelSettings":
        if self.opspilot_mode == "model":
            if not self.llm_base_url or not self.llm_model:
                raise ValueError("Model mode requires LLM_BASE_URL and LLM_MODEL")
            from urllib.parse import urlsplit
            parts = urlsplit(self.llm_base_url)
            if parts.scheme not in {"http", "https"} or not parts.netloc:
                raise ValueError("LLM_BASE_URL must be an HTTP(S) URL")
            if parts.username or parts.password or parts.query or parts.fragment:
                raise ValueError("Put credentials in LLM_API_KEY, not in the base URL")
        return self
