"""Pydantic contracts for the public OpsPilot HTTP API."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, HttpUrl


class QueryRequest(BaseModel):
    question: str = Field(min_length=3, max_length=4000)
    top_k: int = Field(default=5, ge=1, le=20)
    include_sources: bool = True


class SourceView(BaseModel):
    citation_id: str
    title: str
    source_url: str
    score: float
    text: str


class QueryResponse(BaseModel):
    request_id: str
    status: Literal["answer", "clarify", "refuse"]
    answer: str
    clarification: str = ""
    citations: list[str] = Field(default_factory=list)
    sources: list[SourceView] = Field(default_factory=list)
    timings_ms: dict[str, float] = Field(default_factory=dict)


class DocumentCreateRequest(BaseModel):
    title: str = Field(min_length=1, max_length=200)
    source_url: HttpUrl
    product: str = Field(min_length=1, max_length=80)
    content: str = Field(min_length=20, max_length=200_000)


class DocumentResponse(BaseModel):
    document_id: str
    title: str
    source_url: str
    product: str
    content_sha256: str


class FeedbackRequest(BaseModel):
    request_id: str = Field(min_length=8, max_length=100)
    rating: Literal["up", "down"]
    category: Literal["correct", "incorrect", "missing", "unsafe", "other"] = "other"
    comment: str = Field(default="", max_length=500)


class FeedbackResponse(BaseModel):
    accepted: bool
    feedback_id: str


class ErrorBody(BaseModel):
    error_code: str
    message: str
    request_id: str
    details: dict[str, Any] = Field(default_factory=dict)
