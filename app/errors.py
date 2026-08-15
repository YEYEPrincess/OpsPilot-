"""Stable application errors mapped to HTTP status codes."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class ErrorCode(StrEnum):
    INVALID_REQUEST = "INVALID_REQUEST"
    NOT_READY = "SERVICE_NOT_READY"
    QUERY_TIMEOUT = "QUERY_TIMEOUT"
    DOCUMENT_NOT_FOUND = "DOCUMENT_NOT_FOUND"
    INTERNAL_ERROR = "INTERNAL_ERROR"


@dataclass
class APIError(Exception):
    code: ErrorCode
    message: str
    status_code: int
    details: dict[str, Any] = field(default_factory=dict)
