"""Bounded, process-local LRU/TTL cache for already validated answer content."""
from __future__ import annotations

import time
from collections import OrderedDict
from collections.abc import Callable
from copy import deepcopy
from typing import Any


class AnswerCache:
    def __init__(
        self, max_entries: int, ttl_seconds: float, *,
        clock: Callable[[], float] = time.monotonic,
    ):
        self.max_entries = max_entries
        self.ttl_seconds = ttl_seconds
        self.clock = clock
        self.entries: OrderedDict[str, tuple[float, dict[str, Any]]] = OrderedDict()

    def get(self, key: str) -> dict[str, Any] | None:
        if not self.max_entries:
            return None
        item = self.entries.get(key)
        if item is None:
            return None
        expires, value = item
        if expires <= self.clock():
            del self.entries[key]
            return None
        self.entries.move_to_end(key)
        return deepcopy(value)

    def put(self, key: str, value: dict[str, Any]) -> None:
        if not self.max_entries:
            return
        now = self.clock()
        for expired in [k for k, (deadline, _) in self.entries.items() if deadline <= now]:
            del self.entries[expired]
        self.entries[key] = (now + self.ttl_seconds, deepcopy(value))
        self.entries.move_to_end(key)
        while len(self.entries) > self.max_entries:
            self.entries.popitem(last=False)
