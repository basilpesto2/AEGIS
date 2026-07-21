from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from threading import RLock
import time
from typing import Generic, TypeVar


T = TypeVar("T")


@dataclass(frozen=True)
class CacheStats:
    entries: int
    max_entries: int
    ttl_seconds: float
    hits: int
    misses: int
    evictions: int


class BoundedTTLCache(Generic[T]):
    """Small thread-safe LRU cache with a per-entry time-to-live."""

    def __init__(self, *, max_entries: int = 128, ttl_seconds: float = 300.0) -> None:
        if max_entries <= 0:
            raise ValueError("max_entries must be positive.")
        if ttl_seconds <= 0.0:
            raise ValueError("ttl_seconds must be positive.")
        self.max_entries = int(max_entries)
        self.ttl_seconds = float(ttl_seconds)
        self._entries: OrderedDict[str, tuple[float, T]] = OrderedDict()
        self._lock = RLock()
        self._hits = 0
        self._misses = 0
        self._evictions = 0

    def get(self, key: str) -> T | None:
        now = time.monotonic()
        with self._lock:
            item = self._entries.pop(key, None)
            if item is None:
                self._misses += 1
                return None
            expires_at, value = item
            if expires_at <= now:
                self._misses += 1
                self._evictions += 1
                return None
            self._entries[key] = item
            self._hits += 1
            return value

    def put(self, key: str, value: T) -> None:
        expires_at = time.monotonic() + self.ttl_seconds
        with self._lock:
            self._entries.pop(key, None)
            self._entries[key] = (expires_at, value)
            self._purge_expired_locked()
            while len(self._entries) > self.max_entries:
                self._entries.popitem(last=False)
                self._evictions += 1

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()

    def stats(self) -> CacheStats:
        with self._lock:
            self._purge_expired_locked()
            return CacheStats(
                entries=len(self._entries),
                max_entries=self.max_entries,
                ttl_seconds=self.ttl_seconds,
                hits=self._hits,
                misses=self._misses,
                evictions=self._evictions,
            )

    def _purge_expired_locked(self) -> None:
        now = time.monotonic()
        expired = [
            key for key, (expires_at, _) in self._entries.items() if expires_at <= now
        ]
        for key in expired:
            del self._entries[key]
            self._evictions += 1
