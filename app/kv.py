"""Key-value backend: Redis in production, an in-process stand-in for local dev and tests."""
from __future__ import annotations

import logging
import threading
import time
from typing import Any

logger = logging.getLogger(__name__)


def _slice(items: list, start: int, end: int) -> list:
    n = len(items)
    if start < 0:
        start = max(n + start, 0)
    if end < 0:
        end = n + end
    return items[start : end + 1]


class InMemoryKV:
    """Implements the subset of the Redis API this service uses."""

    def __init__(self):
        self._data: dict[str, Any] = {}
        self._expiry: dict[str, float] = {}
        self._lock = threading.Lock()

    def _purge(self, key: str) -> None:
        exp = self._expiry.get(key)
        if exp is not None and exp < time.time():
            self._data.pop(key, None)
            self._expiry.pop(key, None)

    def ping(self) -> bool:
        return True

    def get(self, key: str):
        with self._lock:
            self._purge(key)
            return self._data.get(key)

    def set(self, key: str, value, ex: int | None = None):
        with self._lock:
            self._data[key] = value
            if ex:
                self._expiry[key] = time.time() + ex
            return True

    def incr(self, key: str) -> int:
        with self._lock:
            self._purge(key)
            value = int(self._data.get(key, 0)) + 1
            self._data[key] = value
            return value

    def expire(self, key: str, seconds: int) -> bool:
        with self._lock:
            if key in self._data:
                self._expiry[key] = time.time() + seconds
                return True
            return False

    def rpush(self, key: str, *values) -> int:
        with self._lock:
            self._purge(key)
            lst = self._data.setdefault(key, [])
            lst.extend(values)
            return len(lst)

    def lrange(self, key: str, start: int, end: int) -> list:
        with self._lock:
            self._purge(key)
            return list(_slice(self._data.get(key, []), start, end))

    def ltrim(self, key: str, start: int, end: int) -> bool:
        with self._lock:
            if key in self._data:
                self._data[key] = _slice(self._data[key], start, end)
            return True

    def delete(self, *keys: str) -> int:
        with self._lock:
            removed = 0
            for k in keys:
                if self._data.pop(k, None) is not None:
                    removed += 1
                self._expiry.pop(k, None)
            return removed


def create_kv(redis_url: str | None):
    if not redis_url:
        logger.warning("REDIS_URL not set: using in-process KV store (not for multi-replica production)")
        return InMemoryKV()
    import redis

    return redis.Redis.from_url(redis_url, decode_responses=True, socket_timeout=2, socket_connect_timeout=2)
