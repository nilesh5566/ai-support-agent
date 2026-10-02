from __future__ import annotations

import logging
import time

logger = logging.getLogger(__name__)


class RateLimiter:
    """Fixed-window limiter per API key. Fails open if Redis is unavailable."""

    def __init__(self, kv, limit_per_minute: int):
        self.kv = kv
        self.limit = limit_per_minute

    def check(self, identity: str) -> tuple[bool, int]:
        if self.limit <= 0:
            return True, 0
        key = f"rl:{identity}:{int(time.time() // 60)}"
        try:
            count = int(self.kv.incr(key))
            if count == 1:
                self.kv.expire(key, 65)
        except Exception:
            logger.exception("rate limiter unavailable; allowing request")
            return True, self.limit
        return count <= self.limit, max(0, self.limit - count)
