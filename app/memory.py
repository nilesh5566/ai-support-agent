"""Short-term conversation memory (sliding window in Redis). Long-term history lives in Postgres."""
from __future__ import annotations

import json
import logging

logger = logging.getLogger(__name__)


class ConversationMemory:
    def __init__(self, kv, ttl_seconds: int = 86400, max_messages: int = 20):
        self.kv = kv
        self.ttl = ttl_seconds
        self.max_messages = max_messages

    @staticmethod
    def _key(conversation_id: str) -> str:
        return f"conv:{conversation_id}:messages"

    def load(self, conversation_id: str) -> list[dict]:
        try:
            raw = self.kv.lrange(self._key(conversation_id), 0, -1)
        except Exception:
            logger.exception("memory load failed; continuing without history")
            return []
        return [json.loads(item) for item in raw]

    def append(self, conversation_id: str, role: str, content: str) -> None:
        key = self._key(conversation_id)
        try:
            self.kv.rpush(key, json.dumps({"role": role, "content": content}))
            self.kv.ltrim(key, -self.max_messages, -1)
            self.kv.expire(key, self.ttl)
        except Exception:
            logger.exception("memory append failed")

    def seed(self, conversation_id: str, messages: list[dict]) -> None:
        """Re-hydrate the cache from Postgres (e.g. after Redis eviction)."""
        for m in messages[-self.max_messages :]:
            self.append(conversation_id, m["role"], m["content"])
