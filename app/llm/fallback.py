"""Fallback chain: try providers in order until one answers.

Free tiers have tight rate limits and occasional outages, so a deployment can list several, e.g.
``LLM_PROVIDER=groq`` + ``LLM_FALLBACK_PROVIDERS=gemini,openrouter``. A provider that fails is put on
a short cooldown so the next requests go straight to a healthy one instead of paying the timeout again.
"""
from __future__ import annotations

import logging
import time

from app.llm.base import LLMError, LLMResponse
from app.observability.metrics import LLM_REQUESTS

logger = logging.getLogger(__name__)


class FallbackLLM:
    name = "fallback"

    def __init__(self, providers: list, cooldown_seconds: float = 30.0, clock=time.monotonic):
        if not providers:
            raise ValueError("FallbackLLM needs at least one provider")
        self.providers = providers
        self.cooldown_seconds = cooldown_seconds
        self._clock = clock
        self._down_until: dict[int, float] = {}

    @property
    def model(self) -> str:
        return self.providers[0].model

    @property
    def chain(self) -> list[str]:
        return [f"{p.name}:{p.model}" for p in self.providers]

    def chat(self, messages: list[dict], tools: list[dict], temperature: float = 0.2) -> LLMResponse:
        now = self._clock()
        healthy = [i for i in range(len(self.providers)) if self._down_until.get(i, 0) <= now]
        order = healthy or list(range(len(self.providers)))  # everything cooling down -> try all anyway
        errors: list[str] = []
        for i in order:
            provider = self.providers[i]
            try:
                resp = provider.chat(messages, tools, temperature=temperature)
            except LLMError as exc:
                self._down_until[i] = self._clock() + self.cooldown_seconds
                LLM_REQUESTS.labels(provider.name, "error").inc()
                errors.append(str(exc))
                logger.warning("llm provider failed, trying next", extra={"fields": {
                    "provider": provider.name, "model": provider.model, "error": str(exc)}})
                continue
            self._down_until.pop(i, None)
            resp.provider = resp.provider or provider.name
            resp.model = resp.model or provider.model
            return resp
        raise LLMError("all LLM providers failed: " + "; ".join(errors))
