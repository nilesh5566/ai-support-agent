from __future__ import annotations

from app.llm.fallback import FallbackLLM
from app.llm.mock import MockLLM
from app.llm.openai_provider import OpenAIProvider
from app.llm.providers import resolve


def _build(settings, name: str, *, primary: bool):
    if name == "mock":
        return MockLLM()
    r = resolve(settings, name, primary=primary)
    return OpenAIProvider(r.api_key, r.model, r.base_url, settings.llm_timeout_seconds,
                          name=r.name, max_retries=settings.llm_max_retries)


def fallback_names(settings) -> list[str]:
    names = [n.strip().lower() for n in (settings.llm_fallback_providers or "").split(",") if n.strip()]
    return [n for i, n in enumerate(names) if n != settings.llm_provider and n not in names[:i]]


def create_llm(settings):
    primary = _build(settings, settings.llm_provider, primary=True)
    extras = fallback_names(settings)
    if not extras:
        return primary
    return FallbackLLM([primary, *(_build(settings, n, primary=False) for n in extras)],
                       cooldown_seconds=settings.llm_fallback_cooldown_seconds)
