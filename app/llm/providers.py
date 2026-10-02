"""Catalogue of LLM providers the agent can use.

Every provider here speaks the OpenAI Chat Completions + tool-calling protocol, so one client
implementation (``OpenAIProvider``) serves them all; a preset only supplies the base URL, the env var
that holds the key, and sensible default models. Defaults are just defaults: model catalogues on free
tiers change often, so ``LLM_MODEL`` / ``EMBEDDING_MODEL`` always override them.

Free options (no credit card at the time of writing - check each provider's current limits):
  groq        https://console.groq.com/keys       very fast; generous free tier
  gemini      https://aistudio.google.com/apikey  free tier incl. embeddings (not in EU/UK/CH)
  openrouter  https://openrouter.ai/keys          many ":free" models behind one key
  cerebras    https://cloud.cerebras.ai           very fast; small free context window
  mistral     https://console.mistral.ai          free "Experiment" plan incl. embeddings
  ollama      https://ollama.com                  runs locally, completely free, no key
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ProviderPreset:
    name: str
    label: str
    base_url: str | None          # None -> official OpenAI endpoint
    key_env: str | None           # env var holding the API key (None -> no key needed)
    default_model: str
    embedding_model: str | None = None   # None -> provider has no embeddings endpoint
    embedding_dim: int | None = None     # None -> detected by a probe call at startup
    free_tier: bool = True
    signup_url: str = ""


PRESETS: dict[str, ProviderPreset] = {
    "groq": ProviderPreset(
        "groq", "Groq", "https://api.groq.com/openai/v1", "GROQ_API_KEY",
        default_model="openai/gpt-oss-120b",
        signup_url="https://console.groq.com/keys"),
    "gemini": ProviderPreset(
        "gemini", "Google Gemini (AI Studio)", "https://generativelanguage.googleapis.com/v1beta/openai/",
        "GEMINI_API_KEY", default_model="gemini-flash-latest",
        embedding_model="gemini-embedding-001",
        signup_url="https://aistudio.google.com/apikey"),
    "openrouter": ProviderPreset(
        "openrouter", "OpenRouter", "https://openrouter.ai/api/v1", "OPENROUTER_API_KEY",
        default_model="openai/gpt-oss-120b:free",
        signup_url="https://openrouter.ai/keys"),
    "cerebras": ProviderPreset(
        "cerebras", "Cerebras", "https://api.cerebras.ai/v1", "CEREBRAS_API_KEY",
        default_model="gpt-oss-120b",
        signup_url="https://cloud.cerebras.ai"),
    "mistral": ProviderPreset(
        "mistral", "Mistral AI", "https://api.mistral.ai/v1", "MISTRAL_API_KEY",
        default_model="mistral-small-latest",
        embedding_model="mistral-embed", embedding_dim=1024,
        signup_url="https://console.mistral.ai/api-keys"),
    "ollama": ProviderPreset(
        "ollama", "Ollama (local)", "http://localhost:11434/v1", None,
        default_model="qwen3:8b",
        embedding_model="nomic-embed-text", embedding_dim=768,
        signup_url="https://ollama.com/download"),
    "openai": ProviderPreset(
        "openai", "OpenAI", None, "OPENAI_API_KEY",
        default_model="gpt-4o-mini",
        embedding_model="text-embedding-3-small", embedding_dim=1536,
        free_tier=False, signup_url="https://platform.openai.com/api-keys"),
    # Any other OpenAI-compatible server (vLLM, LM Studio, LiteLLM, Together, Azure...).
    "custom": ProviderPreset(
        "custom", "Custom OpenAI-compatible", None, None,  # key optional: LLM_API_KEY
        default_model="", embedding_model=None, free_tier=False),
}

LLM_PROVIDERS = ("mock", *PRESETS)


@dataclass(frozen=True)
class ResolvedProvider:
    name: str
    base_url: str | None
    api_key: str
    model: str


def _key_for(settings, preset: ProviderPreset) -> str | None:
    if settings.llm_api_key and settings.llm_provider == preset.name:
        return settings.llm_api_key  # generic override for the primary provider
    attr = (preset.key_env or "").lower()
    return getattr(settings, attr, None) if attr else None


def resolve(settings, name: str, *, primary: bool) -> ResolvedProvider:
    """Turn a provider name + settings into concrete connection details (or raise a helpful error)."""
    preset = PRESETS.get(name)
    if preset is None:
        raise ValueError(f"Unknown LLM provider {name!r}. Choose one of: {', '.join(LLM_PROVIDERS)}")

    base_url = preset.base_url
    if name == "ollama":
        base_url = settings.ollama_base_url
    if primary and settings.llm_base_url:
        base_url = settings.llm_base_url
    elif name == "openai" and settings.openai_base_url:
        base_url = settings.openai_base_url
    if name == "custom" and not base_url:
        raise ValueError("LLM_PROVIDER=custom requires LLM_BASE_URL (an OpenAI-compatible /v1 endpoint)")

    model = (settings.llm_model if primary else None) or (
        settings.openai_model if name == "openai" else preset.default_model)
    if not model:
        raise ValueError(f"LLM_PROVIDER={name} requires LLM_MODEL")

    key = _key_for(settings, preset)
    if preset.key_env and not key:
        hint = f" Get a free key at {preset.signup_url}" if preset.free_tier and preset.signup_url else ""
        raise ValueError(f"LLM provider {name!r} requires {preset.key_env}.{hint}")
    return ResolvedProvider(name=name, base_url=base_url, api_key=key or "not-needed", model=model)


def catalogue() -> list[dict]:
    """Public description of the available providers (served by /v1/llm/providers)."""
    return [{"name": p.name, "label": p.label, "free_tier": p.free_tier, "default_model": p.default_model,
             "key_env": p.key_env, "embeddings": p.embedding_model, "signup_url": p.signup_url}
            for p in PRESETS.values()]
