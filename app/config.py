"""Runtime settings, loaded from environment variables (or a .env file)."""
from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_name: str = "AI Support Agent"
    environment: str = "dev"
    log_level: str = "INFO"

    # Storage
    database_url: str = "sqlite:///./data/support_agent.db"
    redis_url: str | None = None  # None -> in-process fallback (dev/tests only)

    # Vector store: "memory" (numpy, dev/tests) or "qdrant" (production)
    vector_backend: Literal["memory", "qdrant"] = "memory"
    qdrant_url: str = ":memory:"
    qdrant_collection: str = "support_kb"

    # LLM. "mock" is a deterministic offline planner so the whole system runs without keys.
    # Free hosted options: groq | gemini | openrouter | cerebras | mistral; local & free: ollama.
    # Paid: openai. Anything else OpenAI-compatible: custom (+ LLM_BASE_URL). See app/llm/providers.py.
    llm_provider: Literal["mock", "groq", "gemini", "openrouter", "cerebras", "mistral", "ollama", "openai",
                          "custom"] = "mock"
    llm_model: str | None = None        # overrides the provider's default model
    llm_base_url: str | None = None     # overrides the provider's endpoint
    llm_api_key: str | None = None      # generic key for the primary provider (instead of e.g. GROQ_API_KEY)
    llm_timeout_seconds: float = 30.0
    llm_max_retries: int = 3            # SDK retries on 429/5xx with backoff (honours Retry-After)
    # Comma-separated providers tried in order if the primary fails, e.g. "gemini,openrouter,mock"
    llm_fallback_providers: str = ""
    llm_fallback_cooldown_seconds: float = 30.0

    # Per-provider keys (only the ones you use need to be set)
    groq_api_key: str | None = None
    gemini_api_key: str | None = None
    openrouter_api_key: str | None = None
    cerebras_api_key: str | None = None
    mistral_api_key: str | None = None
    ollama_base_url: str = "http://localhost:11434/v1"
    openai_api_key: str | None = None
    openai_base_url: str | None = None
    openai_model: str = "gpt-4o-mini"

    # Embeddings for RAG. "hashing" is offline and free. Groq/OpenRouter/Cerebras have no embeddings
    # endpoint, so pair them with hashing (default) or a free embedder: gemini | mistral | ollama.
    embedding_provider: Literal["hashing", "gemini", "mistral", "ollama", "openai", "custom"] = "hashing"
    embedding_model: str | None = None   # overrides the provider's default embedding model
    embedding_base_url: str | None = None
    openai_embedding_model: str = "text-embedding-3-small"

    # Auth: comma separated "key:role" pairs. Roles: client (customer-facing) | admin (support staff/ops)
    api_keys: str = "dev-client-key:client,dev-admin-key:admin"

    agent_config_path: str = "config/agent.yaml"

    # Slack channel (optional)
    slack_signing_secret: str | None = None
    slack_bot_token: str | None = None

    # Escalations: optional webhook (e.g. Slack incoming webhook) notified on human hand-off
    escalation_webhook_url: str | None = None

    rate_limit_per_minute: int = 60
    memory_ttl_seconds: int = 86400
    memory_max_messages: int = 20
    max_upload_bytes: int = 5 * 1024 * 1024

    seed_demo_data: bool = True

    def parsed_api_keys(self) -> dict[str, str]:
        keys: dict[str, str] = {}
        for pair in self.api_keys.split(","):
            pair = pair.strip()
            if not pair:
                continue
            key, _, role = pair.partition(":")
            keys[key.strip()] = role.strip() or "client"
        return keys


@lru_cache
def get_settings() -> Settings:
    return Settings()
