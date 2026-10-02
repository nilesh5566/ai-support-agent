"""Embedding providers.

* HashingEmbedder: deterministic, offline, zero-cost feature-hashing embeddings (unigrams+bigrams).
  Good enough for demos, CI and air-gapped pilots.
* OpenAICompatEmbedder: semantic embeddings via OpenAI, Gemini, Mistral, Ollama or any compatible API.
"""
from __future__ import annotations

import hashlib
import math
from typing import Protocol

import numpy as np

from app.rag.text import tokenize


class Embedder(Protocol):
    name: str
    dim: int

    def embed(self, texts: list[str]) -> list[list[float]]: ...


class HashingEmbedder:
    name = "hashing"

    def __init__(self, dim: int = 512):
        self.dim = dim

    def _vector(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dim, dtype=np.float32)
        words = tokenize(text)
        features = [(w, 1.0) for w in words] + [(f"{a}_{b}", 0.7) for a, b in zip(words, words[1:], strict=False)]
        counts: dict[str, float] = {}
        for feat, weight in features:
            counts[feat] = counts.get(feat, 0.0) + weight
        for feat, weight in counts.items():
            h = int.from_bytes(hashlib.md5(feat.encode()).digest()[:8], "little")
            sign = 1.0 if (h >> 63) & 1 else -1.0
            vec[h % self.dim] += sign * (1.0 + math.log(weight)) if weight >= 1 else sign * weight
        norm = float(np.linalg.norm(vec))
        return vec / norm if norm > 0 else vec

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._vector(t).tolist() for t in texts]


class OpenAICompatEmbedder:
    """Semantic embeddings from any OpenAI-compatible /embeddings endpoint
    (OpenAI, Gemini, Mistral, Ollama, custom)."""

    def __init__(self, api_key: str, model: str, base_url: str | None = None, timeout: float = 30.0,
                 name: str = "openai", dim: int | None = None, batch_size: int = 64, client=None):
        if client is None:
            from openai import OpenAI

            client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout, max_retries=3)
        self.client = client
        self.model = model
        self.batch_size = batch_size
        # The name feeds the Qdrant collection name, so switching embedders never mixes vector spaces.
        self.name = f"{name}-{model}".replace("/", "-").replace(":", "-")
        # Unknown dimension -> probe once (needed up-front to create the Qdrant collection).
        self.dim = dim or len(self.embed(["dimension probe"])[0])

    def embed(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for i in range(0, len(texts), self.batch_size):
            resp = self.client.embeddings.create(model=self.model, input=texts[i : i + self.batch_size])
            out.extend(d.embedding for d in sorted(resp.data, key=lambda d: d.index))
        return out


# Backwards-compatible alias
OpenAIEmbedder = OpenAICompatEmbedder


def create_embedder(settings) -> Embedder:
    from app.llm.providers import PRESETS

    name = settings.embedding_provider
    if name == "hashing":
        return HashingEmbedder()
    preset = PRESETS[name]
    base_url = settings.embedding_base_url or (
        settings.ollama_base_url if name == "ollama"
        else settings.openai_base_url if name == "openai"
        else settings.llm_base_url if name == "custom" else preset.base_url)
    model = settings.embedding_model or (
        settings.openai_embedding_model if name == "openai" else preset.embedding_model)
    if not model:
        raise ValueError(f"EMBEDDING_PROVIDER={name} requires EMBEDDING_MODEL")
    if name == "custom" and not base_url:
        raise ValueError("EMBEDDING_PROVIDER=custom requires EMBEDDING_BASE_URL or LLM_BASE_URL")
    key = (getattr(settings, preset.key_env.lower(), None) if preset.key_env
           else settings.llm_api_key or "not-needed")
    if not key:
        raise ValueError(f"EMBEDDING_PROVIDER={name} requires {preset.key_env}")
    dim = preset.embedding_dim if model == preset.embedding_model else None
    return OpenAICompatEmbedder(key, model, base_url, settings.llm_timeout_seconds, name=name, dim=dim)
