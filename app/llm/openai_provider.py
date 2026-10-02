from __future__ import annotations

import json
import logging
import re

from app.llm.base import LLMError, LLMResponse, ToolCall

logger = logging.getLogger(__name__)

_THINK_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL | re.IGNORECASE)


def _clean(content: str | None) -> str | None:
    """Strip reasoning blocks some open models (Qwen, DeepSeek via Ollama/OpenRouter) put in the answer."""
    if not content:
        return content
    return _THINK_RE.sub("", content).strip() or None


class OpenAIProvider:
    """Chat Completions with function calling against any OpenAI-compatible endpoint.

    The same class serves OpenAI, Groq, Gemini, OpenRouter, Cerebras, Mistral, Ollama and custom
    servers - see ``app/llm/providers.py`` for the presets.
    """

    def __init__(self, api_key: str, model: str, base_url: str | None = None, timeout: float = 30.0,
                 client=None, name: str = "openai", max_retries: int = 3):
        if client is None:
            from openai import OpenAI

            # The SDK retries 408/409/429/5xx with exponential backoff and honours Retry-After,
            # which matters on free tiers with tight per-minute limits.
            client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout, max_retries=max_retries)
        self.client = client
        self.model = model
        self.name = name

    def _create(self, kwargs: dict, retried: bool = False):
        try:
            return self.client.chat.completions.create(**kwargs)
        except Exception as exc:
            status = getattr(exc, "status_code", None)
            body = str(getattr(exc, "body", "") or exc)
            # Open models occasionally emit a malformed tool call; Groq rejects it with 400
            # "tool_use_failed". One retry almost always succeeds.
            if status == 400 and "tool_use_failed" in body and not retried:
                logger.warning("provider rejected malformed tool call, retrying", extra={"fields": {
                    "provider": self.name, "model": self.model}})
                return self._create(kwargs, retried=True)
            detail = f" (HTTP {status})" if status else ""
            raise LLMError(f"{self.name} request failed: {type(exc).__name__}{detail}") from exc

    def chat(self, messages: list[dict], tools: list[dict], temperature: float = 0.2) -> LLMResponse:
        kwargs: dict = {"model": self.model, "messages": messages, "temperature": temperature}
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"
        resp = self._create(kwargs)

        if not getattr(resp, "choices", None):
            raise LLMError(f"{self.name} returned no choices")
        msg = resp.choices[0].message
        calls: list[ToolCall] = []
        for i, tc in enumerate(msg.tool_calls or []):
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                logger.warning("model produced invalid JSON tool arguments for %s", tc.function.name)
                args = {}
            calls.append(ToolCall(id=tc.id or f"call_{i}", name=tc.function.name,
                                  arguments=args if isinstance(args, dict) else {}))
        usage = {}
        if getattr(resp, "usage", None):
            usage = {"prompt": resp.usage.prompt_tokens or 0, "completion": resp.usage.completion_tokens or 0}
        return LLMResponse(content=_clean(msg.content), tool_calls=calls, usage=usage,
                           provider=self.name, model=self.model)
