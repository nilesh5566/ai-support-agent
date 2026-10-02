from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict


@dataclass
class LLMResponse:
    content: str | None
    tool_calls: list[ToolCall] = field(default_factory=list)
    usage: dict = field(default_factory=dict)
    provider: str | None = None  # which provider actually answered (matters with a fallback chain)
    model: str | None = None


class LLMError(RuntimeError):
    pass


class LLMProvider(Protocol):
    name: str
    model: str

    def chat(self, messages: list[dict], tools: list[dict], temperature: float = 0.2) -> LLMResponse: ...
