"""Per-deployment agent configuration (config/agent.yaml).

This is the file a forward-deployed engineer edits per customer: persona, enabled tools,
role -> permission scopes, retrieval thresholds and escalation policy. No code changes needed.
"""
from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, Field


class AgentSection(BaseModel):
    name: str = "Support Assistant"
    company: str = "Acme"
    system_prompt: str = "You are a helpful customer support assistant for {company}."
    max_tool_iterations: int = Field(default=4, ge=1, le=10)
    temperature: float = Field(default=0.2, ge=0, le=2)


class RetrievalConfig(BaseModel):
    top_k: int = Field(default=4, ge=1, le=20)
    min_score: float = Field(default=0.12, ge=0, le=1)


class RoleConfig(BaseModel):
    scopes: list[str] = Field(default_factory=list)


class ToolsConfig(BaseModel):
    enabled: list[str] = Field(default_factory=list)


class EscalationConfig(BaseModel):
    keywords: list[str] = Field(default_factory=list)
    priority: str = "urgent"


class ChannelConfig(BaseModel):
    role: str = "customer"


class AgentConfig(BaseModel):
    agent: AgentSection = Field(default_factory=AgentSection)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    roles: dict[str, RoleConfig] = Field(default_factory=dict)
    tools: ToolsConfig = Field(default_factory=ToolsConfig)
    escalation: EscalationConfig = Field(default_factory=EscalationConfig)
    channels: dict[str, ChannelConfig] = Field(default_factory=dict)

    def scopes_for(self, role: str) -> set[str]:
        cfg = self.roles.get(role)
        return set(cfg.scopes) if cfg else set()

    def role_for_channel(self, channel: str) -> str:
        cfg = self.channels.get(channel)
        return cfg.role if cfg else "customer"


def load_agent_config(path: str | Path) -> AgentConfig:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Agent config not found: {p.resolve()}")
    data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    return AgentConfig.model_validate(data)
