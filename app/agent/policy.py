"""Deterministic guardrails evaluated in code, not left to the prompt."""
from __future__ import annotations

import re
from dataclasses import dataclass

from app.agent_config import EscalationConfig


@dataclass
class PolicyDecision:
    force_escalation: bool = False
    reason: str | None = None
    priority: str = "normal"


def evaluate(message: str, cfg: EscalationConfig) -> PolicyDecision:
    for kw in cfg.keywords:
        if re.search(rf"\b{re.escape(kw.lower())}\b", message.lower()):
            return PolicyDecision(True, f"keyword:{kw}", cfg.priority)
    return PolicyDecision()
