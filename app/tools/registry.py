"""Tool registry: declarative tools with JSON-schema params, permission scopes and audit logging."""
from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from sqlalchemy.orm import Session

from app.models import AuditLog, Customer
from app.observability.metrics import TOOL_CALLS, TOOL_LATENCY
from app.observability.tracing import Trace

logger = logging.getLogger(__name__)


class ToolInputError(ValueError):
    """Raised by handlers for bad arguments; the message is shown to the model."""


@dataclass
class ToolContext:
    session: Session
    role: str
    scopes: set[str]
    channel: str
    conversation_id: str
    customer: Customer | None
    retriever: Any
    trace: Trace
    notifier: Any = None
    sources: list[dict] = field(default_factory=list)
    tickets: list[dict] = field(default_factory=list)
    escalated: bool = False


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict
    any_of_scopes: set[str]
    handler: Callable[..., dict]
    mutating: bool = False

    def allowed(self, scopes: set[str]) -> bool:
        return bool(self.any_of_scopes & scopes)

    def spec(self) -> dict:
        return {"type": "function",
                "function": {"name": self.name, "description": self.description, "parameters": self.parameters}}


class ToolRegistry:
    def __init__(self, enabled: list[str] | None = None):
        self._tools: dict[str, Tool] = {}
        self._enabled = set(enabled) if enabled else None

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def is_enabled(self, name: str) -> bool:
        return name in self._tools and (self._enabled is None or name in self._enabled)

    def list(self) -> list[Tool]:
        return [t for t in self._tools.values() if self.is_enabled(t.name)]

    def specs_for(self, scopes: set[str]) -> list[dict]:
        """Only tools the caller may use are shown to the model (least privilege)."""
        return [t.spec() for t in self.list() if t.allowed(scopes)]

    def _validate(self, tool: Tool, args: dict) -> dict:
        props = tool.parameters.get("properties", {})
        clean = {k: v for k, v in (args or {}).items() if k in props and v is not None}
        missing = [r for r in tool.parameters.get("required", []) if r not in clean or clean[r] in ("", None)]
        if missing:
            raise ToolInputError(f"missing required argument(s): {', '.join(missing)}")
        return clean

    def execute(self, name: str, args: dict, ctx: ToolContext) -> dict:
        tool = self._tools.get(name)
        start = time.perf_counter()
        with ctx.trace.span(f"tool.{name}", role=ctx.role) as span:
            if tool is None or not self.is_enabled(name):
                status, result = "unknown", {"error": f"Tool '{name}' is not available."}
            elif not tool.allowed(ctx.scopes):
                status, result = "denied", {"error": "Permission denied: this action is not allowed for your role."}
            else:
                try:
                    result = tool.handler(ctx, **self._validate(tool, args))
                    status = "error" if "error" in result else "ok"
                except ToolInputError as exc:
                    status, result = "invalid", {"error": str(exc)}
                except Exception:
                    logger.exception("tool %s failed", name)
                    ctx.session.rollback()
                    status, result = "failed", {"error": "The tool failed unexpectedly. Please try again later."}
            span.attributes["status"] = status
            if status != "ok":
                span.status = "error" if status == "failed" else "ok"

        TOOL_CALLS.labels(name, status).inc()
        TOOL_LATENCY.labels(name).observe(time.perf_counter() - start)
        ctx.session.add(AuditLog(conversation_id=ctx.conversation_id, trace_id=ctx.trace.trace_id,
                                 actor_role=ctx.role, tool_name=name, arguments=args or {}, status=status))
        return result
