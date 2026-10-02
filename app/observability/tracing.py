"""Lightweight per-turn tracing.

Every agent turn produces a trace with nested spans (llm.call, tool.*, rag.search ...).
Traces are logged as JSON, returned when `debug=true`, and fetchable via GET /v1/traces/{id}.
The span shape (name, parent, duration, attributes, status) maps 1:1 onto OpenTelemetry
if you later want to export to Jaeger/Tempo/X-Ray.
"""
from __future__ import annotations

import threading
import time
import uuid
from collections import OrderedDict
from contextlib import contextmanager
from dataclasses import dataclass, field


@dataclass
class Span:
    span_id: str
    name: str
    parent_id: str | None
    start_offset_ms: float
    duration_ms: float = 0.0
    status: str = "ok"
    attributes: dict = field(default_factory=dict)


class Trace:
    def __init__(self, trace_id: str | None = None):
        self.trace_id = trace_id or uuid.uuid4().hex
        self.spans: list[Span] = []
        self._stack: list[str] = []
        self._t0 = time.perf_counter()

    @contextmanager
    def span(self, name: str, **attributes):
        s = Span(
            span_id=uuid.uuid4().hex[:16],
            name=name,
            parent_id=self._stack[-1] if self._stack else None,
            start_offset_ms=round((time.perf_counter() - self._t0) * 1000, 2),
            attributes=dict(attributes),
        )
        self._stack.append(s.span_id)
        start = time.perf_counter()
        try:
            yield s
        except Exception as exc:
            s.status = "error"
            s.attributes["error"] = type(exc).__name__
            raise
        finally:
            s.duration_ms = round((time.perf_counter() - start) * 1000, 2)
            self._stack.pop()
            self.spans.append(s)

    def to_dict(self) -> dict:
        spans = sorted(self.spans, key=lambda s: s.start_offset_ms)
        return {
            "trace_id": self.trace_id,
            "spans": [
                {
                    "span_id": s.span_id,
                    "name": s.name,
                    "parent_id": s.parent_id,
                    "start_offset_ms": s.start_offset_ms,
                    "duration_ms": s.duration_ms,
                    "status": s.status,
                    "attributes": s.attributes,
                }
                for s in spans
            ],
        }


class TraceStore:
    """Bounded in-process store of recent traces (per API replica)."""

    def __init__(self, max_items: int = 500):
        self._items: OrderedDict[str, dict] = OrderedDict()
        self._max = max_items
        self._lock = threading.Lock()

    def save(self, trace: Trace) -> None:
        with self._lock:
            self._items[trace.trace_id] = trace.to_dict()
            while len(self._items) > self._max:
                self._items.popitem(last=False)

    def get(self, trace_id: str) -> dict | None:
        with self._lock:
            return self._items.get(trace_id)
