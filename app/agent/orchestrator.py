"""The agent loop: context -> LLM -> tools -> LLM ... -> answer, with guardrails and telemetry."""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.agent import policy as policy_mod
from app.agent_config import AgentConfig
from app.llm.base import LLMError
from app.models import Conversation, Customer, Message
from app.observability.logging import log_event
from app.observability.metrics import CHAT_LATENCY, CHAT_TURNS, LLM_LATENCY, LLM_REQUESTS, LLM_TOKENS
from app.observability.tracing import Trace, TraceStore
from app.tools.registry import ToolContext, ToolRegistry

logger = logging.getLogger(__name__)

FALLBACK_REPLY = ("I'm having trouble answering right now. Please try again in a moment — "
                  "or ask me to open a ticket and a specialist will follow up.")


class ConversationAccessError(PermissionError):
    pass


class ConversationNotFound(LookupError):
    pass


@dataclass
class ChatResult:
    reply: str
    conversation_id: str
    trace_id: str
    sources: list[dict] = field(default_factory=list)
    tool_calls: list[dict] = field(default_factory=list)
    escalated: bool = False
    ticket_number: str | None = None
    latency_ms: float = 0.0
    outcome: str = "answered"
    trace: dict | None = None


class SupportAgent:
    def __init__(self, llm, registry: ToolRegistry, retriever, memory, config: AgentConfig,
                 trace_store: TraceStore, notifier=None):
        self.llm = llm
        self.registry = registry
        self.retriever = retriever
        self.memory = memory
        self.config = config
        self.trace_store = trace_store
        self.notifier = notifier

    # ------------------------------------------------------------------ helpers
    def _system_prompt(self, customer: Customer | None, role: str, channel: str,
                       decision: policy_mod.PolicyDecision) -> str:
        a = self.config.agent
        lines = [a.system_prompt.format(company=a.company, name=a.name).strip(), "", "## Session context",
                 f"- Channel: {channel}", f"- Caller role: {role}"]
        if customer:
            lines.append(f"- Verified customer: {customer.name} (customer id {customer.external_id}, "
                         f"plan {customer.plan}). Account tools default to this customer.")
        else:
            lines.append("- No verified customer is linked to this conversation.")
        if decision.force_escalation:
            lines.append(f"\nPOLICY: This conversation MUST be escalated to a human specialist with priority "
                         f"'{decision.priority}'. Call create_ticket with escalate=true, then tell the customer.")
        return "\n".join(lines)

    def _load_conversation(self, session: Session, conversation_id: str | None, channel: str,
                           customer: Customer | None, create_if_missing: bool) -> tuple[Conversation, bool]:
        if conversation_id:
            conv = session.get(Conversation, conversation_id)
            if conv is not None:
                if conv.customer_id is not None and (customer is None or conv.customer_id != customer.id):
                    raise ConversationAccessError("conversation belongs to a different customer")
                return conv, False
            if not create_if_missing:
                raise ConversationNotFound(conversation_id)
        conv = Conversation(channel=channel, customer_id=customer.id if customer else None)
        if conversation_id:
            conv.id = conversation_id
        session.add(conv)
        session.commit()
        return conv, True

    def _history(self, session: Session, conv: Conversation, is_new: bool) -> list[dict]:
        if is_new:
            return []
        history = self.memory.load(conv.id)
        if not history:  # cache miss (expired/evicted) -> rehydrate from Postgres
            rows = session.scalars(select(Message).where(Message.conversation_id == conv.id)
                                   .order_by(Message.id.desc()).limit(self.memory.max_messages)).all()
            history = [{"role": m.role, "content": m.content} for m in reversed(rows)]
            self.memory.seed(conv.id, history)
        return history

    def _call_llm(self, messages: list[dict], tools: list[dict], trace: Trace, step: int):
        provider = self.llm.name
        start = time.perf_counter()
        with trace.span("llm.call", step=step, provider=provider, model=self.llm.model, tools=len(tools)) as span:
            try:
                resp = self.llm.chat(messages, tools, temperature=self.config.agent.temperature)
            except LLMError:
                LLM_REQUESTS.labels(provider, "error").inc()
                raise
            provider = resp.provider or provider  # with a fallback chain: the provider that actually answered
            span.attributes.update(served_by=provider, served_model=resp.model or self.llm.model,
                                   tool_calls=len(resp.tool_calls), **{f"tokens_{k}": v for k, v in resp.usage.items()})
        LLM_REQUESTS.labels(provider, "ok").inc()
        LLM_LATENCY.labels(provider).observe(time.perf_counter() - start)
        for kind, n in resp.usage.items():
            LLM_TOKENS.labels(provider, kind).inc(n)
        return resp

    # --------------------------------------------------------------------- main
    def handle(self, session: Session, *, message: str, channel: str, role: str,
               conversation_id: str | None = None, customer_email: str | None = None,
               create_if_missing: bool = False, debug: bool = False) -> ChatResult:
        started = time.perf_counter()
        trace = Trace()
        tool_log: list[dict] = []
        outcome = "answered"
        reply: str | None = None

        customer = None
        if customer_email:
            customer = session.scalar(select(Customer).where(func.lower(Customer.email) == customer_email.lower()))

        with trace.span("agent.turn", channel=channel, role=role) as turn_span:
            conv, is_new = self._load_conversation(session, conversation_id, channel, customer, create_if_missing)
            decision = policy_mod.evaluate(message, self.config.escalation)
            scopes = self.config.scopes_for(role)
            ctx = ToolContext(session=session, role=role, scopes=scopes, channel=channel, conversation_id=conv.id,
                              customer=customer, retriever=self.retriever, trace=trace, notifier=self.notifier)
            tools = self.registry.specs_for(scopes)
            messages = [{"role": "system", "content": self._system_prompt(customer, role, channel, decision)},
                        *self._history(session, conv, is_new), {"role": "user", "content": message}]

            try:
                for step in range(self.config.agent.max_tool_iterations):
                    resp = self._call_llm(messages, tools, trace, step)
                    if not resp.tool_calls:
                        reply = resp.content
                        break
                    messages.append({"role": "assistant", "content": resp.content, "tool_calls": [
                        {"id": tc.id, "type": "function",
                         "function": {"name": tc.name, "arguments": json.dumps(tc.arguments)}}
                        for tc in resp.tool_calls]})
                    for tc in resp.tool_calls:
                        result = self.registry.execute(tc.name, tc.arguments, ctx)
                        status = "error" if "error" in result else "ok"
                        tool_log.append({"name": tc.name, "status": status})
                        messages.append({"role": "tool", "tool_call_id": tc.id,
                                         "content": json.dumps(result, default=str)})
                if reply is None:  # iteration budget exhausted -> force a final answer without tools
                    outcome = "max_iterations"
                    reply = self._call_llm(messages, [], trace, step=-1).content
            except LLMError:
                logger.exception("LLM failure")
                outcome = "llm_error"
                reply = FALLBACK_REPLY

            # Guardrail: policy-mandated escalation happens even if the model forgot to do it.
            if decision.force_escalation and not ctx.escalated:
                result = self.registry.execute("create_ticket", {
                    "subject": f"[Auto-escalation] {message[:150]}", "description": message,
                    "priority": decision.priority, "escalate": True}, ctx)
                tool_log.append({"name": "create_ticket", "status": "error" if "error" in result else "ok",
                                 "enforced_by_policy": True})
                if "error" not in result:
                    reply = (f"{reply or ''}\n\nI've escalated this to a human specialist "
                             f"(ticket **{result['ticket_number']}**). They'll contact you shortly.").strip()

            reply = (reply or "").strip() or FALLBACK_REPLY
            session.add_all([Message(conversation_id=conv.id, role="user", content=message, trace_id=trace.trace_id),
                             Message(conversation_id=conv.id, role="assistant", content=reply,
                                     trace_id=trace.trace_id)])
            session.commit()
            self.memory.append(conv.id, "user", message)
            self.memory.append(conv.id, "assistant", reply)

            if ctx.escalated:
                outcome = "escalated" if outcome == "answered" else outcome
            turn_span.attributes.update(outcome=outcome, tools=len(tool_log), sources=len(ctx.sources))

        latency = time.perf_counter() - started
        CHAT_TURNS.labels(channel, outcome).inc()
        CHAT_LATENCY.labels(channel).observe(latency)
        self.trace_store.save(trace)
        log_event(logger, "agent turn complete", conversation_id=conv.id, trace_id=trace.trace_id, channel=channel,
                  role=role, outcome=outcome, tools=[t["name"] for t in tool_log],
                  latency_ms=round(latency * 1000, 1))

        ticket = ctx.tickets[-1]["ticket_number"] if ctx.tickets else None
        return ChatResult(reply=reply, conversation_id=conv.id, trace_id=trace.trace_id, sources=ctx.sources,
                          tool_calls=tool_log, escalated=ctx.escalated, ticket_number=ticket,
                          latency_ms=round(latency * 1000, 1), outcome=outcome,
                          trace=trace.to_dict() if debug else None)
