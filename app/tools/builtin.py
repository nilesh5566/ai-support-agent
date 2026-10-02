"""Built-in tools: knowledge search, CRM lookups and ticketing.

The CRM/ticket handlers talk to the local Postgres tables. In a customer deployment these are the
adapters you swap for Salesforce/Zendesk/HubSpot/internal REST APIs — the agent, auth and audit
layers stay the same.
"""
from __future__ import annotations

from sqlalchemy import func, select

from app.models import Customer, Order, Ticket
from app.observability.metrics import ESCALATIONS, TICKETS_CREATED
from app.tools.registry import Tool, ToolContext, ToolInputError, ToolRegistry

PRIORITIES = ("low", "normal", "high", "urgent")
_PRIORITY_RANK = {p: i for i, p in enumerate(PRIORITIES)}


def _resolve_customer(ctx: ToolContext, email: str | None, any_scope: str) -> Customer | dict:
    """Customers may only see themselves; staff with `*:any` scopes may look anyone up."""
    if email and any_scope in ctx.scopes:
        customer = ctx.session.scalar(select(Customer).where(func.lower(Customer.email) == email.strip().lower()))
        return customer or {"error": f"No customer found with email {email}."}
    if email and (ctx.customer is None or ctx.customer.email.lower() != email.strip().lower()):
        return {"error": "You can only access the account you are signed in with."}
    if ctx.customer is None:
        return {"error": "No verified customer is linked to this conversation. Ask the customer to sign in "
                         "or provide their account email to a support specialist."}
    return ctx.customer


def search_knowledge_base(ctx: ToolContext, query: str, top_k: int = 4) -> dict:
    visibility = ["public"] + (["internal"] if "kb:internal" in ctx.scopes else [])
    with ctx.trace.span("rag.search", visibility=",".join(visibility)) as span:
        results = ctx.retriever.search(query, visibility=visibility, top_k=max(1, min(int(top_k), 8)))
        span.attributes["hits"] = len(results)
    for r in results:
        if not any(s["chunk_id"] == r["chunk_id"] for s in ctx.sources):
            ctx.sources.append({k: r[k] for k in ("chunk_id", "document_id", "title", "section", "score")})
    return {"results": [{k: r[k] for k in ("title", "section", "text", "score")} for r in results]}


def get_customer_profile(ctx: ToolContext, email: str | None = None) -> dict:
    c = _resolve_customer(ctx, email, "customer:any")
    if isinstance(c, dict):
        return c
    open_tickets = ctx.session.scalar(
        select(func.count()).select_from(Ticket).where(Ticket.customer_id == c.id, Ticket.status != "closed"))
    return {"customer_id": c.external_id, "name": c.name, "email": c.email, "plan": c.plan, "status": c.status,
            "customer_since": c.created_at.date().isoformat(), "open_tickets": int(open_tickets or 0)}


def list_orders(ctx: ToolContext, email: str | None = None, limit: int = 5) -> dict:
    c = _resolve_customer(ctx, email, "customer:any")
    if isinstance(c, dict):
        return c
    orders = ctx.session.scalars(select(Order).where(Order.customer_id == c.id)
                                 .order_by(Order.created_at.desc()).limit(max(1, min(int(limit), 20)))).all()
    return {"orders": [{"order_number": o.order_number, "status": o.status,
                        "total": f"{o.total_cents / 100:.2f} {o.currency}", "items": o.items,
                        "date": o.created_at.date().isoformat()} for o in orders]}


def create_ticket(ctx: ToolContext, subject: str, description: str, priority: str = "normal",
                  escalate: bool = False) -> dict:
    priority = str(priority).lower()
    if priority not in PRIORITIES:
        raise ToolInputError(f"priority must be one of {', '.join(PRIORITIES)}")
    escalate = bool(escalate)

    # Idempotency: one open ticket per conversation; later requests update it instead of duplicating.
    existing = ctx.session.scalar(select(Ticket).where(Ticket.conversation_id == ctx.conversation_id,
                                                       Ticket.status.in_(("open", "escalated"))))
    if existing:
        if _PRIORITY_RANK[priority] > _PRIORITY_RANK.get(existing.priority, 1):
            existing.priority = priority
        if escalate and not existing.escalated:
            existing.escalated, existing.status = True, "escalated"
            _notify(ctx, existing, "agent_or_customer_request")
        existing.description = f"{existing.description}\n\n---\n{description}"[:10000]
        ctx.session.flush()
        result = _ticket_dict(existing) | {"deduplicated": True}
    else:
        ticket = Ticket(subject=subject[:200], description=description[:10000], priority=priority,
                        status="escalated" if escalate else "open", escalated=escalate, channel=ctx.channel,
                        conversation_id=ctx.conversation_id, customer_id=ctx.customer.id if ctx.customer else None)
        ctx.session.add(ticket)
        ctx.session.flush()
        TICKETS_CREATED.labels(priority).inc()
        if escalate:
            _notify(ctx, ticket, "agent_or_customer_request")
        result = _ticket_dict(ticket) | {"deduplicated": False}
    ctx.escalated = ctx.escalated or result["escalated"]
    ctx.tickets.append(result)
    return result


def get_ticket(ctx: ToolContext, ticket_number: str) -> dict:
    t = ctx.session.scalar(select(Ticket).where(Ticket.ticket_number == ticket_number.strip().upper()))
    if t is None:
        return {"error": f"Ticket {ticket_number} was not found."}
    if "ticket:any" not in ctx.scopes and (ctx.customer is None or t.customer_id != ctx.customer.id):
        return {"error": f"Ticket {ticket_number} was not found."}  # don't leak existence
    return _ticket_dict(t) | {"updated_at": t.updated_at.isoformat()}


def _ticket_dict(t: Ticket) -> dict:
    return {"ticket_number": t.ticket_number, "subject": t.subject, "status": t.status,
            "priority": t.priority, "escalated": t.escalated}


def _notify(ctx: ToolContext, ticket: Ticket, reason: str) -> None:
    ESCALATIONS.labels(reason).inc()
    if ctx.notifier:
        ctx.notifier.notify(ticket, reason)


def build_registry(enabled: list[str] | None = None) -> ToolRegistry:
    reg = ToolRegistry(enabled)
    reg.register(Tool(
        "search_knowledge_base",
        "Search the company knowledge base (product docs, FAQs, policies, past tickets). Use this before "
        "answering any product or policy question and base your answer on the results.",
        {"type": "object", "properties": {
            "query": {"type": "string", "description": "Natural-language search query"},
            "top_k": {"type": "integer", "minimum": 1, "maximum": 8}},
         "required": ["query"]},
        {"kb:public", "kb:internal"}, search_knowledge_base))
    reg.register(Tool(
        "get_customer_profile",
        "Get the customer's account profile (plan, status, open tickets). Omit email to use the signed-in customer.",
        {"type": "object", "properties": {"email": {"type": "string"}}},
        {"customer:self", "customer:any"}, get_customer_profile))
    reg.register(Tool(
        "list_orders",
        "List the customer's recent orders with status. Omit email to use the signed-in customer.",
        {"type": "object", "properties": {"email": {"type": "string"},
                                          "limit": {"type": "integer", "minimum": 1, "maximum": 20}}},
        {"customer:self", "customer:any"}, list_orders))
    reg.register(Tool(
        "create_ticket",
        "Create a support ticket when the issue can't be resolved from documentation, the customer asks for "
        "a human, or a policy requires it. Set escalate=true to hand off to a human specialist immediately.",
        {"type": "object", "properties": {
            "subject": {"type": "string", "maxLength": 200},
            "description": {"type": "string", "description": "Summary of the issue and what was tried"},
            "priority": {"type": "string", "enum": list(PRIORITIES)},
            "escalate": {"type": "boolean"}},
         "required": ["subject", "description"]},
        {"ticket:create"}, create_ticket, mutating=True))
    reg.register(Tool(
        "get_ticket",
        "Get the status of an existing support ticket by its number (e.g. TCK-1A2B3C4D).",
        {"type": "object", "properties": {"ticket_number": {"type": "string"}}, "required": ["ticket_number"]},
        {"ticket:self", "ticket:any"}, get_ticket))
    return reg
