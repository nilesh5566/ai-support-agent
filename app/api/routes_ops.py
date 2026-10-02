from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.api.deps import Principal, get_db, require_admin, require_api_key
from app.llm.providers import catalogue
from app.models import AuditLog, Conversation, Message, Ticket

router = APIRouter(prefix="/v1", tags=["operations"])


@router.get("/tools")
def list_tools(request: Request, _: Principal = Depends(require_api_key)) -> list[dict]:
    return [{"name": t.name, "description": t.description, "scopes": sorted(t.any_of_scopes),
             "mutating": t.mutating, "parameters": t.parameters} for t in request.app.state.registry.list()]


@router.get("/tickets")
def list_tickets(status_filter: str | None = Query(default=None, alias="status"),
                 limit: int = Query(default=50, ge=1, le=500),
                 _: Principal = Depends(require_admin), db: Session = Depends(get_db)) -> list[dict]:
    q = select(Ticket).order_by(Ticket.id.desc()).limit(limit)
    if status_filter:
        q = q.where(Ticket.status == status_filter)
    return [{"ticket_number": t.ticket_number, "subject": t.subject, "priority": t.priority, "status": t.status,
             "escalated": t.escalated, "channel": t.channel, "conversation_id": t.conversation_id,
             "created_at": t.created_at.isoformat()} for t in db.scalars(q).all()]


@router.get("/conversations/{conversation_id}")
def get_conversation(conversation_id: str, _: Principal = Depends(require_admin),
                     db: Session = Depends(get_db)) -> dict:
    conv = db.get(Conversation, conversation_id)
    if conv is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Conversation not found")
    msgs = db.scalars(select(Message).where(Message.conversation_id == conv.id).order_by(Message.id)).all()
    audits = db.scalars(select(AuditLog).where(AuditLog.conversation_id == conv.id).order_by(AuditLog.id)).all()
    return {"id": conv.id, "channel": conv.channel, "customer_id": conv.customer_id,
            "messages": [{"role": m.role, "content": m.content, "trace_id": m.trace_id,
                          "created_at": m.created_at.isoformat()} for m in msgs],
            "tool_audit": [{"tool": a.tool_name, "status": a.status, "role": a.actor_role, "arguments": a.arguments,
                            "trace_id": a.trace_id} for a in audits]}


@router.get("/traces/{trace_id}")
def get_trace(trace_id: str, request: Request, _: Principal = Depends(require_admin)) -> dict:
    trace = request.app.state.trace_store.get(trace_id)
    if trace is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Trace not found (traces are kept per replica, in memory)")
    return trace


@router.get("/config")
def get_config(request: Request, _: Principal = Depends(require_admin)) -> dict:
    s = request.app.state.settings
    return {"agent": request.app.state.agent.config.model_dump(),
            "runtime": {"llm_provider": s.llm_provider, "model": request.app.state.agent.llm.model,
                        "llm_chain": getattr(request.app.state.agent.llm, "chain", None),
                        "embedding_provider": s.embedding_provider, "vector_backend": s.vector_backend,
                        "redis": bool(s.redis_url), "environment": s.environment}}


@router.get("/llm/providers")
def llm_providers(request: Request, _: Principal = Depends(require_admin)) -> dict:
    """Which LLM providers this build supports, and which one (or chain) is active."""
    s = request.app.state.settings
    llm = request.app.state.agent.llm
    return {"active": {"provider": s.llm_provider, "model": llm.model, "chain": getattr(llm, "chain", None),
                       "embedding_provider": s.embedding_provider},
            "available": catalogue()}
