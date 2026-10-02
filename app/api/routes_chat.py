from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Request, status
from sqlalchemy.orm import Session

from app.agent.orchestrator import ConversationAccessError, ConversationNotFound
from app.api.deps import Principal, get_db, require_api_key
from app.schemas import ChatRequest, ChatResponse

router = APIRouter(prefix="/v1", tags=["chat"])


@router.post("/chat", response_model=ChatResponse)
def chat(body: ChatRequest, request: Request, principal: Principal = Depends(require_api_key),
         db: Session = Depends(get_db)) -> ChatResponse:
    if body.as_staff and principal.role != "admin":
        raise HTTPException(status.HTTP_403_FORBIDDEN, "as_staff requires an admin API key")
    agent = request.app.state.agent
    role = "staff" if body.as_staff else agent.config.role_for_channel(body.channel)
    try:
        result = agent.handle(db, message=body.message, channel=body.channel, role=role,
                              conversation_id=body.conversation_id, customer_email=body.customer_email,
                              debug=body.debug)
    except ConversationNotFound:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Conversation not found") from None
    except ConversationAccessError:
        raise HTTPException(status.HTTP_403_FORBIDDEN, "Conversation belongs to another customer") from None
    return ChatResponse(reply=result.reply, conversation_id=result.conversation_id, trace_id=result.trace_id,
                        sources=result.sources, tool_calls=result.tool_calls, escalated=result.escalated,
                        ticket_number=result.ticket_number, outcome=result.outcome, latency_ms=result.latency_ms,
                        trace=result.trace)
