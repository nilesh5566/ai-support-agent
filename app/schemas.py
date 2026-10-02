from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, Field, field_validator

_EMAIL = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=4000)
    conversation_id: str | None = Field(default=None, max_length=80)
    customer_email: str | None = Field(
        default=None, description="Identity asserted by the calling (authenticated) application")
    channel: Literal["web", "api"] = "api"
    as_staff: bool = Field(default=False, description="Admin keys only: run with the staff role")
    debug: bool = Field(default=False, description="Include the full trace in the response")

    @field_validator("customer_email")
    @classmethod
    def _email(cls, v: str | None) -> str | None:
        if v is None or v == "":
            return None
        if not _EMAIL.match(v):
            raise ValueError("invalid email address")
        return v.strip().lower()

    @field_validator("message")
    @classmethod
    def _message(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("message must not be blank")
        return v


class Source(BaseModel):
    document_id: int | None = None
    title: str | None = None
    section: str | None = None
    score: float


class ChatResponse(BaseModel):
    reply: str
    conversation_id: str
    trace_id: str
    sources: list[Source]
    tool_calls: list[dict]
    escalated: bool
    ticket_number: str | None
    outcome: str
    latency_ms: float
    trace: dict | None = None


class TextDocumentIn(BaseModel):
    title: str = Field(min_length=1, max_length=300)
    content: str = Field(min_length=1, max_length=2_000_000)
    source: str | None = Field(default=None, max_length=500)
    visibility: Literal["public", "internal"] = "public"


class DocumentOut(BaseModel):
    id: int
    title: str
    source: str
    content_type: str
    visibility: str
    chunk_count: int


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=1000)
    top_k: int = Field(default=4, ge=1, le=20)
    include_internal: bool = True
