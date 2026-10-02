from __future__ import annotations

import hashlib
from collections.abc import Iterator
from dataclasses import dataclass

from fastapi import Depends, Header, HTTPException, Request, status
from sqlalchemy.orm import Session

from app.observability.metrics import RATE_LIMITED


@dataclass
class Principal:
    key_id: str  # hash prefix, never the raw key
    role: str  # client | admin


def get_db(request: Request) -> Iterator[Session]:
    yield from request.app.state.db.session()


def require_api_key(request: Request, x_api_key: str | None = Header(default=None)) -> Principal:
    if not x_api_key:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Missing X-API-Key header")
    role = request.app.state.api_keys.get(x_api_key)
    if role is None:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid API key")
    principal = Principal(hashlib.sha256(x_api_key.encode()).hexdigest()[:12], role)
    allowed, remaining = request.app.state.rate_limiter.check(principal.key_id)
    request.state.rate_limit_remaining = remaining
    if not allowed:
        RATE_LIMITED.inc()
        raise HTTPException(status.HTTP_429_TOO_MANY_REQUESTS, "Rate limit exceeded", headers={"Retry-After": "60"})
    return principal


def require_admin(principal: Principal = Depends(require_api_key)) -> Principal:
    if principal.role != "admin":
        raise HTTPException(status.HTTP_403_FORBIDDEN, "Admin API key required")
    return principal
