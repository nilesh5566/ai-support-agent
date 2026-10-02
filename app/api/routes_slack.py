"""Slack Events API channel: internal support staff @mention the bot in Slack.

Requests are verified with the Slack signing secret, acknowledged within 3s, and processed in
the background. Each Slack thread maps to one conversation.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import logging
import re
import time

import httpx
from fastapi import APIRouter, BackgroundTasks, HTTPException, Request, status

router = APIRouter(prefix="/v1/slack", tags=["channels"])
logger = logging.getLogger(__name__)
_MENTION = re.compile(r"<@[A-Z0-9]+>")


def verify_slack_signature(secret: str, timestamp: str | None, body: bytes, signature: str | None) -> bool:
    if not timestamp or not signature:
        return False
    try:
        if abs(time.time() - int(timestamp)) > 300:
            return False
    except ValueError:
        return False
    base = f"v0:{timestamp}:".encode() + body
    expected = "v0=" + hmac.new(secret.encode(), base, hashlib.sha256).hexdigest()
    return hmac.compare_digest(expected, signature)


def _process_event(app, event: dict) -> None:
    text = _MENTION.sub("", event.get("text", "")).strip()
    if not text:
        return
    thread_ts = event.get("thread_ts") or event.get("ts")
    conversation_id = f"slack-{event.get('channel')}-{thread_ts}"[:80]
    agent = app.state.agent
    session = app.state.db.SessionLocal()
    try:
        result = agent.handle(session, message=text, channel="slack", role=agent.config.role_for_channel("slack"),
                              conversation_id=conversation_id, create_if_missing=True)
    except Exception:
        logger.exception("slack event processing failed")
        return
    finally:
        session.close()
    token = app.state.settings.slack_bot_token
    if not token:
        logger.warning("SLACK_BOT_TOKEN not set; cannot post reply")
        return
    try:
        httpx.post("https://slack.com/api/chat.postMessage", timeout=10,
                   headers={"Authorization": f"Bearer {token}"},
                   json={"channel": event.get("channel"), "thread_ts": thread_ts, "text": result.reply})
    except httpx.HTTPError:
        logger.exception("failed to post Slack reply")


@router.post("/events")
async def slack_events(request: Request, background: BackgroundTasks) -> dict:
    secret = request.app.state.settings.slack_signing_secret
    if not secret:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "Slack channel is not configured")
    body = await request.body()
    if not verify_slack_signature(secret, request.headers.get("X-Slack-Request-Timestamp"), body,
                                  request.headers.get("X-Slack-Signature")):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid Slack signature")
    try:
        payload = json.loads(body)
    except json.JSONDecodeError:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Invalid JSON") from None

    if payload.get("type") == "url_verification":
        return {"challenge": payload.get("challenge")}
    if request.headers.get("X-Slack-Retry-Num"):
        return {"ok": True}  # we already acknowledged the original delivery
    event = payload.get("event") or {}
    if payload.get("type") == "event_callback" and event.get("type") in {"app_mention", "message"} \
            and not event.get("bot_id") and not event.get("subtype"):
        background.add_task(_process_event, request.app, event)
    return {"ok": True}
