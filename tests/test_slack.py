import hashlib
import hmac
import json
import time

SECRET = "test-signing-secret"


def _signed(body: dict, secret: str = SECRET, ts: int | None = None):
    raw = json.dumps(body).encode()
    ts = str(ts or int(time.time()))
    sig = "v0=" + hmac.new(secret.encode(), f"v0:{ts}:".encode() + raw, hashlib.sha256).hexdigest()
    return raw, {"X-Slack-Request-Timestamp": ts, "X-Slack-Signature": sig, "Content-Type": "application/json"}


def test_slack_not_configured(client):
    assert client.post("/v1/slack/events", json={}).status_code == 503


def test_slack_url_verification_and_signature(app_factory):
    c = app_factory(slack_signing_secret=SECRET)
    raw, headers = _signed({"type": "url_verification", "challenge": "xyz"})
    r = c.post("/v1/slack/events", content=raw, headers=headers)
    assert r.status_code == 200 and r.json() == {"challenge": "xyz"}

    raw, headers = _signed({"type": "url_verification", "challenge": "xyz"}, secret="wrong")
    assert c.post("/v1/slack/events", content=raw, headers=headers).status_code == 401

    raw, headers = _signed({"type": "url_verification", "challenge": "xyz"}, ts=int(time.time()) - 3600)
    assert c.post("/v1/slack/events", content=raw, headers=headers).status_code == 401


def test_slack_mention_runs_agent_as_staff(app_factory):
    c = app_factory(slack_signing_secret=SECRET)
    event = {"type": "event_callback", "event": {
        "type": "app_mention", "text": "<@U123> what are the refund approval limits for tier 1 agents?",
        "channel": "C42", "ts": "1700000000.000100"}}
    raw, headers = _signed(event)
    assert c.post("/v1/slack/events", content=raw, headers=headers).json() == {"ok": True}
    conv = c.get("/v1/conversations/slack-C42-1700000000.000100", headers={"X-API-Key": "dev-admin-key"})
    assert conv.status_code == 200
    body = conv.json()
    assert body["channel"] == "slack" and "$100" in body["messages"][-1]["content"]
