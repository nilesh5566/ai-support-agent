from __future__ import annotations

import logging

import httpx

logger = logging.getLogger(__name__)


class EscalationNotifier:
    """Posts escalations to a webhook (Slack incoming-webhook compatible). Never blocks the turn on failure."""

    def __init__(self, webhook_url: str | None):
        self.webhook_url = webhook_url

    def notify(self, ticket, reason: str) -> None:
        logger.warning("ticket escalated", extra={"fields": {
            "ticket": ticket.ticket_number, "priority": ticket.priority, "reason": reason}})
        if not self.webhook_url:
            return
        text = (f":rotating_light: Escalation {ticket.ticket_number} [{ticket.priority}] via {ticket.channel}: "
                f"{ticket.subject}")
        try:
            httpx.post(self.webhook_url, json={"text": text}, timeout=3.0)
        except httpx.HTTPError:
            logger.exception("escalation webhook failed")
