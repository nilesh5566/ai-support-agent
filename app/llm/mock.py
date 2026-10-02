"""Deterministic, offline LLM stand-in.

It speaks the same tool-calling protocol as the OpenAI provider: on the first pass of a turn it
plans tool calls from simple intent rules; once tool results are in, it composes a grounded answer
from them. This lets the full platform (RAG, tools, auth, escalation, metrics) run and be tested
without any API key. Switch LLM_PROVIDER to groq, gemini, openrouter, ollama... for real reasoning.
"""
from __future__ import annotations

import json
import re
import uuid

from app.llm.base import LLMResponse, ToolCall
from app.rag.text import tokenize

_EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
_TICKET = re.compile(r"\bTCK-[A-Z0-9]{4,}\b", re.IGNORECASE)
_SENTENCES = re.compile(r"(?<=[.!?])\s+|\n+")

ACCOUNT_WORDS = ("my account", "my plan", "my subscription", "my billing", "my invoice", "my profile", "am i on",
                 "my status", "account status", "account for", "account of")
ORDER_WORDS = ("my order", "my orders", "my shipment", "my delivery", "my package", "where is my", "order status",
               "orders for")
HUMAN_WORDS = ("human", "real person", "agent", "representative", "escalate", "speak to", "talk to", "manager",
               "supervisor", "lawyer", "legal", "chargeback")
TICKET_WORDS = ("open a ticket", "create a ticket", "file a ticket", "raise a ticket", "complaint", "i want a refund",
                "refund me", "request a refund", "not working", "broken", "stopped working", "defective", "outage",
                "urgent")
GREETINGS = {"hi", "hello", "hey", "thanks", "thank you", "good morning", "ok", "okay"}


def _has(text: str, words) -> bool:
    return any(w in text for w in words)


class MockLLM:
    name = "mock"
    model = "mock-rules-v1"

    def chat(self, messages: list[dict], tools: list[dict], temperature: float = 0.2) -> LLMResponse:
        tool_names = {t["function"]["name"] for t in tools or []}
        user_idx = max(i for i, m in enumerate(messages) if m["role"] == "user")
        user_text = messages[user_idx]["content"]
        system = messages[0]["content"] if messages and messages[0]["role"] == "system" else ""
        turn = messages[user_idx + 1 :]
        usage = {"prompt": sum(len(str(m.get("content") or "")) for m in messages) // 4}

        if tool_names and not any(m["role"] == "tool" for m in turn):
            calls = self._plan(user_text, tool_names, system)
            if calls:
                return LLMResponse(None, calls, {**usage, "completion": 20 * len(calls)})

        reply = self._compose(user_text, self._collect_results(turn))
        return LLMResponse(reply, [], {**usage, "completion": len(reply) // 4})

    # ---------------------------------------------------------------- planning
    def _plan(self, text: str, tools: set[str], system: str) -> list[ToolCall]:
        low = text.lower().strip()
        if low.strip("!.? ") in GREETINGS:
            return []
        calls: list[tuple[str, dict]] = []
        email = _EMAIL.search(text)
        who = {"email": email.group(0)} if email else {}
        ticket = _TICKET.search(text)

        if ticket and "get_ticket" in tools:
            calls.append(("get_ticket", {"ticket_number": ticket.group(0).upper()}))
        if _has(low, ACCOUNT_WORDS) and "get_customer_profile" in tools:
            calls.append(("get_customer_profile", who))
        if _has(low, ORDER_WORDS) and "list_orders" in tools:
            calls.append(("list_orders", who))

        must_escalate = "must be escalated" in system.lower()
        wants_human = _has(low, HUMAN_WORDS)
        if (must_escalate or wants_human or _has(low, TICKET_WORDS)) and "create_ticket" in tools:
            if must_escalate or _has(low, ("urgent", "outage", "lawyer", "legal", "chargeback", "security")):
                priority = "urgent"
            elif _has(low, ("refund", "broken", "not working", "stopped working", "defective", "outage")):
                priority = "high"
            else:
                priority = "normal"
            subject = re.split(r"[.!?\n]", text.strip())[0][:120] or "Customer request"
            calls.append(("create_ticket", {
                "subject": subject, "description": text[:2000], "priority": priority,
                "escalate": must_escalate or wants_human,
            }))

        # Safety/legal escalations go straight to a human; don't stall them with doc lookups.
        if "search_knowledge_base" in tools and not must_escalate and not (ticket and len(tokenize(text)) <= 4):
            calls.append(("search_knowledge_base", {"query": text[:500]}))
        return [ToolCall(id=f"call_{uuid.uuid4().hex[:12]}", name=n, arguments=a) for n, a in calls]

    # ------------------------------------------------------------- composition
    @staticmethod
    def _collect_results(turn: list[dict]) -> dict[str, dict]:
        names: dict[str, str] = {}
        results: dict[str, dict] = {}
        for m in turn:
            if m["role"] == "assistant":
                for tc in m.get("tool_calls") or []:
                    names[tc["id"]] = tc["function"]["name"]
            elif m["role"] == "tool":
                try:
                    results[names.get(m["tool_call_id"], "unknown")] = json.loads(m["content"])
                except (json.JSONDecodeError, TypeError):
                    continue
        return results

    @staticmethod
    def _best_sentences(query: str, text: str, limit: int = 3) -> str:
        q = set(tokenize(query))
        sentences = [s.strip(" -*\t") for s in _SENTENCES.split(text) if len(s.strip(" -*\t")) > 3]
        if not sentences:
            return text[:400]
        scored = sorted(((len(q & set(tokenize(s))), i) for i, s in enumerate(sentences)), reverse=True)
        keep = sorted(i for score, i in scored[:limit] if score > 0) or [0, 1][: len(sentences)]
        return " ".join(sentences[i].rstrip(".") + "." for i in keep)

    def _compose(self, user_text: str, results: dict[str, dict]) -> str:
        parts: list[str] = []
        if not results and user_text.lower().strip("!.? ") in GREETINGS:
            return "Hi! I'm the support assistant. Ask me about our products, your account, orders or tickets."

        if (t := results.get("get_ticket")) is not None:
            parts.append(f"I couldn't look up that ticket: {t['error']}" if "error" in t else
                         f"Ticket **{t['ticket_number']}** ({t['subject']}) is currently **{t['status']}** "
                         f"with {t['priority']} priority.")
        if (c := results.get("get_customer_profile")) is not None:
            parts.append(f"I couldn't access account details: {c['error']}" if "error" in c else
                         f"I found your account, {c['name']}: you're on the **{c['plan']}** plan and your account "
                         f"status is **{c['status']}**.")
        if (o := results.get("list_orders")) is not None:
            if "error" in o:
                parts.append(f"I couldn't access orders: {o['error']}")
            elif not o.get("orders"):
                parts.append("I don't see any orders on your account.")
            else:
                lines = [f"- {x['order_number']}: {x['status']} ({x['total']})" for x in o["orders"]]
                parts.append("Here are your most recent orders:\n" + "\n".join(lines))
        if (kb := results.get("search_knowledge_base")) is not None:
            hits = kb.get("results") or []
            # With account data already in the answer, only add docs that are clearly relevant.
            if hits and (not parts or hits[0]["score"] >= 0.2):
                top = hits[0]
                section = (top.get("section") or "").split(" > ")[-1]
                label = f"{top['title']} > {section}" if section and section != top["title"] else top["title"]
                parts.append(self._best_sentences(user_text, top["text"]) + f"\n\n_Source: {label}_")
            elif len(parts) == 0 and "create_ticket" not in results:
                parts.append("I couldn't find an answer to that in our documentation. "
                             "I can open a ticket so a specialist can help — just say the word.")
        if (tk := results.get("create_ticket")) is not None:
            if "error" in tk:
                parts.append(f"I wasn't able to create a ticket: {tk['error']}")
            else:
                verb = "updated your existing ticket" if tk.get("deduplicated") else "created ticket"
                tail = " and escalated it to a human specialist" if tk.get("escalated") else ""
                parts.append(f"I've {verb} **{tk['ticket_number']}** ({tk['priority']} priority){tail}. "
                             "Our team will follow up by email.")
        return "\n\n".join(parts) or "Could you tell me a bit more about what you need help with?"
