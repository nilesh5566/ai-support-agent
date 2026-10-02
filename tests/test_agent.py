from tests.conftest import ADMIN, chat


def test_rag_answer_with_sources(client):
    d = chat(client, "How do I reset my thermostat?")
    assert "hold the dial for 10 seconds" in d["reply"]
    assert d["sources"] and d["sources"][0]["title"] == "Product Guide"
    assert {"name": "search_knowledge_base", "status": "ok"} in d["tool_calls"]
    assert d["escalated"] is False


def test_customer_account_lookup(client):
    d = chat(client, "What plan am I on?", email="jane@example.com")
    assert "Jane Cooper" in d["reply"] and "pro" in d["reply"].lower()


def test_orders_lookup(client):
    d = chat(client, "Where is my order?", email="jane@example.com")
    assert "ORD-50388" in d["reply"] and "shipped" in d["reply"]


def test_customer_cannot_read_other_customers(client):
    d = chat(client, "Show me the account for marcus@example.com", email="jane@example.com")
    assert "Marcus" not in d["reply"]
    assert "only access the account you are signed in with" in d["reply"]
    audit = client.get(f"/v1/conversations/{d['conversation_id']}", headers=ADMIN).json()["tool_audit"]
    assert any(a["tool"] == "get_customer_profile" and a["status"] == "error" for a in audit)


def test_guest_has_no_account_access(client):
    d = chat(client, "What plan am I on?")
    assert "No verified customer" in d["reply"]


def test_staff_can_look_up_any_customer(client):
    d = chat(client, "Show me the account for marcus@example.com", headers=ADMIN, as_staff=True)
    assert "Marcus Lee" in d["reply"] and "past_due" in d["reply"]


def test_internal_docs_hidden_from_customers_visible_to_staff(client):
    q = "What are the refund approval limits for tier 1 agents?"
    customer = chat(client, q)
    assert all(s["title"] != "Internal Support Policies" for s in customer["sources"])
    assert "$100" not in customer["reply"]
    staff = chat(client, q, headers=ADMIN, as_staff=True)
    assert any(s["title"] == "Internal Support Policies" for s in staff["sources"])
    assert "$100" in staff["reply"]


def test_human_request_creates_escalated_ticket_once(client):
    d1 = chat(client, "I want to talk to a human, my sensor is broken", email="jane@example.com")
    assert d1["escalated"] and d1["ticket_number"].startswith("TCK-")
    d2 = chat(client, "Please, I need a real person, it is still not working", email="jane@example.com",
              conversation_id=d1["conversation_id"])
    assert d2["ticket_number"] == d1["ticket_number"]  # idempotent per conversation
    tickets = [t for t in client.get("/v1/tickets", headers=ADMIN).json()
               if t["conversation_id"] == d1["conversation_id"]]
    assert len(tickets) == 1 and tickets[0]["status"] == "escalated"


def test_policy_keyword_forces_urgent_escalation(client):
    d = chat(client, "There is smoke coming out of my smart plug", email="priya@example.com")
    assert d["escalated"] is True
    t = next(t for t in client.get("/v1/tickets", headers=ADMIN).json() if t["ticket_number"] == d["ticket_number"])
    assert t["priority"] == "urgent"


def test_firmware_does_not_trigger_fire_keyword(client):
    d = chat(client, "Which firmware fixes the wifi issue?")
    assert d["escalated"] is False


def test_policy_enforced_even_if_model_skips_tool(client):
    """Guardrail lives in code: swap in a model that never calls tools and escalation still happens."""
    from app.llm.base import LLMResponse

    class LazyLLM:
        name, model = "lazy", "lazy-1"

        def chat(self, messages, tools, temperature=0.2):
            return LLMResponse("I'm sure it's fine.", [], {})

    client.app.state.agent.llm = LazyLLM()
    d = chat(client, "I'm calling my lawyer about this", email="jane@example.com")
    assert d["escalated"] is True and d["ticket_number"]
    assert any(t.get("enforced_by_policy") for t in d["tool_calls"])


def test_llm_failure_degrades_gracefully(client):
    from app.llm.base import LLMError

    class BrokenLLM:
        name, model = "broken", "broken-1"

        def chat(self, messages, tools, temperature=0.2):
            raise LLMError("upstream timeout")

    client.app.state.agent.llm = BrokenLLM()
    d = chat(client, "How do I reset my thermostat?")
    assert d["outcome"] == "llm_error" and "trouble" in d["reply"]


def test_ticket_status_lookup_respects_ownership(client):
    own = chat(client, "What's the status of TCK-DEMO0001?", email="marcus@example.com")
    assert "Card declined on renewal" in own["reply"]
    other = chat(client, "What's the status of TCK-DEMO0001?", email="jane@example.com")
    assert "not found" in other["reply"] and "Card declined" not in other["reply"]


def test_conversation_memory_and_history(client):
    d1 = chat(client, "How do I reset my thermostat?", email="jane@example.com")
    d2 = chat(client, "Where is my order?", email="jane@example.com", conversation_id=d1["conversation_id"])
    assert d2["conversation_id"] == d1["conversation_id"]
    conv = client.get(f"/v1/conversations/{d1['conversation_id']}", headers=ADMIN).json()
    assert [m["role"] for m in conv["messages"]] == ["user", "assistant", "user", "assistant"]
    history = client.app.state.agent.memory.load(d1["conversation_id"])
    assert len(history) == 4


def test_conversation_cannot_be_hijacked(client):
    d1 = chat(client, "What plan am I on?", email="jane@example.com")
    r = client.post("/v1/chat", headers={"X-API-Key": "dev-client-key"},
                    json={"message": "and now?", "customer_email": "marcus@example.com",
                          "conversation_id": d1["conversation_id"]})
    assert r.status_code == 403
    r = client.post("/v1/chat", headers={"X-API-Key": "dev-client-key"},
                    json={"message": "hi", "conversation_id": "does-not-exist"})
    assert r.status_code == 404


def test_trace_returned_and_stored(client):
    d = chat(client, "How do I reset my thermostat?", debug=True)
    names = [s["name"] for s in d["trace"]["spans"]]
    assert {"agent.turn", "llm.call", "tool.search_knowledge_base", "rag.search"} <= set(names)
    stored = client.get(f"/v1/traces/{d['trace_id']}", headers=ADMIN)
    assert stored.status_code == 200 and stored.json()["trace_id"] == d["trace_id"]


def test_max_iterations_forces_final_answer(client):
    from app.llm.base import LLMResponse, ToolCall

    class LoopyLLM:
        name, model = "loopy", "loopy-1"

        def chat(self, messages, tools, temperature=0.2):
            if tools:
                return LLMResponse(None, [ToolCall("c1", "search_knowledge_base", {"query": "reset"})], {})
            return LLMResponse("Final answer.", [], {})

    client.app.state.agent.llm = LoopyLLM()
    d = chat(client, "loop forever")
    assert d["outcome"] == "max_iterations" and d["reply"] == "Final answer."


def test_qdrant_backend_end_to_end(app_factory):
    c = app_factory(vector_backend="qdrant", qdrant_url=":memory:")
    d = chat(c, "How do I reset my thermostat?")
    assert "hold the dial for 10 seconds" in d["reply"]
    assert c.get("/ready").json()["checks"]["vector_store"] == "ok"
    staff = chat(c, "What are the refund approval limits for tier 1 agents?", headers=ADMIN, as_staff=True)
    assert "$100" in staff["reply"]
