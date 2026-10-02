from tests.conftest import ADMIN, CLIENT, chat


def test_health_ready_metrics(client):
    body = client.get("/health").json()
    assert body["status"] == "ok" and body["llm"]["provider"] == "mock"
    r = client.get("/ready")
    assert r.status_code == 200 and r.json()["status"] == "ready"
    chat(client, "How do I reset my thermostat?")
    metrics = client.get("/metrics").text
    for name in ("agent_chat_turns_total", "agent_tool_calls_total", "agent_rag_retrieval_seconds",
                 "http_request_duration_seconds"):
        assert name in metrics


def test_web_ui_served(client):
    r = client.get("/")
    assert r.status_code == 200 and "Acme Smart Home support" in r.text


def test_auth_required_and_roles(client):
    assert client.post("/v1/chat", json={"message": "hi"}).status_code == 401
    assert client.post("/v1/chat", headers={"X-API-Key": "nope"}, json={"message": "hi"}).status_code == 401
    assert client.get("/v1/documents", headers=CLIENT).status_code == 403
    assert client.get("/v1/documents", headers=ADMIN).status_code == 200
    r = client.post("/v1/chat", headers=CLIENT, json={"message": "hi", "as_staff": True})
    assert r.status_code == 403


def test_request_validation(client):
    assert client.post("/v1/chat", headers=CLIENT, json={"message": "   "}).status_code == 422
    assert client.post("/v1/chat", headers=CLIENT, json={"message": "x" * 4001}).status_code == 422
    assert client.post("/v1/chat", headers=CLIENT,
                       json={"message": "hi", "customer_email": "not-an-email"}).status_code == 422


def test_request_id_header(client):
    r = client.get("/health", headers={"X-Request-ID": "abc123"})
    assert r.headers["X-Request-ID"] == "abc123"


def test_rate_limit(app_factory):
    c = app_factory(rate_limit_per_minute=3)
    codes = [c.get("/v1/tools", headers=CLIENT).status_code for _ in range(5)]
    assert codes[:3] == [200, 200, 200] and codes[3:] == [429, 429]


def test_tools_listing(client):
    names = {t["name"] for t in client.get("/v1/tools", headers=CLIENT).json()}
    assert names == {"search_knowledge_base", "get_customer_profile", "list_orders", "create_ticket", "get_ticket"}


def test_config_endpoint(client):
    cfg = client.get("/v1/config", headers=ADMIN).json()
    assert cfg["runtime"]["llm_provider"] == "mock"
    assert "staff" in cfg["agent"]["roles"]
