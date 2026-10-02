"""Free/hosted LLM provider support: presets, fallback chain, and an end-to-end run where the real
OpenAI SDK talks HTTP to a fake Groq / Gemini endpoint (no network, no keys)."""
import json

import httpx
import openai
import pytest

from app.config import Settings
from app.llm.base import LLMError, LLMResponse
from app.llm.factory import create_llm
from app.llm.fallback import FallbackLLM
from app.llm.openai_provider import OpenAIProvider
from app.llm.providers import PRESETS, resolve
from app.rag.embeddings import HashingEmbedder, OpenAICompatEmbedder, create_embedder
from tests.conftest import ADMIN, chat

KEYS = dict(groq_api_key="gsk_test", gemini_api_key="gm_test", openrouter_api_key="or_test",
            cerebras_api_key="cb_test", mistral_api_key="ms_test", openai_api_key="sk_test")


def settings(**kw) -> Settings:
    return Settings(_env_file=None, **{**KEYS, **kw})


# --------------------------------------------------------------------------------- presets
@pytest.mark.parametrize("name,host,model", [
    ("groq", "api.groq.com", "openai/gpt-oss-120b"),
    ("gemini", "generativelanguage.googleapis.com", "gemini-flash-latest"),
    ("openrouter", "openrouter.ai", "openai/gpt-oss-120b:free"),
    ("cerebras", "api.cerebras.ai", "gpt-oss-120b"),
    ("mistral", "api.mistral.ai", "mistral-small-latest"),
    ("ollama", "localhost:11434", "qwen3:8b"),
])
def test_presets_resolve(name, host, model):
    r = resolve(settings(llm_provider=name), name, primary=True)
    assert host in r.base_url and r.model == model and r.api_key


def test_model_and_url_overrides():
    s = settings(llm_provider="groq", llm_model="qwen/qwen3-32b", llm_base_url="http://proxy:4000/v1")
    r = resolve(s, "groq", primary=True)
    assert (r.model, r.base_url) == ("qwen/qwen3-32b", "http://proxy:4000/v1")
    # overrides apply to the primary only, never to fallbacks
    assert resolve(s, "gemini", primary=False).model == PRESETS["gemini"].default_model


def test_missing_key_gives_signup_hint():
    with pytest.raises(ValueError, match="GROQ_API_KEY.*console.groq.com"):
        create_llm(Settings(_env_file=None, llm_provider="groq", groq_api_key=None))


def test_generic_llm_api_key_and_custom_provider():
    r = resolve(Settings(_env_file=None, llm_provider="groq", groq_api_key=None, llm_api_key="gsk_generic"),
                "groq", primary=True)
    assert r.api_key == "gsk_generic"
    with pytest.raises(ValueError, match="LLM_BASE_URL"):
        resolve(settings(llm_provider="custom", llm_model="x"), "custom", primary=True)
    r = resolve(settings(llm_provider="custom", llm_model="local-model", llm_base_url="http://vllm:8000/v1"),
                "custom", primary=True)
    assert r.base_url == "http://vllm:8000/v1" and r.api_key == "not-needed"


def test_factory_builds_fallback_chain():
    llm = create_llm(settings(llm_provider="groq", llm_fallback_providers="gemini, groq, mock, gemini"))
    assert isinstance(llm, FallbackLLM)
    assert llm.chain == ["groq:openai/gpt-oss-120b", "gemini:gemini-flash-latest", "mock:mock-rules-v1"]
    assert isinstance(create_llm(settings(llm_provider="groq")), OpenAIProvider)


# ---------------------------------------------------------------------------- fallback chain
class Flaky:
    def __init__(self, name, fail):
        self.name, self.model, self.fail, self.calls = name, f"{name}-model", fail, 0

    def chat(self, messages, tools, temperature=0.2):
        self.calls += 1
        if self.fail:
            raise LLMError(f"{self.name} 429")
        return LLMResponse(content=f"hi from {self.name}")


def test_fallback_uses_next_provider_and_cools_down_the_failed_one():
    now = [0.0]
    a, b = Flaky("a", fail=True), Flaky("b", fail=False)
    llm = FallbackLLM([a, b], cooldown_seconds=30, clock=lambda: now[0])
    r = llm.chat([], [])
    assert (r.content, r.provider, r.model) == ("hi from b", "b", "b-model")
    llm.chat([], [])
    assert a.calls == 1  # skipped while cooling down
    now[0] = 31
    llm.chat([], [])
    assert a.calls == 2  # retried after cooldown


def test_fallback_raises_when_all_fail():
    with pytest.raises(LLMError, match="all LLM providers failed"):
        FallbackLLM([Flaky("a", True), Flaky("b", True)]).chat([], [])


# ------------------------------------------------------------------- HTTP-level fake provider
class FakeOpenAICompatServer:
    """Minimal /chat/completions + /embeddings implementation that plans one tool call per turn."""

    def __init__(self, expect_host: str, expect_key: str, fail_status: int | None = None):
        self.expect_host, self.expect_key, self.fail_status = expect_host, expect_key, fail_status
        self.requests: list[dict] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        assert request.url.host == self.expect_host, request.url
        assert request.headers["authorization"] == f"Bearer {self.expect_key}"
        body = json.loads(request.content)
        self.requests.append({"path": request.url.path, "body": body})
        if self.fail_status:
            return httpx.Response(self.fail_status, json={"error": {"message": "rate limited"}})
        if request.url.path.endswith("/embeddings"):
            data = [{"object": "embedding", "index": i, "embedding": [0.1] * 8} for i, _ in enumerate(body["input"])]
            return httpx.Response(200, json={"object": "list", "data": data, "model": body["model"],
                                             "usage": {"prompt_tokens": 1, "total_tokens": 1}})
        msgs = body["messages"]
        if msgs[-1]["role"] == "tool":
            result = json.loads(msgs[-1]["content"])
            message = {"role": "assistant", "content": "<think>summarise</think>"
                       f"You have {len(result.get('orders', []))} orders on file."}
        else:
            assert any(t["function"]["name"] == "list_orders" for t in body.get("tools", []))
            message = {"role": "assistant", "content": None, "tool_calls": [{
                "id": "call_abc123", "type": "function",
                "function": {"name": "list_orders", "arguments": "{}"}}]}
        return httpx.Response(200, json={
            "id": "chatcmpl-1", "object": "chat.completion", "created": 0, "model": body["model"],
            "choices": [{"index": 0, "message": message, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 50, "completion_tokens": 10, "total_tokens": 60}})


@pytest.fixture
def fake_servers(monkeypatch):
    servers: dict[str, FakeOpenAICompatServer] = {}
    real = openai.OpenAI

    def patched(*args, base_url=None, **kwargs):
        transport = httpx.MockTransport(lambda req: servers[req.url.host](req))
        kwargs["max_retries"] = 0
        return real(*args, base_url=base_url, http_client=httpx.Client(transport=transport), **kwargs)

    monkeypatch.setattr(openai, "OpenAI", patched)
    return servers


def test_end_to_end_chat_through_groq(app_factory, fake_servers):
    fake_servers["api.groq.com"] = server = FakeOpenAICompatServer("api.groq.com", "gsk_test")
    client = app_factory(llm_provider="groq", groq_api_key="gsk_test")
    assert client.get("/health").json()["llm"] == {"provider": "groq", "model": "openai/gpt-oss-120b",
                                                    "chain": None}
    d = chat(client, "What are my orders?", email="jane@example.com", debug=True)
    assert d["reply"].startswith("You have ") and "<think>" not in d["reply"]
    assert [t["name"] for t in d["tool_calls"]] == ["list_orders"]
    first = server.requests[0]["body"]
    assert first["model"] == "openai/gpt-oss-120b" and server.requests[0]["path"] == "/openai/v1/chat/completions"
    llm_spans = [s for s in d["trace"]["spans"] if s["name"] == "llm.call"]
    assert llm_spans and all(s["attributes"]["served_by"] == "groq" for s in llm_spans)
    providers = client.get("/v1/llm/providers", headers=ADMIN).json()
    assert providers["active"]["provider"] == "groq" and {p["name"] for p in providers["available"]} >= {
        "groq", "gemini", "openrouter", "ollama"}


def test_rate_limited_groq_falls_back_to_gemini(app_factory, fake_servers):
    fake_servers["api.groq.com"] = FakeOpenAICompatServer("api.groq.com", "gsk_test", fail_status=429)
    fake_servers["generativelanguage.googleapis.com"] = gemini = FakeOpenAICompatServer(
        "generativelanguage.googleapis.com", "gm_test")
    client = app_factory(llm_provider="groq", groq_api_key="gsk_test", gemini_api_key="gm_test",
                         llm_fallback_providers="gemini")
    d = chat(client, "What are my orders?", email="jane@example.com", debug=True)
    assert d["outcome"] != "llm_error" and d["reply"].startswith("You have ")
    assert gemini.requests[0]["body"]["model"] == "gemini-flash-latest"
    assert {s["attributes"].get("served_by") for s in d["trace"]["spans"] if s["name"] == "llm.call"} == {"gemini"}


def test_all_providers_down_degrades_gracefully(app_factory, fake_servers):
    fake_servers["api.groq.com"] = FakeOpenAICompatServer("api.groq.com", "gsk_test", fail_status=503)
    client = app_factory(llm_provider="groq", groq_api_key="gsk_test")
    d = chat(client, "What are my orders?", email="jane@example.com")
    assert d["reply"]  # orchestrator's LLMError fallback still answers politely


def test_semantic_embedders(fake_servers):
    fake_servers["generativelanguage.googleapis.com"] = FakeOpenAICompatServer(
        "generativelanguage.googleapis.com", "gm_test")
    emb = create_embedder(settings(embedding_provider="gemini"))
    assert isinstance(emb, OpenAICompatEmbedder) and emb.dim == 8  # dimension probed at startup
    assert emb.name == "gemini-gemini-embedding-001"
    assert len(emb.embed(["a", "b", "c"])) == 3
    assert isinstance(create_embedder(settings()), HashingEmbedder)
    ollama = create_embedder(settings(embedding_provider="ollama"))
    assert ollama.dim == 768 and "11434" in str(ollama.client.base_url)


def test_malformed_tool_call_is_retried_once():
    class Exc(Exception):
        status_code = 400
        body = {"error": {"code": "tool_use_failed"}}

    class Completions:
        calls = 0

        def create(self, **kw):
            Completions.calls += 1
            if Completions.calls == 1:
                raise Exc("bad tool call")
            msg = type("M", (), {"content": "ok", "tool_calls": None})
            return type("R", (), {"choices": [type("C", (), {"message": msg})], "usage": None})

    client = type("Cl", (), {"chat": type("Ch", (), {"completions": Completions()})})
    r = OpenAIProvider("k", "m", client=client, name="groq").chat([{"role": "user", "content": "x"}], [])
    assert r.content == "ok" and Completions.calls == 2 and r.provider == "groq"
