import numpy as np
import pytest

from app.kv import InMemoryKV
from app.llm.openai_provider import OpenAIProvider
from app.memory import ConversationMemory
from app.observability.redact import redact
from app.rag.chunker import chunk_text
from app.rag.embeddings import HashingEmbedder
from app.rag.vectorstore import InMemoryVectorStore, QdrantVectorStore, VectorRecord


def test_chunker_keeps_section_path_and_limits_size():
    text = "# Guide\n\n## Setup\n" + ("Sentence about setup. " * 120) + "\n\n## Billing\nPay monthly."
    chunks = chunk_text(text, max_chars=500, overlap=80)
    assert all(len(c.text) <= 600 for c in chunks)
    assert chunks[0].section == "Guide > Setup" and chunks[-1].section == "Guide > Billing"


def test_chunker_sibling_headings_do_not_nest():
    chunks = chunk_text("## A\none\n## B\ntwo")
    assert [c.section for c in chunks] == ["A", "B"]


def test_hashing_embedder_similarity():
    e = HashingEmbedder()
    q, rel, irr = (np.array(v) for v in e.embed(["reset the thermostat", "Resetting your thermostat",
                                                  "invoice purchase order"]))
    assert q @ rel > q @ irr
    assert abs(np.linalg.norm(q) - 1) < 1e-5


@pytest.mark.parametrize("store_factory", [
    lambda: InMemoryVectorStore(4),
    lambda: QdrantVectorStore(":memory:", "test", 4),
])
def test_vector_stores(store_factory):
    store = store_factory()
    store.upsert([VectorRecord(1, [1, 0, 0, 0], {"document_id": 10, "visibility": "public"}),
                  VectorRecord(2, [0.9, 0.1, 0, 0], {"document_id": 11, "visibility": "internal"}),
                  VectorRecord(3, [0, 1, 0, 0], {"document_id": 11, "visibility": "internal"})])
    assert store.count() == 3
    hits = store.search([1, 0, 0, 0], top_k=2)
    assert [h.id for h in hits] == [1, 2]
    assert [h.id for h in store.search([1, 0, 0, 0], top_k=5, visibility=["public"])] == [1]
    store.delete_document(11)
    assert store.count() == 1


def test_inmemory_kv_list_semantics():
    kv = InMemoryKV()
    for i in range(5):
        kv.rpush("k", str(i))
    kv.ltrim("k", -3, -1)
    assert kv.lrange("k", 0, -1) == ["2", "3", "4"]
    assert kv.incr("n") == 1 and kv.incr("n") == 2


def test_memory_window():
    mem = ConversationMemory(InMemoryKV(), max_messages=3)
    for i in range(5):
        mem.append("c", "user", f"m{i}")
    assert [m["content"] for m in mem.load("c")] == ["m2", "m3", "m4"]


def test_redaction():
    out = redact("mail jane@example.com, card 4111 1111 1111 1111, call +1 415 555 0100, ticket TCK-1A2B3C4D")
    assert "jane@" not in out and "4111" not in out and "555" not in out
    assert "TCK-1A2B3C4D" in out


def test_openai_provider_parses_tool_calls():
    from types import SimpleNamespace as NS

    captured = {}

    class FakeCompletions:
        def create(self, **kwargs):
            captured.update(kwargs)
            msg = NS(content=None, tool_calls=[
                NS(id="call_1", function=NS(name="get_ticket", arguments='{"ticket_number": "TCK-1"}')),
                NS(id="call_2", function=NS(name="search_knowledge_base", arguments="not json"))])
            return NS(choices=[NS(message=msg)], usage=NS(prompt_tokens=12, completion_tokens=5))

    provider = OpenAIProvider("k", "gpt-test", client=NS(chat=NS(completions=FakeCompletions())))
    tools = [{"type": "function", "function": {"name": "get_ticket", "parameters": {}}}]
    resp = provider.chat([{"role": "user", "content": "hi"}], tools)
    assert captured["model"] == "gpt-test" and captured["tool_choice"] == "auto"
    assert resp.tool_calls[0].arguments == {"ticket_number": "TCK-1"}
    assert resp.tool_calls[1].arguments == {}
    assert resp.usage == {"prompt": 12, "completion": 5}
