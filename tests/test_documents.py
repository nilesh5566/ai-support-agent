import json

from tests.conftest import ADMIN, CLIENT, chat


def test_seeded_documents(client):
    docs = client.get("/v1/documents", headers=ADMIN).json()
    titles = {d["title"] for d in docs}
    assert {"Product Guide", "Frequently Asked Questions", "Resolved Support Tickets",
            "Internal Support Policies"} <= titles
    assert next(d for d in docs if d["title"] == "Internal Support Policies")["visibility"] == "internal"


def test_upload_search_delete(client):
    content = b"# Pool Heater\n\n## Winter mode\nThe Acme Pool Heater enters frost protection below 3 degrees."
    r = client.post("/v1/documents", headers=ADMIN, files={"file": ("pool_heater.md", content, "text/markdown")},
                    data={"visibility": "public"})
    assert r.status_code == 201, r.text
    doc = r.json()
    assert doc["chunk_count"] >= 1

    d = chat(client, "Does the pool heater have frost protection?")
    assert "frost protection" in d["reply"]

    # Re-uploading identical content is idempotent
    again = client.post("/v1/documents", headers=ADMIN, files={"file": ("pool_heater.md", content, "text/markdown")})
    assert again.json()["id"] == doc["id"]

    assert client.delete(f"/v1/documents/{doc['id']}", headers=ADMIN).status_code == 204
    hits = client.post("/v1/search", headers=ADMIN, json={"query": "pool heater frost protection"}).json()["results"]
    assert all(h["document_id"] != doc["id"] for h in hits)
    assert client.delete(f"/v1/documents/{doc['id']}", headers=ADMIN).status_code == 404


def test_new_version_replaces_old(client):
    up = lambda body: client.post("/v1/documents", headers=ADMIN,  # noqa: E731
                                  files={"file": ("policy.txt", body, "text/plain")}).json()
    up(b"Returns are accepted within 10 days.")
    up(b"Returns are accepted within 45 days.")
    docs = [d for d in client.get("/v1/documents", headers=ADMIN).json() if d["source"] == "policy.txt"]
    assert len(docs) == 1
    hits = client.post("/v1/search", headers=ADMIN, json={"query": "returns accepted within days"}).json()["results"]
    texts = " ".join(h["text"] for h in hits)
    assert "45 days" in texts and "10 days" not in texts


def test_text_and_json_ingestion(client):
    r = client.post("/v1/documents/text", headers=ADMIN,
                    json={"title": "Holiday hours", "content": "Support is closed on 25 December.",
                          "visibility": "internal"})
    assert r.status_code == 201 and r.json()["visibility"] == "internal"
    faq = json.dumps([{"question": "Do you ship to Canada?", "answer": "Yes, Canada shipping takes 7 days."}])
    r = client.post("/v1/documents", headers=ADMIN, files={"file": ("ca.json", faq.encode(), "application/json")})
    assert r.status_code == 201 and r.json()["content_type"] == "json"


def test_upload_validation(client):
    bad = client.post("/v1/documents", headers=ADMIN, files={"file": ("x.exe", b"MZ", "application/octet-stream")})
    assert bad.status_code == 400 and "unsupported" in bad.json()["detail"]
    empty = client.post("/v1/documents", headers=ADMIN, files={"file": ("e.md", b"   ", "text/markdown")})
    assert empty.status_code == 400
    broken = client.post("/v1/documents", headers=ADMIN, files={"file": ("b.json", b"{nope", "application/json")})
    assert broken.status_code == 400
    assert client.post("/v1/documents", headers=CLIENT, files={"file": ("a.md", b"# hi", "text/markdown")}
                       ).status_code == 403


def test_upload_size_limit(app_factory):
    c = app_factory(max_upload_bytes=100)
    r = c.post("/v1/documents", headers=ADMIN, files={"file": ("big.txt", b"a " * 200, "text/plain")})
    assert r.status_code == 413
