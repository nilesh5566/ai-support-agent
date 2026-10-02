from __future__ import annotations

from app.observability.metrics import RAG_HITS, RAG_LATENCY, RAG_TOP_SCORE


class Retriever:
    def __init__(self, embedder, store, top_k: int = 4, min_score: float = 0.12):
        self.embedder = embedder
        self.store = store
        self.top_k = top_k
        self.min_score = min_score

    def search(self, query: str, visibility: list[str] | None, top_k: int | None = None) -> list[dict]:
        with RAG_LATENCY.time():
            vector = self.embedder.embed([query])[0]
            hits = self.store.search(vector, top_k or self.top_k, visibility)
        RAG_TOP_SCORE.observe(hits[0].score if hits else 0.0)
        relevant = [h for h in hits if h.score >= self.min_score]
        RAG_HITS.observe(len(relevant))
        return [
            {
                "chunk_id": h.id,
                "document_id": h.payload.get("document_id"),
                "title": h.payload.get("title"),
                "section": h.payload.get("section"),
                "visibility": h.payload.get("visibility"),
                "text": h.payload.get("text"),
                "score": round(h.score, 4),
            }
            for h in relevant
        ]
