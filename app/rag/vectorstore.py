"""Vector store abstraction: in-memory (numpy) for dev/tests, Qdrant for production."""
from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from typing import Protocol

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class VectorRecord:
    id: int
    vector: list[float]
    payload: dict


@dataclass
class SearchHit:
    id: int
    score: float
    payload: dict


class VectorStore(Protocol):
    def upsert(self, records: list[VectorRecord]) -> None: ...
    def search(self, vector: list[float], top_k: int, visibility: list[str] | None = None) -> list[SearchHit]: ...
    def delete_document(self, document_id: int) -> None: ...
    def count(self) -> int: ...
    def ping(self) -> None: ...


class InMemoryVectorStore:
    def __init__(self, dim: int):
        self.dim = dim
        self._records: dict[int, VectorRecord] = {}
        self._lock = threading.Lock()

    def upsert(self, records: list[VectorRecord]) -> None:
        with self._lock:
            for r in records:
                if len(r.vector) != self.dim:
                    raise ValueError(f"vector dim {len(r.vector)} != {self.dim}")
                self._records[r.id] = r

    def search(self, vector: list[float], top_k: int, visibility: list[str] | None = None) -> list[SearchHit]:
        with self._lock:
            candidates = [
                r for r in self._records.values() if visibility is None or r.payload.get("visibility") in visibility
            ]
        if not candidates:
            return []
        matrix = np.asarray([r.vector for r in candidates], dtype=np.float32)
        q = np.asarray(vector, dtype=np.float32)
        q_norm = np.linalg.norm(q) or 1.0
        norms = np.linalg.norm(matrix, axis=1)
        norms[norms == 0] = 1.0
        scores = matrix @ q / (norms * q_norm)
        order = np.argsort(-scores)[:top_k]
        return [SearchHit(candidates[i].id, float(scores[i]), candidates[i].payload) for i in order]

    def delete_document(self, document_id: int) -> None:
        with self._lock:
            for rid in [rid for rid, r in self._records.items() if r.payload.get("document_id") == document_id]:
                del self._records[rid]

    def count(self) -> int:
        return len(self._records)

    def ping(self) -> None:
        return None


class QdrantVectorStore:
    def __init__(self, url: str, collection: str, dim: int, retries: int = 15):
        from qdrant_client import QdrantClient, models

        self.models = models
        self.collection = collection
        self.dim = dim
        self.local = url == ":memory:"
        self.client = QdrantClient(location=":memory:") if self.local else QdrantClient(url=url, timeout=10)
        for attempt in range(1, retries + 1):
            try:
                self._ensure_collection()
                break
            except Exception as exc:  # qdrant container may still be booting
                if attempt == retries:
                    raise
                logger.warning("qdrant not ready (%s), retry %d/%d", type(exc).__name__, attempt, retries)
                time.sleep(2)

    def _ensure_collection(self) -> None:
        m = self.models
        if not self.client.collection_exists(self.collection):
            self.client.create_collection(
                collection_name=self.collection,
                vectors_config=m.VectorParams(size=self.dim, distance=m.Distance.COSINE),
            )
            if not self.local:  # payload indexes only matter on a Qdrant server
                for field_name, schema in (("visibility", m.PayloadSchemaType.KEYWORD),
                                           ("document_id", m.PayloadSchemaType.INTEGER)):
                    self.client.create_payload_index(self.collection, field_name=field_name, field_schema=schema)

    def upsert(self, records: list[VectorRecord]) -> None:
        if not records:
            return
        points = [self.models.PointStruct(id=r.id, vector=r.vector, payload=r.payload) for r in records]
        self.client.upsert(collection_name=self.collection, points=points, wait=True)

    def search(self, vector: list[float], top_k: int, visibility: list[str] | None = None) -> list[SearchHit]:
        m = self.models
        query_filter = None
        if visibility is not None:
            query_filter = m.Filter(must=[m.FieldCondition(key="visibility", match=m.MatchAny(any=visibility))])
        res = self.client.query_points(
            collection_name=self.collection, query=vector, limit=top_k, query_filter=query_filter, with_payload=True
        )
        return [SearchHit(int(p.id), float(p.score), dict(p.payload or {})) for p in res.points]

    def delete_document(self, document_id: int) -> None:
        m = self.models
        self.client.delete(
            collection_name=self.collection,
            points_selector=m.FilterSelector(
                filter=m.Filter(must=[m.FieldCondition(key="document_id", match=m.MatchValue(value=document_id))])
            ),
            wait=True,
        )

    def count(self) -> int:
        return int(self.client.count(collection_name=self.collection, exact=True).count)

    def ping(self) -> None:
        self.client.get_collection(self.collection)


def create_vector_store(settings, embedder) -> VectorStore:
    if settings.vector_backend == "qdrant":
        name = f"{settings.qdrant_collection}_{embedder.name}_{embedder.dim}"
        return QdrantVectorStore(settings.qdrant_url, name, embedder.dim)
    return InMemoryVectorStore(embedder.dim)
