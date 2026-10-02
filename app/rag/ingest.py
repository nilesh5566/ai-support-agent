"""Document ingestion: parse -> chunk -> embed -> index (Postgres keeps the source of truth)."""
from __future__ import annotations

import csv
import hashlib
import io
import json
import logging
from pathlib import PurePath

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import Document, DocumentChunk
from app.observability.metrics import DOCS_INGESTED
from app.rag.chunker import chunk_text
from app.rag.vectorstore import VectorRecord

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".md", ".markdown", ".txt", ".csv", ".json"}
VISIBILITIES = {"public", "internal"}


class IngestionError(ValueError):
    pass


def _csv_to_text(raw: str) -> str:
    reader = csv.DictReader(io.StringIO(raw))
    blocks = []
    for i, row in enumerate(reader, 1):
        title_key = next((k for k in ("subject", "question", "title") if row.get(k)), None)
        title = row[title_key] if title_key else f"Record {i}"
        lines = [f"{k.strip().capitalize()}: {(v or '').strip()}" for k, v in row.items() if k and k != title_key]
        blocks.append(f"## {title}\n" + "\n".join(lines))
    return "\n\n".join(blocks)


def _json_to_text(raw: str) -> str:
    data = json.loads(raw)
    if isinstance(data, dict) and isinstance(data.get("faqs"), list):
        data = data["faqs"]
    if isinstance(data, list) and all(isinstance(x, dict) and "question" in x and "answer" in x for x in data):
        return "\n\n".join(f"## {x['question']}\n{x['answer']}" for x in data)
    return json.dumps(data, indent=2)


class IngestionService:
    def __init__(self, embedder, store):
        self.embedder = embedder
        self.store = store

    def ingest_text(
        self, session: Session, *, title: str, content: str, source: str,
        visibility: str = "public", content_type: str = "text",
    ) -> Document:
        if visibility not in VISIBILITIES:
            raise IngestionError(f"visibility must be one of {sorted(VISIBILITIES)}")
        content = content.strip()
        if not content:
            raise IngestionError("document is empty")
        checksum = hashlib.sha256(f"{visibility}:{content}".encode()).hexdigest()

        existing = session.scalar(select(Document).where(Document.checksum == checksum))
        if existing:
            return existing  # idempotent re-upload
        previous = session.scalars(select(Document).where(Document.source == source)).all()
        for doc in previous:  # new version of the same source replaces the old one
            self.delete_document(session, doc.id, commit=False)

        doc = Document(title=title, source=source, content_type=content_type, visibility=visibility, checksum=checksum)
        session.add(doc)
        session.flush()

        chunks = chunk_text(content)
        rows = [DocumentChunk(document_id=doc.id, chunk_index=c.index, section=c.section, text=c.text) for c in chunks]
        session.add_all(rows)
        session.flush()
        self._index(doc, rows)
        doc.chunk_count = len(rows)
        session.commit()
        DOCS_INGESTED.labels(content_type).inc()
        logger.info("ingested document", extra={"fields": {"doc_id": doc.id, "chunks": len(rows)}})
        return doc

    def ingest_file(self, session: Session, filename: str, data: bytes, visibility: str = "public",
                    title: str | None = None) -> Document:
        ext = PurePath(filename).suffix.lower()
        if ext not in SUPPORTED_EXTENSIONS:
            raise IngestionError(f"unsupported file type '{ext}'. Supported: {sorted(SUPPORTED_EXTENSIONS)}")
        try:
            raw = data.decode("utf-8-sig")
        except UnicodeDecodeError as exc:
            raise IngestionError("file must be UTF-8 text") from exc
        try:
            if ext == ".csv":
                content, ctype = _csv_to_text(raw), "csv"
            elif ext == ".json":
                content, ctype = _json_to_text(raw), "json"
            else:
                content, ctype = raw, "markdown" if ext in {".md", ".markdown"} else "text"
        except (json.JSONDecodeError, csv.Error) as exc:
            raise IngestionError(f"could not parse {ext} file: {exc}") from exc
        words = PurePath(filename).stem.replace("_", " ").replace("-", " ").split()
        small = {"and", "or", "of", "the", "a", "an", "to", "for", "in", "on"}
        default_title = " ".join(w if i and w.lower() in small else w.capitalize() for i, w in enumerate(words))
        return self.ingest_text(session, title=title or default_title, content=content, source=filename,
                                visibility=visibility, content_type=ctype)

    def _index(self, doc: Document, rows: list[DocumentChunk]) -> None:
        if not rows:
            return
        texts = [f"{doc.title}. {r.section}\n{r.text}" for r in rows]
        vectors = self.embedder.embed(texts)
        self.store.upsert([
            VectorRecord(
                id=r.id, vector=v,
                payload={"document_id": doc.id, "title": doc.title, "section": r.section, "text": r.text,
                         "visibility": doc.visibility, "source": doc.source},
            )
            for r, v in zip(rows, vectors, strict=True)
        ])

    def delete_document(self, session: Session, document_id: int, commit: bool = True) -> bool:
        doc = session.get(Document, document_id)
        if not doc:
            return False
        self.store.delete_document(document_id)
        session.delete(doc)
        if commit:
            session.commit()
        else:
            session.flush()
        return True

    def rebuild_index(self, session: Session) -> int:
        """Re-index all chunks from Postgres (used for the in-memory store, or to repair Qdrant)."""
        total = 0
        for doc in session.scalars(select(Document)).all():
            rows = session.scalars(select(DocumentChunk).where(DocumentChunk.document_id == doc.id)).all()
            self._index(doc, list(rows))
            total += len(rows)
        return total
