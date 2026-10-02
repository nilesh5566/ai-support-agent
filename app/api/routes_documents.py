from __future__ import annotations

from typing import Literal

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.api.deps import Principal, get_db, require_admin
from app.models import Document
from app.rag.ingest import IngestionError
from app.schemas import DocumentOut, SearchRequest, TextDocumentIn

router = APIRouter(prefix="/v1", tags=["knowledge base"])


def _out(doc: Document) -> DocumentOut:
    return DocumentOut(id=doc.id, title=doc.title, source=doc.source, content_type=doc.content_type,
                       visibility=doc.visibility, chunk_count=doc.chunk_count)


@router.post("/documents", response_model=DocumentOut, status_code=status.HTTP_201_CREATED)
async def upload_document(request: Request, file: UploadFile = File(...),
                          visibility: Literal["public", "internal"] = Form("public"),
                          title: str | None = Form(None),
                          _: Principal = Depends(require_admin), db: Session = Depends(get_db)) -> DocumentOut:
    limit = request.app.state.settings.max_upload_bytes
    data = await file.read(limit + 1)
    if len(data) > limit:
        raise HTTPException(status.HTTP_413_CONTENT_TOO_LARGE, f"File exceeds {limit} bytes")
    try:
        doc = request.app.state.ingestion.ingest_file(db, file.filename or "upload.txt", data, visibility, title)
    except IngestionError as exc:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, str(exc)) from None
    return _out(doc)


@router.post("/documents/text", response_model=DocumentOut, status_code=status.HTTP_201_CREATED)
def add_text_document(body: TextDocumentIn, request: Request, _: Principal = Depends(require_admin),
                      db: Session = Depends(get_db)) -> DocumentOut:
    try:
        doc = request.app.state.ingestion.ingest_text(
            db, title=body.title, content=body.content, source=body.source or f"text:{body.title}",
            visibility=body.visibility, content_type="text")
    except IngestionError as exc:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, str(exc)) from None
    return _out(doc)


@router.get("/documents", response_model=list[DocumentOut])
def list_documents(_: Principal = Depends(require_admin), db: Session = Depends(get_db)) -> list[DocumentOut]:
    return [_out(d) for d in db.scalars(select(Document).order_by(Document.id)).all()]


@router.delete("/documents/{document_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_document(document_id: int, request: Request, _: Principal = Depends(require_admin),
                    db: Session = Depends(get_db)) -> None:
    if not request.app.state.ingestion.delete_document(db, document_id):
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Document not found")


@router.post("/search")
def search(body: SearchRequest, request: Request, _: Principal = Depends(require_admin)) -> dict:
    visibility = ["public", "internal"] if body.include_internal else ["public"]
    return {"results": request.app.state.retriever.search(body.query, visibility, body.top_k)}
