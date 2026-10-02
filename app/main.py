"""Application factory: wires storage, RAG, LLM, tools, channels and observability together."""
from __future__ import annotations

import logging
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from sqlalchemy import func, select

from app.agent.orchestrator import SupportAgent
from app.agent_config import load_agent_config
from app.api import routes_chat, routes_documents, routes_ops, routes_slack
from app.config import Settings, get_settings
from app.db import Database
from app.kv import create_kv
from app.llm.factory import create_llm
from app.memory import ConversationMemory
from app.models import DocumentChunk
from app.notifier import EscalationNotifier
from app.observability.logging import configure_logging, request_id_ctx
from app.observability.metrics import HTTP_LATENCY, HTTP_REQUESTS
from app.observability.tracing import TraceStore
from app.rag.embeddings import create_embedder
from app.rag.ingest import IngestionService
from app.rag.retriever import Retriever
from app.rag.vectorstore import create_vector_store
from app.ratelimit import RateLimiter
from app.seed import seed_demo_data
from app.tools.builtin import build_registry

logger = logging.getLogger("app")
STATIC_DIR = Path(__file__).resolve().parent / "static"


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        configure_logging(settings.log_level)
        st = app.state
        st.settings = settings
        st.api_keys = settings.parsed_api_keys()
        st.db = Database(settings.database_url)
        st.db.create_all()
        st.kv = create_kv(settings.redis_url)
        st.rate_limiter = RateLimiter(st.kv, settings.rate_limit_per_minute)
        st.trace_store = TraceStore()

        agent_cfg = load_agent_config(settings.agent_config_path)
        embedder = create_embedder(settings)
        store = create_vector_store(settings, embedder)
        st.ingestion = IngestionService(embedder, store)
        st.retriever = Retriever(embedder, store, agent_cfg.retrieval.top_k, agent_cfg.retrieval.min_score)
        st.registry = build_registry(agent_cfg.tools.enabled)
        st.vector_store = store
        st.agent = SupportAgent(
            llm=create_llm(settings), registry=st.registry, retriever=st.retriever,
            memory=ConversationMemory(st.kv, settings.memory_ttl_seconds, settings.memory_max_messages),
            config=agent_cfg, trace_store=st.trace_store,
            notifier=EscalationNotifier(settings.escalation_webhook_url))

        with st.db.SessionLocal() as session:
            seeded = settings.seed_demo_data and seed_demo_data(session, st.ingestion)
            chunks = session.scalar(select(func.count()).select_from(DocumentChunk)) or 0
            if not seeded and chunks and store.count() < chunks:  # cold in-memory store / repaired Qdrant
                indexed = st.ingestion.rebuild_index(session)
                logger.info("vector index rebuilt", extra={"fields": {"chunks": indexed}})
        llm = st.agent.llm
        logger.info("startup complete", extra={"fields": {
            "llm": settings.llm_provider, "model": llm.model,
            "llm_chain": getattr(llm, "chain", None), "vector_backend": settings.vector_backend,
            "embedding": settings.embedding_provider, "tools": [t.name for t in st.registry.list()]}})
        yield
        st.db.engine.dispose()

    app = FastAPI(title=settings.app_name, version="1.0.0", lifespan=lifespan,
                  description="Configurable AI customer-operations agent: RAG + authorized tools + observability.")

    @app.middleware("http")
    async def observability_middleware(request: Request, call_next):
        rid = request.headers.get("X-Request-ID") or uuid.uuid4().hex
        token = request_id_ctx.set(rid)
        start = time.perf_counter()
        status_code = 500
        try:
            response = await call_next(request)
            status_code = response.status_code
        except Exception:
            logger.exception("unhandled error")
            response = JSONResponse({"detail": "Internal server error", "request_id": rid}, status_code=500)
        finally:
            route = request.scope.get("route")
            path = getattr(route, "path", "unmatched")
            HTTP_REQUESTS.labels(request.method, path, str(status_code)).inc()
            HTTP_LATENCY.labels(request.method, path).observe(time.perf_counter() - start)
            request_id_ctx.reset(token)
        response.headers["X-Request-ID"] = rid
        remaining = getattr(request.state, "rate_limit_remaining", None)
        if remaining is not None:
            response.headers["X-RateLimit-Remaining"] = str(remaining)
        return response

    app.include_router(routes_chat.router)
    app.include_router(routes_documents.router)
    app.include_router(routes_ops.router)
    app.include_router(routes_slack.router)

    @app.get("/health", tags=["ops"])
    def health(request: Request) -> dict:
        llm = request.app.state.agent.llm
        return {"status": "ok", "llm": {"provider": request.app.state.settings.llm_provider, "model": llm.model,
                                         "chain": getattr(llm, "chain", None)}}

    @app.get("/ready", tags=["ops"])
    def ready(request: Request) -> JSONResponse:
        checks: dict[str, str] = {}
        for name, fn in (("database", request.app.state.db.ping), ("kv", request.app.state.kv.ping),
                         ("vector_store", request.app.state.vector_store.ping)):
            try:
                fn()
                checks[name] = "ok"
            except Exception as exc:
                checks[name] = f"error: {type(exc).__name__}"
        ok = all(v == "ok" for v in checks.values())
        return JSONResponse({"status": "ready" if ok else "degraded", "checks": checks},
                            status_code=200 if ok else 503)

    @app.get("/metrics", include_in_schema=False)
    def metrics() -> Response:
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    return app


app = create_app()
