"""Prometheus metrics. Scraped from GET /metrics and visualised in Grafana."""
from prometheus_client import Counter, Histogram

HTTP_REQUESTS = Counter("http_requests_total", "HTTP requests", ["method", "path", "status"])
HTTP_LATENCY = Histogram("http_request_duration_seconds", "HTTP request latency", ["method", "path"])

CHAT_TURNS = Counter("agent_chat_turns_total", "Agent conversation turns", ["channel", "outcome"])
CHAT_LATENCY = Histogram(
    "agent_turn_duration_seconds", "End-to-end agent turn latency", ["channel"],
    buckets=(0.05, 0.1, 0.25, 0.5, 1, 2, 4, 8, 15, 30, 60),
)

LLM_REQUESTS = Counter("agent_llm_requests_total", "LLM calls", ["provider", "status"])
LLM_LATENCY = Histogram(
    "agent_llm_latency_seconds", "LLM call latency", ["provider"],
    buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1, 2, 4, 8, 15, 30),
)
LLM_TOKENS = Counter("agent_llm_tokens_total", "LLM tokens", ["provider", "kind"])

TOOL_CALLS = Counter("agent_tool_calls_total", "Tool executions", ["tool", "status"])
TOOL_LATENCY = Histogram("agent_tool_latency_seconds", "Tool execution latency", ["tool"])

RAG_LATENCY = Histogram("agent_rag_retrieval_seconds", "Retrieval latency")
RAG_HITS = Histogram("agent_rag_hits", "Relevant chunks per query", buckets=(0, 1, 2, 3, 4, 6, 8, 12))
RAG_TOP_SCORE = Histogram(
    "agent_rag_top_score", "Best similarity score per query",
    buckets=(0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0),
)

ESCALATIONS = Counter("agent_escalations_total", "Human escalations", ["reason"])
TICKETS_CREATED = Counter("agent_tickets_created_total", "Tickets created by the agent", ["priority"])
DOCS_INGESTED = Counter("agent_documents_ingested_total", "Documents ingested", ["content_type"])
RATE_LIMITED = Counter("agent_rate_limited_total", "Requests rejected by the rate limiter")
