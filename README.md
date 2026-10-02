# AI Customer Operations Agent

A configurable AI support agent that connects a company's documentation and operational systems, retrieves relevant context with RAG, executes **authorized** actions through tools, and runs as an observable, production-shaped service.

> Not "a chatbot": the agent answers from the company's own knowledge base with citations, looks up customer data through permission-checked tools, opens and escalates tickets, enforces escalation policy in code, and emits metrics, structured logs, traces and an audit trail for every action it takes.

**Example deployment (included as demo data):** *Acme Smart Home* uploads its product guide, billing policy, shipping/returns policy, FAQ, resolved tickets and internal staff policies. Customers chat on the website; support staff ask the same agent in Slack and get answers that include internal-only policies.

```
                Customer / Support staff
                         │
             ┌───────────┼────────────┐
             ▼           ▼            ▼
          Web UI      REST API     Slack Events
             └───────────┼────────────┘
                         ▼
          FastAPI  ── auth (API keys + roles) ── rate limit (Redis)
                         │
                         ▼
                 Agent orchestrator ──── policy guardrails (code, not prompt)
                         │
            ┌────────────┼──────────────┐
            ▼            ▼              ▼
       RAG retrieval   Tool registry    Memory
       (Qdrant)        (scoped, audited) (Redis window + Postgres history)
            │            │
            ▼            ▼
        Documents    CRM / Orders / Tickets (Postgres adapters → swap for Salesforce, Zendesk…)
                         │
                         ▼
                  Answer + actions
                         │
                         ▼
     Prometheus metrics · JSON logs (PII-redacted) · per-turn traces · audit log → Grafana
```

---

## Contents

1. [Quick start](#1-quick-start)
2. [What it does](#2-what-it-does)
3. [API reference](#3-api-reference)
4. [Configuring a deployment](#4-configuring-a-deployment-agentyaml)
5. [Connecting a real company's systems](#5-connecting-a-real-companys-systems)
6. [Security model](#6-security-model)
7. [Observability](#7-observability)
8. [Testing](#8-testing)
9. [CI/CD and AWS deployment](#9-cicd-and-aws-deployment)
10. [Project structure](#10-project-structure)
11. [Design decisions and trade-offs](#11-design-decisions-and-trade-offs)
12. [Known limitations and roadmap](#12-known-limitations-and-roadmap)

---

## 1. Quick start

### Option A: Python only (no Docker, no API keys)

Runs with SQLite, an in-process KV store, in-memory vectors and the offline mock LLM, so it works on any machine.

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt
uvicorn app.main:app --reload --port 8000
```

Open **http://localhost:8000** for the chat console, or **http://localhost:8000/docs** for interactive API docs. Demo customers, orders and documents are seeded on first start.

### Option B: Full stack with Docker Compose

API + PostgreSQL + Redis + Qdrant + Prometheus + Grafana:

```bash
docker compose up --build
```

| Service | URL |
|---|---|
| Chat console | http://localhost:8000 |
| API docs (Swagger) | http://localhost:8000/docs |
| Prometheus | http://localhost:9090 |
| Grafana (admin / admin) | http://localhost:3000 → "AI Support Agent" dashboard |

### Use a real LLM (free options included)

The agent works with any provider that speaks the OpenAI tool-calling protocol. Pick one with
`LLM_PROVIDER` and set that provider's key. Several have free tiers, and Ollama runs locally for free.

| `LLM_PROVIDER` | Cost | Default model | Key | Get a key |
|---|---|---|---|---|
| `groq` | free tier | `openai/gpt-oss-120b` | `GROQ_API_KEY` | https://console.groq.com/keys |
| `gemini` | free tier (not in EU/UK/CH) | `gemini-flash-latest` | `GEMINI_API_KEY` | https://aistudio.google.com/apikey |
| `openrouter` | free `:free` models | `openai/gpt-oss-120b:free` | `OPENROUTER_API_KEY` | https://openrouter.ai/keys |
| `cerebras` | free tier | `gpt-oss-120b` | `CEREBRAS_API_KEY` | https://cloud.cerebras.ai |
| `mistral` | free plan | `mistral-small-latest` | `MISTRAL_API_KEY` | https://console.mistral.ai/api-keys |
| `ollama` | free, local | `qwen3:8b` | none | https://ollama.com/download |
| `openai` | paid | `gpt-4o-mini` | `OPENAI_API_KEY` | https://platform.openai.com/api-keys |
| `custom` | any | set `LLM_MODEL` | `LLM_API_KEY` (optional) | vLLM, LM Studio, LiteLLM, Together, Azure… via `LLM_BASE_URL` |
| `mock` (default) | free, offline | rule-based | none | n/a |

Free-tier limits and model lists change often. Treat the defaults as starting points and override them with `LLM_MODEL`.
Use a model that supports tool calling, because the agent depends on it.

**Quickest setup (Groq):**

```bash
cp .env.example .env
# in .env:
LLM_PROVIDER=groq
GROQ_API_KEY=gsk_...
python scripts/check_llm.py      # verifies key, model answer and tool calling
uvicorn app.main:app --reload
```

The chat console header shows the active provider and model.

**Local and free with Ollama:** run `ollama pull qwen3:8b` (and `ollama pull nomic-embed-text` for embeddings), then set
`LLM_PROVIDER=ollama`. In Docker Compose the API container reaches Ollama on your host at `host.docker.internal`.

**Fallback chain for rate limits.** Free tiers have tight per-minute limits, so you can list backups:

```bash
LLM_PROVIDER=groq
LLM_FALLBACK_PROVIDERS=gemini,mock   # tried in order when the primary fails or returns 429/5xx
GROQ_API_KEY=gsk_...
GEMINI_API_KEY=...
```

Each provider is retried with backoff (the SDK honours `Retry-After`) before the chain moves on. A provider that failed
is skipped for `LLM_FALLBACK_COOLDOWN_SECONDS` (default 30), so the next requests don't wait on it again. Each trace
records which provider served every LLM call (`served_by`), and `agent_llm_requests_total{provider,status}` counts
successes and failures per provider. `mock` as the last link means the agent always answers.

**Embeddings (for document search).** Groq, OpenRouter and Cerebras have no embeddings API, so either keep the free
offline `hashing` embedder (default) or use a free semantic one:

```bash
EMBEDDING_PROVIDER=gemini     # gemini-embedding-001 (uses GEMINI_API_KEY)
EMBEDDING_PROVIDER=mistral    # mistral-embed (uses MISTRAL_API_KEY)
EMBEDDING_PROVIDER=ollama     # nomic-embed-text (local)
```

LLM and embedding providers are independent. For example, `LLM_PROVIDER=groq` with `EMBEDDING_PROVIDER=gemini` is a
good all-free combination. Qdrant collections are namespaced by embedder name and dimension, so vectors from different
embedders are never mixed. On restart, the index for a new embedder is rebuilt from Postgres. After switching to
semantic embeddings, re-tune `retrieval.min_score` in `agent.yaml` (see section 4).

Other settings: `LLM_MODEL` overrides the model, `LLM_BASE_URL` overrides the endpoint (e.g. a LiteLLM proxy), and
`LLM_API_KEY` is a generic key for the primary provider. Run `python scripts/check_llm.py --list` to print every
provider. Open models sometimes wrap their reasoning in `<think>…</think>`; it is stripped from replies. A malformed
tool call rejected by the provider (Groq's `tool_use_failed`) is retried once.

### Demo credentials

| API key | Role | Can do |
|---|---|---|
| `dev-client-key` | client | Chat as a customer (public docs, own account only) |
| `dev-admin-key` | admin | Upload docs, view tickets/conversations/traces, chat as staff |

Demo customers: `jane@example.com` (Pro), `marcus@example.com` (Starter, past due, has ticket `TCK-DEMO0001`), `priya@example.com` (Enterprise).

---

## 2. What it does

| Capability | How |
|---|---|
| **Answers from company knowledge** | Heading-aware chunking → embeddings → vector search with a relevance threshold. Answers cite the source document and section. If nothing relevant is found the agent says so instead of guessing. |
| **Multi-format ingestion** | Markdown, text, CSV (e.g. exported tickets) and JSON (FAQ lists). Re-uploading identical content is idempotent; uploading a new version of the same file replaces the old one in both Postgres and the vector store. |
| **Public vs internal knowledge** | Each document is `public` or `internal`. Customers can only retrieve public documents; staff (Slack / admin) also retrieve internal policies. Enforced as a vector-store filter, not a prompt instruction. |
| **Customer data via tools** | `get_customer_profile`, `list_orders`, `get_ticket`: a customer can only ever see their own data; staff can look up anyone. |
| **Actions** | `create_ticket` with priority and human escalation; idempotent per conversation (follow-ups update the existing ticket instead of creating duplicates); escalations notify a webhook (e.g. a Slack channel). |
| **Policy guardrails** | Configured keywords (legal threats, chargebacks, safety such as "smoke") always produce an urgent escalated ticket, **even if the model doesn't call the tool**. |
| **Conversation memory** | Sliding window in Redis for speed; full history in Postgres; Redis is rehydrated from Postgres on a cache miss. |
| **Channels** | Web UI, REST API, Slack Events API (signature-verified, thread = conversation). |
| **Resilience** | LLM failures degrade to a safe reply instead of a 500; tool exceptions are contained and reported to the model; max tool-iteration budget with a forced final answer; Redis outages fail open for rate limiting and memory. |
| **Observability** | Prometheus metrics, Grafana dashboard, alert rules, JSON logs with request IDs and PII redaction, per-turn traces with a waterfall in the UI, audit log of every tool call. |

### Try these in the chat console

| Message | Signed in as | What happens |
|---|---|---|
| How do I reset my thermostat? | anyone | RAG answer with citation |
| Where is my order? | Jane | `list_orders` + shipping policy |
| Show me the account for marcus@example.com | Jane | **Denied**: customers can't read other accounts (audited) |
| What's the status of TCK-DEMO0001? | Marcus / Jane | Found for Marcus; "not found" for Jane (existence isn't leaked) |
| I want to talk to a human, my sensor is broken | Jane | Escalated ticket, high priority |
| There is smoke coming out of my plug | anyone | Policy guardrail → urgent escalation |
| What are the refund approval limits for tier 1 agents? | toggle *Staff mode* | Staff gets the internal policy ($100); customers don't |

### About the mock LLM

`LLM_PROVIDER=mock` (the default) is a deterministic rule-based planner that speaks the **same tool-calling protocol** as the OpenAI provider: it plans tool calls from intent keywords, then composes an answer from the tool results. It exists so the whole platform (retrieval, authorization, escalation, metrics, tests, CI) runs with no API key and no cost. It is not meant to be smart; switch to `groq` (free) or another provider for real reasoning. Everything outside the LLM call is identical in both modes.

---

## 3. API reference

All `/v1` endpoints require `X-API-Key`. Full schemas are at `/docs`.

| Method | Path | Key | Purpose |
|---|---|---|---|
| POST | `/v1/chat` | any | Send a message; returns reply, sources, tool calls, ticket, trace id |
| POST | `/v1/documents` | admin | Upload a file (multipart: `file`, `visibility`, optional `title`) |
| POST | `/v1/documents/text` | admin | Add a document from JSON text |
| GET | `/v1/documents` | admin | List documents |
| DELETE | `/v1/documents/{id}` | admin | Remove a document and its vectors |
| POST | `/v1/search` | admin | Debug retrieval (raw chunks + scores) |
| GET | `/v1/tickets?status=escalated` | admin | Tickets created by the agent |
| GET | `/v1/conversations/{id}` | admin | Message history + tool audit trail |
| GET | `/v1/traces/{trace_id}` | admin | Span-level trace of one turn |
| GET | `/v1/tools` | any | Enabled tools with their scopes and JSON schemas |
| GET | `/v1/config` | admin | Active agent config and runtime backends |
| GET | `/v1/llm/providers` | admin | Supported LLM providers and the active provider / fallback chain |
| POST | `/v1/slack/events` | Slack signature | Slack Events API endpoint |
| GET | `/health`, `/ready` | none | Liveness (+ active LLM provider/model); readiness checks DB, KV and vector store |
| GET | `/metrics` | none | Prometheus metrics |

### Examples

```bash
# Customer question (the calling app asserts which customer is signed in)
curl -s localhost:8000/v1/chat -H 'X-API-Key: dev-client-key' -H 'Content-Type: application/json' \
  -d '{"message": "Where is my order?", "customer_email": "jane@example.com"}'
```

```json
{
  "reply": "Here are your most recent orders:\n- ORD-50388: shipped (59.00 USD)\n- ORD-50021: delivered (249.00 USD)\n\nOrders ship from our warehouse within 2 business days. ...",
  "conversation_id": "be85dcc5721c4b029f8aaa05aef1e765",
  "trace_id": "55c5afcefbc645acad9af3a2f1e4958b",
  "sources": [{"document_id": 6, "title": "Shipping and Returns", "section": "... > Shipping", "score": 0.2425}],
  "tool_calls": [{"name": "list_orders", "status": "ok"}, {"name": "search_knowledge_base", "status": "ok"}],
  "escalated": false,
  "ticket_number": null,
  "outcome": "answered",
  "latency_ms": 14.2
}
```

Continue the conversation by sending the returned `conversation_id`. Add `"debug": true` to include the full trace.

```bash
# Upload documentation
curl -s localhost:8000/v1/documents -H 'X-API-Key: dev-admin-key' \
  -F file=@./warranty.md -F visibility=public

# Bulk-upload a folder (files named internal_* become staff-only)
python scripts/ingest_docs.py ./customer_docs --url http://localhost:8000 --key dev-admin-key

# Ask as support staff (internal docs + any customer)
curl -s localhost:8000/v1/chat -H 'X-API-Key: dev-admin-key' -H 'Content-Type: application/json' \
  -d '{"message": "Show me the account for marcus@example.com", "as_staff": true}'
```

### Slack setup

1. Create a Slack app; under *Event Subscriptions* set the request URL to `https://<your-host>/v1/slack/events` and subscribe to `app_mention`.
2. Add the `chat:write` and `app_mentions:read` bot scopes and install the app.
3. Set `SLACK_SIGNING_SECRET` and `SLACK_BOT_TOKEN`.

Requests are verified with HMAC-SHA256 and a 5-minute replay window, acknowledged immediately, and processed in the background. Slack retries are ignored to avoid duplicate answers. Each thread maps to one conversation, and the Slack channel runs with the `staff` role by default.

---

## 4. Configuring a deployment (`agent.yaml`)

Everything that changes per customer lives in `config/agent.yaml`, so a new deployment is a config change, not a code change:

```yaml
agent:
  name: "Acme Support Assistant"
  company: "Acme Smart Home"
  max_tool_iterations: 4
  system_prompt: |
    You are {name}, the customer support agent for {company}. ...

retrieval:
  top_k: 4
  min_score: 0.12        # below this, a chunk counts as "not found"

tools:
  enabled: [search_knowledge_base, get_customer_profile, list_orders, create_ticket, get_ticket]

roles:                   # role -> scopes; tools declare the scopes they need
  customer: { scopes: [kb:public, customer:self, ticket:create, ticket:self] }
  staff:    { scopes: [kb:public, kb:internal, customer:any, ticket:create, ticket:any] }

channels:                # channel -> default role
  web:   { role: customer }
  api:   { role: customer }
  slack: { role: staff }

escalation:              # always escalate, enforced in code
  priority: urgent
  keywords: [lawyer, legal action, chargeback, data breach, smoke, carbon monoxide]
```

Runtime settings (database, Redis, vector backend, LLM provider, keys) are environment variables; see `.env.example`.

**Tuning `min_score`:** use `POST /v1/search` with real customer questions and look at the scores of right vs wrong chunks. Hashing embeddings score lower in absolute terms than semantic embeddings (Gemini, Mistral, OpenAI), so re-tune after switching providers. The Grafana *Knowledge gaps* panel shows how often searches find nothing, which tells you which docs the customer is missing.

---

## 5. Connecting a real company's systems

The local Postgres `customers`, `orders` and `tickets` tables stand in for a customer's real systems. Each tool handler in `app/tools/builtin.py` is a thin adapter, so connecting to the real thing means replacing the body of one function. The agent loop, authorization, auditing, metrics and tests stay the same.

```python
# Example: back create_ticket with Zendesk instead of Postgres
def create_ticket(ctx, subject, description, priority="normal", escalate=False):
    resp = zendesk.post("/api/v2/tickets", json={"ticket": {
        "subject": subject, "comment": {"body": description},
        "priority": priority, "requester": {"email": ctx.customer.email},
        "tags": ["ai-agent", "escalated"] if escalate else ["ai-agent"]}})
    resp.raise_for_status()
    t = resp.json()["ticket"]
    return {"ticket_number": str(t["id"]), "subject": subject, "status": t["status"],
            "priority": priority, "escalated": escalate}
```

Adding a new tool is one `Tool(...)` registration with a JSON schema and the scope it requires, plus adding its name to `tools.enabled`. Typical next tools: `issue_refund` (with an amount cap per role), `check_service_status`, `update_shipping_address`, `lookup_invoice`.

**A typical rollout:**

1. Ingest the docs the support team already uses; review `/v1/search` results for the top 50 real questions; tune `min_score`.
2. Start internal-only (Slack, staff role) so agents validate answers before customers see them.
3. Connect read-only tools first (CRM, orders), then mutating ones (tickets, refunds) behind scopes.
4. Turn on the web channel for customers; watch escalation rate, knowledge gaps and tool error rates in Grafana.

---

## 6. Security model

- **Authentication:** API keys mapped to roles (`client`, `admin`). Only a hash prefix of the key is used in logs and rate-limit buckets.
- **Identity:** the calling application (your website backend, after it has logged the user in) passes `customer_email`. The agent never trusts identity claims made in the chat text itself.
- **Least privilege:** the model is only *shown* the tools its role may use, and every call is re-checked server-side. A prompt injection that names a hidden tool gets "not available", and a customer asking for another account gets "you can only access the account you are signed in with".
- **Data isolation:** customers can only read their own profile, orders and tickets; a ticket owned by someone else returns "not found" rather than "forbidden" so existence isn't leaked. Conversations are bound to the customer who started them (403 otherwise).
- **Knowledge isolation:** internal documents are excluded by a vector-store filter for customer roles.
- **Audit:** every tool call (allowed, denied, invalid or failed) is written to `audit_logs` with role, arguments, status and trace id.
- **Logs:** emails, card numbers and phone numbers are redacted before they are written.
- **Abuse controls:** per-key rate limiting (Redis), message length limits, upload size and type limits, Slack signature verification with replay protection.
- **Container:** non-root user, health check, secrets via AWS Secrets Manager in production.

---

## 7. Observability

### Metrics (`/metrics`)

| Metric | Why it matters |
|---|---|
| `agent_chat_turns_total{channel,outcome}` | Volume, plus answered / escalated / llm_error / max_iterations |
| `agent_turn_duration_seconds` | End-to-end latency per channel |
| `agent_llm_requests_total`, `agent_llm_latency_seconds`, `agent_llm_tokens_total` | Provider health, latency, cost |
| `agent_tool_calls_total{tool,status}`, `agent_tool_latency_seconds` | Which integrations are used, failing or denied |
| `agent_rag_hits`, `agent_rag_top_score`, `agent_rag_retrieval_seconds` | Retrieval quality and knowledge gaps |
| `agent_escalations_total`, `agent_tickets_created_total{priority}` | Human workload created by the agent |
| `http_requests_total`, `http_request_duration_seconds` | Standard RED metrics per route |

The Grafana dashboard (`deploy/grafana/dashboards/support-agent.json`) is provisioned automatically. Alert rules in `deploy/prometheus/alerts.yml` cover high error rate, LLM failures, slow turns, knowledge gaps and escalation spikes.

### Traces

Every turn produces a trace with nested spans (`agent.turn` → `llm.call` → `tool.*` → `rag.search`) including durations, token counts and statuses. Traces are logged, returned with `debug: true`, fetchable at `/v1/traces/{id}`, and drawn as a waterfall in the chat console. The span model maps directly onto OpenTelemetry.

### Logs

One JSON object per line with `request_id`, `conversation_id`, `trace_id`, outcome, tools used and latency, ready for CloudWatch Logs Insights, Loki or Datadog:

```json
{"level": "INFO", "msg": "agent turn complete", "request_id": "28df94c0...", "conversation_id": "a91b93e7...",
 "trace_id": "ab3e3aec...", "channel": "api", "role": "customer", "outcome": "answered",
 "tools": ["search_knowledge_base"], "latency_ms": 13.7}
```

---

## 8. Testing

```bash
pytest -v          # 61 tests, ~8 seconds, no network or API keys needed
ruff check .
```

The suite covers: RAG answers and citations; account tools; cross-customer access denial; guest access; staff access; internal-doc isolation; escalation and ticket idempotency; policy guardrail enforcement **with a model that refuses to call tools**; graceful LLM failure; max-iteration handling; ticket ownership; conversation memory and hijack protection; traces; document upload/versioning/deletion/validation/size limits; Slack signature verification and the Slack → agent flow; rate limiting; both vector stores (in-memory and Qdrant); chunking; embeddings; Redis list semantics; PII redaction; OpenAI tool-call parsing; every provider preset (endpoint, default model, missing-key errors with sign-up hints); the fallback chain and cooldown; a full chat where the real OpenAI SDK sends HTTP to a fake Groq endpoint; Groq → Gemini fallback on HTTP 429; graceful failure when every provider is down; semantic embedders with dimension probing; retry of malformed tool calls.

Run the same suite against real PostgreSQL and Redis:

```bash
TEST_DATABASE_URL=postgresql+psycopg://support:support@localhost:5432/support \
TEST_REDIS_URL=redis://localhost:6379/0 pytest
```

Smoke-test any running deployment:

```bash
python scripts/smoke_test.py --url http://localhost:8000
```

---

## 9. CI/CD and AWS deployment

`.github/workflows/ci.yml`:

1. **lint**: `ruff check`.
2. **test**: the full suite twice (matrix): in-process backends, and real PostgreSQL 16 + Redis 7 service containers.
3. **docker**: builds the image, runs it, executes `scripts/smoke_test.py` against the container, validates `docker-compose.yml`.
4. **deploy** (on `main`, opt-in): GitHub OIDC → AWS (no stored keys), build and push to ECR, render `deploy/aws/task-definition.json` with the new image, deploy to ECS Fargate and wait for stability.

To enable deploys, set repository variable `AWS_DEPLOY_ENABLED=true` plus variables `AWS_REGION`, `ECR_REPOSITORY`, `ECS_CLUSTER`, `ECS_SERVICE`, and secret `AWS_DEPLOY_ROLE_ARN`.

### Reference AWS architecture

| Component | AWS service |
|---|---|
| API containers | ECS Fargate behind an Application Load Balancer (HTTPS via ACM) |
| Relational data | RDS for PostgreSQL (Multi-AZ) |
| Memory / rate limiting | ElastiCache for Redis |
| Vector database | Qdrant on ECS with EFS storage, or Qdrant Cloud |
| Secrets | Secrets Manager (injected via the task definition) |
| Images | ECR |
| Logs | CloudWatch Logs (`awslogs` driver) |
| Metrics / dashboards | Amazon Managed Prometheus + Amazon Managed Grafana (same dashboard JSON) |
| Network | Private subnets for tasks and data stores; only the ALB is public |

Replace the `<ACCOUNT_ID>` / `<REGION>` placeholders in the task definition. The infrastructure itself (VPC, RDS, ElastiCache, ECS service, IAM roles) is expected to come from your IaC tool of choice; it isn't included here.

---

## 10. Project structure

```
app/
  main.py               app factory: wiring, middleware, health/ready/metrics
  config.py             environment settings
  agent_config.py       per-deployment config (agent.yaml) models
  agent/
    orchestrator.py     agent loop: context → LLM → tools → answer, guardrails, telemetry
    policy.py           deterministic escalation policy
  llm/                  provider presets (Groq, Gemini, OpenRouter, Cerebras, Mistral, Ollama, OpenAI, custom),
                        OpenAI-compatible client, fallback chain, offline mock
  rag/                  chunker, embeddings, vector stores (memory/Qdrant), ingestion, retriever
  tools/
    registry.py         tool definitions, scope checks, validation, audit, metrics
    builtin.py          knowledge, CRM, orders and ticketing tools (swappable adapters)
  api/                  chat, documents, ops, Slack routes; auth and rate-limit deps
  observability/        JSON logging, PII redaction, Prometheus metrics, tracing
  memory.py, kv.py, ratelimit.py, notifier.py, seed.py, models.py, schemas.py, db.py
  static/index.html     web chat console with activity, sources and trace panels
config/agent.yaml       the file you edit per customer
sample_data/docs/       demo knowledge base (md, json FAQ, csv tickets, internal policy)
scripts/                ingest_docs.py (bulk upload), smoke_test.py (deployment check), check_llm.py (provider check)
deploy/                 Prometheus config + alerts, Grafana provisioning + dashboard, AWS task definition
tests/                  61 tests
Dockerfile, docker-compose.yml, .github/workflows/ci.yml, Makefile, .env.example
```

---

## 11. Design decisions and trade-offs

- **Guardrails in code, not prompts.** Authorization, internal-doc filtering, conversation ownership and mandatory escalation are enforced outside the model. The prompt describes the policy; the code guarantees it. A test swaps in a model that never calls tools and the escalation still happens.
- **Postgres is the source of truth for documents.** The vector store is a derived index that can be rebuilt (`IngestionService.rebuild_index`), which happens automatically on startup when the index is empty or behind.
- **One client, many providers.** Every hosted model is reached through the OpenAI-compatible protocol, so adding a provider is a preset (URL, key name, default model), not new code. Teams can start free (Groq/Gemini/Ollama) and move to a paid or self-hosted model by changing environment variables.
- **Pluggable everything behind small interfaces.** LLM provider, embedder, vector store and KV store each have a dev implementation and a production one, which keeps tests fast and deterministic and lets the system run in air-gapped pilots.
- **Synchronous SQLAlchemy in FastAPI's threadpool.** Simpler to reason about and test than async ORM code; the hot path is dominated by LLM latency, not database I/O.
- **Idempotent side effects.** Ticket creation is deduplicated per conversation and document uploads are deduplicated by checksum, because LLMs retry and users repeat themselves.

---

## 12. Known limitations and roadmap

Honest notes on what this is and isn't:

- **Mock LLM and hashing embeddings are for development.** They make the system runnable and testable offline, but answer quality comes from a real model (e.g. `LLM_PROVIDER=groq`, free) and semantic embeddings (e.g. `EMBEDDING_PROVIDER=gemini`, free).
- **Schema management uses `create_all`.** Add Alembic migrations before evolving the schema in production.
- **Traces are kept in memory per replica** (and in logs). Export to OpenTelemetry/X-Ray for a cross-replica view.
- **Single tenant per deployment.** One company per stack is the FDE-style model here; multi-tenancy would add a `tenant_id` to every table, collection and cache key.
- **Hosted providers were tested against fake OpenAI-compatible endpoints, not the live services.** The build environment has no internet access to Groq, Gemini, etc. The real OpenAI SDK, HTTP requests, tool calling, fallback and error handling were all exercised, but run `python scripts/check_llm.py` with your own key to confirm a live provider and model.
- **The Docker image and AWS deploy job were not executed in the build environment** (no Docker daemon there). The image's contents and start command were verified by running them in a clean virtualenv, and the compose/workflow/task-definition files are syntax-checked.

Roadmap: streaming responses (SSE), hybrid search (BM25 + vectors) with a re-ranker, PDF/HTML/Confluence/Notion connectors, an offline evaluation set with answer-faithfulness scoring in CI, human-in-the-loop approval for refunds above a threshold, OpenTelemetry export, per-tenant config.

---

### One-line summary for a resume

Built a configurable AI customer-operations agent (FastAPI, PostgreSQL, Redis, Qdrant, LLM function calling across Groq/Gemini/OpenAI/Ollama with automatic provider fallback) that answers from enterprise documentation via RAG with citations, executes role-scoped and audited actions against CRM and ticketing APIs, enforces escalation policy in code, and ships with Prometheus/Grafana observability, a 61-test suite, Docker Compose and a GitHub Actions → AWS ECS pipeline.
