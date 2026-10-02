"""Verify your LLM (and embedding) provider settings before starting the server.

    python scripts/check_llm.py                      # uses .env / environment
    LLM_PROVIDER=groq GROQ_API_KEY=gsk_... python scripts/check_llm.py
    python scripts/check_llm.py --list               # show supported providers

Checks: the key/endpoint works, the model answers, and the model can call a tool
(the agent depends on tool calling), plus one embedding call if a semantic embedder is set.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import Settings  # noqa: E402
from app.llm.base import LLMError  # noqa: E402
from app.llm.factory import create_llm  # noqa: E402
from app.llm.providers import catalogue  # noqa: E402
from app.rag.embeddings import create_embedder  # noqa: E402

TOOL = {"type": "function", "function": {
    "name": "get_order_status", "description": "Look up the status of an order by its number.",
    "parameters": {"type": "object", "properties": {"order_number": {"type": "string"}},
                   "required": ["order_number"]}}}


class _SkipToolCheck(Exception):
    pass


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true", help="list supported providers and exit")
    args = ap.parse_args()
    if args.list:
        for p in catalogue():
            free = "free tier" if p["free_tier"] else ("any      " if p["name"] == "custom" else "paid     ")
            key = p["key_env"] or ("LLM_API_KEY (opt.)" if p["name"] == "custom" else "none")
            print(f"  {p['name']:<11} {free}  default model: {p['default_model'] or '(set LLM_MODEL)':<28} "
                  f"key: {key:<19} {p['signup_url']}")
        return 0

    s = Settings()
    print(f"LLM provider: {s.llm_provider}   fallbacks: {s.llm_fallback_providers or '-'}")
    try:
        llm = create_llm(s)
    except ValueError as exc:
        print(f"  CONFIG ERROR: {exc}")
        return 2
    print(f"  model: {llm.model}" + (f"   chain: {llm.chain}" if hasattr(llm, "chain") else ""))
    ok = True
    try:
        t = time.perf_counter()
        r = llm.chat([{"role": "user", "content": "Reply with the single word: pong"}], [], temperature=0)
        print(f"  [PASS] plain answer in {time.perf_counter() - t:.2f}s via {r.provider or llm.name}: "
              f"{(r.content or '').strip()[:60]!r}")
        if s.llm_provider == "mock" and not s.llm_fallback_providers:
            print("  [INFO] mock is the offline rule-based planner; set LLM_PROVIDER to use a real model")
            raise _SkipToolCheck
        t = time.perf_counter()
        r = llm.chat([{"role": "system", "content": "Use tools when they can answer the question."},
                      {"role": "user", "content": "What's the status of order ORD-1042?"}], [TOOL], temperature=0)
        if r.tool_calls:
            print(f"  [PASS] tool calling in {time.perf_counter() - t:.2f}s: "
                  f"{r.tool_calls[0].name}({r.tool_calls[0].arguments})")
        else:
            ok = False
            print("  [WARN] model answered without calling the tool - choose a model with tool-use support")
    except _SkipToolCheck:
        pass
    except LLMError as exc:
        print(f"  [FAIL] {exc}  (cause: {exc.__cause__!r})")
        return 1

    print(f"Embedding provider: {s.embedding_provider}")
    try:
        emb = create_embedder(s)
        v = emb.embed(["How do I reset my thermostat?"])[0]
        print(f"  [PASS] {emb.name}: {len(v)}-dim vectors")
    except Exception as exc:  # noqa: BLE001 - diagnostic script
        print(f"  [FAIL] {type(exc).__name__}: {exc}")
        return 1
    print("All good." if ok else "Works, with warnings.")
    return 0 if ok else 3


if __name__ == "__main__":
    raise SystemExit(main())
