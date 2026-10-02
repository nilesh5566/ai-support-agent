#!/usr/bin/env python3
"""End-to-end smoke test against a running deployment (used by CI after `docker run`)."""
import argparse
import sys
import time

import httpx


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--client-key", default="dev-client-key")
    ap.add_argument("--wait", type=int, default=60, help="seconds to wait for readiness")
    args = ap.parse_args()

    deadline = time.time() + args.wait
    while True:
        try:
            if httpx.get(f"{args.url}/ready", timeout=3).status_code == 200:
                break
        except httpx.HTTPError:
            pass
        if time.time() > deadline:
            print("service never became ready")
            return 1
        time.sleep(2)

    checks = []
    r = httpx.post(f"{args.url}/v1/chat", headers={"X-API-Key": args.client_key}, timeout=60,
                   json={"message": "How do I reset my thermostat?"})
    checks.append(("knowledge answer", r.status_code == 200 and bool(r.json().get("sources"))))
    r = httpx.post(f"{args.url}/v1/chat", headers={"X-API-Key": args.client_key}, timeout=60,
                   json={"message": "What plan am I on?", "customer_email": "jane@example.com"})
    checks.append(("account tool", r.status_code == 200 and
                   any(t["name"] == "get_customer_profile" for t in r.json()["tool_calls"])))
    checks.append(("auth enforced", httpx.post(f"{args.url}/v1/chat", json={"message": "hi"}).status_code == 401))
    checks.append(("metrics exposed", "agent_chat_turns_total" in httpx.get(f"{args.url}/metrics").text))

    for name, ok in checks:
        print(f"{'PASS' if ok else 'FAIL'}  {name}")
    return 0 if all(ok for _, ok in checks) else 1


if __name__ == "__main__":
    sys.exit(main())
