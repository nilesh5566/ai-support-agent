#!/usr/bin/env python3
"""Bulk-upload a folder of documents into a running agent.

  python scripts/ingest_docs.py ./customer_docs --url http://localhost:8000 --key dev-admin-key
  Files named internal_* are uploaded with internal visibility (staff only).
"""
import argparse
import sys
from pathlib import Path

import httpx

SUPPORTED = {".md", ".markdown", ".txt", ".csv", ".json"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("folder", type=Path)
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--key", default="dev-admin-key")
    ap.add_argument("--internal", action="store_true", help="mark every file as internal")
    args = ap.parse_args()

    files = sorted(p for p in args.folder.rglob("*") if p.suffix.lower() in SUPPORTED)
    if not files:
        print(f"No supported files in {args.folder}")
        return 1
    failures = 0
    with httpx.Client(base_url=args.url, headers={"X-API-Key": args.key}, timeout=120) as client:
        for path in files:
            visibility = "internal" if args.internal or path.name.startswith("internal_") else "public"
            r = client.post("/v1/documents", files={"file": (path.name, path.read_bytes())},
                            data={"visibility": visibility})
            if r.status_code == 201:
                d = r.json()
                print(f"ok    {path.name:40s} -> doc {d['id']} ({d['chunk_count']} chunks, {visibility})")
            else:
                failures += 1
                print(f"FAIL  {path.name:40s} {r.status_code} {r.text[:200]}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
