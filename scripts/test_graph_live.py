"""Live verification of memu-graph against a real Ollama + Cognee/Kuzu stack.

Not a unit test — requires memu-graph actually running (real LLM calls,
real graph-store writes). Run from CI after `docker compose -f
docker-compose.full.yml up -d ollama ollama-pull memu-graph` and a health
wait, per .github/workflows/core-tests.yml.

Exercises the full ingest -> cognify -> query -> forget cycle described in
kai-pm/MEMORY_GRAPH_DESIGN.md Phase A. CI treats failures here as a warning,
not a build-breaker (see DECISIONS.md) — Cognee's Kuzu extension download
from extension.kuzudb.com is an external network dependency outside our
control, and graph-extraction quality from a 0.5B model is not something
we want gating merges.
"""
from __future__ import annotations

import os
import sys
import time

import requests

BASE_URL = os.getenv("MEMU_GRAPH_URL", "http://localhost:8061")
SOURCE_ID = "graph-live-test-001"


def main() -> int:
    health = requests.get(f"{BASE_URL}/health", timeout=10)
    health.raise_for_status()
    print(f"health: {health.json()}")

    ingest = requests.post(
        f"{BASE_URL}/graph/ingest",
        json={
            "text": "Kai is a sovereign AI system. Kai's memory subsystem uses "
                     "Postgres with pgvector for vector search and Cognee for "
                     "graph-structured entity relationships.",
            "source_id": SOURCE_ID,
            "category": "test",
        },
        timeout=120,
    )
    print(f"ingest status={ingest.status_code} body={ingest.text[:500]}")
    ingest.raise_for_status()
    ingest_body = ingest.json()
    if ingest_body.get("status") != "ingested":
        print(f"FAIL: unexpected ingest status: {ingest_body}")
        return 1

    # Cognify runs as part of /graph/ingest synchronously, but give the
    # graph store a moment before querying, same as any eventually-visible
    # index.
    time.sleep(2)

    query = requests.get(
        f"{BASE_URL}/graph/query",
        params={"q": "What does Kai's memory subsystem use for vector search?"},
        timeout=60,
    )
    print(f"query status={query.status_code} body={query.text[:1000]}")
    query.raise_for_status()
    query_body = query.json()
    if "results" not in query_body:
        print(f"FAIL: query response missing 'results': {query_body}")
        return 1
    print(f"query returned {len(query_body['results'])} result(s)")

    forget = requests.post(
        f"{BASE_URL}/graph/forget",
        json={"source_id": SOURCE_ID},
        timeout=60,
    )
    print(f"forget status={forget.status_code} body={forget.text[:500]}")
    forget.raise_for_status()
    forget_body = forget.json()
    if forget_body.get("status") not in ("forgotten", "not_found"):
        print(f"FAIL: unexpected forget status: {forget_body}")
        return 1

    print("PASS: ingest -> cognify -> query -> forget cycle completed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
