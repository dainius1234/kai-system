"""Live verification of memu-graph against a real Ollama + Cognee/Kuzu stack.

Not a unit test — requires memu-graph actually running (real LLM calls,
real graph-store writes). Run from CI after `docker compose -f
docker-compose.full.yml up -d ollama ollama-pull memu-graph` and a health
wait, per .github/workflows/core-tests.yml.

**Runs inside the container**, via `docker compose exec`. memu-graph is
on `agent-net`, which is declared `internal: true`, so
`http://localhost:8061` from the runner has reached nothing since
`e4655bc` removed the host-port bindings. Hence stdlib `urllib` rather
than `requests`: the service image carries no HTTP client beyond what its
own healthcheck uses, and adding a pip install to a container just to
probe it is a dependency this does not need.

Exercises the full ingest -> cognify -> query -> forget cycle described in
kai-pm/MEMORY_GRAPH_DESIGN.md Phase A. CI treats failures here as a warning,
not a build-breaker (see DECISIONS.md) — Cognee's Kuzu extension download
from extension.kuzudb.com is an external network dependency outside our
control, and graph-extraction quality from a 0.5B model is not something
we want gating merges.
"""
from __future__ import annotations

import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

BASE_URL = os.getenv("MEMU_GRAPH_URL", "http://localhost:8061")
SOURCE_ID = "graph-live-test-001"


class _Response:
    """Just enough of the `requests` shape for the calls below.

    Kept deliberately small: this replaces four call sites, not a
    library, and a fuller shim would be code nobody exercises.
    """

    def __init__(self, status: int, body: bytes, url: str):
        self.status_code = status
        self._body = body
        self.text = body.decode("utf-8", "replace")
        self._url = url

    def json(self):
        return json.loads(self._body)

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError(
                f"{self._url} returned HTTP {self.status_code}: {self.text[:300]}")


def _request(method: str, url: str, payload=None, timeout: int = 30) -> _Response:
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(url, data=data, method=method)
    if data is not None:
        req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return _Response(resp.status, resp.read(), url)
    except urllib.error.HTTPError as exc:
        return _Response(exc.code, exc.read(), url)
    except urllib.error.URLError as exc:
        # Deliberately raised, not turned into a status-0 response: a
        # synthetic 0 would sail past `raise_for_status()` (0 < 400) and
        # an unreachable service would read as a healthy one.
        raise RuntimeError(f"{url} is unreachable: {exc.reason}") from None


def _get(url: str, timeout: int = 30) -> _Response:
    return _request("GET", url, None, timeout)


def _post(url: str, payload: dict, timeout: int = 30) -> _Response:
    return _request("POST", url, payload, timeout)


def main() -> int:
    health = _get(f"{BASE_URL}/health", timeout=10)
    health.raise_for_status()
    print(f"health: {health.json()}")

    ingest = _post(
        f"{BASE_URL}/graph/ingest",
        {
            "text": "Kai is a sovereign AI system. Kai's memory subsystem uses "
            "Postgres with pgvector for vector search and Cognee for "
            "graph-structured entity relationships.",
            "source_id": SOURCE_ID,
            "category": "test",
        },
        timeout=300,  # D53: qwen2.5:3b needs up to 3-4 LLM calls at 30-45s each for cognify
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

    question = "What does Kai's memory subsystem use for vector search?"
    query = _get(
        f"{BASE_URL}/graph/query?q={urllib.parse.quote(question)}",
        timeout=120,  # D53: qwen2.5:3b graph search also needs headroom on CPU
    )
    print(f"query status={query.status_code} body={query.text[:1000]}")
    query.raise_for_status()
    query_body = query.json()
    if "results" not in query_body:
        print(f"FAIL: query response missing 'results': {query_body}")
        return 1
    print(f"query returned {len(query_body['results'])} result(s)")

    forget = _post(
        f"{BASE_URL}/graph/forget",
        {"source_id": SOURCE_ID},
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
    try:
        sys.exit(main())
    except Exception as exc:                       # a readable failure
        print(f"FAIL: {exc}")
        sys.exit(1)
