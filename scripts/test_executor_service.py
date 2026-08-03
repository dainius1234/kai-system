"""Executor service tests — including that it fails closed.

This suite used to assert `POST /execute` returns 200 with no
authentication configured. G-03 made side-effecting endpoints fail closed
without `KAI_SERVICE_TOKEN`, so the assertion has been failing ever
since — and I repeatedly filed it under "pre-existing, needs a running
stack". It needs no stack at all: it runs against `TestClient`
in-process. The failure was a test asserting the behaviour from **before**
the hardening, exactly as `test_memu_routes` once asserted that a
persistence failure returns 200.

So it now asserts both halves, which is the stronger claim:

  - with no token configured, `/execute` refuses with 503
  - with a token configured and presented, it executes

Testing only the second would let the fail-closed behaviour regress
silently; testing only the first would let the endpoint stop working.
"""
from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

passed = 0
failed = 0


def check(name: str, condition: bool, detail: str = "") -> None:
    global passed, failed
    if condition:
        passed += 1
    else:
        failed += 1
        print(f"  FAIL: {name}" + (f" — {detail}" if detail else ""))


def _client() -> TestClient:
    """Load executor/app.py fresh so it re-reads the environment."""
    for name in ("executor_app",):
        sys.modules.pop(name, None)
    spec = importlib.util.spec_from_file_location(
        "executor_app", ROOT / "executor" / "app.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return TestClient(module.app)


def test_health_and_alive_need_no_credentials():
    client = _client()
    for path in ("/health", "/alive"):
        resp = client.get(path)
        check(f"{path} answers 200", resp.status_code == 200,
              str(resp.status_code))
        check(f"{path} reports ok", resp.json().get("status") == "ok",
              resp.text[:80])


def test_execute_fails_closed_without_a_token():
    """The behaviour G-03 introduced, now asserted rather than tripped over."""
    for var in ("KAI_SERVICE_TOKEN", "KAI_ALLOW_UNAUTHENTICATED"):
        os.environ.pop(var, None)
    client = _client()
    resp = client.post("/execute", json={"tool": "noop", "params": {},
                                         "task_id": "t1", "device": "cpu"})
    check("/execute refuses with 503 when no token is configured",
          resp.status_code == 503, str(resp.status_code))
    check("the refusal says it fails closed by design",
          "fails closed" in resp.text, resp.text[:120])


def test_execute_succeeds_with_a_token():
    os.environ["KAI_SERVICE_TOKEN"] = "test-token-for-executor-suite"
    try:
        client = _client()
        resp = client.post(
            "/execute",
            json={"tool": "noop", "params": {}, "task_id": "t1",
                  "device": "cpu"},
            headers={"Authorization": "Bearer test-token-for-executor-suite"})
        check("/execute answers 200 for an authenticated caller",
              resp.status_code == 200, f"{resp.status_code}: {resp.text[:120]}")
        if resp.status_code == 200:
            check("the task completes",
                  resp.json().get("status") == "completed", resp.text[:120])
    finally:
        os.environ.pop("KAI_SERVICE_TOKEN", None)


def test_execute_refuses_a_wrong_token():
    os.environ["KAI_SERVICE_TOKEN"] = "test-token-for-executor-suite"
    try:
        client = _client()
        resp = client.post(
            "/execute",
            json={"tool": "noop", "params": {}, "task_id": "t1",
                  "device": "cpu"},
            headers={"Authorization": "Bearer not-the-right-token"})
        check("a wrong token is rejected", resp.status_code in (401, 403),
              str(resp.status_code))
    finally:
        os.environ.pop("KAI_SERVICE_TOKEN", None)


def run() -> None:
    test_health_and_alive_need_no_credentials()
    test_execute_fails_closed_without_a_token()
    test_execute_succeeds_with_a_token()
    test_execute_refuses_a_wrong_token()


if __name__ == "__main__":
    run()
    print(f"\n{'='*60}")
    print(f"Executor Service Tests: {passed} passed, {failed} failed")
    if failed:
        print("EXIT GATE: FAIL")
        sys.exit(1)
    print("EXIT GATE: PASS")
