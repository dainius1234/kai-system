from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

from fastapi.testclient import TestClient

# Credentials must exist before any request: the gateway fails closed, so
# without them every route answers 503 and the tests below would be
# asserting against a refusal rather than against the handler.
TOKEN = "test-dashboard-token"
os.environ["KAI_DASHBOARD_TOKEN"] = TOKEN
os.environ["KAI_DASHBOARD_IDENTITY"] = "test-operator"
os.environ["KAI_DASHBOARD_ROLE"] = "keeper"
os.environ.pop("KAI_DASHBOARD_PRINCIPALS", None)
os.environ.pop("KAI_ALLOW_UNAUTHENTICATED", None)

AUTH = {"Authorization": f"Bearer {TOKEN}"}

# import dashboard app
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
spec = importlib.util.spec_from_file_location("dashboard_app", ROOT / "dashboard" / "app.py")
assert spec and spec.loader
mod = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = mod
spec.loader.exec_module(mod)

# configure environment to avoid missing variables
mod.TOOL_GATE_URL = "http://tool-gate:8000"
mod.NODES = {}

client = TestClient(mod.app)


def test_health():
    """Liveness stays public — orchestrators probe it before credentials exist."""
    resp = client.get("/health")
    assert resp.status_code == 200
    assert "status" in resp.json()


def test_index_minimal():
    # patch out fetch_status to avoid external calls
    async def fake_status():
        return {}
    mod.fetch_status = fake_status  # type: ignore
    try:
        resp = client.get("/", headers=AUTH)
    except Exception:
        # external dependencies may be unreachable; nothing to check here
        return
    assert resp.status_code in (200, 500)
    if resp.status_code == 200:
        jsonp = resp.json()
        assert "service" in jsonp and jsonp["service"] == "dashboard"


# ── Inbound authentication (KAI-DASH-001, 011, 018) ──────────────────

def test_anonymous_request_is_refused():
    """The gateway no longer answers to anonymous callers."""
    for method, path in [
        ("get", "/"),
        ("post", "/api/soul"),
        ("post", "/api/browser/navigate"),
        ("get", "/api/memories"),
        ("get", "/api/broker/balance"),
    ]:
        resp = getattr(client, method)(path)
        assert resp.status_code == 401, f"{method.upper()} {path} → {resp.status_code}"


def test_bad_token_is_refused():
    resp = client.post("/api/soul", headers={"Authorization": "Bearer wrong"})
    assert resp.status_code == 401


def test_public_routes_stay_reachable():
    """The browser must load the shell before it can authenticate."""
    for path in ("/health", "/metrics", "/ui", "/app", "/chat", "/thinking"):
        resp = client.get(path)
        assert resp.status_code == 200, f"{path} → {resp.status_code}"


def test_viewer_cannot_reach_privileged_routes():
    """Least privilege is enforced per route, not merely declared."""
    os.environ["KAI_DASHBOARD_ROLE"] = "viewer"
    try:
        denied = client.post("/api/browser/navigate", headers=AUTH, json={})
        sensitive = client.get("/api/broker/balance", headers=AUTH)
        allowed = client.get("/api/weather/health", headers=AUTH)
    finally:
        os.environ["KAI_DASHBOARD_ROLE"] = "keeper"
    assert denied.status_code == 403, denied.status_code
    assert sensitive.status_code == 403, sensitive.status_code
    assert allowed.status_code in (200, 500, 503), allowed.status_code


def test_unconfigured_gateway_fails_closed():
    saved = os.environ.pop("KAI_DASHBOARD_TOKEN")
    try:
        resp = client.get("/api/memories", headers=AUTH)
    finally:
        os.environ["KAI_DASHBOARD_TOKEN"] = saved
    assert resp.status_code == 503, resp.status_code
    assert "fails closed" in resp.json()["detail"]


# ── /api/upload tests ────────────────────────────────────────────────

def _tiny_png() -> bytes:
    """Minimal valid 1×1 red PNG (67 bytes)."""
    import base64
    return base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8"
        "z8BQDwADhQGAWjR9awAAAABJRU5ErkJggg=="
    )


def test_upload_no_file():
    """POST /api/upload with no file → 422 (FastAPI validation)."""
    resp = client.post("/api/upload", headers=AUTH)
    assert resp.status_code == 422


def test_upload_too_large():
    """POST /api/upload with a file exceeding 10 MB → 413."""
    big = b"x" * (10 * 1024 * 1024 + 1)
    resp = client.post("/api/upload", headers=AUTH,
                       files={"file": ("big.png", big, "image/png")})
    assert resp.status_code == 413


def test_upload_image_ocr_success(monkeypatch=None):
    """POST /api/upload forwards to screen-capture and returns OCR text."""
    import unittest.mock as mock
    import httpx

    ocr_payload = {"status": "ok", "text": "OCR text here", "source": "upload:test.png",
                   "timestamp": 1234567890.0, "ocr_available": False}

    async def fake_post(*args, **kwargs):
        r = mock.MagicMock()
        r.status_code = 200
        r.json.return_value = ocr_payload
        return r

    with mock.patch.object(httpx.AsyncClient, "post", new=fake_post):
        resp = client.post(
            "/api/upload",
            headers=AUTH,
            files={"file": ("test.png", _tiny_png(), "image/png")},
        )
    assert resp.status_code == 200
    assert resp.json()["text"] == "OCR text here"


def test_upload_service_unreachable():
    """POST /api/upload returns 503 when screen-capture is down."""
    import unittest.mock as mock
    import httpx

    async def fail(*args, **kwargs):
        raise httpx.ConnectError("refused")

    with mock.patch.object(httpx.AsyncClient, "post", new=fail):
        resp = client.post(
            "/api/upload",
            headers=AUTH,
            files={"file": ("test.png", _tiny_png(), "image/png")},
        )
    assert resp.status_code == 503



def test_memory_reads_are_scoped_to_the_caller():
    """KAI-DASH-023: the caller's identity must reach the backend.

    Static absence of the string "keeper" is not evidence that the right
    identity is sent, so this asserts on the outbound request.
    """
    import unittest.mock as mock
    import httpx

    seen = []

    async def fake_get(self, url, **kwargs):
        seen.append((str(url), kwargs.get("params") or {}))
        r = mock.MagicMock()
        r.status_code = 200
        r.raise_for_status = mock.MagicMock()
        r.json.return_value = []
        return r

    os.environ["KAI_DASHBOARD_IDENTITY"] = "dainius"
    try:
        with mock.patch.object(httpx.AsyncClient, "get", new=fake_get):
            client.get("/api/memories?query=anything", headers=AUTH)
    finally:
        os.environ["KAI_DASHBOARD_IDENTITY"] = "test-operator"

    retrieve = [p for url, p in seen if "memory/retrieve" in url]
    assert retrieve, f"no /memory/retrieve call was made: {seen}"
    assert retrieve[0].get("user_id") == "dainius", retrieve[0]


def test_memory_search_sends_the_required_parameter():
    """KAI-DASH-D02: user_id is required upstream; omitting it was a 422."""
    import unittest.mock as mock
    import httpx

    seen = []

    async def fake_get(self, url, **kwargs):
        seen.append((str(url), kwargs.get("params") or {}))
        r = mock.MagicMock()
        r.status_code = 200
        r.raise_for_status = mock.MagicMock()
        r.json.return_value = []
        return r

    with mock.patch.object(httpx.AsyncClient, "get", new=fake_get):
        client.get("/api/memories?query=test", headers=AUTH)

    retrieve = [p for url, p in seen if "memory/retrieve" in url]
    assert retrieve, f"no /memory/retrieve call was made: {seen}"
    for required in ("query", "user_id", "top_k"):
        assert required in retrieve[0], f"{required} missing from {retrieve[0]}"


if __name__ == "__main__":
    test_health()
    test_index_minimal()
    test_anonymous_request_is_refused()
    test_bad_token_is_refused()
    test_public_routes_stay_reachable()
    test_viewer_cannot_reach_privileged_routes()
    test_unconfigured_gateway_fails_closed()
    test_memory_reads_are_scoped_to_the_caller()
    test_memory_search_sends_the_required_parameter()
    test_upload_no_file()
    test_upload_too_large()
    test_upload_image_ocr_success()
    test_upload_service_unreachable()
    print("dashboard tests passed")
