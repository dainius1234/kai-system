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


# ── Tracks E-I: bounds, headers, media types, audit ──────────────────

def test_oversized_body_is_refused():
    """KAI-DASH-017: an unbounded proxy body makes this an amplifier."""
    big = {"payload": "x" * (300 * 1024)}
    resp = client.post("/api/soul", headers=AUTH, json=big)
    assert resp.status_code == 413, resp.status_code


def test_deeply_nested_body_is_refused():
    payload = current = {}
    for _ in range(40):
        current["next"] = {}
        current = current["next"]
    resp = client.post("/api/soul", headers=AUTH, json=payload)
    assert resp.status_code == 413, resp.status_code


def test_high_cardinality_body_is_refused():
    payload = {f"k{i}": i for i in range(2000)}
    resp = client.post("/api/soul", headers=AUTH, json=payload)
    assert resp.status_code == 413, resp.status_code


def test_normal_body_is_not_refused():
    """The bound must not be so tight it blocks ordinary use."""
    import unittest.mock as mock
    import httpx

    async def fake_post(self, url, **kwargs):
        r = mock.MagicMock()
        r.status_code = 200
        r.raise_for_status = mock.MagicMock()
        r.json.return_value = {"ok": True}
        return r

    with mock.patch.object(httpx.AsyncClient, "post", new=fake_post):
        resp = client.post("/api/soul", headers=AUTH,
                           json={"content": "a reasonable soul edit"})
    assert resp.status_code != 413, resp.status_code


def test_html_carries_browser_security_headers():
    """KAI-DASH-088: no CSP, frame or referrer protections."""
    resp = client.get("/app")
    assert resp.status_code == 200
    for header in ("Content-Security-Policy", "X-Frame-Options",
                   "Referrer-Policy", "X-Content-Type-Options"):
        assert header in resp.headers, f"{header} missing"
    csp = resp.headers["Content-Security-Policy"]
    assert "frame-ancestors 'none'" in csp, csp
    # An inline-script escape hatch would defeat the policy's main job.
    assert "'unsafe-inline'" not in csp.split("script-src")[1].split(";")[0], csp


def test_json_responses_do_not_carry_html_headers():
    """The policy belongs on documents, not on every API payload."""
    resp = client.get("/health")
    assert resp.status_code == 200
    assert "Content-Security-Policy" not in resp.headers


def test_health_discloses_no_topology():
    """KAI-DASH-069: health leaked the gate URL and policy hash."""
    payload = client.get("/health").json()
    for leaked in ("tool_gate_url", "policy_version", "policy_hash"):
        assert leaked not in payload, f"{leaked} still disclosed"


def test_go_no_go_is_not_a_200_when_not_go():
    """KAI-DASH-080: nothing downstream could enforce an advisory NO_GO."""
    resp = client.get("/go-no-go", headers=AUTH)
    payload = resp.json()
    if payload.get("decision") == "GO":
        assert resp.status_code == 200
    else:
        assert resp.status_code == 503, (resp.status_code, payload.get("decision"))


def test_audit_records_a_credential_derived_actor():
    """KAI-DASH-096: audit recorded only method, path and status."""
    seen = []
    original = mod.audit.log
    mod.audit.log = lambda level, message: seen.append(message)
    try:
        client.get("/health", headers=AUTH)
    finally:
        mod.audit.log = original
    assert seen, "nothing was audited"
    assert any("actor=" in line for line in seen), seen
    # The credential itself must never reach the log.
    assert not any(TOKEN in line for line in seen), "token leaked into audit"


def test_oversized_upload_is_refused_during_the_read():
    """KAI-DASH-045: the limit used to fire after the body was buffered."""
    big = b"x" * (11 * 1024 * 1024)
    resp = client.post("/api/upload", headers=AUTH,
                       files={"file": ("big.png", big, "image/png")})
    assert resp.status_code == 413, resp.status_code


def test_audio_and_vision_uploads_are_bounded():
    """KAI-DASH-046/047: these had no size check at all."""
    big = b"x" * (11 * 1024 * 1024)
    for path, name, mime in [
        ("/api/audio/transcribe", "big.webm", "audio/webm"),
        ("/api/vision/analyze", "big.jpg", "image/jpeg"),
        ("/api/vision/presence", "big.jpg", "image/jpeg"),
    ]:
        resp = client.post(path, headers=AUTH,
                           files={"file": (name, big, mime)})
        assert resp.status_code == 413, f"{path} -> {resp.status_code}"


def test_traversal_filename_is_canonicalised():
    """KAI-DASH-051: the raw name reached services that write to disk."""
    import unittest.mock as mock
    import httpx

    seen = {}

    async def fake_post(self, url, **kwargs):
        seen.update(kwargs.get("files") or {})
        r = mock.MagicMock()
        r.status_code = 200
        r.raise_for_status = mock.MagicMock()
        r.json.return_value = {"text": "ok"}
        return r

    with mock.patch.object(httpx.AsyncClient, "post", new=fake_post):
        client.post("/api/upload", headers=AUTH,
                    files={"file": ("../../etc/passwd.png", b"x", "image/png")})
    assert seen, "no upload was forwarded"
    forwarded = seen["file"][0]
    assert "/" not in forwarded and ".." not in forwarded, forwarded


def test_error_details_do_not_leak_internal_hosts():
    """KAI-DASH-052: exception text carried internal URLs to the caller."""
    import unittest.mock as mock
    import httpx

    async def fail(*args, **kwargs):
        raise httpx.ConnectError("connection to http://screen-capture:8059 refused")

    with mock.patch.object(httpx.AsyncClient, "post", new=fail):
        resp = client.post("/api/upload", headers=AUTH,
                           files={"file": ("a.png", b"x", "image/png")})
    body = resp.text
    assert "screen-capture:8059" not in body, body
    assert "http://" not in body, body


def test_query_limits_reject_out_of_range_values():
    """KAI-DASH-093: negative and extreme values sailed through."""
    for path in ("/api/memories/recent?top_k=-1",
                 "/api/memories/recent?top_k=999999",
                 "/api/logs?limit=0"):
        resp = client.get(path, headers=AUTH)
        assert resp.status_code == 422, f"{path} -> {resp.status_code}"


def test_valid_query_limits_still_work():
    resp = client.get("/api/memories/recent?top_k=5", headers=AUTH)
    assert resp.status_code != 422, resp.status_code


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
    test_oversized_body_is_refused()
    test_deeply_nested_body_is_refused()
    test_high_cardinality_body_is_refused()
    test_normal_body_is_not_refused()
    test_html_carries_browser_security_headers()
    test_json_responses_do_not_carry_html_headers()
    test_health_discloses_no_topology()
    test_go_no_go_is_not_a_200_when_not_go()
    test_audit_records_a_credential_derived_actor()
    test_oversized_upload_is_refused_during_the_read()
    test_audio_and_vision_uploads_are_bounded()
    test_traversal_filename_is_canonicalised()
    test_error_details_do_not_leak_internal_hosts()
    test_query_limits_reject_out_of_range_values()
    test_valid_query_limits_still_work()
    test_upload_no_file()
    test_upload_too_large()
    test_upload_image_ocr_success()
    test_upload_service_unreachable()
    print("dashboard tests passed")
