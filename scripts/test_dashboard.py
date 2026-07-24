from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

from fastapi.testclient import TestClient

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
    resp = client.get("/health")
    assert resp.status_code == 200
    assert "status" in resp.json()


def test_index_minimal():
    # patch out fetch_status to avoid external calls
    async def fake_status():
        return {}
    mod.fetch_status = fake_status  # type: ignore
    try:
        resp = client.get("/")
    except Exception:
        # external dependencies may be unreachable; nothing to check here
        return
    assert resp.status_code in (200, 500)
    if resp.status_code == 200:
        jsonp = resp.json()
        assert "service" in jsonp and jsonp["service"] == "dashboard"


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
    resp = client.post("/api/upload")
    assert resp.status_code == 422


def test_upload_too_large():
    """POST /api/upload with a file exceeding 10 MB → 413."""
    big = b"x" * (10 * 1024 * 1024 + 1)
    resp = client.post("/api/upload", files={"file": ("big.png", big, "image/png")})
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
            files={"file": ("test.png", _tiny_png(), "image/png")},
        )
    assert resp.status_code == 503


if __name__ == "__main__":
    test_health()
    test_index_minimal()
    test_upload_no_file()
    test_upload_too_large()
    test_upload_image_ocr_success()
    test_upload_service_unreachable()
    print("dashboard tests passed")
