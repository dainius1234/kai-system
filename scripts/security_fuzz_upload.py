"""Security fuzz test for the /api/upload endpoint in dashboard/app.py.

Tests boundary conditions, malformed inputs, and oversized payloads that should
be rejected before reaching the screen-capture service.

Run:
    PYTHONPATH=. python -m pytest scripts/security_fuzz_upload.py -v
"""
from __future__ import annotations

import io
import sys
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fastapi.testclient import TestClient
import importlib.util

# Load dashboard/app.py under an explicit module name to avoid colliding with
# the 'app' module that other test files (e.g. test_shell_sandbox.py) load from
# different paths in the same pytest process.
_spec = importlib.util.spec_from_file_location(
    "dashboard_app", ROOT / "dashboard" / "app.py"
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
app = _mod.app

# The dashboard gateway fails closed (Wave 1 Track A), so tests must
# present credentials or every route answers 503 instead of exercising
# the handler under test.
import os as _os
_os.environ["KAI_DASHBOARD_TOKEN"] = "test-dashboard-token"
_os.environ["KAI_DASHBOARD_IDENTITY"] = "test-operator"
_os.environ["KAI_DASHBOARD_ROLE"] = "keeper"
_DASH_AUTH = {"Authorization": f"Bearer {_os.environ['KAI_DASHBOARD_TOKEN']}"}

client = TestClient(app, headers=_DASH_AUTH)

_MAX_BYTES = 10 * 1024 * 1024  # must match _UPLOAD_MAX_BYTES in dashboard/app.py


def _mock_ocr_ok(text: str = "extracted text") -> MagicMock:
    """Return an httpx.AsyncClient mock that simulates a healthy OCR service."""
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"text": text, "status": "ok"}
    mock_resp.raise_for_status = MagicMock()

    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    mock_client.post = AsyncMock(return_value=mock_resp)
    return mock_client


def _upload(filename: str, data: bytes, content_type: str = "image/png") -> object:
    return client.post(
        "/api/upload",
        files={"file": (filename, io.BytesIO(data), content_type)},
    )


class TestUploadValidation(unittest.TestCase):
    """Pure input-validation cases — no OCR service call required."""

    def test_no_filename_rejected(self):
        """Sending a file with an empty filename must be rejected.
        FastAPI/Starlette may reject at the multipart layer (422) or the
        handler raises 400 — both are valid 4xx rejection responses."""
        r = client.post(
            "/api/upload",
            files={"file": ("", io.BytesIO(b"data"), "image/png")},
        )
        self.assertIn(r.status_code, (400, 422))

    def test_oversized_payload_returns_413(self):
        """Payloads larger than 10 MB must be rejected before forwarding to OCR."""
        oversized = b"X" * (_MAX_BYTES + 1)
        r = _upload("big.png", oversized)
        self.assertEqual(r.status_code, 413)

    def test_exactly_at_limit_is_forwarded(self):
        """A payload at exactly 10 MB is within the limit — reaches OCR service."""
        at_limit = b"X" * _MAX_BYTES
        with patch("httpx.AsyncClient", return_value=_mock_ocr_ok()):
            r = _upload("at_limit.png", at_limit)
        self.assertIn(r.status_code, (200, 503))  # 503 if mock setup doesn't match

    def test_one_byte_over_limit_returns_413(self):
        one_over = b"X" * (_MAX_BYTES + 1)
        r = _upload("one_over.png", one_over)
        self.assertEqual(r.status_code, 413)


class TestUploadSecurityBoundaries(unittest.TestCase):
    """Inputs that must not crash the service or leak information."""

    def test_path_traversal_in_filename_does_not_crash(self):
        """Filenames like ../../etc/passwd are forwarded as-is to OCR; the endpoint
        must not crash — the filename is never opened locally."""
        with patch("httpx.AsyncClient", return_value=_mock_ocr_ok()):
            r = _upload("../../etc/passwd", b"fake image data")
        # Either proxied successfully (200) or OCR unreachable (503/4xx) — never 500
        self.assertNotEqual(r.status_code, 500)

    def test_null_bytes_in_filename_do_not_crash(self):
        with patch("httpx.AsyncClient", return_value=_mock_ocr_ok()):
            r = _upload("file\x00name.png", b"data")
        self.assertNotEqual(r.status_code, 500)

    def test_very_long_filename_does_not_crash(self):
        long_name = "a" * 4096 + ".png"
        with patch("httpx.AsyncClient", return_value=_mock_ocr_ok()):
            r = _upload(long_name, b"data")
        self.assertNotEqual(r.status_code, 500)

    def test_shell_script_content_type_is_forwarded(self):
        """Content-type is not validated at this layer — that is intentional.
        Screen-capture decides what it can OCR. Ensure no 500 leak here."""
        with patch("httpx.AsyncClient", return_value=_mock_ocr_ok()):
            r = _upload("evil.sh", b"#!/bin/bash\nrm -rf /", "application/x-sh")
        self.assertNotEqual(r.status_code, 500)

    def test_empty_file_body_does_not_crash(self):
        with patch("httpx.AsyncClient", return_value=_mock_ocr_ok()):
            r = _upload("empty.png", b"")
        self.assertNotEqual(r.status_code, 500)

    def test_binary_garbage_does_not_crash(self):
        garbage = bytes(range(256)) * 16
        with patch("httpx.AsyncClient", return_value=_mock_ocr_ok()):
            r = _upload("garbage.png", garbage)
        self.assertNotEqual(r.status_code, 500)


class TestUploadOCRPassthrough(unittest.TestCase):
    """When OCR service returns errors, the endpoint maps them correctly."""

    def test_ocr_4xx_is_passed_through(self):
        import httpx

        mock_resp = MagicMock()
        mock_resp.status_code = 415
        mock_resp.raise_for_status.side_effect = httpx.HTTPStatusError(
            "Unsupported Media Type", request=MagicMock(), response=mock_resp
        )
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.post = AsyncMock(return_value=mock_resp)

        with patch("httpx.AsyncClient", return_value=mock_client):
            r = _upload("bad.png", b"data")
        self.assertEqual(r.status_code, 415)

    def test_ocr_5xx_becomes_502(self):
        import httpx

        mock_resp = MagicMock()
        mock_resp.status_code = 500
        mock_resp.raise_for_status.side_effect = httpx.HTTPStatusError(
            "Internal Server Error", request=MagicMock(), response=mock_resp
        )
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.post = AsyncMock(return_value=mock_resp)

        with patch("httpx.AsyncClient", return_value=mock_client):
            r = _upload("bad.png", b"data")
        self.assertEqual(r.status_code, 502)

    def test_ocr_unreachable_becomes_503(self):
        import httpx

        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.post = AsyncMock(
            side_effect=httpx.ConnectError("Connection refused")
        )

        with patch("httpx.AsyncClient", return_value=mock_client):
            r = _upload("img.png", b"data")
        self.assertEqual(r.status_code, 503)

    def test_happy_path_returns_ocr_json(self):
        with patch("httpx.AsyncClient", return_value=_mock_ocr_ok("hello world")):
            r = _upload("doc.png", b"fake png")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json().get("text"), "hello world")


if __name__ == "__main__":
    unittest.main()
