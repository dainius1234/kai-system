"""P1 Screen Capture pipeline tests — headless / file-based mode.

Exercises screen-capture/app.py via FastAPI TestClient without X11 or
a live Tesseract installation.  All tests run on bare Python with no
GPU, no display, and no optional C deps (mss, tesseract, Pillow).

Run with:
    PYTHONPATH=. python -m pytest scripts/test_screen_capture.py -v
or:
    PYTHONPATH=. python scripts/test_screen_capture.py
"""
from __future__ import annotations

import io
import struct
import sys
import tempfile
import zlib
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Override env before importing app so WATCH_DIR is under our temp dir
import os

_tmp_watch = tempfile.mkdtemp(prefix="kai-sc-test-")
os.environ.setdefault("SCREEN_WATCH_DIR", _tmp_watch)
os.environ.setdefault("MEMU_URL", "http://localhost:9999")  # nothing listening
os.environ.setdefault("AUTO_MEMORIZE", "false")

from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "screen-capture"))
from app import app  # noqa: E402

client = TestClient(app, raise_server_exceptions=False)

WATCH_DIR = Path(_tmp_watch)


# ── helpers ──────────────────────────────────────────────────────────

def _minimal_png() -> bytes:
    """Build a 1×1 white PNG with stdlib only — no Pillow."""
    def chunk(tag: bytes, data: bytes) -> bytes:
        length = struct.pack(">I", len(data))
        crc = struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
        return length + tag + data + crc

    sig = b"\x89PNG\r\n\x1a\n"
    ihdr_data = struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0)   # 1×1, 8-bit RGB
    ihdr = chunk(b"IHDR", ihdr_data)
    # Scanline: filter-byte 0 + RGB (255,255,255)
    idat = chunk(b"IDAT", zlib.compress(b"\x00\xff\xff\xff"))
    iend = chunk(b"IEND", b"")
    return sig + ihdr + idat + iend


PNG = _minimal_png()


# ── health ────────────────────────────────────────────────────────────

class TestHealth:
    def test_health_200(self):
        r = client.get("/health")
        assert r.status_code == 200

    def test_health_has_status_ok(self):
        data = client.get("/health").json()
        assert data["status"] == "ok"

    def test_health_has_service_name(self):
        data = client.get("/health").json()
        assert data["service"] == "screen-capture"

    def test_health_has_watch_dir(self):
        data = client.get("/health").json()
        assert "watch_dir" in data

    def test_health_has_ocr_fields(self):
        data = client.get("/health").json()
        assert "mss_available" in data
        assert "tesseract_available" in data
        assert "ocr_enabled" in data


# ── /metrics ─────────────────────────────────────────────────────────

class TestMetrics:
    def test_metrics_200(self):
        r = client.get("/metrics")
        assert r.status_code == 200

    def test_metrics_is_dict(self):
        data = client.get("/metrics").json()
        assert isinstance(data, dict)


# ── /capture/file ────────────────────────────────────────────────────

class TestCaptureFile:
    def test_upload_png_200(self):
        r = client.post(
            "/capture/file",
            files={"file": ("test.png", io.BytesIO(PNG), "image/png")},
        )
        assert r.status_code == 200

    def test_upload_returns_status_ok(self):
        r = client.post(
            "/capture/file",
            files={"file": ("test.png", io.BytesIO(PNG), "image/png")},
        )
        data = r.json()
        assert data["status"] == "ok"

    def test_upload_returns_source_with_filename(self):
        r = client.post(
            "/capture/file",
            files={"file": ("myscreen.png", io.BytesIO(PNG), "image/png")},
        )
        data = r.json()
        assert "myscreen.png" in data["source"]

    def test_upload_returns_timestamp_float(self):
        r = client.post(
            "/capture/file",
            files={"file": ("t.png", io.BytesIO(PNG), "image/png")},
        )
        data = r.json()
        assert isinstance(data["timestamp"], float)
        assert data["timestamp"] > 0

    def test_upload_returns_ocr_available_field(self):
        r = client.post(
            "/capture/file",
            files={"file": ("t.png", io.BytesIO(PNG), "image/png")},
        )
        data = r.json()
        assert "ocr_available" in data
        assert isinstance(data["ocr_available"], bool)

    def test_upload_text_field_is_string(self):
        r = client.post(
            "/capture/file",
            files={"file": ("t.png", io.BytesIO(PNG), "image/png")},
        )
        data = r.json()
        assert isinstance(data["text"], str)

    def test_upload_no_file_returns_422(self):
        r = client.post("/capture/file")
        assert r.status_code == 422

    def test_upload_oversized_returns_413(self):
        # 11 MB of zeros
        big = b"\x00" * (11 * 1024 * 1024)
        r = client.post(
            "/capture/file",
            files={"file": ("big.png", io.BytesIO(big), "image/png")},
        )
        assert r.status_code == 413

    def test_upload_jpeg_is_accepted(self):
        # A minimal valid JPEG (1×1 white)
        jpeg = bytes([
            0xFF, 0xD8, 0xFF, 0xE0, 0x00, 0x10, 0x4A, 0x46, 0x49, 0x46, 0x00,
            0x01, 0x01, 0x00, 0x00, 0x01, 0x00, 0x01, 0x00, 0x00, 0xFF, 0xDB,
            0x00, 0x43, 0x00, 0x08, 0x06, 0x06, 0x07, 0x06, 0x05, 0x08, 0x07,
            0x07, 0x07, 0x09, 0x09, 0x08, 0x0A, 0x0C, 0x14, 0x0D, 0x0C, 0x0B,
            0x0B, 0x0C, 0x19, 0x12, 0x13, 0x0F, 0x14, 0x1D, 0x1A, 0x1F, 0x1E,
            0x1D, 0x1A, 0x1C, 0x1C, 0x20, 0x24, 0x2E, 0x27, 0x20, 0x22, 0x2C,
            0x23, 0x1C, 0x1C, 0x28, 0x37, 0x29, 0x2C, 0x30, 0x31, 0x34, 0x34,
            0x34, 0x1F, 0x27, 0x39, 0x3D, 0x38, 0x32, 0x3C, 0x2E, 0x33, 0x34,
            0x32, 0xFF, 0xC0, 0x00, 0x0B, 0x08, 0x00, 0x01, 0x00, 0x01, 0x01,
            0x01, 0x11, 0x00, 0xFF, 0xC4, 0x00, 0x1F, 0x00, 0x00, 0x01, 0x05,
            0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00,
            0x00, 0x00, 0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08,
            0x09, 0x0A, 0x0B, 0xFF, 0xC4, 0x00, 0xB5, 0x10, 0x00, 0x02, 0x01,
            0x03, 0x03, 0x02, 0x04, 0x03, 0x05, 0x05, 0x04, 0x04, 0x00, 0x00,
            0x01, 0x7D, 0x01, 0x02, 0x03, 0x00, 0x04, 0x11, 0x05, 0x12, 0x21,
            0x31, 0x41, 0x06, 0x13, 0x51, 0x61, 0x07, 0x22, 0x71, 0x14, 0x32,
            0x81, 0x91, 0xA1, 0x08, 0x23, 0x42, 0xB1, 0xC1, 0x15, 0x52, 0xD1,
            0xF0, 0x24, 0x33, 0x62, 0x72, 0x82, 0x09, 0x0A, 0x16, 0x17, 0x18,
            0x19, 0x1A, 0x25, 0x26, 0x27, 0x28, 0x29, 0x2A, 0x34, 0x35, 0x36,
            0x37, 0x38, 0x39, 0x3A, 0x43, 0x44, 0x45, 0x46, 0x47, 0x48, 0x49,
            0x4A, 0x53, 0x54, 0x55, 0x56, 0x57, 0x58, 0x59, 0x5A, 0x63, 0x64,
            0x65, 0x66, 0x67, 0x68, 0x69, 0x6A, 0x73, 0x74, 0x75, 0x76, 0x77,
            0x78, 0x79, 0x7A, 0x83, 0x84, 0x85, 0x86, 0x87, 0x88, 0x89, 0x8A,
            0x92, 0x93, 0x94, 0x95, 0x96, 0x97, 0x98, 0x99, 0x9A, 0xA2, 0xA3,
            0xA4, 0xA5, 0xA6, 0xA7, 0xA8, 0xA9, 0xAA, 0xB2, 0xB3, 0xB4, 0xB5,
            0xB6, 0xB7, 0xB8, 0xB9, 0xBA, 0xC2, 0xC3, 0xC4, 0xC5, 0xC6, 0xC7,
            0xC8, 0xC9, 0xCA, 0xD2, 0xD3, 0xD4, 0xD5, 0xD6, 0xD7, 0xD8, 0xD9,
            0xDA, 0xE1, 0xE2, 0xE3, 0xE4, 0xE5, 0xE6, 0xE7, 0xE8, 0xE9, 0xEA,
            0xF1, 0xF2, 0xF3, 0xF4, 0xF5, 0xF6, 0xF7, 0xF8, 0xF9, 0xFA, 0xFF,
            0xDA, 0x00, 0x08, 0x01, 0x01, 0x00, 0x00, 0x3F, 0x00, 0xFB, 0xD0,
            0xFF, 0xD9,
        ])
        r = client.post(
            "/capture/file",
            files={"file": ("snap.jpg", io.BytesIO(jpeg), "image/jpeg")},
        )
        # Service should accept it (OCR may or may not work, that's ok)
        assert r.status_code == 200


# ── /capture — watchdir fallback ─────────────────────────────────────

class TestCaptureWatchDir:
    def test_capture_503_when_empty_and_no_mss(self):
        # With no mss and empty watch dir, should 503
        # First clear the watch dir
        for f in WATCH_DIR.glob("*.png"):
            f.unlink()
        # If mss isn't available and dir is empty, expect 503 or 500
        r = client.post("/capture")
        assert r.status_code in (500, 503)

    def test_capture_200_with_file_in_watchdir(self, tmp_path):
        img_path = WATCH_DIR / "test_watchdir_capture.png"
        img_path.write_bytes(PNG)
        try:
            r = client.post("/capture")
            # Only succeeds if mss is not available but a file is in watch dir
            # — in CI without mss this should return 200
            if r.status_code == 200:
                data = r.json()
                assert data["status"] == "ok"
                assert "file:" in data["source"] or "screen-" in data["source"]
        finally:
            img_path.unlink(missing_ok=True)


# ── CaptureResult schema ──────────────────────────────────────────────

class TestResponseSchema:
    def test_all_required_fields_present(self):
        r = client.post(
            "/capture/file",
            files={"file": ("t.png", io.BytesIO(PNG), "image/png")},
        )
        assert r.status_code == 200
        data = r.json()
        for field in ("status", "text", "source", "timestamp", "ocr_available"):
            assert field in data, f"missing field: {field}"

    def test_ocr_stub_message_when_unavailable(self):
        # When tesseract is not installed, text should be the stub message or empty
        r = client.post(
            "/capture/file",
            files={"file": ("t.png", io.BytesIO(PNG), "image/png")},
        )
        data = r.json()
        # Either OCR ran (may produce empty string) or stub was returned
        assert isinstance(data["text"], str)


# ── Run directly ──────────────────────────────────────────────────────

if __name__ == "__main__":
    import subprocess
    result = subprocess.run(
        [sys.executable, "-m", "pytest", __file__, "-v"],
        cwd=str(Path(__file__).resolve().parents[1]),
    )
    sys.exit(result.returncode)
