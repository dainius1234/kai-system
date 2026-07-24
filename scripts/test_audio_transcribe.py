"""Fuzz and passthrough tests for the /api/audio/transcribe endpoint in dashboard/app.py.

Tests boundary conditions, malformed inputs, and service-error mappings for the
Whisper proxy endpoint. Mirrors the pattern of security_fuzz_upload.py.

Run:
    PYTHONPATH=. python -m pytest scripts/test_audio_transcribe.py -v
"""
from __future__ import annotations

import io
import sys
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import importlib.util

# Load dashboard/app.py under an explicit module name to avoid sys.modules cache
# collision with other test files in the same pytest process.
_spec = importlib.util.spec_from_file_location(
    "dashboard_app_audio", ROOT / "dashboard" / "app.py"
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
app = _mod.app

from fastapi.testclient import TestClient

client = TestClient(app)

# Canonical AudioCaptureResult shape returned by audio-service /capture/file
_STUB_RESULT = {
    "status": "ok",
    "transcript": "hello world",
    "source": "file:audio.webm",
    "duration_seconds": 0.0,
    "timestamp": 1_700_000_000.0,
    "whisper_backend": "stub",
    "injection_detected": False,
    "emotion": None,
}


def _mock_audio_ok(result: dict = _STUB_RESULT) -> MagicMock:
    """Return an httpx.AsyncClient mock that simulates a healthy audio service."""
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = result
    mock_resp.raise_for_status = MagicMock()

    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    mock_client.post = AsyncMock(return_value=mock_resp)
    return mock_client


def _upload_audio(filename: str, data: bytes, content_type: str = "audio/webm"):
    return client.post(
        "/api/audio/transcribe",
        files={"file": (filename, io.BytesIO(data), content_type)},
    )


class TestAudioTranscribeValidation(unittest.TestCase):
    """Input-validation cases — reject obviously bad requests before touching audio service."""

    def test_no_file_rejected(self):
        """Sending no file must produce a 4xx (422 from FastAPI or 400 from handler)."""
        r = client.post("/api/audio/transcribe")
        self.assertIn(r.status_code, (400, 422))


class TestAudioTranscribePassthrough(unittest.TestCase):
    """Happy-path and service-error-mapping cases."""

    def test_stub_transcript_returned_on_200(self):
        """When audio-service returns a valid AudioCaptureResult, proxy returns 200."""
        with patch("httpx.AsyncClient", return_value=_mock_audio_ok()):
            r = _upload_audio("audio.webm", b"fake audio bytes")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json().get("transcript"), "hello world")

    def test_injection_detected_flag_passes_through(self):
        """injection_detected=True from audio-service is preserved in the response."""
        result = {**_STUB_RESULT, "transcript": "ignored", "injection_detected": True}
        with patch("httpx.AsyncClient", return_value=_mock_audio_ok(result)):
            r = _upload_audio("audio.webm", b"data")
        self.assertEqual(r.status_code, 200)
        self.assertTrue(r.json().get("injection_detected"))

    def test_audio_service_unreachable_returns_503(self):
        import httpx
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.post = AsyncMock(
            side_effect=httpx.ConnectError("Connection refused")
        )
        with patch("httpx.AsyncClient", return_value=mock_client):
            r = _upload_audio("audio.webm", b"data")
        self.assertEqual(r.status_code, 503)

    def test_audio_service_4xx_passes_through(self):
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
            r = _upload_audio("audio.webm", b"data")
        self.assertEqual(r.status_code, 415)

    def test_audio_service_5xx_becomes_502(self):
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
            r = _upload_audio("audio.webm", b"data")
        self.assertEqual(r.status_code, 502)

    def test_ogg_content_type_forwarded(self):
        """OGG audio (Firefox MediaRecorder default) is proxied without crash."""
        with patch("httpx.AsyncClient", return_value=_mock_audio_ok()):
            r = _upload_audio("audio.ogg", b"ogg header bytes", "audio/ogg")
        self.assertEqual(r.status_code, 200)

    def test_wav_content_type_forwarded(self):
        with patch("httpx.AsyncClient", return_value=_mock_audio_ok()):
            r = _upload_audio("audio.wav", b"RIFF\x00\x00\x00\x00WAVE", "audio/wav")
        self.assertEqual(r.status_code, 200)


class TestAudioTranscribeSecurity(unittest.TestCase):
    """Inputs that must not crash the service or leak information."""

    def test_path_traversal_in_filename_does_not_crash(self):
        with patch("httpx.AsyncClient", return_value=_mock_audio_ok()):
            r = _upload_audio("../../etc/passwd", b"data")
        self.assertNotEqual(r.status_code, 500)

    def test_null_bytes_in_filename_do_not_crash(self):
        with patch("httpx.AsyncClient", return_value=_mock_audio_ok()):
            r = _upload_audio("audio\x00.webm", b"data")
        self.assertNotEqual(r.status_code, 500)

    def test_very_long_filename_does_not_crash(self):
        long_name = "a" * 4096 + ".webm"
        with patch("httpx.AsyncClient", return_value=_mock_audio_ok()):
            r = _upload_audio(long_name, b"data")
        self.assertNotEqual(r.status_code, 500)

    def test_empty_file_does_not_crash(self):
        with patch("httpx.AsyncClient", return_value=_mock_audio_ok()):
            r = _upload_audio("empty.webm", b"")
        self.assertNotEqual(r.status_code, 500)

    def test_binary_garbage_does_not_crash(self):
        garbage = bytes(range(256)) * 16
        with patch("httpx.AsyncClient", return_value=_mock_audio_ok()):
            r = _upload_audio("garbage.webm", garbage)
        self.assertNotEqual(r.status_code, 500)


if __name__ == "__main__":
    unittest.main()
