"""Tests for perception/vision/app.py.

OpenCV and DeepFace are mocked — no camera or ML model required.

Run:
    PYTHONPATH=. python -m pytest scripts/test_vision_service.py -v
"""
from __future__ import annotations

import importlib.util
import io
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.module_stubs import stubbed  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "vision_service_app", ROOT / "perception" / "vision" / "app.py"
)
_mod = importlib.util.module_from_spec(_spec)

# This file's docstring says OpenCV and DeepFace are mocked, and every test
# below does mock them — but the *load* used whatever cv2 the machine had.
# So the result depended on the environment: absent locally (the app's
# ImportError branch, fine), present-but-partial in CI (AttributeError on
# cv2.CascadeClassifier, collection aborted, nothing ran).
#
# Stubbing the load makes the two agree. Scoped, so nothing downstream sees
# a fake cv2 — see scripts/module_stubs.py.
with stubbed({} if "cv2" in sys.modules else {"cv2": MagicMock()}):
    _spec.loader.exec_module(_mod)
app = _mod.app

from fastapi.testclient import TestClient

client = TestClient(app)

# Minimal valid JPEG header (20 bytes) — enough to not be "empty"
_FAKE_JPEG = bytes([
    0xFF, 0xD8, 0xFF, 0xE0, 0x00, 0x10, 0x4A, 0x46, 0x49, 0x46,
    0x00, 0x01, 0x01, 0x00, 0x00, 0x01, 0x00, 0x01, 0x00, 0x00,
])


def _upload_frame(data: bytes = _FAKE_JPEG, filename: str = "frame.jpg", ct: str = "image/jpeg"):
    return client.post(
        "/analyze/frame",
        files={"file": (filename, io.BytesIO(data), ct)},
    )


def _upload_presence(data: bytes = _FAKE_JPEG):
    return client.post(
        "/analyze/presence",
        files={"file": ("frame.jpg", io.BytesIO(data), "image/jpeg")},
    )


class TestHealth(unittest.TestCase):
    def test_health_returns_ok(self):
        r = client.get("/health")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["status"], "ok")

    def test_metrics_returns_dict(self):
        r = client.get("/metrics")
        self.assertEqual(r.status_code, 200)
        self.assertIsInstance(r.json(), dict)


class TestAnalyzeFrameStubMode(unittest.TestCase):
    """When OpenCV is not available the service returns a stub response, not a 500."""

    def test_stub_response_when_opencv_absent(self):
        with patch.object(_mod, "_OPENCV_OK", False):
            r = _upload_frame()
        self.assertEqual(r.status_code, 200)
        data = r.json()
        self.assertFalse(data["presence"])
        self.assertEqual(data["backend"], "stub")

    def test_empty_file_rejected(self):
        r = _upload_frame(data=b"")
        self.assertEqual(r.status_code, 400)

    def test_garbage_bytes_do_not_crash(self):
        with patch.object(_mod, "_OPENCV_OK", False):
            r = _upload_frame(data=bytes(range(256)) * 4)
        self.assertNotEqual(r.status_code, 500)


class TestAnalyzeFrameWithOpenCV(unittest.TestCase):
    """Mock OpenCV to return a valid image and known face detections."""

    def _mock_cv2(self, face_rects=None):
        import numpy as np
        mock_cv2 = MagicMock()
        # imdecode returns a fake 100x100 BGR array
        mock_cv2.imdecode.return_value = np.zeros((100, 100, 3), dtype=np.uint8)
        mock_cv2.cvtColor.return_value = np.zeros((100, 100), dtype=np.uint8)
        mock_cv2.COLOR_BGR2GRAY = 6
        cascade = MagicMock()
        rects = face_rects if face_rects is not None else []
        cascade.detectMultiScale.return_value = rects
        mock_cv2.CascadeClassifier.return_value = cascade
        mock_cv2.data = MagicMock()
        mock_cv2.data.haarcascades = ""
        return mock_cv2

    def test_no_face_detected(self):
        fake_img = object()  # opaque sentinel — _detect_faces is mocked anyway
        with patch.object(_mod, "_OPENCV_OK", True), \
             patch.object(_mod, "_DEEPFACE_OK", False), \
             patch.object(_mod, "_decode_image", return_value=fake_img), \
             patch.object(_mod, "_detect_faces", return_value=[]):
            r = _upload_frame()
        self.assertEqual(r.status_code, 200)
        data = r.json()
        self.assertFalse(data["presence"])
        self.assertEqual(data["face_count"], 0)
        self.assertIsNone(data["dominant_emotion"])

    def test_face_detected(self):
        fake_img = object()
        fake_face = {"x": 10, "y": 10, "w": 50, "h": 50}
        with patch.object(_mod, "_OPENCV_OK", True), \
             patch.object(_mod, "_DEEPFACE_OK", False), \
             patch.object(_mod, "_decode_image", return_value=fake_img), \
             patch.object(_mod, "_detect_faces", return_value=[fake_face]):
            r = _upload_frame()
        self.assertEqual(r.status_code, 200)
        data = r.json()
        self.assertTrue(data["presence"])
        self.assertEqual(data["face_count"], 1)
        self.assertEqual(data["faces"][0]["w"], 50)

    def test_emotion_returned_when_deepface_available(self):
        fake_img = object()
        fake_face = {"x": 10, "y": 10, "w": 50, "h": 50}
        mock_emotion = {"dominant": "happy", "scores": {"happy": 0.9, "neutral": 0.1}}
        with patch.object(_mod, "_OPENCV_OK", True), \
             patch.object(_mod, "_DEEPFACE_OK", True), \
             patch.object(_mod, "_decode_image", return_value=fake_img), \
             patch.object(_mod, "_detect_faces", return_value=[fake_face]), \
             patch.object(_mod, "_detect_emotion", return_value=mock_emotion):
            r = _upload_frame()
        self.assertEqual(r.status_code, 200)
        data = r.json()
        self.assertEqual(data["dominant_emotion"], "happy")
        self.assertIn("happy", data["emotions"])


class TestAnalyzePresence(unittest.TestCase):
    def test_presence_stub_when_no_opencv(self):
        with patch.object(_mod, "_OPENCV_OK", False):
            r = _upload_presence()
        self.assertEqual(r.status_code, 200)
        data = r.json()
        self.assertFalse(data["present"])
        self.assertEqual(data["backend"], "stub")

    def test_presence_true_when_face_found(self):
        fake_img = object()
        with patch.object(_mod, "_OPENCV_OK", True), \
             patch.object(_mod, "_decode_image", return_value=fake_img), \
             patch.object(_mod, "_detect_faces", return_value=[{"x": 0, "y": 0, "w": 40, "h": 40}]):
            r = _upload_presence()
        self.assertEqual(r.status_code, 200)
        data = r.json()
        self.assertTrue(data["present"])
        self.assertGreater(data["confidence"], 0)

    def test_empty_frame_rejected(self):
        r = _upload_presence(data=b"")
        self.assertEqual(r.status_code, 400)

    def test_binary_garbage_does_not_crash(self):
        with patch.object(_mod, "_OPENCV_OK", False):
            r = _upload_presence(data=bytes(range(256)) * 8)
        self.assertNotEqual(r.status_code, 500)


if __name__ == "__main__":
    unittest.main()
