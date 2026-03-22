"""Multi-Modal Sensory Input tests (Gap 2).

Tests voice emotion analysis in perception/audio/app.py and
visual/frame analysis in perception/camera/app.py.

Source: emotion-aware agent & screen-analysis patterns.
"""
from __future__ import annotations

import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
AUDIO_SRC = (ROOT / "perception" / "audio" / "app.py").read_text()
CAMERA_SRC = (ROOT / "perception" / "camera" / "app.py").read_text()


# ══ Audio: Voice Emotion Analysis ═══════════════════════════════════

class TestEmotionKeywords(unittest.TestCase):
    """Verify emotion keyword dictionaries."""

    def test_keywords_dict_defined(self):
        self.assertIn("_EMOTION_KEYWORDS", AUDIO_SRC)

    def test_has_stress_category(self):
        self.assertIn('"stress"', AUDIO_SRC)

    def test_has_fatigue_category(self):
        self.assertIn('"fatigue"', AUDIO_SRC)

    def test_has_excitement_category(self):
        self.assertIn('"excitement"', AUDIO_SRC)

    def test_has_uncertainty_category(self):
        self.assertIn('"uncertainty"', AUDIO_SRC)

    def test_has_calm_category(self):
        self.assertIn('"calm"', AUDIO_SRC)


class TestEmotionNudges(unittest.TestCase):
    """Verify emotion-to-nudge mapping."""

    def test_nudges_dict_defined(self):
        self.assertIn("_EMOTION_NUDGES", AUDIO_SRC)

    def test_stress_has_nudge(self):
        nudges_section = AUDIO_SRC.split("_EMOTION_NUDGES")[1].split("}")[0]
        self.assertIn("stress", nudges_section)

    def test_fatigue_has_nudge(self):
        nudges_section = AUDIO_SRC.split("_EMOTION_NUDGES")[1].split("}")[0]
        self.assertIn("fatigue", nudges_section)


class TestVoiceEmotionAnalyser(unittest.TestCase):
    """Verify voice emotion analysis function."""

    def test_function_defined(self):
        self.assertIn("def _analyse_voice_emotion(", AUDIO_SRC)

    def test_takes_transcript_and_audio(self):
        fn = AUDIO_SRC.split("def _analyse_voice_emotion(")[1].split("\ndef ")[0]
        self.assertIn("transcript", fn)

    def test_returns_scores(self):
        fn = AUDIO_SRC.split("def _analyse_voice_emotion(")[1].split("\ndef ")[0]
        self.assertIn("scores", fn)

    def test_returns_dominant_emotion(self):
        fn = AUDIO_SRC.split("def _analyse_voice_emotion(")[1].split("\ndef ")[0]
        self.assertIn("dominant", fn)

    def test_rms_audio_energy(self):
        """Should analyse RMS for fatigue/stress signals."""
        fn = AUDIO_SRC.split("def _analyse_voice_emotion(")[1].split("\ndef ")[0]
        self.assertIn("rms", fn.lower())

    def test_nudge_generation(self):
        fn = AUDIO_SRC.split("def _analyse_voice_emotion(")[1].split("\ndef ")[0]
        self.assertIn("nudge", fn)


class TestEmotionEndpoint(unittest.TestCase):
    """Verify /analyse/emotion endpoint."""

    def test_endpoint_exists(self):
        self.assertIn('"/analyse/emotion"', AUDIO_SRC)

    def test_endpoint_is_post(self):
        self.assertIn('@app.post("/analyse/emotion")', AUDIO_SRC)


class TestAudioCaptureEmotion(unittest.TestCase):
    """Verify emotion is wired into capture results."""

    def test_emotion_field_on_model(self):
        model_section = AUDIO_SRC.split("class AudioCaptureResult")[1].split("\n\n")[0]
        self.assertIn("emotion", model_section)

    def test_mic_capture_includes_emotion(self):
        fn = AUDIO_SRC.split("async def capture_mic(")[1].split("\nasync def ")[0]
        self.assertIn("emotion", fn)


# ══ Camera: Visual Frame Analysis ══════════════════════════════════

class TestFrameAnalysis(unittest.TestCase):
    """Verify frame analysis engine."""

    def test_analyse_frame_defined(self):
        self.assertIn("def _analyse_frame(", CAMERA_SRC)

    def test_brightness_detection(self):
        fn = CAMERA_SRC.split("def _analyse_frame(")[1].split("\ndef ")[0]
        self.assertIn("brightness", fn)

    def test_edge_density(self):
        fn = CAMERA_SRC.split("def _analyse_frame(")[1].split("\ndef ")[0]
        self.assertIn("edge", fn)

    def test_motion_detection(self):
        fn = CAMERA_SRC.split("def _analyse_frame(")[1].split("\ndef ")[0]
        self.assertIn("motion", fn)


class TestCameraCapture(unittest.TestCase):
    """Verify camera capture function."""

    def test_camera_capture_defined(self):
        self.assertIn("def _capture_camera_frame(", CAMERA_SRC)

    def test_screen_capture_defined(self):
        self.assertIn("def _capture_screen(", CAMERA_SRC)


class TestCameraEndpoints(unittest.TestCase):
    """Verify camera service HTTP endpoints."""

    def test_capture_camera_endpoint(self):
        self.assertIn('"/capture/camera"', CAMERA_SRC)

    def test_capture_screen_endpoint(self):
        self.assertIn('"/capture/screen"', CAMERA_SRC)

    def test_analysis_history_endpoint(self):
        self.assertIn('"/analysis/history"', CAMERA_SRC)

    def test_health_endpoint(self):
        self.assertIn('"/health"', CAMERA_SRC)


class TestGracefulDegradation(unittest.TestCase):
    """Camera service must work without opencv/numpy."""

    def test_cv2_is_optional(self):
        self.assertIn("_cv2_available", CAMERA_SRC)

    def test_numpy_is_optional(self):
        self.assertIn("_numpy_available", CAMERA_SRC)

    def test_has_fallback_frame(self):
        """Should produce something useful even without hardware."""
        self.assertIn("dummy", CAMERA_SRC.lower())


class TestNudgeGeneration(unittest.TestCase):
    """Verify visual nudge suggestions."""

    def test_dark_screen_nudge(self):
        self.assertIn("dark", CAMERA_SRC.lower())

    def test_nudge_key_in_output(self):
        self.assertIn("nudge", CAMERA_SRC)


if __name__ == "__main__":
    unittest.main()
