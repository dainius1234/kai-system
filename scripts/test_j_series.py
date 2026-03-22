"""Tests for J-Series Jewels — all 7 features.

J1: Live Canvas Visualization (dashboard HTML structure)
J2: Wake-word "Kai" + Intent Judge
J3: Auto-Redaction PII
J4: Proactive Low-Latency Voice
J5: Memory Viewer GUI (dashboard HTML structure)
J6: SOUL.md + AGENTS.md
J7: Skills Auto-Install Hub
"""
from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

# Ensure repo root is on path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ═══════════════════════════════════════════════════════════════════════
# J2: Wake-word "Kai" + Intent Judge
# ═══════════════════════════════════════════════════════════════════════


class TestJ2WakeWord(unittest.TestCase):
    """J2: Wake-word detection and intent classification."""

    def setUp(self):
        os.environ.setdefault("WHISPER_BACKEND", "stub")
        os.environ.setdefault("AUDIO_DIR", "/tmp/audio-test")
        os.environ.setdefault("LOG_PATH", "/tmp/test-audio.json.log")
        # Import the module fresh
        from perception.audio.app import detect_wake_word, WAKE_WORD_RE
        self.detect = detect_wake_word
        self.regex = WAKE_WORD_RE

    def test_wake_word_detected_in_simple_address(self):
        self.assertTrue(self.detect("Kai, what's the weather?"))

    def test_wake_word_detected_case_insensitive(self):
        self.assertTrue(self.detect("hey KAI tell me something"))

    def test_wake_word_not_detected_in_unrelated(self):
        self.assertFalse(self.detect("the sky is blue today"))

    def test_wake_word_detected_mid_sentence(self):
        self.assertTrue(self.detect("I told kai about the project"))

    def test_wake_word_not_triggered_by_partial_match(self):
        # "kaiser" should NOT trigger — word boundary
        self.assertFalse(self.detect("I ate a kaiser roll"))

    def test_wake_word_at_end_of_sentence(self):
        self.assertTrue(self.detect("thanks kai"))

    def test_wake_word_with_punctuation(self):
        self.assertTrue(self.detect("Kai! Help me out"))

    def test_empty_input(self):
        self.assertFalse(self.detect(""))


class TestJ2IntentClassification(unittest.TestCase):
    """J2: Intent classification fallback heuristic."""

    def setUp(self):
        os.environ.setdefault("WHISPER_BACKEND", "stub")
        os.environ.setdefault("AUDIO_DIR", "/tmp/audio-test")
        os.environ.setdefault("LOG_PATH", "/tmp/test-audio.json.log")

    def test_heuristic_command_at_start(self):
        """When LLM is unavailable, 'Kai' at start → command."""
        import asyncio
        from perception.audio.app import classify_intent

        # Patch query_specialist at its source so lazy import picks up the mock
        with patch("common.llm.query_specialist", side_effect=Exception("no LLM")):
            result = asyncio.run(
                classify_intent("Kai what time is it")
            )
            self.assertEqual(result, "command")

    def test_heuristic_mention_mid_sentence(self):
        """When LLM unavailable and 'kai' not at start → unknown (not command)."""
        import asyncio
        from perception.audio.app import classify_intent

        with patch("common.llm.query_specialist", side_effect=Exception("no LLM")):
            result = asyncio.run(
                classify_intent("i was talking to someone about kai yesterday")
            )
            self.assertEqual(result, "unknown")

    def test_heuristic_command_after_comma(self):
        """'Hey, Kai' pattern → command."""
        import asyncio
        from perception.audio.app import classify_intent

        with patch("common.llm.query_specialist", side_effect=Exception("no LLM")):
            result = asyncio.run(
                classify_intent("hey, kai please help me")
            )
            self.assertEqual(result, "command")


class TestJ2Endpoints(unittest.TestCase):
    """J2: Wake-word API endpoints exist."""

    def test_wake_word_detect_endpoint_exists(self):
        from perception.audio.app import app
        routes = [r.path for r in app.routes]
        self.assertIn("/wake-word/detect", routes)

    def test_wake_word_history_endpoint_exists(self):
        from perception.audio.app import app
        routes = [r.path for r in app.routes]
        self.assertIn("/wake-word/history", routes)


# ═══════════════════════════════════════════════════════════════════════
# J3: Auto-Redaction PII
# ═══════════════════════════════════════════════════════════════════════


class TestJ3PIIDetection(unittest.TestCase):
    """J3: PII detection and redaction."""

    def setUp(self):
        from common.runtime import detect_pii, redact_pii
        self.detect = detect_pii
        self.redact = redact_pii

    def test_detect_email(self):
        counts = self.detect("contact me at john@example.com please")
        self.assertIn("email", counts)
        self.assertEqual(counts["email"], 1)

    def test_detect_multiple_emails(self):
        counts = self.detect("john@example.com and jane@test.co.uk")
        self.assertEqual(counts["email"], 2)

    def test_detect_credit_card(self):
        counts = self.detect("card number 4111-1111-1111-1111")
        self.assertIn("credit_card", counts)

    def test_detect_credit_card_spaces(self):
        counts = self.detect("card 4111 1111 1111 1111")
        self.assertIn("credit_card", counts)

    def test_detect_uk_ni_number(self):
        counts = self.detect("my NI number is AB123456C")
        self.assertIn("uk_ni_number", counts)

    def test_detect_api_token(self):
        counts = self.detect("api_key=sk_live_abc123def456ghi")
        self.assertIn("api_token", counts)

    def test_detect_no_pii_in_clean_text(self):
        counts = self.detect("the quick brown fox jumps over the lazy dog")
        self.assertEqual(len(counts), 0)

    def test_redact_email(self):
        text, counts = self.redact("email me at user@example.com")
        self.assertNotIn("user@example.com", text)
        self.assertIn("[REDACTED-EMAIL]", text)
        self.assertEqual(counts["email"], 1)

    def test_redact_credit_card(self):
        text, counts = self.redact("card: 4111-1111-1111-1111")
        self.assertNotIn("4111", text)
        self.assertIn("[REDACTED-CREDIT_CARD]", text)

    def test_redact_preserves_clean_text(self):
        original = "hello world"
        text, counts = self.redact(original)
        self.assertEqual(text, original)
        self.assertEqual(len(counts), 0)

    def test_redact_multiple_types(self):
        text, counts = self.redact(
            "email: test@test.com card: 4111-1111-1111-1111 ni: AB123456C"
        )
        self.assertIn("email", counts)
        self.assertIn("credit_card", counts)
        self.assertIn("uk_ni_number", counts)
        self.assertNotIn("test@test.com", text)

    def test_redact_uk_postcode(self):
        text, counts = self.redact("address: 10 Downing St, SW1A 2AA")
        self.assertIn("uk_postcode", counts)
        self.assertIn("[REDACTED-UK_POSTCODE]", text)

    def test_detect_bearer_token(self):
        counts = self.detect("bearer=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9")
        self.assertIn("api_token", counts)


class TestJ3VerifierEndpoint(unittest.TestCase):
    """J3: PII redaction endpoint in verifier."""

    def test_redact_endpoint_exists(self):
        from verifier.app import app
        routes = [r.path for r in app.routes]
        self.assertIn("/redact", routes)


# ═══════════════════════════════════════════════════════════════════════
# J4: Proactive Low-Latency Voice
# ═══════════════════════════════════════════════════════════════════════


class TestJ4SpeakOrNot(unittest.TestCase):
    """J4: Speak-or-not gate combining audio + video signals."""

    def setUp(self):
        from perception.camera.app import _speak_or_not
        self._speak_or_not = _speak_or_not
        # Reset cooldown
        import perception.camera.app as cam_mod
        cam_mod._last_proactive_ts = 0.0

    def test_should_speak_on_high_stress(self):
        audio = {"scores": {"stress": 0.6}, "dominant": "stress", "dominant_score": 0.6}
        video = {"brightness": 128, "motion_level": 5, "motion_detected": False}
        result = self._speak_or_not(audio, video)
        self.assertTrue(result["should_speak"])
        self.assertIn("high_stress_detected", result["reason"])

    def test_should_not_speak_on_calm(self):
        import perception.camera.app as cam_mod
        cam_mod._last_proactive_ts = 0.0
        audio = {"scores": {"calm": 0.8}, "dominant": "calm", "dominant_score": 0.8}
        video = {"brightness": 128, "motion_level": 5, "motion_detected": True}
        result = self._speak_or_not(audio, video)
        self.assertFalse(result["should_speak"])

    def test_cooldown_prevents_rapid_fire(self):
        import time
        import perception.camera.app as cam_mod
        cam_mod._last_proactive_ts = time.time()  # just triggered
        audio = {"scores": {"stress": 0.9}, "dominant": "stress", "dominant_score": 0.9}
        video = {"brightness": 10, "motion_level": 0, "motion_detected": False}
        result = self._speak_or_not(audio, video)
        self.assertFalse(result["should_speak"])
        self.assertEqual(result["reason"], "cooldown_active")

    def test_dark_environment_adds_urgency(self):
        import perception.camera.app as cam_mod
        cam_mod._last_proactive_ts = 0.0
        audio = {"scores": {"fatigue": 0.3}, "dominant": "fatigue", "dominant_score": 0.3}
        video = {"brightness": 20, "motion_level": 0, "motion_detected": False}
        result = self._speak_or_not(audio, video)
        self.assertGreater(result["urgency"], 0.3)

    def test_fatigue_triggers_speech(self):
        import perception.camera.app as cam_mod
        cam_mod._last_proactive_ts = 0.0
        audio = {"scores": {"fatigue": 0.5}, "dominant": "fatigue", "dominant_score": 0.5}
        video = {"brightness": 50, "motion_level": 0, "motion_detected": False}
        result = self._speak_or_not(audio, video)
        self.assertTrue(result["should_speak"])

    def test_suggested_message_present_when_speaking(self):
        import perception.camera.app as cam_mod
        cam_mod._last_proactive_ts = 0.0
        audio = {"scores": {"stress": 0.8}, "dominant": "stress", "dominant_score": 0.8}
        video = {"brightness": 128, "motion_level": 5, "motion_detected": True}
        result = self._speak_or_not(audio, video)
        self.assertIsNotNone(result["suggested_message"])


class TestJ4Endpoints(unittest.TestCase):
    """J4: Proactive voice endpoints exist."""

    def test_proactive_evaluate_endpoint(self):
        from perception.camera.app import app
        routes = [r.path for r in app.routes]
        self.assertIn("/proactive/evaluate", routes)

    def test_proactive_auto_endpoint(self):
        from perception.camera.app import app
        routes = [r.path for r in app.routes]
        self.assertIn("/proactive/auto", routes)


# ═══════════════════════════════════════════════════════════════════════
# J6: SOUL.md + AGENTS.md
# ═══════════════════════════════════════════════════════════════════════


class TestJ6SoulAndAgents(unittest.TestCase):
    """J6: SOUL.md and AGENTS.md identity files."""

    def test_soul_file_exists(self):
        soul_path = Path(__file__).parent.parent / "data" / "SOUL.md"
        self.assertTrue(soul_path.exists(), "data/SOUL.md must exist")

    def test_soul_contains_core_values(self):
        soul_path = Path(__file__).parent.parent / "data" / "SOUL.md"
        content = soul_path.read_text()
        self.assertIn("Core Values", content)
        self.assertIn("Honesty", content)
        self.assertIn("Loyalty", content)

    def test_soul_contains_personality(self):
        soul_path = Path(__file__).parent.parent / "data" / "SOUL.md"
        content = soul_path.read_text()
        self.assertIn("Personality", content)
        self.assertIn("WORK mode", content)
        self.assertIn("PUB mode", content)

    def test_agents_file_exists(self):
        agents_path = Path(__file__).parent.parent / "data" / "AGENTS.md"
        self.assertTrue(agents_path.exists(), "data/AGENTS.md must exist")

    def test_agents_contains_registry(self):
        agents_path = Path(__file__).parent.parent / "data" / "AGENTS.md"
        content = agents_path.read_text()
        self.assertIn("Active Agents", content)
        self.assertIn("Memory Recall", content)
        self.assertIn("Tax Advisory", content)

    def test_agents_contains_wake_word_agent(self):
        agents_path = Path(__file__).parent.parent / "data" / "AGENTS.md"
        content = agents_path.read_text()
        self.assertIn("Wake-word", content)

    def test_agents_contains_pii_redactor(self):
        agents_path = Path(__file__).parent.parent / "data" / "AGENTS.md"
        content = agents_path.read_text()
        self.assertIn("PII Redactor", content)


class TestJ6LanggraphEndpoints(unittest.TestCase):
    """J6: SOUL/AGENTS endpoints in langgraph."""

    @classmethod
    def setUpClass(cls):
        # langgraph/app.py uses local imports like kai_config
        langgraph_dir = os.path.join(os.path.dirname(__file__), "..", "langgraph")
        if langgraph_dir not in sys.path:
            sys.path.insert(0, langgraph_dir)

    def test_soul_endpoint_exists(self):
        os.environ.setdefault("LOG_PATH", "/tmp/test-langgraph.json.log")
        from langgraph.app import app
        routes = [r.path for r in app.routes]
        self.assertIn("/soul", routes)

    def test_agents_registry_endpoint_exists(self):
        from langgraph.app import app
        routes = [r.path for r in app.routes]
        self.assertIn("/agents-registry", routes)


class TestJ6SoulLoader(unittest.TestCase):
    """J6: SOUL.md loader function."""

    @classmethod
    def setUpClass(cls):
        langgraph_dir = os.path.join(os.path.dirname(__file__), "..", "langgraph")
        if langgraph_dir not in sys.path:
            sys.path.insert(0, langgraph_dir)

    def test_load_soul_returns_content(self):
        os.environ.setdefault("LOG_PATH", "/tmp/test-langgraph.json.log")
        from langgraph.app import _load_soul, _soul_text
        result = _load_soul()
        # Should find data/SOUL.md (we're running from repo root via PYTHONPATH=.)
        self.assertTrue(len(result) > 0 or len(_soul_text) > 0)


# ═══════════════════════════════════════════════════════════════════════
# J7: Skills Auto-Install Hub
# ═══════════════════════════════════════════════════════════════════════


class TestJ7SkillLoader(unittest.TestCase):
    """J7: Skill file parser and loader."""

    def test_parse_skill_file(self):
        from langgraph.router import _parse_skill_file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
            f.write(
                "# Skill: Test Skill\n\n"
                "## Trigger patterns\n"
                '- "do the thing"\n'
                '- "make it happen"\n\n'
                "## Action\n"
                "POST /test/endpoint\n\n"
                "## Response template\n"
                "Done: {result}\n"
            )
            f.flush()
            skill = _parse_skill_file(Path(f.name))
        os.unlink(f.name)

        self.assertIsNotNone(skill)
        self.assertEqual(skill.name, "Test Skill")
        self.assertEqual(len(skill.trigger_patterns), 2)
        self.assertIn("do the thing", skill.trigger_strings)
        self.assertIn("POST /test/endpoint", skill.action)

    def test_parse_skill_file_no_triggers_returns_none(self):
        from langgraph.router import _parse_skill_file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
            f.write("# Skill: Empty\n\n## Action\nNothing\n")
            f.flush()
            skill = _parse_skill_file(Path(f.name))
        os.unlink(f.name)
        self.assertIsNone(skill)

    def test_match_skill(self):
        from langgraph.router import _parse_skill_file, _loaded_skills, match_skill
        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
            f.write(
                "# Skill: Greeter\n\n"
                "## Trigger patterns\n"
                '- "good morning"\n'
                '- "hello there"\n\n'
                "## Action\n"
                "Return greeting\n"
            )
            f.flush()
            skill = _parse_skill_file(Path(f.name))
        os.unlink(f.name)

        _loaded_skills.clear()
        _loaded_skills.append(skill)

        matched = match_skill("good morning everyone")
        self.assertIsNotNone(matched)
        self.assertEqual(matched.name, "Greeter")

        no_match = match_skill("goodbye")
        self.assertIsNone(no_match)

    def test_load_skills_finds_sample_skills(self):
        from langgraph.router import load_skills
        skills = load_skills()
        # Should find daily_brief.md and invoice.md in data/skills/
        names = [s.name for s in skills]
        self.assertIn("Daily Brief", names)
        self.assertIn("Invoice Generator", names)

    def test_list_skills_returns_dicts(self):
        from langgraph.router import load_skills, list_skills
        load_skills()
        result = list_skills()
        self.assertIsInstance(result, list)
        if result:
            self.assertIn("name", result[0])
            self.assertIn("triggers", result[0])


class TestJ7Endpoints(unittest.TestCase):
    """J7: Skills Hub API endpoints."""

    @classmethod
    def setUpClass(cls):
        langgraph_dir = os.path.join(os.path.dirname(__file__), "..", "langgraph")
        if langgraph_dir not in sys.path:
            sys.path.insert(0, langgraph_dir)

    def test_skills_endpoint_exists(self):
        os.environ.setdefault("LOG_PATH", "/tmp/test-langgraph.json.log")
        from langgraph.app import app
        routes = [r.path for r in app.routes]
        self.assertIn("/skills", routes)

    def test_skills_reload_endpoint_exists(self):
        from langgraph.app import app
        routes = [r.path for r in app.routes]
        self.assertIn("/skills/reload", routes)

    def test_skills_match_endpoint_exists(self):
        from langgraph.app import app
        routes = [r.path for r in app.routes]
        self.assertIn("/skills/match", routes)


# ═══════════════════════════════════════════════════════════════════════
# J1: Live Canvas Visualization (dashboard HTML structure)
# ═══════════════════════════════════════════════════════════════════════


class TestJ1LiveCanvas(unittest.TestCase):
    """J1: Canvas view exists in dashboard HTML."""

    def setUp(self):
        self.html_path = Path(__file__).parent.parent / "dashboard" / "static" / "app.html"
        self.html = self.html_path.read_text()

    def test_canvas_nav_item_exists(self):
        self.assertIn('data-view="canvas"', self.html)

    def test_canvas_view_section_exists(self):
        self.assertIn('id="canvasView"', self.html)

    def test_canvas_element_exists(self):
        self.assertIn('id="liveCanvas"', self.html)

    def test_canvas_mode_buttons_exist(self):
        self.assertIn('btnMindmap', self.html)
        self.assertIn('btnEmotion', self.html)
        self.assertIn('btnPlan', self.html)

    def test_canvas_js_functions_exist(self):
        self.assertIn('function canvasMode(', self.html)
        self.assertIn('function refreshCanvas(', self.html)
        self.assertIn('function drawMindMap(', self.html)
        self.assertIn('function drawEmotionTimeline(', self.html)
        self.assertIn('function drawPlanFlow(', self.html)

    def test_canvas_uses_dom_purify_for_xss(self):
        """Ensure DOMPurify is used for XSS protection."""
        self.assertIn('DOMPurify', self.html)


# ═══════════════════════════════════════════════════════════════════════
# J5: Memory Viewer GUI (dashboard HTML structure)
# ═══════════════════════════════════════════════════════════════════════


class TestJ5MemoryDiary(unittest.TestCase):
    """J5: Memory Diary view exists in dashboard HTML."""

    def setUp(self):
        self.html_path = Path(__file__).parent.parent / "dashboard" / "static" / "app.html"
        self.html = self.html_path.read_text()

    def test_diary_nav_item_exists(self):
        self.assertIn('data-view="diary"', self.html)

    def test_diary_view_section_exists(self):
        self.assertIn('id="diaryView"', self.html)

    def test_diary_search_input_exists(self):
        self.assertIn('id="diarySearch"', self.html)

    def test_diary_category_filter_exists(self):
        self.assertIn('id="diaryCat"', self.html)

    def test_diary_importance_slider_exists(self):
        self.assertIn('id="diaryImportance"', self.html)

    def test_diary_stats_cards_exist(self):
        self.assertIn('id="diaryTotal"', self.html)
        self.assertIn('id="diaryCategories"', self.html)
        self.assertIn('id="diaryShown"', self.html)

    def test_diary_js_functions_exist(self):
        self.assertIn('function loadDiaryStats(', self.html)
        self.assertIn('function searchDiary(', self.html)

    def test_diary_sanitizes_output(self):
        """Ensure DOMPurify is used for output sanitization."""
        # Count DOMPurify.sanitize usages in diary section
        self.assertIn('DOMPurify.sanitize(cat)', self.html)


if __name__ == "__main__":
    unittest.main()
