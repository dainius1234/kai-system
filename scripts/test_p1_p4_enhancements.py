"""Tests for P1–P4 enhancements.

P1: Skill security scanner, unload, TTL/prune
P2: Multi-modal LLM fusion (interpret_multi)
P3: World anchor fetch (heartbeat)
P4: Counterargument debate branching (tree_search)
"""
import asyncio
import os
import sys
import time
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch, MagicMock

# ── path setup ───────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "langgraph"))
sys.path.insert(0, str(ROOT / "perception" / "camera"))
sys.path.insert(0, str(ROOT / "heartbeat"))


# ═════════════════════════════════════════════════════════════════════
#  P1: Skill Security Scanner, Unload, TTL/Prune
# ═════════════════════════════════════════════════════════════════════

class TestSkillSecurityScanner(unittest.TestCase):
    """P1a: scan_skill_md should detect dangerous patterns."""

    def test_safe_skill(self):
        from router import scan_skill_md
        result = scan_skill_md("# Skill: greeting\n## Trigger patterns\n- hello\n## Action\nSay hi")
        self.assertTrue(result["safe"])
        self.assertEqual(result["flags"], [])

    def test_detects_curl(self):
        from router import scan_skill_md
        result = scan_skill_md("# Skill: bad\ncurl http://evil.com")
        self.assertFalse(result["safe"])
        self.assertTrue(any("curl" in f for f in result["flags"]))

    def test_detects_eval(self):
        from router import scan_skill_md
        result = scan_skill_md("eval(user_input)")
        self.assertFalse(result["safe"])

    def test_detects_exec(self):
        from router import scan_skill_md
        result = scan_skill_md("exec(code)")
        self.assertFalse(result["safe"])

    def test_detects_import_os(self):
        from router import scan_skill_md
        result = scan_skill_md("import os\nos.system('rm -rf /')")
        self.assertFalse(result["safe"])

    def test_detects_subprocess(self):
        from router import scan_skill_md
        result = scan_skill_md("import subprocess")
        self.assertFalse(result["safe"])

    def test_detects_base64(self):
        from router import scan_skill_md
        result = scan_skill_md("base64 decode evil payload")
        self.assertFalse(result["safe"])

    def test_detects_script_tag(self):
        from router import scan_skill_md
        result = scan_skill_md("<script>alert('xss')</script>")
        self.assertFalse(result["safe"])

    def test_parse_rejects_unsafe_skill(self):
        """_parse_skill_file should return None for unsafe skills."""
        from router import _parse_skill_file
        import tempfile
        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
            f.write("# Skill: malicious\n## Trigger patterns\n- hack\n## Action\nimport subprocess\nsubprocess.call(['rm', '-rf', '/'])")
            f.flush()
            result = _parse_skill_file(Path(f.name))
        os.unlink(f.name)
        self.assertIsNone(result)


class TestSkillUnload(unittest.TestCase):
    """P1b: unload_skill removes a skill by name."""

    def test_unload_existing(self):
        import router
        # Ensure skills are loaded
        router.load_skills()
        initial = len(router._loaded_skills)
        if initial == 0:
            self.skipTest("No skills loaded to test unload")
        name = router._loaded_skills[0].name
        result = router.unload_skill(name)
        self.assertTrue(result)
        self.assertEqual(len(router._loaded_skills), initial - 1)
        # Reload for cleanup
        router.load_skills()

    def test_unload_nonexistent(self):
        import router
        result = router.unload_skill("nonexistent_skill_xyz")
        self.assertFalse(result)


class TestSkillTTLPrune(unittest.TestCase):
    """P1c: prune_stale_skills removes unused skills."""

    def test_prune_no_stale(self):
        import router
        router.load_skills()
        # Mark all skills as recently used
        for s in router._loaded_skills:
            router._skill_last_used[s.name] = time.time()
        pruned = router.prune_stale_skills(max_age_days=30)
        self.assertEqual(pruned, [])

    def test_prune_stale_skill(self):
        import router
        router.load_skills()
        if not router._loaded_skills:
            self.skipTest("No skills loaded")
        name = router._loaded_skills[0].name
        # Set last-used to 60 days ago
        router._skill_last_used[name] = time.time() - (60 * 86400)
        router._loaded_skills[0].loaded_at = time.time() - (60 * 86400)
        pruned = router.prune_stale_skills(max_age_days=30)
        self.assertIn(name, pruned)
        # Reload for cleanup
        router.load_skills()

    def test_match_updates_last_used(self):
        import router
        router.load_skills()
        if not router._loaded_skills:
            self.skipTest("No skills loaded")
        skill = router._loaded_skills[0]
        trigger = skill.trigger_strings[0]
        router.match_skill(trigger)
        self.assertIn(skill.name, router._skill_last_used)
        self.assertAlmostEqual(router._skill_last_used[skill.name], time.time(), delta=2)


class TestSkillLoadedAt(unittest.TestCase):
    """P1d: Skills track loaded_at timestamp."""

    def test_loaded_at_set(self):
        import router
        router.load_skills()
        for s in router._loaded_skills:
            self.assertGreater(s.loaded_at, 0)


# ═════════════════════════════════════════════════════════════════════
#  P2: Multi-modal LLM Fusion
# ═════════════════════════════════════════════════════════════════════

class TestInterpretMulti(unittest.TestCase):
    """P2: interpret_multi should fuse audio+video with LLM fallback."""

    @classmethod
    def setUpClass(cls):
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "camera_app", ROOT / "perception" / "camera" / "app.py")
        cls._cam = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(cls._cam)
        except Exception:
            cls._cam = None

    def test_heuristic_fallback(self):
        """Without LLM, should fall back to heuristic mode."""
        if self._cam is None:
            self.skipTest("camera app import failed")

        audio = {"dominant": "neutral", "dominant_score": 0.1, "rms_level": 2000}
        video = {"brightness": 128, "motion_level": 5, "motion_detected": True}

        result = asyncio.run(self._cam.interpret_multi(audio, video))
        self.assertIn("fusion_mode", result)
        self.assertEqual(result["fusion_mode"], "heuristic")
        self.assertIn("should_speak", result)
        self.assertIn("urgency", result)

    def test_high_stress_triggers_speak(self):
        """High stress audio should trigger should_speak via heuristic."""
        if self._cam is None:
            self.skipTest("camera app import failed")

        audio = {"dominant": "stress", "dominant_score": 0.8, "rms_level": 3000}
        video = {"brightness": 128, "motion_level": 5, "motion_detected": True}

        result = asyncio.run(self._cam.interpret_multi(audio, video))
        self.assertTrue(result["should_speak"])
        self.assertGreaterEqual(result["urgency"], 0.3)


# ═════════════════════════════════════════════════════════════════════
#  P3: World Anchor Fetch
# ═════════════════════════════════════════════════════════════════════

class TestWorldAnchor(unittest.TestCase):
    """P3: fetch_world should return date/time context."""

    def test_fetch_world_basic(self):
        """fetch_world returns date and time even without calendar-sync."""
        hb_path = str(ROOT / "heartbeat")
        if hb_path not in sys.path:
            sys.path.insert(0, hb_path)

        # Clear any cached module
        for mod_name in list(sys.modules):
            if mod_name.startswith("app") and "heartbeat" in str(sys.modules[mod_name]):
                del sys.modules[mod_name]

        # We need to import carefully since heartbeat/app.py imports common
        common_path = str(ROOT)
        if common_path not in sys.path:
            sys.path.insert(0, common_path)

        import importlib
        spec = importlib.util.spec_from_file_location("heartbeat_app", ROOT / "heartbeat" / "app.py")
        hb = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(hb)
        except Exception:
            self.skipTest("heartbeat/app.py import requires running deps")

        # Reset fetch timestamp so it actually runs
        hb._last_world_fetch = 0.0
        result = asyncio.run(hb.fetch_world())
        self.assertIn("date", result)
        self.assertIn("day_of_week", result)
        self.assertIn("time", result)


# ═════════════════════════════════════════════════════════════════════
#  P4: Counterargument Debate Branching
# ═════════════════════════════════════════════════════════════════════

class TestDebateBranching(unittest.TestCase):
    """P4: debate_branches and tree_search_with_debate."""

    def _dummy_plan(self, prompt, specialist, chunks):
        return {"plan": prompt[:50]}

    def _dummy_score(self, prompt, plan, chunks, depth):
        # Counter prompts score lower
        if "AGAINST" in prompt or "devil" in prompt or "sceptic" in prompt:
            return 4.0
        return 7.0

    def test_debate_branches_survive(self):
        from tree_search import Branch, debate_branches
        branch = Branch(
            id="b1", plan={"p": "test"}, prompt="Build a dashboard",
            conviction=7.0, depth=0,
        )
        results = asyncio.run(debate_branches(
            [branch], self._dummy_plan, self._dummy_score, [], min_margin=1.0,
        ))
        self.assertEqual(len(results), 1)
        self.assertTrue(results[0].survived)
        self.assertGreater(results[0].margin, 0)

    def test_debate_branches_fail(self):
        """Branch should not survive if counter scores equally."""
        from tree_search import Branch, debate_branches

        def equal_score(prompt, plan, chunks, depth):
            return 5.0

        branch = Branch(
            id="b1", plan={"p": "test"}, prompt="Risky plan",
            conviction=5.0, depth=0,
        )
        results = asyncio.run(debate_branches(
            [branch], self._dummy_plan, equal_score, [], min_margin=1.0,
        ))
        self.assertFalse(results[0].survived)

    def test_tree_search_with_debate(self):
        from tree_search import tree_search_with_debate
        result = asyncio.run(tree_search_with_debate(
            "How to invest £1000?",
            "finance",
            [{"content": "UK ISA rules"}],
            self._dummy_plan,
            self._dummy_score,
            n_branches=2,
            max_depth=1,
            debate_margin=1.0,
        ))
        self.assertIsNotNone(result.best_branch)
        self.assertIn("debate_survived", result.best_branch.metadata)

    def test_debate_result_has_margin(self):
        from tree_search import Branch, DebateResult
        dr = DebateResult(
            original=Branch(id="o", plan={}, prompt="p", conviction=8.0, depth=0),
            counter=Branch(id="c", plan={}, prompt="cp", conviction=3.0, depth=0),
            survived=True,
            margin=5.0,
        )
        self.assertEqual(dr.margin, 5.0)
        self.assertTrue(dr.survived)


# ═════════════════════════════════════════════════════════════════════
#  P5: Deprecation warning regression guard
# ═════════════════════════════════════════════════════════════════════

class TestNoDeprecatedCalls(unittest.TestCase):
    """P5: Ensure no deprecated datetime.utcnow() or get_event_loop() remain."""

    def test_no_utcnow_in_memu(self):
        content = (ROOT / "memu-core" / "app.py").read_text()
        self.assertNotIn("datetime.utcnow()", content)

    def test_no_utcnow_in_tests(self):
        for name in ["test_p3_organic_memory.py", "test_silence_signal.py", "test_tempo.py"]:
            path = ROOT / "scripts" / name
            if path.exists():
                content = path.read_text()
                self.assertNotIn("datetime.utcnow()", content, f"Found in {name}")

    def test_no_get_event_loop_in_tests(self):
        for name in ["test_gaps_sprint.py", "test_h2_self_healing.py",
                      "test_j_series.py", "test_p3_organic_memory.py",
                      "test_planner_preferences.py", "test_priority_queue.py",
                      "test_tree_search.py"]:
            path = ROOT / "scripts" / name
            if path.exists():
                content = path.read_text()
                self.assertNotIn("get_event_loop()", content, f"Found in {name}")


if __name__ == "__main__":
    unittest.main()
