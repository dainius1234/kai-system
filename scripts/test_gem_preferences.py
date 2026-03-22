"""Tests for P5 — GEM Cognitive Alignment (preference extraction).

Validates extract_preference() in kai_config and the planner's
preference injection path.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "langgraph"))

# Redis is optional — stub if not available
if "redis" not in sys.modules:
    from types import ModuleType
    _r = ModuleType("redis")

    class _FakeRedis:
        @classmethod
        def from_url(cls, *a, **kw):
            return cls()

        def ping(self):
            return True
    _r.Redis = _FakeRedis
    sys.modules["redis"] = _r

from kai_config import extract_preference, TopicBoundary, build_knowledge_boundary


class TestPreferenceExtraction(unittest.TestCase):
    """P5 GEM: extract_preference() from corrections."""

    def test_extracts_preference_from_correction(self):
        """When Kai's output is corrected, should produce a preference."""
        original = "The drawdown limit is 5% with a monthly review cycle."
        correction = "Actually the drawdown limit is 10% with quarterly review."
        user_input = "What is the drawdown limit?"
        pref = extract_preference(original, correction, user_input)
        self.assertIsNotNone(pref)
        self.assertIn("keeper", pref.lower())

    def test_returns_none_for_empty_correction(self):
        pref = extract_preference("some output", "", "some input")
        self.assertIsNone(pref)

    def test_returns_none_for_empty_original(self):
        pref = extract_preference("", "some correction", "some input")
        self.assertIsNone(pref)

    def test_returns_none_for_identical_text(self):
        pref = extract_preference("same text here", "same text here", "ask about X")
        self.assertIsNone(pref)

    def test_preference_includes_topic_context(self):
        """Preference should mention topic keywords."""
        original = "The VAT registration threshold is £85,000 per year."
        correction = "The VAT registration threshold is now £90,000 per year."
        user_input = "What is the current VAT registration threshold?"
        pref = extract_preference(original, correction, user_input)
        self.assertIsNotNone(pref)
        self.assertIn("topic=", pref.lower())

    def test_correction_with_added_words(self):
        """Correction adds new terms → 'wants emphasis on'."""
        original = "Set risk limit to medium."
        correction = "Set risk limit to medium with daily reporting and alerts."
        user_input = "How should I set risk limits?"
        pref = extract_preference(original, correction, user_input)
        self.assertIsNotNone(pref)

    def test_correction_with_removed_words(self):
        """Correction removes terms → 'does NOT want'."""
        original = "Use aggressive trading strategy with maximum leverage."
        correction = "Use trading strategy."
        user_input = "What trading strategy?"
        pref = extract_preference(original, correction, user_input)
        self.assertIsNotNone(pref)


class TestKnowledgeBoundary(unittest.TestCase):
    """P6: build_knowledge_boundary() from episode history."""

    def _make_episodes(self, topic_words, count, success_rate=0.5, avg_conviction=5.0):
        """Generate synthetic episodes for a topic."""
        import time
        episodes = []
        for i in range(count):
            is_success = i < int(count * success_rate)
            episodes.append({
                "episode_id": f"ep-{topic_words}-{i}",
                "input": f"Tell me about {topic_words} in detail please",
                "output": f"Here is info about {topic_words}",
                "ts": time.time() - (i * 86400),
                "outcome_score": 0.9 if is_success else 0.2,
                "final_conviction": avg_conviction,
                "learning_value": 0.5,
            })
        return episodes

    def test_identifies_knowledge_gaps(self):
        """Topics with low success rate should be flagged as gaps."""
        episodes = (
            self._make_episodes("construction safety", 4, success_rate=0.25, avg_conviction=4.0)
            + self._make_episodes("executor policies", 4, success_rate=0.9, avg_conviction=8.0)
        )
        boundaries = build_knowledge_boundary(episodes, min_episodes=2)
        gaps = [b for b in boundaries if b.is_gap]
        non_gaps = [b for b in boundaries if not b.is_gap]
        self.assertGreater(len(gaps), 0, "Should have at least one gap")
        self.assertGreater(len(non_gaps), 0, "Should have at least one non-gap")

    def test_empty_episodes(self):
        """No episodes = no boundaries."""
        boundaries = build_knowledge_boundary([], min_episodes=2)
        self.assertEqual(len(boundaries), 0)

    def test_min_episodes_filter(self):
        """Topics with fewer episodes than min_episodes are excluded."""
        episodes = self._make_episodes("rare topic testing something", 1, success_rate=1.0)
        boundaries = build_knowledge_boundary(episodes, min_episodes=2)
        self.assertEqual(len(boundaries), 0)

    def test_gap_has_probe_question(self):
        """Knowledge gaps should have a non-empty probe question."""
        episodes = self._make_episodes("unknown domain area", 5, success_rate=0.2, avg_conviction=3.0)
        boundaries = build_knowledge_boundary(episodes, min_episodes=2)
        gaps = [b for b in boundaries if b.is_gap]
        for g in gaps:
            self.assertNotEqual(g.probe_question, "", f"Gap {g.topic} missing probe question")

    def test_gaps_sorted_first(self):
        """Gaps appear before non-gaps in the result."""
        episodes = (
            self._make_episodes("weak topic area", 3, success_rate=0.0, avg_conviction=3.0)
            + self._make_episodes("strong topic area", 3, success_rate=1.0, avg_conviction=9.0)
        )
        boundaries = build_knowledge_boundary(episodes, min_episodes=2)
        if len(boundaries) >= 2:
            first_is_gap = boundaries[0].is_gap
            self.assertTrue(first_is_gap, "Gaps should be sorted to the front")

    def test_topic_boundary_fields(self):
        """TopicBoundary dataclass should have all expected fields."""
        b = TopicBoundary(topic="test")
        self.assertEqual(b.total_episodes, 0)
        self.assertEqual(b.successes, 0)
        self.assertEqual(b.failures, 0)
        self.assertEqual(b.avg_conviction, 0.0)
        self.assertFalse(b.is_gap)
        self.assertEqual(b.probe_question, "")


if __name__ == "__main__":
    unittest.main()
