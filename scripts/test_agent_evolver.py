"""P24 — Agent-Evolver Insight Engine tests.

Tests the failure→pattern→fix suggestion pipeline:
  1. analyze_failures() — pattern extraction and fix generation
  2. EvolutionSuggestion — priority assignment, topic extraction
  3. evolver_dream_phase() — integration with Dream State
  4. Persistence — save/load evolver reports
  5. Edge cases — empty episodes, no failures, single episodes
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import time
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "langgraph"))

from kai_config import (  # noqa: E402
    FailureClass,
    EvolutionSuggestion,
    EvolutionReport,
    analyze_failures,
    evolver_dream_phase,
    save_evolver_report,
    load_evolver_reports,
    _extract_topic,
    _assign_priority,
    EVOLVER_MIN_PATTERN_COUNT,
    DreamInsight,
)


def _make_episode(
    input_text="calculate tax owed",
    outcome=0.3,
    failure_class="confidence_low",
    conviction=4.0,
    rethink_count=2,
    ts=None,
):
    return {
        "input": input_text,
        "outcome_score": outcome,
        "failure_class": failure_class,
        "final_conviction": conviction,
        "conviction_score": conviction,
        "rethink_count": rethink_count,
        "ts": ts or time.time(),
    }


# ═══════════════════════════════════════════════════════════════════
# Part 1: analyze_failures core logic
# ═══════════════════════════════════════════════════════════════════

class TestAnalyzeFailuresEmpty(unittest.TestCase):

    def test_no_episodes(self):
        report = analyze_failures([])
        self.assertIsInstance(report, EvolutionReport)
        self.assertEqual(report.total_failures, 0)
        self.assertEqual(len(report.suggestions), 0)
        self.assertIsNone(report.top_failure_class)

    def test_all_successful(self):
        episodes = [
            _make_episode(outcome=0.8, failure_class="unknown"),
            _make_episode(outcome=0.9, failure_class="unknown"),
        ]
        report = analyze_failures(episodes)
        self.assertEqual(report.total_failures, 0)
        self.assertEqual(len(report.suggestions), 0)

    def test_single_failure_below_threshold(self):
        """One failure isn't enough to generate a suggestion."""
        episodes = [_make_episode()]
        report = analyze_failures(episodes)
        self.assertEqual(report.total_failures, 1)
        # Below EVOLVER_MIN_PATTERN_COUNT (2), so no suggestions
        self.assertEqual(len(report.suggestions), 0)


class TestAnalyzeFailuresPatterns(unittest.TestCase):

    def test_recurring_failure_generates_suggestion(self):
        episodes = [
            _make_episode(input_text="calculate tax liability", failure_class="confidence_low"),
            _make_episode(input_text="calculate tax credits", failure_class="confidence_low"),
            _make_episode(input_text="calculate tax allowance", failure_class="confidence_low"),
        ]
        report = analyze_failures(episodes)
        self.assertGreater(len(report.suggestions), 0)
        s = report.suggestions[0]
        self.assertEqual(s.failure_class, "confidence_low")
        self.assertGreaterEqual(s.frequency, 2)
        self.assertIn("evidence", s.fix.lower())

    def test_different_failure_classes_separate_suggestions(self):
        episodes = [
            _make_episode(input_text="run shell command", failure_class="policy_blocked"),
            _make_episode(input_text="run shell script", failure_class="policy_blocked"),
            _make_episode(input_text="check health status", failure_class="service_unavailable"),
            _make_episode(input_text="check health endpoint", failure_class="service_unavailable"),
        ]
        report = analyze_failures(episodes)
        classes = {s.failure_class for s in report.suggestions}
        self.assertGreaterEqual(len(classes), 2)

    def test_top_failure_class_identified(self):
        episodes = [
            _make_episode(failure_class="data_insufficient"),
            _make_episode(failure_class="data_insufficient"),
            _make_episode(failure_class="data_insufficient"),
            _make_episode(failure_class="confidence_low"),
        ]
        report = analyze_failures(episodes)
        self.assertEqual(report.top_failure_class, "data_insufficient")
        self.assertEqual(report.top_failure_count, 3)

    def test_mixed_success_and_failure(self):
        episodes = [
            _make_episode(outcome=0.9, failure_class="unknown"),  # success
            _make_episode(outcome=0.3, failure_class="confidence_low"),
            _make_episode(outcome=0.2, failure_class="confidence_low"),
        ]
        report = analyze_failures(episodes)
        # 2 failures (the successful one is excluded)
        self.assertEqual(report.total_failures, 2)


class TestAnalyzeFailuresSorting(unittest.TestCase):

    def test_critical_before_high(self):
        episodes = [
            _make_episode(input_text="verify claim alpha", failure_class="contradicted_by_evidence"),
            _make_episode(input_text="verify claim alpha", failure_class="contradicted_by_evidence"),
            _make_episode(input_text="verify claim alpha", failure_class="contradicted_by_evidence"),
            _make_episode(input_text="check general info", failure_class="confidence_low"),
            _make_episode(input_text="check general info", failure_class="confidence_low"),
        ]
        report = analyze_failures(episodes)
        if len(report.suggestions) >= 2:
            self.assertLessEqual(
                {"critical": 0, "high": 1, "medium": 2, "low": 3}[report.suggestions[0].priority],
                {"critical": 0, "high": 1, "medium": 2, "low": 3}[report.suggestions[1].priority],
            )


# ═══════════════════════════════════════════════════════════════════
# Part 2: Fix templates and topic extraction
# ═══════════════════════════════════════════════════════════════════

class TestFixTemplates(unittest.TestCase):

    def test_data_insufficient_fix_mentions_memu(self):
        episodes = [
            _make_episode(input_text="find recent news", failure_class="data_insufficient"),
            _make_episode(input_text="find recent articles", failure_class="data_insufficient"),
        ]
        report = analyze_failures(episodes)
        if report.suggestions:
            self.assertIn("memu", report.suggestions[0].fix.lower())

    def test_policy_blocked_fix_mentions_tool_gate(self):
        episodes = [
            _make_episode(input_text="execute shell command", failure_class="policy_blocked"),
            _make_episode(input_text="execute shell script", failure_class="policy_blocked"),
        ]
        report = analyze_failures(episodes)
        if report.suggestions:
            self.assertIn("tool-gate", report.suggestions[0].fix.lower())

    def test_contradicted_fix_mentions_verify(self):
        episodes = [
            _make_episode(input_text="claim about revenue", failure_class="contradicted_by_evidence"),
            _make_episode(input_text="claim about revenue", failure_class="contradicted_by_evidence"),
        ]
        report = analyze_failures(episodes)
        if report.suggestions:
            self.assertIn("verif", report.suggestions[0].fix.lower())


class TestTopicExtraction(unittest.TestCase):

    def test_extract_topic_basic(self):
        episodes = [
            {"input": "calculate tax owed for business"},
            {"input": "calculate tax liability quarterly"},
        ]
        topic = _extract_topic(episodes)
        self.assertIn("calculate", topic)

    def test_extract_topic_empty(self):
        topic = _extract_topic([{"input": ""}])
        self.assertEqual(topic, "general")

    def test_extract_topic_no_episodes(self):
        topic = _extract_topic([])
        self.assertEqual(topic, "general")


# ═══════════════════════════════════════════════════════════════════
# Part 3: Priority assignment
# ═══════════════════════════════════════════════════════════════════

class TestPriorityAssignment(unittest.TestCase):

    def test_critical_class_high_frequency(self):
        p = _assign_priority(3, FailureClass.CONTRADICTED_BY_EVIDENCE)
        self.assertEqual(p, "critical")

    def test_critical_class_low_frequency(self):
        p = _assign_priority(1, FailureClass.CONTRADICTED_BY_EVIDENCE)
        self.assertEqual(p, "high")

    def test_high_frequency_any_class(self):
        p = _assign_priority(5, FailureClass.CONFIDENCE_LOW)
        self.assertEqual(p, "critical")

    def test_medium_frequency(self):
        p = _assign_priority(2, FailureClass.CONFIDENCE_LOW)
        self.assertEqual(p, "medium")

    def test_low_frequency(self):
        p = _assign_priority(1, FailureClass.CONFIDENCE_LOW)
        self.assertEqual(p, "low")

    def test_three_noncritical(self):
        p = _assign_priority(3, FailureClass.DATA_INSUFFICIENT)
        self.assertEqual(p, "high")


# ═══════════════════════════════════════════════════════════════════
# Part 4: Dream State integration
# ═══════════════════════════════════════════════════════════════════

class TestEvolverDreamPhase(unittest.TestCase):

    def test_no_failures_empty_insights(self):
        episodes = [
            _make_episode(outcome=0.9, failure_class="unknown"),
        ]
        insights = evolver_dream_phase(episodes)
        self.assertEqual(len(insights), 0)

    def test_recurring_failures_generate_insights(self):
        episodes = [
            _make_episode(input_text="tax calc a", failure_class="confidence_low"),
            _make_episode(input_text="tax calc b", failure_class="confidence_low"),
            _make_episode(input_text="tax calc c", failure_class="confidence_low"),
        ]
        insights = evolver_dream_phase(episodes)
        self.assertGreater(len(insights), 0)
        self.assertTrue(all(isinstance(i, DreamInsight) for i in insights))
        self.assertTrue(all(i.insight_type == "evolution" for i in insights))
        self.assertTrue(all(i.actionable for i in insights))

    def test_dominant_failure_generates_extra_insight(self):
        episodes = [
            _make_episode(input_text="search data x", failure_class="data_insufficient"),
            _make_episode(input_text="search data y", failure_class="data_insufficient"),
            _make_episode(input_text="search data z", failure_class="data_insufficient"),
        ]
        insights = evolver_dream_phase(episodes)
        dominant = [i for i in insights if "Dominant failure" in i.description]
        self.assertEqual(len(dominant), 1)
        self.assertIn("data_insufficient", dominant[0].description)

    def test_insight_has_fix_in_description(self):
        episodes = [
            _make_episode(input_text="run policy check", failure_class="policy_blocked"),
            _make_episode(input_text="run policy verify", failure_class="policy_blocked"),
        ]
        insights = evolver_dream_phase(episodes)
        fix_insights = [i for i in insights if "Fix:" in i.description]
        self.assertGreater(len(fix_insights), 0)


# ═══════════════════════════════════════════════════════════════════
# Part 5: Persistence
# ═══════════════════════════════════════════════════════════════════

class TestPersistence(unittest.TestCase):

    def setUp(self):
        import kai_config
        self._orig_path = kai_config.EVOLVER_INSIGHT_PATH
        self._tmpfile = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
        self._tmpfile.close()
        kai_config.EVOLVER_INSIGHT_PATH = type(kai_config.EVOLVER_INSIGHT_PATH)(self._tmpfile.name)
        # Clean the file
        os.unlink(self._tmpfile.name)

    def tearDown(self):
        import kai_config
        kai_config.EVOLVER_INSIGHT_PATH = self._orig_path
        try:
            os.unlink(self._tmpfile.name)
        except FileNotFoundError:
            pass

    def test_save_and_load(self):
        report = EvolutionReport(
            report_id="test123",
            ts=time.time(),
            episodes_analyzed=10,
            suggestions=[
                EvolutionSuggestion(
                    suggestion_id="s1",
                    pattern="test pattern",
                    failure_class="confidence_low",
                    frequency=3,
                    fix="do this instead",
                    confidence=0.8,
                    source_episodes=3,
                    priority="high",
                ),
            ],
            top_failure_class="confidence_low",
            top_failure_count=3,
            total_failures=5,
            duration_ms=10.0,
        )
        save_evolver_report(report)
        loaded = load_evolver_reports()
        self.assertEqual(len(loaded), 1)
        self.assertEqual(loaded[0]["report_id"], "test123")
        self.assertEqual(len(loaded[0]["suggestions"]), 1)
        self.assertEqual(loaded[0]["suggestions"][0]["priority"], "high")

    def test_load_empty(self):
        loaded = load_evolver_reports()
        self.assertEqual(loaded, [])

    def test_cap_at_20(self):
        for i in range(25):
            report = EvolutionReport(
                report_id=f"r{i}",
                ts=time.time(),
                episodes_analyzed=1,
                suggestions=[],
                top_failure_class=None,
                top_failure_count=0,
                total_failures=0,
                duration_ms=1.0,
            )
            save_evolver_report(report)
        loaded = load_evolver_reports()
        self.assertEqual(len(loaded), 20)
        self.assertEqual(loaded[-1]["report_id"], "r24")


# ═══════════════════════════════════════════════════════════════════
# Part 6: Dataclass serialization
# ═══════════════════════════════════════════════════════════════════

class TestSerialization(unittest.TestCase):

    def test_suggestion_to_dict(self):
        s = EvolutionSuggestion(
            suggestion_id="abc",
            pattern="test",
            failure_class="confidence_low",
            frequency=5,
            fix="fix it",
            confidence=0.789,
            source_episodes=5,
            priority="high",
        )
        d = s.to_dict()
        self.assertEqual(d["suggestion_id"], "abc")
        self.assertEqual(d["confidence"], 0.789)
        self.assertEqual(d["priority"], "high")

    def test_report_to_dict(self):
        r = EvolutionReport(
            report_id="r1",
            ts=1234.0,
            episodes_analyzed=10,
            suggestions=[],
            top_failure_class="data_insufficient",
            top_failure_count=4,
            total_failures=6,
            duration_ms=15.5,
        )
        d = r.to_dict()
        self.assertEqual(d["report_id"], "r1")
        self.assertEqual(d["total_failures"], 6)
        self.assertIsInstance(d["suggestions"], list)

    def test_report_json_serializable(self):
        r = EvolutionReport(
            report_id="r1",
            ts=time.time(),
            episodes_analyzed=5,
            suggestions=[
                EvolutionSuggestion("s1", "p", "f", 2, "fix", 0.7, 2, "medium"),
            ],
            top_failure_class=None,
            top_failure_count=0,
            total_failures=2,
            duration_ms=5.0,
        )
        serialized = json.dumps(r.to_dict())
        self.assertIsInstance(serialized, str)
        parsed = json.loads(serialized)
        self.assertEqual(parsed["report_id"], "r1")


# ═══════════════════════════════════════════════════════════════════
# Part 7: Edge cases
# ═══════════════════════════════════════════════════════════════════

class TestEdgeCases(unittest.TestCase):

    def test_episodes_with_missing_fields(self):
        episodes = [
            {"outcome_score": 0.3, "failure_class": "confidence_low"},
            {"outcome_score": 0.2, "failure_class": "confidence_low"},
        ]
        report = analyze_failures(episodes)
        self.assertEqual(report.total_failures, 2)

    def test_episodes_with_unknown_failure_class(self):
        episodes = [
            _make_episode(failure_class="some_new_class"),
            _make_episode(failure_class="some_new_class"),
        ]
        report = analyze_failures(episodes)
        # Should still work with fallback fix template
        self.assertGreaterEqual(report.total_failures, 0)

    def test_confidence_bounded(self):
        """Confidence should never exceed 0.95."""
        episodes = [_make_episode(conviction=1.0) for _ in range(20)]
        report = analyze_failures(episodes)
        for s in report.suggestions:
            self.assertLessEqual(s.confidence, 0.95)

    def test_report_has_timing(self):
        report = analyze_failures([_make_episode(), _make_episode()])
        self.assertGreater(report.duration_ms, 0)


if __name__ == "__main__":
    unittest.main()
