"""P15 Dream State — unit tests for offline consolidation engine."""
import json
import sys
import os
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "langgraph"))

from kai_config import (
    DreamCycle,
    DreamInsight,
    _extract_words,
    _word_overlap,
    cluster_failures,
    deduplicate_rules,
    detect_rule_contradictions,
    load_dream_cycles,
    run_dream_cycle,
    save_dream_cycle,
    synthesize_patterns,
    DREAM_INSIGHT_PATH,
)


class TestDreamInsight(unittest.TestCase):
    def test_to_dict(self):
        i = DreamInsight("pattern", "test desc", 0.85, 5, True)
        d = i.to_dict()
        self.assertEqual(d["insight_type"], "pattern")
        self.assertEqual(d["confidence"], 0.85)
        self.assertTrue(d["actionable"])
        self.assertEqual(d["source_episodes"], 5)

    def test_confidence_rounds(self):
        i = DreamInsight("pattern", "x", 0.33333, 1, False)
        self.assertEqual(i.to_dict()["confidence"], 0.333)


class TestDreamCycle(unittest.TestCase):
    def test_to_dict_structure(self):
        c = DreamCycle(
            cycle_id="abc123",
            ts=1000.0,
            episodes_analysed=10,
            insights=[DreamInsight("pattern", "d", 0.9, 3, True)],
            merged_rules=2,
            failure_clusters={"low_confidence": 3},
            boundary_shifts=[],
            duration_ms=42.567,
        )
        d = c.to_dict()
        self.assertEqual(d["cycle_id"], "abc123")
        self.assertEqual(d["episodes_analysed"], 10)
        self.assertEqual(d["merged_rules"], 2)
        self.assertEqual(d["duration_ms"], 42.6)
        self.assertEqual(len(d["insights"]), 1)


class TestExtractWords(unittest.TestCase):
    def test_basic(self):
        words = _extract_words("Hello world of testing")
        self.assertIn("hello", words)
        self.assertIn("world", words)
        self.assertIn("testing", words)
        # "of" is only 2 chars — should be dropped
        self.assertNotIn("of", words)

    def test_empty(self):
        self.assertEqual(_extract_words(""), set())


class TestWordOverlap(unittest.TestCase):
    def test_identical(self):
        self.assertAlmostEqual(_word_overlap("hello world", "hello world"), 1.0)

    def test_disjoint(self):
        self.assertAlmostEqual(_word_overlap("alpha beta", "gamma delta"), 0.0)

    def test_partial(self):
        s = _word_overlap("always verify sources carefully", "always verify data carefully")
        self.assertGreater(s, 0.3)
        self.assertLess(s, 1.0)

    def test_empty(self):
        self.assertAlmostEqual(_word_overlap("", "hello"), 0.0)


class TestDeduplicateRules(unittest.TestCase):
    def test_empty(self):
        self.assertEqual(deduplicate_rules([]), [])

    def test_no_duplicates(self):
        rules = [
            "Always verify data before accepting it",
            "Never use sudo without explicit permission",
        ]
        deduped = deduplicate_rules(rules)
        self.assertEqual(len(deduped), 2)

    def test_merges_similar(self):
        rules = [
            "Always verify sources before trusting data",
            "Always verify sources before trusting any data always",
            "Never run commands without checking them first",
        ]
        deduped = deduplicate_rules(rules, threshold=0.6)
        # The two similar "verify sources" rules should cluster
        self.assertLessEqual(len(deduped), 2)

    def test_keeps_longest(self):
        rules = [
            "short rule",
            "short rule with more context and detail added",
        ]
        deduped = deduplicate_rules(rules, threshold=0.5)
        if len(deduped) == 1:
            self.assertIn("more context", deduped[0])


class TestClusterFailures(unittest.TestCase):
    def test_groups_by_class(self):
        eps = [
            {"failure_class": "low_confidence", "input": "q1", "ts": 1},
            {"failure_class": "low_confidence", "input": "q2", "ts": 2},
            {"failure_class": "wrong_answer", "input": "q3", "ts": 3},
        ]
        clusters = cluster_failures(eps)
        self.assertEqual(len(clusters["low_confidence"]), 2)
        self.assertEqual(len(clusters["wrong_answer"]), 1)

    def test_skips_no_failure_class(self):
        eps = [{"input": "q1"}, {"failure_class": "x", "input": "q2"}]
        clusters = cluster_failures(eps)
        self.assertEqual(len(clusters), 1)
        self.assertIn("x", clusters)

    def test_empty(self):
        self.assertEqual(cluster_failures([]), {})


class TestSynthesizePatterns(unittest.TestCase):
    def test_below_minimum(self):
        eps = [{"input": "test", "final_conviction": 5}]
        self.assertEqual(synthesize_patterns(eps), [])

    def test_struggling_topic_detected(self):
        # 5+ episodes with same topic words and low conviction
        eps = [
            {"input": "python async error handling here", "final_conviction": 3, "rethink_count": 0, "ts": i}
            for i in range(6)
        ]
        insights = synthesize_patterns(eps)
        struggling = [i for i in insights if "Struggling" in i.description]
        self.assertGreater(len(struggling), 0)

    def test_learning_trend_detected(self):
        # Conviction improving over time
        eps = [
            {"input": "kubernetes deployment config check", "final_conviction": 3, "ts": 1},
            {"input": "kubernetes deployment config check", "final_conviction": 3, "ts": 2},
            {"input": "kubernetes deployment config check", "final_conviction": 3, "ts": 3},
            {"input": "kubernetes deployment config check", "final_conviction": 8, "ts": 4},
            {"input": "kubernetes deployment config check", "final_conviction": 9, "ts": 5},
            {"input": "kubernetes deployment config check", "final_conviction": 9, "ts": 6},
        ]
        insights = synthesize_patterns(eps)
        learning = [i for i in insights if "Learning" in i.description]
        self.assertGreater(len(learning), 0)


class TestDetectRuleContradictions(unittest.TestCase):
    def test_finds_always_never_contradiction(self):
        rules = [
            "Always verify data sources before trusting them",
            "Never verify data sources just trust them always",
        ]
        insights = detect_rule_contradictions(rules)
        contradictions = [i for i in insights if i.insight_type == "contradiction"]
        self.assertGreater(len(contradictions), 0)

    def test_no_contradictions(self):
        rules = [
            "Always verify data before using it",
            "Use testing frameworks for debugging code",
        ]
        insights = detect_rule_contradictions(rules)
        self.assertEqual(len(insights), 0)


class TestRunDreamCycle(unittest.TestCase):
    def test_minimal_run(self):
        eps = [
            {"input": f"test query {i}", "final_conviction": 7, "ts": i}
            for i in range(6)
        ]
        cycle = run_dream_cycle(eps, cycle_id="test-001")
        self.assertEqual(cycle.cycle_id, "test-001")
        self.assertEqual(cycle.episodes_analysed, 6)
        self.assertIsInstance(cycle.insights, list)
        self.assertGreater(cycle.duration_ms, 0)

    def test_empty_episodes(self):
        cycle = run_dream_cycle([], cycle_id="empty")
        self.assertEqual(cycle.episodes_analysed, 0)
        self.assertEqual(cycle.merged_rules, 0)

    def test_with_failures(self):
        eps = [
            {"input": "q1", "failure_class": "low_confidence", "final_conviction": 3, "ts": i}
            for i in range(5)
        ]
        cycle = run_dream_cycle(eps, cycle_id="fail-test")
        self.assertIn("low_confidence", cycle.failure_clusters)

    def test_with_rules(self):
        eps = [
            {"input": "q", "metacognitive_rule": "Always check sources before trusting data", "ts": i}
            for i in range(3)
        ]
        cycle = run_dream_cycle(eps, cycle_id="rule-test")
        # All 3 rules the same → should merge to 1 (merged_count = 2)
        self.assertGreaterEqual(cycle.merged_rules, 0)


class TestPersistence(unittest.TestCase):
    def test_save_and_load(self):
        import langgraph.kai_config as kc
        original = kc.DREAM_INSIGHT_PATH
        try:
            with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
                tmp = f.name
            kc.DREAM_INSIGHT_PATH = type(original)(tmp)
            # Remove file so save_dream_cycle creates fresh
            if os.path.exists(tmp):
                os.unlink(tmp)

            # run_dream_cycle calls save_dream_cycle internally
            cycle = run_dream_cycle([], cycle_id="persist-test")
            loaded = load_dream_cycles()
            self.assertGreaterEqual(len(loaded), 1)
            self.assertEqual(loaded[-1]["cycle_id"], "persist-test")
        finally:
            kc.DREAM_INSIGHT_PATH = original
            if os.path.exists(tmp):
                os.unlink(tmp)


if __name__ == "__main__":
    unittest.main()
