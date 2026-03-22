"""Tests for P13: Recursive Self-Improvement Gate — snapshot + evaluate."""
import sys, os, time, json, tempfile, unittest
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "langgraph"))

from kai_config import (
    PerformanceSnapshot,
    capture_snapshot,
    save_snapshot,
    load_latest_snapshot,
    evaluate_improvement,
    IMPROVEMENT_TOLERANCE,
)


def _ep(conviction: float = 8.0, outcome: float = 0.8, rethinks: int = 0) -> dict:
    return {
        "episode_id": "test",
        "input": "test input",
        "output": "test output",
        "ts": time.time(),
        "outcome_score": outcome,
        "conviction_score": conviction,
        "final_conviction": conviction,
        "rethink_count": rethinks,
    }


class TestPerformanceSnapshot(unittest.TestCase):
    def test_snapshot_fields(self):
        snap = PerformanceSnapshot(
            timestamp=time.time(),
            avg_conviction=8.0,
            avg_outcome=0.8,
            failure_rate=0.1,
            total_episodes=10,
            rethink_rate=0.2,
            label="test",
        )
        self.assertEqual(snap.avg_conviction, 8.0)
        self.assertEqual(snap.label, "test")

    def test_to_dict_roundtrip(self):
        snap = PerformanceSnapshot(
            timestamp=123.0,
            avg_conviction=7.5,
            avg_outcome=0.75,
            failure_rate=0.15,
            total_episodes=20,
            rethink_rate=0.1,
            label="roundtrip",
        )
        d = snap.to_dict()
        restored = PerformanceSnapshot.from_dict(d)
        self.assertEqual(restored.avg_conviction, 7.5)
        self.assertEqual(restored.total_episodes, 20)
        self.assertEqual(restored.label, "roundtrip")

    def test_from_dict_defaults(self):
        snap = PerformanceSnapshot.from_dict({})
        self.assertEqual(snap.avg_conviction, 0.0)
        self.assertEqual(snap.total_episodes, 0)
        self.assertEqual(snap.label, "")


class TestCaptureSnapshot(unittest.TestCase):
    def test_empty_episodes(self):
        snap = capture_snapshot([])
        self.assertEqual(snap.total_episodes, 0)
        self.assertEqual(snap.avg_conviction, 0.0)

    def test_normal_episodes(self):
        eps = [_ep(8.0, 0.9), _ep(7.0, 0.3), _ep(9.0, 0.8)]
        snap = capture_snapshot(eps, label="test")
        self.assertEqual(snap.total_episodes, 3)
        self.assertAlmostEqual(snap.avg_conviction, 8.0, places=1)
        self.assertAlmostEqual(snap.failure_rate, 0.333, places=2)
        self.assertEqual(snap.label, "test")

    def test_rethink_rate(self):
        eps = [_ep(rethinks=0), _ep(rethinks=2), _ep(rethinks=0), _ep(rethinks=1)]
        snap = capture_snapshot(eps)
        self.assertAlmostEqual(snap.rethink_rate, 0.5, places=1)

    def test_all_failures(self):
        eps = [_ep(outcome=0.2), _ep(outcome=0.1)]
        snap = capture_snapshot(eps)
        self.assertEqual(snap.failure_rate, 1.0)


class TestSaveLoadSnapshot(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.path = os.path.join(self.tmpdir, "snapshots.json")
        import kai_config
        self._orig = kai_config.IMPROVEMENT_SNAPSHOT_PATH
        kai_config.IMPROVEMENT_SNAPSHOT_PATH = __import__("pathlib").Path(self.path)

    def tearDown(self):
        import kai_config
        kai_config.IMPROVEMENT_SNAPSHOT_PATH = self._orig

    def test_save_and_load(self):
        snap = PerformanceSnapshot(
            timestamp=time.time(),
            avg_conviction=8.0,
            avg_outcome=0.8,
            failure_rate=0.1,
            total_episodes=10,
            rethink_rate=0.2,
            label="save-test",
        )
        save_snapshot(snap)
        loaded = load_latest_snapshot()
        self.assertIsNotNone(loaded)
        self.assertEqual(loaded.label, "save-test")
        self.assertEqual(loaded.total_episodes, 10)

    def test_load_from_empty(self):
        loaded = load_latest_snapshot()
        self.assertIsNone(loaded)

    def test_multiple_saves(self):
        for i in range(3):
            snap = PerformanceSnapshot(
                timestamp=time.time() + i,
                avg_conviction=7.0 + i,
                avg_outcome=0.7,
                failure_rate=0.1,
                total_episodes=i + 1,
                rethink_rate=0.1,
                label=f"snap-{i}",
            )
            save_snapshot(snap)
        loaded = load_latest_snapshot()
        self.assertEqual(loaded.label, "snap-2")
        # verify file has 3 entries
        data = json.loads(open(self.path).read())
        self.assertEqual(len(data), 3)


class TestEvaluateImprovement(unittest.TestCase):
    def _snap(self, conv=8.0, outcome=0.8, fail=0.1, rethink=0.2):
        return PerformanceSnapshot(
            timestamp=time.time(),
            avg_conviction=conv,
            avg_outcome=outcome,
            failure_rate=fail,
            total_episodes=10,
            rethink_rate=rethink,
        )

    def test_no_change_approved(self):
        before = self._snap()
        after = self._snap()
        verdict = evaluate_improvement(before, after)
        self.assertTrue(verdict.approved)
        self.assertEqual(len(verdict.degraded_metrics), 0)
        self.assertIn("neutral", verdict.recommendation.lower())

    def test_improvement_approved(self):
        before = self._snap(conv=7.0, outcome=0.6, fail=0.3)
        after = self._snap(conv=8.5, outcome=0.85, fail=0.05)
        verdict = evaluate_improvement(before, after)
        self.assertTrue(verdict.approved)
        self.assertGreater(len(verdict.improved_metrics), 0)

    def test_degradation_rejected(self):
        before = self._snap(conv=8.0, outcome=0.8, fail=0.1)
        after = self._snap(conv=5.0, outcome=0.4, fail=0.5)
        verdict = evaluate_improvement(before, after)
        self.assertFalse(verdict.approved)
        self.assertGreater(len(verdict.degraded_metrics), 0)
        self.assertIn("revert", verdict.recommendation.lower())

    def test_conviction_drop_rejected(self):
        before = self._snap(conv=8.0)
        after = self._snap(conv=7.0)  # drop of 1.0, tolerance=0.1
        verdict = evaluate_improvement(before, after)
        self.assertFalse(verdict.approved)
        self.assertIn("avg_conviction", verdict.degraded_metrics)

    def test_failure_rate_increase_rejected(self):
        before = self._snap(fail=0.1)
        after = self._snap(fail=0.4)  # increase of 0.3
        verdict = evaluate_improvement(before, after)
        self.assertFalse(verdict.approved)
        self.assertIn("failure_rate", verdict.degraded_metrics)

    def test_within_tolerance_approved(self):
        before = self._snap(conv=8.0)
        after = self._snap(conv=7.95)  # drop of 0.05, within 0.1 tolerance
        verdict = evaluate_improvement(before, after)
        self.assertTrue(verdict.approved)

    def test_custom_tolerance(self):
        before = self._snap(conv=8.0)
        after = self._snap(conv=7.5)  # drop 0.5
        # with tolerance=1.0, this is fine
        verdict = evaluate_improvement(before, after, tolerance=1.0)
        self.assertTrue(verdict.approved)
        # with tolerance=0.1, this fails
        verdict = evaluate_improvement(before, after, tolerance=0.1)
        self.assertFalse(verdict.approved)

    def test_rethink_rate_increase_rejected(self):
        before = self._snap(rethink=0.1)
        after = self._snap(rethink=0.5)
        verdict = evaluate_improvement(before, after)
        self.assertFalse(verdict.approved)
        self.assertIn("rethink_rate", verdict.degraded_metrics)

    def test_delta_values(self):
        before = self._snap(conv=8.0, outcome=0.8, fail=0.1)
        after = self._snap(conv=9.0, outcome=0.9, fail=0.2)
        verdict = evaluate_improvement(before, after)
        self.assertAlmostEqual(verdict.delta_conviction, 1.0, places=2)
        self.assertAlmostEqual(verdict.delta_outcome, 0.1, places=2)
        self.assertAlmostEqual(verdict.delta_failure_rate, 0.1, places=2)

    def test_default_tolerance_value(self):
        self.assertEqual(IMPROVEMENT_TOLERANCE, 0.1)


if __name__ == "__main__":
    unittest.main()
