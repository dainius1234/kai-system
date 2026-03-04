"""Tests for P11: Operator Tempo Modeling — pace detection + style hints."""
import sys, os, time, unittest
from unittest.mock import patch, MagicMock
from datetime import datetime, timedelta

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "memu-core"))

# We test the tempo logic by calling the endpoint via the FastAPI test client
# but the logic uses store.search() — so we mock the store.

# Import the app after path setup
os.environ.setdefault("VECTOR_STORE", "memory")
import app as memu_app


def _make_record(ts_offset_seconds: float) -> MagicMock:
    """Create a mock MemoryRecord with a timestamp offset from now."""
    ts = datetime.utcnow() - timedelta(seconds=ts_offset_seconds)
    rec = MagicMock()
    rec.timestamp = ts.isoformat()
    rec.poisoned = False
    rec.category = "general"
    return rec


class TestTempoConfig(unittest.TestCase):
    def test_defaults(self):
        self.assertEqual(memu_app.TEMPO_RAPID_THRESHOLD, 30.0)
        self.assertEqual(memu_app.TEMPO_NORMAL_THRESHOLD, 300.0)
        self.assertEqual(memu_app.TEMPO_REFLECTIVE_THRESHOLD, 1800.0)
        self.assertEqual(memu_app.TEMPO_MIN_INTERACTIONS, 3)


class TestTempoInsufficientData(unittest.TestCase):
    def test_empty_store_returns_insufficient(self):
        """With no records, should return insufficient_data."""
        from fastapi.testclient import TestClient
        client = TestClient(memu_app.app)
        with patch.object(memu_app, "store") as mock_store:
            mock_store.search.return_value = []
            resp = client.get("/memory/tempo?hours=1")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["status"], "insufficient_data")
        self.assertEqual(data["tempo"], "unknown")

    def test_two_records_insufficient(self):
        """Two records is below the min_interactions threshold."""
        from fastapi.testclient import TestClient
        client = TestClient(memu_app.app)
        records = [_make_record(60), _make_record(30)]
        with patch.object(memu_app, "store") as mock_store:
            mock_store.search.return_value = records
            resp = client.get("/memory/tempo?hours=1")
        data = resp.json()
        self.assertEqual(data["status"], "insufficient_data")


class TestTempoClassification(unittest.TestCase):
    def _get_tempo(self, gap_seconds: float, count: int = 5) -> dict:
        """Helper: create `count` records spaced `gap_seconds` apart."""
        from fastapi.testclient import TestClient
        client = TestClient(memu_app.app)
        records = [_make_record(gap_seconds * i) for i in range(count)]
        with patch.object(memu_app, "store") as mock_store:
            mock_store.search.return_value = records
            resp = client.get("/memory/tempo?hours=24")
        return resp.json()

    def test_rapid_tempo(self):
        data = self._get_tempo(10)  # 10s gaps
        self.assertEqual(data["status"], "ok")
        self.assertEqual(data["tempo"], "rapid")
        self.assertIn("concise", data["style_hint"].lower())

    def test_normal_tempo(self):
        data = self._get_tempo(120)  # 2min gaps
        self.assertEqual(data["status"], "ok")
        self.assertEqual(data["tempo"], "normal")

    def test_reflective_tempo(self):
        data = self._get_tempo(600)  # 10min gaps
        self.assertEqual(data["status"], "ok")
        self.assertEqual(data["tempo"], "reflective")
        self.assertIn("thorough", data["style_hint"].lower())

    def test_idle_tempo(self):
        data = self._get_tempo(3600)  # 1hr gaps
        self.assertEqual(data["status"], "ok")
        self.assertEqual(data["tempo"], "idle")
        self.assertIn("summarise", data["style_hint"].lower())


class TestTempoDistribution(unittest.TestCase):
    def test_distribution_keys(self):
        """Response should include all four gap categories."""
        from fastapi.testclient import TestClient
        client = TestClient(memu_app.app)
        records = [_make_record(10 * i) for i in range(5)]
        with patch.object(memu_app, "store") as mock_store:
            mock_store.search.return_value = records
            resp = client.get("/memory/tempo?hours=24")
        data = resp.json()
        if data["status"] == "ok":
            dist = data["distribution"]
            for key in ("rapid", "normal", "reflective", "idle"):
                self.assertIn(key, dist)

    def test_avg_and_median_gap(self):
        from fastapi.testclient import TestClient
        client = TestClient(memu_app.app)
        records = [_make_record(10 * i) for i in range(5)]
        with patch.object(memu_app, "store") as mock_store:
            mock_store.search.return_value = records
            resp = client.get("/memory/tempo?hours=24")
        data = resp.json()
        if data["status"] == "ok":
            self.assertIn("avg_gap_seconds", data)
            self.assertIn("median_gap_seconds", data)
            self.assertGreater(data["avg_gap_seconds"], 0)


class TestTempoBurstDetection(unittest.TestCase):
    def test_burst_with_rapid_sequence(self):
        """4+ rapid-fire records should detect at least 1 burst."""
        from fastapi.testclient import TestClient
        client = TestClient(memu_app.app)
        # 6 records, 5s apart (all rapid)
        records = [_make_record(5 * i) for i in range(6)]
        with patch.object(memu_app, "store") as mock_store:
            mock_store.search.return_value = records
            resp = client.get("/memory/tempo?hours=24")
        data = resp.json()
        if data["status"] == "ok":
            self.assertGreaterEqual(data["burst_episodes"], 1)

    def test_no_burst_with_slow_pace(self):
        """Records spaced 10min apart shouldn't trigger bursts."""
        from fastapi.testclient import TestClient
        client = TestClient(memu_app.app)
        records = [_make_record(600 * i) for i in range(5)]
        with patch.object(memu_app, "store") as mock_store:
            mock_store.search.return_value = records
            resp = client.get("/memory/tempo?hours=24")
        data = resp.json()
        if data["status"] == "ok":
            self.assertEqual(data["burst_episodes"], 0)


class TestTempoWindowParam(unittest.TestCase):
    def test_hours_param_respected(self):
        """Records outside the window should be excluded."""
        from fastapi.testclient import TestClient
        client = TestClient(memu_app.app)
        # all records 3 hours old — outside 1-hour window
        records = [_make_record(10800 + 10 * i) for i in range(5)]
        with patch.object(memu_app, "store") as mock_store:
            mock_store.search.return_value = records
            resp = client.get("/memory/tempo?hours=1")
        data = resp.json()
        self.assertEqual(data["status"], "insufficient_data")


if __name__ == "__main__":
    unittest.main()
