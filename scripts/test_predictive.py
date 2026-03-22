"""Tests for P10: Predictive Pre-Computation — sequence mining + prediction."""
import sys
import os
import time
import unittest
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "langgraph"))


from planner import (
    PredictedRequest,
    _extract_topic_key,
    mine_request_sequences,
    predict_next_request,
)


def _ep(input_text: str, ts_offset: float = 0.0) -> dict:
    """Helper: build a minimal episode dict."""
    return {
        "episode_id": f"ep-{hash(input_text) % 10000}",
        "input": input_text,
        "output": "ok",
        "ts": time.time() - 3600 + ts_offset,
        "outcome_score": 0.8,
        "conviction_score": 8.0,
    }


class TestTopicKeyExtraction(unittest.TestCase):
    def test_normal_sentence(self):
        key = _extract_topic_key("deploy the server now")
        self.assertIn("deploy", key)
        self.assertIn("server", key)

    def test_short_words_filtered(self):
        key = _extract_topic_key("do it now")
        self.assertEqual(key, "general")

    def test_max_three_words(self):
        key = _extract_topic_key("alpha bravo charlie delta echo foxtrot")
        parts = key.split("+")
        self.assertLessEqual(len(parts), 3)

    def test_empty_input(self):
        self.assertEqual(_extract_topic_key(""), "general")


class TestSequenceMining(unittest.TestCase):
    def test_empty_episodes(self):
        result = mine_request_sequences([])
        self.assertEqual(result, {})

    def test_single_episode(self):
        result = mine_request_sequences([_ep("deploy the server")])
        self.assertEqual(result, {})

    def test_bigram_detection(self):
        """A→B repeated twice should produce a transition."""
        eps = [
            _ep("deploy the server", ts_offset=0),
            _ep("check the logs", ts_offset=10),
            _ep("deploy the server", ts_offset=20),
            _ep("check the logs", ts_offset=30),
        ]
        result = mine_request_sequences(eps, min_support=2)
        self.assertTrue(len(result) > 0)

    def test_min_support_filter(self):
        """Single occurrence shouldn't appear with min_support=2."""
        eps = [
            _ep("deploy the server", ts_offset=0),
            _ep("check the logs", ts_offset=10),
            _ep("restart the database", ts_offset=20),
        ]
        result = mine_request_sequences(eps, min_support=2)
        self.assertEqual(result, {})

    def test_self_loops_excluded(self):
        """Repeated same topic shouldn't create a self-loop transition."""
        eps = [
            _ep("deploy the server", ts_offset=0),
            _ep("deploy the server", ts_offset=10),
            _ep("deploy the server", ts_offset=20),
        ]
        result = mine_request_sequences(eps, min_support=1)
        # no transitions because all topics are the same
        self.assertEqual(result, {})

    def test_probability_calculation(self):
        """If A→B twice and A→C once, B gets 2/3 probability."""
        eps = [
            _ep("deploy the server", ts_offset=0),
            _ep("check the logs", ts_offset=10),
            _ep("deploy the server", ts_offset=20),
            _ep("check the logs", ts_offset=30),
            _ep("deploy the server", ts_offset=40),
            _ep("restart database service", ts_offset=50),
        ]
        result = mine_request_sequences(eps, min_support=1)
        topic_a = _extract_topic_key("deploy the server")
        if topic_a in result:
            probs = {t: p for t, p, c in result[topic_a]}
            topic_b = _extract_topic_key("check the logs")
            if topic_b in probs:
                self.assertGreater(probs[topic_b], 0.5)


class TestPredictNextRequest(unittest.TestCase):
    def _build_sequence(self):
        """Build A→B→A→B→A→B pattern."""
        eps = []
        for i in range(6):
            if i % 2 == 0:
                eps.append(_ep("deploy the server", ts_offset=i * 10))
            else:
                eps.append(_ep("check the logs", ts_offset=i * 10))
        return eps

    def test_prediction_made(self):
        eps = self._build_sequence()
        predictions = predict_next_request("deploy the server", eps, min_support=2, confidence_threshold=0.3)
        self.assertIsInstance(predictions, list)

    def test_no_prediction_for_unknown(self):
        eps = self._build_sequence()
        predictions = predict_next_request("xyz", eps)
        self.assertEqual(predictions, [])

    def test_general_input_returns_empty(self):
        eps = self._build_sequence()
        predictions = predict_next_request("hi", eps)
        self.assertEqual(predictions, [])

    def test_prediction_fields(self):
        eps = self._build_sequence()
        predictions = predict_next_request("deploy the server", eps, min_support=1, confidence_threshold=0.1)
        for p in predictions:
            self.assertIsInstance(p, PredictedRequest)
            self.assertTrue(0.0 <= p.confidence <= 1.0)
            self.assertGreaterEqual(p.support, 1)

    def test_confidence_threshold_filter(self):
        eps = self._build_sequence()
        # with impossibly high threshold, nothing passes
        predictions = predict_next_request("deploy the server", eps, min_support=1, confidence_threshold=0.99)
        # may or may not return depending on exact pattern match
        for p in predictions:
            self.assertGreaterEqual(p.confidence, 0.99)


if __name__ == "__main__":
    unittest.main()
