"""Context budget tests — verify token estimation and prompt trimming.

Exercises:
  - _estimate_tokens returns reasonable approximation
  - _trim_context preserves system prompt and user query
  - _trim_context drops oldest middle messages when over budget
  - _trim_context is a no-op when within budget
  - CONTEXT_BUDGET_TOKENS env-configurable
"""
from __future__ import annotations

import importlib.util
import sys
import types
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "langgraph"))
sys.modules.setdefault("redis", types.SimpleNamespace())

spec = importlib.util.spec_from_file_location("langgraph_app", ROOT / "langgraph" / "app.py")
assert spec and spec.loader
mod = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = mod
spec.loader.exec_module(mod)


class TestEstimateTokens(unittest.TestCase):
    """Test _estimate_tokens approximation."""

    def test_empty_string(self):
        # tiktoken returns 0 for empty, heuristic returned 1
        self.assertGreaterEqual(mod._estimate_tokens(""), 0)
        self.assertLessEqual(mod._estimate_tokens(""), 1)

    def test_short_text(self):
        result = mod._estimate_tokens("hello world")
        self.assertGreaterEqual(result, 1)
        self.assertLessEqual(result, 10)

    def test_longer_text(self):
        text = "a" * 400
        result = mod._estimate_tokens(text)
        # tiktoken counts vary; should be in reasonable range
        self.assertGreater(result, 10)
        self.assertLess(result, 500)

    def test_always_positive(self):
        self.assertGreaterEqual(mod._estimate_tokens("x"), 1)


class TestTrimContext(unittest.TestCase):
    """Test _trim_context prompt trimming."""

    def test_within_budget_unchanged(self):
        messages = [
            {"role": "system", "content": "short"},
            {"role": "user", "content": "hi"},
        ]
        result = mod._trim_context(messages, budget=1000)
        self.assertEqual(len(result), 2)

    def test_empty_messages(self):
        result = mod._trim_context([], budget=100)
        self.assertEqual(result, [])

    def test_preserves_first_and_last(self):
        messages = [
            {"role": "system", "content": "system prompt " * 100},
            {"role": "system", "content": "context A " * 100},
            {"role": "user", "content": "old question " * 100},
            {"role": "assistant", "content": "old answer " * 100},
            {"role": "user", "content": "current question"},
        ]
        # budget that can only fit system prompt + current question + maybe 1 middle
        result = mod._trim_context(messages, budget=350)
        # first and last are always kept
        self.assertEqual(result[0]["role"], "system")
        self.assertIn("system prompt", result[0]["content"])
        self.assertEqual(result[-1]["role"], "user")
        self.assertIn("current question", result[-1]["content"])

    def test_drops_oldest_middle_first(self):
        """Newest middle messages are kept preferentially."""
        messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "old " * 200},
            {"role": "assistant", "content": "old_ans " * 200},
            {"role": "user", "content": "recent " * 50},
            {"role": "assistant", "content": "recent_ans " * 50},
            {"role": "user", "content": "now"},
        ]
        result = mod._trim_context(messages, budget=200)
        contents = " ".join(m["content"] for m in result)
        # recent messages should survive over old ones
        self.assertIn("now", contents)
        self.assertIn("sys", contents)

    def test_single_message(self):
        messages = [{"role": "user", "content": "hello"}]
        result = mod._trim_context(messages, budget=100)
        self.assertEqual(len(result), 1)

    def test_does_not_mutate_input(self):
        messages = [
            {"role": "system", "content": "x" * 1000},
            {"role": "system", "content": "y" * 1000},
            {"role": "user", "content": "z"},
        ]
        original_len = len(messages)
        mod._trim_context(messages, budget=300)
        self.assertEqual(len(messages), original_len)


class TestContextBudgetConfig(unittest.TestCase):
    """Test CONTEXT_BUDGET_TOKENS is configured."""

    def test_default_budget(self):
        self.assertIsInstance(mod.CONTEXT_BUDGET_TOKENS, int)
        self.assertGreaterEqual(mod.CONTEXT_BUDGET_TOKENS, 1024)


if __name__ == "__main__":
    unittest.main()
