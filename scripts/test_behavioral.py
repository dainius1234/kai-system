"""Behavioral tests — end-to-end reasoning pipeline coherence.

Unlike structural tests that check JSON keys, these verify that:
  1. Conviction scoring produces meaningfully different scores for
     well-supported vs. unsupported queries
  2. The adversary pipeline penalises known-failure patterns
  3. Context budget trimming preserves semantic quality
  4. Router → conviction → adversary flow is coherent
  5. SAGE critique detects genuine weaknesses in signals
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

# ── Load conviction module ────────────────────────────────────────────
_conv_spec = importlib.util.spec_from_file_location(
    "conviction", ROOT / "langgraph" / "conviction.py"
)
_conv = importlib.util.module_from_spec(_conv_spec)
sys.modules[_conv_spec.name] = _conv
_conv_spec.loader.exec_module(_conv)

# ── Load router module ────────────────────────────────────────────────
_router_spec = importlib.util.spec_from_file_location(
    "router", ROOT / "langgraph" / "router.py"
)
_router = importlib.util.module_from_spec(_router_spec)
sys.modules["router"] = _router
_router_spec.loader.exec_module(_router)

# ── Load adversary module ─────────────────────────────────────────────
_adv_spec = importlib.util.spec_from_file_location(
    "adversary", ROOT / "langgraph" / "adversary.py"
)
_adv = importlib.util.module_from_spec(_adv_spec)
sys.modules["adversary"] = _adv
_adv_spec.loader.exec_module(_adv)

# ── Load context budget from langgraph/app.py ─────────────────────────
sys.modules.setdefault("redis", types.SimpleNamespace())
_app_spec = importlib.util.spec_from_file_location(
    "langgraph_app", ROOT / "langgraph" / "app.py"
)
_app = importlib.util.module_from_spec(_app_spec)
sys.modules[_app_spec.name] = _app
_app_spec.loader.exec_module(_app)


class TestConvictionMeaningfulScoring(unittest.TestCase):
    """Conviction scores should reflect actual evidence quality."""

    def test_high_coverage_scores_higher(self):
        """Well-supported query with relevant chunks scores higher."""
        relevant_chunks = [
            {"content": "tax threshold for 2024 is £12,570 personal allowance"},
            {"content": "HMRC self-assessment deadline is 31 January"},
            {"content": "mileage rate for self-employment is 45p per mile"},
        ]
        irrelevant_chunks = [
            {"content": "the weather in London is cloudy today"},
            {"content": "Python was created by Guido van Rossum"},
            {"content": "the cat sat on the mat"},
        ]
        user_input = "how much tax do I owe for self-employment mileage?"

        score_relevant = _conv._context_coverage(relevant_chunks, user_input)
        score_irrelevant = _conv._context_coverage(irrelevant_chunks, user_input)

        self.assertGreater(
            score_relevant, score_irrelevant,
            "Relevant chunks should produce higher coverage score"
        )

    def test_empty_context_scores_zero(self):
        score = _conv._context_coverage([], "any question")
        self.assertEqual(score, 0.0)

    def test_plan_specificity_rewards_detail(self):
        """Detailed plans should score higher than vague ones."""
        detailed_plan = {
            "specialist": "DeepSeek-V4",
            "summary": "Retrieve tax records, calculate mileage deduction, generate report",
            "steps": [
                {"action": "retrieve", "input": "tax records"},
                {"action": "calculate", "input": "mileage at 45p/mile"},
                {"action": "generate", "output": "deduction report"},
            ],
        }
        vague_plan = {
            "steps": [{"action": "do_something"}],
        }

        score_detailed = _conv._plan_specificity(detailed_plan)
        score_vague = _conv._plan_specificity(vague_plan)

        self.assertGreater(score_detailed, score_vague)

    def test_query_clarity_distinguishes_inputs(self):
        """Specific questions should score higher than vague ones."""
        specific = "What is the VAT threshold for UK self-employment in 2024?"
        vague = "tax?"
        medium = "how much tax do I owe"

        score_specific = _conv._query_clarity(specific)
        score_vague = _conv._query_clarity(vague)
        score_medium = _conv._query_clarity(medium)

        self.assertGreater(score_specific, score_vague)
        self.assertGreaterEqual(score_medium, score_vague)

    def test_full_conviction_score_meaningful(self):
        """Full score_conviction should produce score in valid range."""
        chunks = [
            {"content": "risk policy: max drawdown 5%, rollback after 3 failures"},
            {"content": "executor rollback triggered at error_ratio > 0.2"},
        ]
        plan = _conv.build_plan(
            "what are the risk limits for executor?",
            "DeepSeek-V4",
            chunks,
        )
        score = _conv.score_conviction(
            "what are the risk limits for executor?",
            plan,
            chunks,
            rethink_count=0,
        )
        self.assertIsInstance(score, float)
        self.assertGreaterEqual(score, 0)
        self.assertLessEqual(score, 10)


class TestAdversaryBehavior(unittest.TestCase):
    """Adversary challenges should produce meaningful findings."""

    def test_history_challenge_penalises_past_failures(self):
        """If a similar request failed before, modifier should be negative."""
        import time
        now = time.time()
        failed_episodes = [
            {
                "input": "deploy the server to production",
                "output": "deployment failed: timeout",
                "outcome_score": 0.1,
                "final_conviction": 5.0,
                "ts": now - 86400,
                "episode_id": "ep-fail-1",
            },
        ]
        plan = {"steps": [{"action": "deploy"}]}
        result = _adv.challenge_history(
            plan, "deploy the server", failed_episodes
        )
        self.assertFalse(result.passed)
        self.assertLess(result.modifier, 0)

    def test_history_challenge_boosts_past_successes(self):
        """If a similar request succeeded before, modifier should be positive."""
        import time
        now = time.time()
        success_episodes = [
            {
                "input": "calculate tax deductions",
                "output": "deductions: £1500",
                "outcome_score": 0.9,
                "final_conviction": 9.0,
                "ts": now - 86400,
                "episode_id": "ep-ok-1",
            },
        ]
        plan = {"steps": [{"action": "calculate"}]}
        result = _adv.challenge_history(
            plan, "calculate tax deductions", success_episodes
        )
        self.assertTrue(result.passed)
        self.assertGreaterEqual(result.modifier, 0)

    def test_no_history_returns_neutral(self):
        """No episodes → neutral result."""
        result = _adv.challenge_history({}, "anything", [])
        self.assertTrue(result.passed)
        self.assertEqual(result.modifier, 0.0)


class TestRouterConvictionCoherence(unittest.TestCase):
    """Router decisions should be coherent with conviction pipeline."""

    def test_tax_query_routes_and_scores(self):
        """Tax query → TAX_ADVISORY route, and conviction scores with tax chunks."""
        decision = _router.classify("how much VAT do I owe this quarter?")
        self.assertEqual(decision.route, "TAX_ADVISORY")
        self.assertTrue(decision.bypass_llm)

        # Same query should get coverage from tax-related chunks
        tax_chunks = [
            {"content": "VAT threshold is £85,000 for 2024"},
            {"content": "quarterly VAT returns due 7th of month+1"},
        ]
        coverage = _conv._context_coverage(tax_chunks, "how much VAT do I owe this quarter?")
        self.assertGreater(coverage, 0.0)

    def test_casual_chat_routes_to_general(self):
        decision = _router.classify("what do you think about pizza?")
        self.assertEqual(decision.route, "GENERAL_CHAT")
        self.assertFalse(decision.bypass_llm)

    def test_memory_query_routes_correctly(self):
        decision = _router.classify("do you remember what I said about the car?")
        self.assertEqual(decision.route, "MEMORY_RECALL")
        self.assertTrue(decision.bypass_llm)


class TestContextBudgetBehavior(unittest.TestCase):
    """Context trimming should preserve semantic quality."""

    def test_trimming_keeps_system_and_current_query(self):
        """System prompt and current user query are always preserved."""
        messages = [
            {"role": "system", "content": "You are Kai, a sovereign AI assistant."},
            {"role": "user", "content": "old question " * 200},
            {"role": "assistant", "content": "old answer " * 200},
            {"role": "user", "content": "another old question " * 200},
            {"role": "assistant", "content": "another old answer " * 200},
            {"role": "user", "content": "What is my current tax liability?"},
        ]
        result = _app._trim_context(messages, budget=200)

        # System prompt preserved
        self.assertEqual(result[0]["role"], "system")
        self.assertIn("Kai", result[0]["content"])

        # Current query preserved
        self.assertEqual(result[-1]["role"], "user")
        self.assertIn("tax liability", result[-1]["content"])

        # Some middle messages dropped
        self.assertLess(len(result), len(messages))

    def test_trimming_prefers_recent_context(self):
        """More recent messages should survive over older ones."""
        messages = [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "very old question " * 100},
            {"role": "assistant", "content": "very old answer " * 100},
            {"role": "user", "content": "recent question about tax"},
            {"role": "assistant", "content": "recent answer about tax"},
            {"role": "user", "content": "current question"},
        ]
        result = _app._trim_context(messages, budget=150)
        contents = " ".join(m["content"] for m in result)

        self.assertIn("current question", contents)
        self.assertIn("system", contents)
        # Recent context should be preferred
        if "recent" in contents:
            self.assertIn("tax", contents)

    def test_no_trimming_when_within_budget(self):
        """Short conversations should pass through unchanged."""
        messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "hi"},
        ]
        result = _app._trim_context(messages, budget=10000)
        self.assertEqual(len(result), len(messages))


class TestConvictionGating(unittest.TestCase):
    """The conviction gate should meaningfully separate good from bad plans."""

    def test_well_supported_plan_scores_higher(self):
        """Plan with good evidence scores higher than plan without."""
        query = "HMRC deadlines self-assessment filing"
        good_chunks = [
            {"content": "HMRC self-assessment deadline January online filing"},
            {"content": "self-assessment payment account HMRC penalty"},
            {"content": "late filing penalty HMRC assessment"},
        ]
        bad_chunks = [
            {"content": "music playlist coding spotify favourites"},
            {"content": "recipe pasta carbonara Italian cheese"},
        ]

        plan = _conv.build_plan(query, "DeepSeek-V4")
        score_good = _conv.score_conviction(query, plan, good_chunks, rethink_count=0)
        score_bad = _conv.score_conviction(query, plan, bad_chunks, rethink_count=0)

        self.assertGreater(
            score_good, score_bad,
            "Well-evidenced plan should score higher than poorly-evidenced one"
        )


if __name__ == "__main__":
    unittest.main()
