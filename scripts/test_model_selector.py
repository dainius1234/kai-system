"""Tests for HP2 MoE-Style Model Selector."""
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "langgraph"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from model_selector import (
    ModelProfile, estimate_complexity, select_model,
    get_profile, register_model, list_models, _PROFILES,
)


class TestEstimateComplexity(unittest.TestCase):
    def test_simple_query(self):
        c = estimate_complexity("hello")
        self.assertLess(c, 0.3)

    def test_complex_query(self):
        c = estimate_complexity(
            "analyse the trade-off between risk assessment approaches "
            "and evaluate which strategy would be optimal for a multi-step "
            "business plan involving refactoring the architecture"
        )
        self.assertGreater(c, 0.5)

    def test_medium_query(self):
        c = estimate_complexity("what is the best way to debug this issue?")
        self.assertGreater(c, 0.1)

    def test_empty_query(self):
        c = estimate_complexity("")
        self.assertEqual(c, 0.0)

    def test_capped_at_one(self):
        # stuff every trigger keyword in
        c = estimate_complexity(
            "analyse compare evaluate design architect debug refactor "
            "optimise optimize strategy multi-step trade-off risk assessment "
            "business plan? Long sentence one. Sentence two. Sentence three. "
            "Sentence four. Sentence five."
        )
        self.assertLessEqual(c, 1.0)

    def test_question_mark_boost(self):
        c1 = estimate_complexity("explain this")
        c2 = estimate_complexity("explain this?")
        self.assertGreater(c2, c1)


class TestSelectModel(unittest.TestCase):
    def test_single_available_model(self):
        m = select_model("GENERAL_CHAT", "hello", available=["Ollama"])
        self.assertEqual(m, "Ollama")

    def test_empty_available_uses_all_models(self):
        m = select_model("GENERAL_CHAT", "hello", available=[])
        # With empty list, falls back to all registered models
        self.assertIn(m, list_models())

    def test_route_matching_boosts_score(self):
        # DeepSeek-V4's strengths include FACT_CHECK
        m = select_model("FACT_CHECK", "verify this claim", available=["Ollama", "DeepSeek-V4"])
        self.assertEqual(m, "DeepSeek-V4")

    def test_complex_task_prefers_quality(self):
        # Complex task should prefer higher quality_tier
        m = select_model(
            "EXECUTE_ACTION",
            "analyse and evaluate a multi-step strategy for the risk assessment "
            "of refactoring the trade-off between approaches",
            available=["Ollama", "DeepSeek-V4"],
        )
        self.assertEqual(m, "DeepSeek-V4")

    def test_simple_task_with_speed_preference(self):
        m = select_model("GENERAL_CHAT", "hi", available=["Ollama", "DeepSeek-V4"], prefer_speed=True)
        self.assertEqual(m, "Ollama")

    def test_long_input_boosts_large_context(self):
        long_input = "x " * 1500  # >2000 chars
        m = select_model("GENERAL_CHAT", long_input, available=["Ollama", "Kimi-2.5"])
        self.assertEqual(m, "Kimi-2.5")

    def test_moe_bonus_for_complex_tasks(self):
        # DeepSeek-V4 has moe_expert_count=256, complex task should trigger MoE bonus
        m = select_model(
            "MULTI_SIGNAL",
            "evaluate and analyse the trade-off in this strategy",
            available=["Kimi-2.5", "DeepSeek-V4"],
        )
        self.assertEqual(m, "DeepSeek-V4")

    def test_default_models_all_registered(self):
        models = list_models()
        self.assertIn("Ollama", models)
        self.assertIn("DeepSeek-V4", models)
        self.assertIn("Kimi-2.5", models)
        self.assertIn("Dolphin", models)

    def test_no_user_input(self):
        m = select_model("GENERAL_CHAT", "", available=["Ollama", "Kimi-2.5"])
        self.assertIn(m, ["Ollama", "Kimi-2.5"])


class TestModelProfile(unittest.TestCase):
    def test_get_profile(self):
        p = get_profile("Ollama")
        self.assertIsNotNone(p)
        self.assertEqual(p.name, "Ollama")
        self.assertEqual(p.vram_gb, 0.4)

    def test_get_unknown_profile(self):
        p = get_profile("NonExistent")
        self.assertIsNone(p)

    def test_register_model(self):
        custom = ModelProfile(
            name="TestModel",
            strengths=["GENERAL_CHAT"],
            max_context=8192,
            speed_tier=1,
            quality_tier=2,
        )
        register_model(custom)
        p = get_profile("TestModel")
        self.assertIsNotNone(p)
        self.assertEqual(p.name, "TestModel")
        # cleanup
        if "TestModel" in _PROFILES:
            del _PROFILES["TestModel"]

    def test_deepseek_is_moe(self):
        p = get_profile("DeepSeek-V4")
        self.assertGreater(p.moe_expert_count, 0)

    def test_ollama_is_dense(self):
        p = get_profile("Ollama")
        self.assertEqual(p.moe_expert_count, 0)


if __name__ == "__main__":
    unittest.main()
