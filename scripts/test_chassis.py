"""Tests for the plug-and-play chassis layer: model_registry + prompt_templates.

Validates token counting accuracy, model-aware context budgets,
prompt template scaling, and LLM response validation.
"""
from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


class TestModelRegistry(unittest.TestCase):
    """Model registry — the source of truth for plug-and-play."""

    def test_active_model_default(self):
        from common.model_registry import active_model
        self.assertIsInstance(active_model(), str)
        self.assertTrue(len(active_model()) > 0)

    def test_get_known_model(self):
        from common.model_registry import get_model_spec
        spec = get_model_spec("qwen2:0.5b")
        self.assertEqual(spec.context_window, 4096)
        self.assertEqual(spec.quality_tier, 1)

    def test_get_large_model(self):
        from common.model_registry import get_model_spec
        spec = get_model_spec("qwen2.5:7b")
        self.assertEqual(spec.context_window, 32768)
        self.assertTrue(spec.quality_tier >= 2)
        self.assertTrue(spec.supports_json)

    def test_get_huge_context_model(self):
        from common.model_registry import get_model_spec
        spec = get_model_spec("kimi-2.5")
        self.assertEqual(spec.context_window, 131072)
        self.assertTrue(spec.supports_vision)

    def test_unknown_model_gets_defaults(self):
        from common.model_registry import get_model_spec
        spec = get_model_spec("totally-fake-model:99b")
        self.assertEqual(spec.context_window, 4096)  # conservative default

    def test_prefix_match(self):
        """Models with quantization suffixes should match base spec."""
        from common.model_registry import get_model_spec
        spec = get_model_spec("qwen2.5:7b-q4_K_M")
        self.assertEqual(spec.context_window, 32768)

    def test_context_budget_small_model(self):
        from common.model_registry import context_budget
        budget = context_budget("qwen2:0.5b")
        self.assertEqual(budget, 4096 - 1024)  # 3072

    def test_context_budget_large_model(self):
        from common.model_registry import context_budget
        budget = context_budget("qwen2.5:7b")
        self.assertEqual(budget, 32768 - 4096)  # 28672

    def test_model_timeout_varies(self):
        from common.model_registry import model_timeout
        small = model_timeout("qwen2:0.5b")
        large = model_timeout("llama3.3:70b")
        self.assertLess(small, large)

    def test_list_models_not_empty(self):
        from common.model_registry import list_models
        models = list_models()
        self.assertGreater(len(models), 10)


class TestTokenCounting(unittest.TestCase):
    """Token counting — accuracy matters for context budgets."""

    def test_empty_string(self):
        from common.model_registry import count_tokens
        self.assertEqual(count_tokens(""), 0)

    def test_hello_world(self):
        from common.model_registry import count_tokens
        tokens = count_tokens("Hello, world!")
        # tiktoken: 4 tokens. Heuristic: ~4. Both should be close.
        self.assertGreater(tokens, 0)
        self.assertLess(tokens, 20)

    def test_long_text_accuracy(self):
        """Token count should be within 20% of tiktoken for English text."""
        from common.model_registry import count_tokens, _HAS_TIKTOKEN
        text = "The quick brown fox jumps over the lazy dog. " * 50
        tokens = count_tokens(text)
        if _HAS_TIKTOKEN:
            # With tiktoken, should be very accurate
            self.assertGreater(tokens, 400)
            self.assertLess(tokens, 600)
        else:
            # Heuristic should at least be in the right ballpark
            self.assertGreater(tokens, 200)

    def test_count_messages_tokens(self):
        from common.model_registry import count_messages_tokens
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello"},
        ]
        tokens = count_messages_tokens(messages)
        self.assertGreater(tokens, 5)
        # Should include per-message overhead
        self.assertGreater(tokens, 2 + 4 + 4)  # 2 priming + 2*4 overhead

    def test_count_tokens_never_negative(self):
        from common.model_registry import count_tokens
        for text in ["", " ", "a", "hello world", "x" * 10000]:
            self.assertGreaterEqual(count_tokens(text), 0)


class TestPromptTemplates(unittest.TestCase):
    """Prompt templates — model-aware prompt construction."""

    def test_system_prompt_tier1_is_short(self):
        from common.prompt_templates import build_system_prompt
        prompt = build_system_prompt("PUB", model="qwen2:0.5b")
        self.assertIn("Kai", prompt)
        self.assertNotIn("Reasoning guidelines", prompt)  # too small

    def test_system_prompt_tier2_has_reasoning(self):
        from common.prompt_templates import build_system_prompt
        prompt = build_system_prompt("PUB", model="qwen2.5:7b")
        self.assertIn("Reasoning guidelines", prompt)
        self.assertIn("step-by-step", prompt)

    def test_system_prompt_tier3_has_json(self):
        from common.prompt_templates import build_system_prompt
        prompt = build_system_prompt("WORK", model="deepseek-v4")
        self.assertIn("JSON", prompt)

    def test_work_mode_mentions_uk(self):
        from common.prompt_templates import build_system_prompt
        prompt = build_system_prompt("WORK", model="qwen2.5:7b")
        self.assertIn("UK", prompt)

    def test_extra_context_injected(self):
        from common.prompt_templates import build_system_prompt
        prompt = build_system_prompt("PUB", extra_context="Tax year 2025-26")
        self.assertIn("Tax year 2025-26", prompt)

    def test_build_chat_messages_structure(self):
        from common.prompt_templates import build_chat_messages
        msgs = build_chat_messages(
            "System prompt here",
            [{"role": "user", "content": "Hi"},
             {"role": "assistant", "content": "Hello"}],
            "What's the weather?",
        )
        self.assertEqual(msgs[0]["role"], "system")
        self.assertEqual(msgs[-1]["role"], "user")
        self.assertEqual(msgs[-1]["content"], "What's the weather?")
        self.assertEqual(len(msgs), 4)  # system + 2 history + user

    def test_evidence_injected_into_system(self):
        from common.prompt_templates import build_chat_messages
        msgs = build_chat_messages("Base", [], "Q", evidence="Some facts")
        self.assertIn("Some facts", msgs[0]["content"])

    def test_fact_check_prompt_scales(self):
        from common.prompt_templates import fact_check_prompt
        # Tiny model: short prompt
        os.environ["OLLAMA_MODEL"] = "qwen2:0.5b"
        short = fact_check_prompt("Earth is flat")
        os.environ["OLLAMA_MODEL"] = "qwen2.5:7b"
        long = fact_check_prompt("Earth is flat")
        self.assertGreater(len(long), len(short))
        self.assertIn("Verdict", long)
        # Restore
        os.environ.pop("OLLAMA_MODEL", None)

    def test_planning_prompt_structure(self):
        from common.prompt_templates import planning_prompt
        os.environ["OLLAMA_MODEL"] = "qwen2.5:7b"
        prompt = planning_prompt("Build a website", "Budget £500")
        self.assertIn("step-by-step", prompt)
        self.assertIn("Budget", prompt)
        os.environ.pop("OLLAMA_MODEL", None)

    def test_register_custom_template(self):
        from common.prompt_templates import register_template, get_template
        register_template("greeting", "Hello {{name}}")
        self.assertEqual(get_template("greeting"), "Hello {{name}}")
        self.assertIsNone(get_template("nonexistent"))


class TestLLMResponseValidation(unittest.TestCase):
    """LLM response validation — catch bad outputs."""

    def test_empty_response_detected(self):
        from common.llm import _validate_llm_response
        result = _validate_llm_response("")
        self.assertIn("empty", result.lower())

    def test_whitespace_only_detected(self):
        from common.llm import _validate_llm_response
        result = _validate_llm_response("   \n  ")
        self.assertIn("empty", result.lower())

    def test_valid_response_passes_through(self):
        from common.llm import _validate_llm_response
        result = _validate_llm_response("This is a good answer.")
        self.assertEqual(result, "This is a good answer.")

    def test_error_json_detected(self):
        from common.llm import _validate_llm_response
        result = _validate_llm_response('{"error": "model not found"}')
        self.assertIn("error", result.lower())


class TestFusionAgreement(unittest.TestCase):
    """Fusion agreement — upgraded from pure Jaccard."""

    def _import_fusion(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "fusion_app",
            os.path.join(os.path.dirname(__file__), "..", "fusion-engine", "app.py"),
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    def test_jaccard_identical_texts(self):
        mod = self._import_fusion()
        texts = ["quantum physics analysis theory", "quantum physics analysis theory"]
        self.assertGreater(mod._jaccard_agreement(texts), 0.99)

    def test_jaccard_different_texts(self):
        mod = self._import_fusion()
        texts = ["quantum physics theory", "banana bread recipe baking"]
        score = mod._jaccard_agreement(texts)
        self.assertLess(score, 0.3)

    def test_measure_agreement_single_response(self):
        mod = self._import_fusion()
        responses = [mod.SpecialistResponse(
            specialist="A", response="hello", latency_ms=1.0, source="live"
        )]
        self.assertEqual(mod._measure_agreement(responses), 1.0)

    def test_measure_agreement_error_responses(self):
        mod = self._import_fusion()
        responses = [
            mod.SpecialistResponse(
                specialist="A", response="ok", latency_ms=1.0, source="live"),
            mod.SpecialistResponse(
                specialist="B", response="err", latency_ms=1.0, source="error"),
        ]
        self.assertEqual(mod._measure_agreement(responses), 0.0)


class TestModelAwareContextBudget(unittest.TestCase):
    """Context budget adapts to model size."""

    def test_small_model_small_budget(self):
        from common.model_registry import context_budget
        b = context_budget("qwen2:0.5b")
        self.assertLessEqual(b, 4096)

    def test_large_model_large_budget(self):
        from common.model_registry import context_budget
        b = context_budget("qwen2.5:7b")
        self.assertGreater(b, 20000)

    def test_huge_model_huge_budget(self):
        from common.model_registry import context_budget
        b = context_budget("kimi-2.5")
        self.assertGreater(b, 100000)

    def test_budget_always_positive(self):
        from common.model_registry import context_budget, list_models
        for name in list_models():
            self.assertGreater(context_budget(name), 0)


if __name__ == "__main__":
    unittest.main()
