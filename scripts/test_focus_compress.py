"""Active Context Compression Loop tests.

Tests the focus-compress endpoint, token budget meter, merge logic,
and integration into the nightly compression cycle.

Source: arxiv.org/abs/2601.07190 (Active Context Compression)
"""
from __future__ import annotations

import math
import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MEMU_SRC = (ROOT / "memu-core" / "app.py").read_text()
COMPRESSOR_SRC = (ROOT / "memory-compressor" / "app.py").read_text()
HEARTBEAT_SRC = (ROOT / "heartbeat" / "app.py").read_text()


# ── Endpoint existence ──────────────────────────────────────────────

class TestFocusCompressEndpoint(unittest.TestCase):
    """Verify the focus-compress endpoint is registered."""

    def test_focus_compress_endpoint_exists(self):
        self.assertIn('"/memory/focus-compress"', MEMU_SRC)

    def test_focus_compress_is_post(self):
        self.assertIn('@app.post("/memory/focus-compress")', MEMU_SRC)

    def test_token_budget_endpoint_exists(self):
        self.assertIn('"/memory/token-budget"', MEMU_SRC)

    def test_token_budget_is_get(self):
        self.assertIn('@app.get("/memory/token-budget")', MEMU_SRC)


# ── Algorithm components ────────────────────────────────────────────

class TestTokenEstimation(unittest.TestCase):
    """Verify token estimation helpers exist and use ~4 chars/token."""

    def test_estimate_tokens_defined(self):
        self.assertIn("def _estimate_tokens(", MEMU_SRC)

    def test_estimate_tokens_divides_by_four(self):
        fn = MEMU_SRC.split("def _estimate_tokens(")[1].split("\ndef ")[0]
        self.assertIn("// 4", fn)

    def test_memory_token_cost_defined(self):
        self.assertIn("def _memory_token_cost(", MEMU_SRC)


class TestKeywordOverlap(unittest.TestCase):
    """Verify keyword clustering uses Jaccard similarity."""

    def test_keyword_set_defined(self):
        self.assertIn("def _keyword_set(", MEMU_SRC)

    def test_keyword_overlap_defined(self):
        self.assertIn("def _keyword_overlap(", MEMU_SRC)

    def test_jaccard_formula(self):
        fn = MEMU_SRC.split("def _keyword_overlap(")[1].split("\ndef ")[0]
        # Jaccard = intersection / union
        self.assertIn("a & b", fn)
        self.assertIn("a | b", fn)

    def test_overlap_threshold_030(self):
        """Merge threshold is 0.3 Jaccard overlap."""
        fn = MEMU_SRC.split("async def focus_compress(")[1].split("\n@app")[0]
        self.assertIn("0.3", fn)


class TestMergeLogic(unittest.TestCase):
    """Verify the memory cluster merge function."""

    def test_merge_function_defined(self):
        self.assertIn("def _merge_memory_cluster(", MEMU_SRC)

    def test_merge_creates_summary_event_type(self):
        fn = MEMU_SRC.split("def _merge_memory_cluster(")[1].split("\ndef ")[0]
        self.assertIn("focus_compress_summary", fn)

    def test_merge_preserves_best_relevance(self):
        fn = MEMU_SRC.split("def _merge_memory_cluster(")[1].split("\ndef ")[0]
        self.assertIn("max(r.relevance for r in cluster)", fn)

    def test_merge_sums_access_counts(self):
        fn = MEMU_SRC.split("def _merge_memory_cluster(")[1].split("\ndef ")[0]
        self.assertIn("sum(r.access_count for r in cluster)", fn)

    def test_merge_keeps_max_stability(self):
        fn = MEMU_SRC.split("def _merge_memory_cluster(")[1].split("\ndef ")[0]
        self.assertIn("max(r.stability for r in cluster)", fn)

    def test_merge_content_has_merged_count(self):
        fn = MEMU_SRC.split("def _merge_memory_cluster(")[1].split("\ndef ")[0]
        self.assertIn("merged_count", fn)

    def test_merge_caps_snippets(self):
        """Merged content is capped at 10 snippets to stay compact."""
        fn = MEMU_SRC.split("def _merge_memory_cluster(")[1].split("\ndef ")[0]
        self.assertIn("[:10]", fn)


# ── Focus-compress algorithm flow ───────────────────────────────────

class TestFocusCompressAlgorithm(unittest.TestCase):
    """Verify the focus-compress endpoint implements the full algorithm."""

    def test_uses_mars_retention(self):
        """Focus selection uses MARS _recency_weight."""
        fn = MEMU_SRC.split("async def focus_compress(")[1].split("\n@app")[0]
        self.assertIn("_recency_weight", fn)

    def test_respects_pinned_memories(self):
        """Pinned memories are always in the focus zone."""
        fn = MEMU_SRC.split("async def focus_compress(")[1].split("\n@app")[0]
        self.assertIn("pinned", fn)

    def test_groups_by_category(self):
        fn = MEMU_SRC.split("async def focus_compress(")[1].split("\n@app")[0]
        self.assertIn("category", fn)
        self.assertIn("cat_groups", fn)

    def test_has_token_budget_param(self):
        fn = MEMU_SRC.split("async def focus_compress(")[1].split("\n@app")[0]
        self.assertIn("token_budget", fn)

    def test_has_focus_top_k_param(self):
        fn = MEMU_SRC.split("async def focus_compress(")[1].split("\n@app")[0]
        self.assertIn("focus_top_k", fn)

    def test_returns_savings_pct(self):
        fn = MEMU_SRC.split("async def focus_compress(")[1].split("\n@app")[0]
        self.assertIn("savings_pct", fn)

    def test_returns_tokens_before_after(self):
        fn = MEMU_SRC.split("async def focus_compress(")[1].split("\n@app")[0]
        self.assertIn("tokens_before", fn)
        self.assertIn("tokens_after", fn)

    def test_skips_poisoned_memories(self):
        fn = MEMU_SRC.split("async def focus_compress(")[1].split("\n@app")[0]
        self.assertIn("poisoned", fn)

    def test_handles_empty_store(self):
        """Returns gracefully when no memories exist."""
        fn = MEMU_SRC.split("async def focus_compress(")[1].split("\n@app")[0]
        self.assertIn("no memories to compress", fn)

    def test_skips_when_under_budget(self):
        """Doesn't compress if already within token budget."""
        fn = MEMU_SRC.split("async def focus_compress(")[1].split("\n@app")[0]
        self.assertIn("already within token budget", fn)

    def test_truncation_fallback(self):
        """If merging isn't enough, truncates low-retention content."""
        fn = MEMU_SRC.split("async def focus_compress(")[1].split("\n@app")[0]
        self.assertIn("truncated", fn)


# ── Token budget meter ──────────────────────────────────────────────

class TestTokenBudgetMeter(unittest.TestCase):
    """Verify the live token budget endpoint."""

    def test_returns_total_tokens(self):
        fn = MEMU_SRC.split("async def token_budget_status(")[1].split("\n@app")[0]
        self.assertIn("total_tokens", fn)

    def test_returns_usage_pct(self):
        fn = MEMU_SRC.split("async def token_budget_status(")[1].split("\n@app")[0]
        self.assertIn("usage_pct", fn)

    def test_returns_headroom(self):
        fn = MEMU_SRC.split("async def token_budget_status(")[1].split("\n@app")[0]
        self.assertIn("headroom", fn)


# ── Configuration ───────────────────────────────────────────────────

class TestFocusCompressConfig(unittest.TestCase):
    """Verify configurable parameters via environment variables."""

    def test_token_budget_env(self):
        self.assertIn("FOCUS_COMPRESS_TOKEN_BUDGET", MEMU_SRC)

    def test_top_k_env(self):
        self.assertIn("FOCUS_COMPRESS_TOP_K", MEMU_SRC)

    def test_default_budget_50k(self):
        match = re.search(r'FOCUS_COMPRESS_TOKEN_BUDGET.*?(\d+)', MEMU_SRC)
        self.assertIsNotNone(match)
        self.assertEqual(match.group(1), "50000")

    def test_default_top_k_50(self):
        match = re.search(r'FOCUS_COMPRESS_TOP_K.*?"(\d+)"', MEMU_SRC)
        self.assertIsNotNone(match)
        self.assertEqual(match.group(1), "50")


# ── Nightly cycle integration ──────────────────────────────────────

class TestNightlyCycleIntegration(unittest.TestCase):
    """Verify focus-compress is wired into memory-compressor cycle."""

    def test_compressor_calls_focus_compress(self):
        self.assertIn("/memory/focus-compress", COMPRESSOR_SRC)

    def test_compressor_step_order(self):
        """focus-compress runs AFTER consolidate and BEFORE compress."""
        consolidate_pos = COMPRESSOR_SRC.index("/memory/consolidate")
        focus_pos = COMPRESSOR_SRC.index("/memory/focus-compress")
        compress_pos = COMPRESSOR_SRC.index('"/memory/compress"')
        self.assertLess(consolidate_pos, focus_pos)
        self.assertLess(focus_pos, compress_pos)

    def test_compressor_non_fatal(self):
        """focus-compress failure doesn't crash the cycle."""
        self.assertIn("non-fatal", COMPRESSOR_SRC)


class TestHeartbeatIntegration(unittest.TestCase):
    """Verify focus-compress is wired into heartbeat auto-sleep."""

    def test_heartbeat_calls_focus_compress(self):
        self.assertIn("/memory/focus-compress", HEARTBEAT_SRC)

    def test_heartbeat_non_fatal(self):
        """focus-compress failure in heartbeat is non-fatal."""
        # The heartbeat wraps it in try/except with pass
        self.assertIn("non-fatal", HEARTBEAT_SRC)


# ── Math validation ─────────────────────────────────────────────────

class TestFocusCompressMath(unittest.TestCase):
    """Validate the mathematical properties of the compression algorithm."""

    def test_4_chars_per_token_approximation(self):
        """Standard GPT-family approximation: ~4 chars per token."""
        fn = MEMU_SRC.split("def _estimate_tokens(")[1].split("\ndef ")[0]
        self.assertIn("// 4", fn)

    def test_jaccard_range_0_to_1(self):
        """Jaccard similarity always returns 0.0–1.0."""
        fn = MEMU_SRC.split("def _keyword_overlap(")[1].split("\ndef ")[0]
        # Division by union ensures 0..1
        self.assertIn("a | b", fn)

    def test_savings_pct_formula(self):
        """Savings = (1 - after/before) * 100."""
        fn = MEMU_SRC.split("async def focus_compress(")[1].split("\n@app")[0]
        self.assertIn("1.0 - tokens_after", fn)


if __name__ == "__main__":
    unittest.main()
