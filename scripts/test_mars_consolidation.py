"""MARS Memory Consolidation tests.

Tests the Ebbinghaus forgetting curve with stability parameter,
conscience-filtered pruning, and nightly consolidation cycle.
"""
from __future__ import annotations

import math
import re
import textwrap
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MEMU_SRC = (ROOT / "memu-core" / "app.py").read_text()
COMPRESSOR_SRC = (ROOT / "memory-compressor" / "app.py").read_text()


# ── Pattern Tests (static analysis of source code) ──────────────────

class TestMARSRecencyFormula(unittest.TestCase):
    """Verify _recency_weight uses MARS retention formula R = e^{-τ/S}."""

    def test_uses_stability_parameter(self):
        fn = MEMU_SRC.split("def _recency_weight(")[1].split("\ndef ")[0]
        self.assertIn("stability", fn)

    def test_uses_tau_over_S(self):
        fn = MEMU_SRC.split("def _recency_weight(")[1].split("\ndef ")[0]
        self.assertIn("tau", fn)
        self.assertIn("math.exp(", fn)

    def test_no_half_life_in_formula(self):
        """MARS uses stability S directly, not RECENCY_HALF_LIFE_DAYS."""
        fn = MEMU_SRC.split("def _recency_weight(")[1].split("\ndef ")[0]
        self.assertNotIn("RECENCY_HALF_LIFE_DAYS", fn)

    def test_stability_floor(self):
        """Stability should have a safety floor to avoid division by zero."""
        fn = MEMU_SRC.split("def _recency_weight(")[1].split("\ndef ")[0]
        self.assertIn("max(stability", fn)


class TestMARSStabilityGrowth(unittest.TestCase):
    """Verify stability grows on memory access (retrieval strengthens)."""

    def test_stability_grows_on_access(self):
        fn = MEMU_SRC.split("def retrieve_ranked(")[1].split("\ndef ")[0]
        self.assertIn("stability", fn)
        self.assertIn("math.sqrt", fn)

    def test_stability_capped(self):
        fn = MEMU_SRC.split("def retrieve_ranked(")[1].split("\ndef ")[0]
        self.assertIn("365.0", fn)

    def test_stability_persisted(self):
        fn = MEMU_SRC.split("def retrieve_ranked(")[1].split("\ndef ")[0]
        self.assertIn("stability=record.stability", fn)


class TestMARSMemoryRecord(unittest.TestCase):
    """Verify MemoryRecord has stability field."""

    def test_stability_field_exists(self):
        self.assertIn("stability: float = 1.0", MEMU_SRC)

    def test_stability_in_pg_schema(self):
        schema = MEMU_SRC.split("CREATE TABLE IF NOT EXISTS memories")[1].split(");")[0]
        self.assertIn("stability float DEFAULT 1.0", schema)

    def test_stability_in_select_cols(self):
        self.assertIn("stability", MEMU_SRC.split("_SELECT_COLS")[1].split(")")[0])

    def test_stability_in_insert(self):
        insert = MEMU_SRC.split("INSERT INTO memories")[1].split("ON CONFLICT")[0]
        self.assertIn("stability", insert)

    def test_stability_in_allowed_update(self):
        fn = MEMU_SRC.split("class PGVectorStore")[1].split("def update_record")[1].split("def ")[0]
        self.assertIn("stability", fn)

    def test_stability_migration(self):
        """Stability column should be in the migration list."""
        migration = MEMU_SRC.split("for col, typ, default in [")[1].split("]:")[0]
        self.assertIn('"stability"', migration)


class TestMARSConsolidateEndpoint(unittest.TestCase):
    """Verify /memory/consolidate endpoint exists with proper structure."""

    def test_consolidate_endpoint_exists(self):
        self.assertIn('"/memory/consolidate"', MEMU_SRC)

    def test_consolidate_prunes(self):
        fn = MEMU_SRC.split("def mars_consolidate")[1].split("\n@app.")[0]
        self.assertIn("MARS_PRUNE_THRESHOLD", fn)
        self.assertIn("pruned", fn)

    def test_consolidate_strengthens(self):
        fn = MEMU_SRC.split("def mars_consolidate")[1].split("\n@app.")[0]
        self.assertIn("strengthened", fn)

    def test_consolidate_fades(self):
        fn = MEMU_SRC.split("def mars_consolidate")[1].split("\n@app.")[0]
        self.assertIn("faded", fn)

    def test_conscience_filter(self):
        """Conscience-linked memories should survive pruning."""
        fn = MEMU_SRC.split("def mars_consolidate")[1].split("\n@app.")[0]
        self.assertIn("conscience_saved", fn)
        self.assertIn("_formed_values", fn)

    def test_uses_delete_record(self):
        fn = MEMU_SRC.split("def mars_consolidate")[1].split("\n@app.")[0]
        self.assertIn("delete_record", fn)

    def test_returns_stats(self):
        fn = MEMU_SRC.split("def mars_consolidate")[1].split("\n@app.")[0]
        for key in ["pruned", "faded", "strengthened", "conscience_saved", "skipped"]:
            self.assertIn(f'"{key}"', fn)


class TestMARSDecayEndpoint(unittest.TestCase):
    """Verify /memory/decay uses MARS stability."""

    def test_decay_uses_stability(self):
        fn = MEMU_SRC.split("def apply_spaced_repetition_decay")[1].split("\n@app.")[0]
        self.assertIn("stability", fn)

    def test_decay_calls_recency_with_stability(self):
        fn = MEMU_SRC.split("def apply_spaced_repetition_decay")[1].split("\n@app.")[0]
        self.assertIn("stability=stab", fn)


class TestDeleteRecord(unittest.TestCase):
    """Verify delete_record exists on both store implementations."""

    def test_protocol_has_delete_record(self):
        self.assertIn("def delete_record(self, record_id: str) -> bool:", MEMU_SRC)

    def test_pg_store_has_delete_record(self):
        pg_section = MEMU_SRC.split("class PGVectorStore")[1].split("class InMemoryVectorStore")[0]
        self.assertIn("def delete_record(", pg_section)
        self.assertIn("DELETE FROM memories WHERE id = %s", pg_section)

    def test_in_memory_store_has_delete_record(self):
        mem_section = MEMU_SRC.split("class InMemoryVectorStore")[1]
        self.assertIn("def delete_record(", mem_section)


class TestCompressorMARSIntegration(unittest.TestCase):
    """Verify memory-compressor calls /memory/consolidate in nightly cycle."""

    def test_consolidation_step_in_cycle(self):
        self.assertIn("/memory/consolidate", COMPRESSOR_SRC)

    def test_consolidation_before_compress(self):
        """MARS consolidation _call_memu should appear before compress _call_memu."""
        cycle_fn = COMPRESSOR_SRC.split("async def run_compression_cycle")[1].split("\nasync def ")[0]
        # Find the actual _call_memu calls (not comments)
        consolidate_pos = cycle_fn.index('_call_memu("/memory/consolidate")')
        compress_pos = cycle_fn.index('_call_memu("/memory/compress")')
        self.assertLess(consolidate_pos, compress_pos)

    def test_pruned_count_in_result(self):
        self.assertIn('"pruned"', COMPRESSOR_SRC)


# ── Runtime Tests (actual math validation) ──────────────────────────

class TestMARSMath(unittest.TestCase):
    """Validate the MARS retention formula directly."""

    def test_fresh_memory_full_retention(self):
        """R = e^{0/1} = 1.0 for a brand new memory."""
        tau = 0.0
        S = 1.0
        R = math.exp(-tau / S)
        self.assertAlmostEqual(R, 1.0)

    def test_one_day_low_stability(self):
        """R = e^{-1/1} ≈ 0.368 — rapid forgetting."""
        tau = 1.0
        S = 1.0
        R = math.exp(-tau / S)
        self.assertAlmostEqual(R, 0.3679, places=3)

    def test_one_day_high_stability(self):
        """R = e^{-1/30} ≈ 0.967 — well-rehearsed memory barely decays."""
        tau = 1.0
        S = 30.0
        R = math.exp(-tau / S)
        self.assertAlmostEqual(R, 0.9672, places=3)

    def test_stability_growth_formula(self):
        """S_new = S * (1 + 0.1 * sqrt(interval_days + 1))."""
        S_old = 1.0
        interval_days = 3.0
        S_new = S_old * (1.0 + 0.1 * math.sqrt(interval_days + 1.0))
        self.assertAlmostEqual(S_new, 1.2, places=1)

    def test_ten_rehearsals_compound(self):
        """After 10 rehearsals at 1-day intervals, S should be ~3.4+."""
        S = 1.0
        for _ in range(10):
            S = S * (1.0 + 0.1 * math.sqrt(1.0 + 1.0))
            S = min(S, 365.0)
        self.assertGreater(S, 3.0)
        self.assertLess(S, 10.0)

    def test_prune_threshold_math(self):
        """A memory with S=1.0 at age 4 days has R ≈ 0.018 < 0.02."""
        tau = 4.0
        S = 1.0
        R = math.exp(-tau / S)
        self.assertLess(R, 0.02)  # would be pruned

    def test_rehearsed_survives_prune(self):
        """A memory with S=10 at age 4 days has R ≈ 0.67 — survives."""
        tau = 4.0
        S = 10.0
        R = math.exp(-tau / S)
        self.assertGreater(R, 0.5)  # far above prune threshold


if __name__ == "__main__":
    unittest.main()
