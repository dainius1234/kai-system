"""Bio-inspired Self-Healing Phase tests (Gap 5).

Tests the 4-phase ReCiSt healing engine in common/resilience.py:
  Phase 1 — Containment
  Phase 2 — Diagnosis
  Phase 3 — Meta-Cognitive
  Phase 4 — Knowledge

Source: Multi-agent self-healing patterns (ReCiSt model).
"""
from __future__ import annotations

import asyncio
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESILIENCE_SRC = (ROOT / "common" / "resilience.py").read_text()


# ── Phase Constants ─────────────────────────────────────────────────

class TestPhaseConstants(unittest.TestCase):
    """Verify all 4 phase constants are defined."""

    def test_containment_phase(self):
        self.assertIn('PHASE_CONTAINMENT', RESILIENCE_SRC)

    def test_diagnosis_phase(self):
        self.assertIn('PHASE_DIAGNOSIS', RESILIENCE_SRC)

    def test_meta_cognitive_phase(self):
        self.assertIn('PHASE_META_COGNITIVE', RESILIENCE_SRC)

    def test_knowledge_phase(self):
        self.assertIn('PHASE_KNOWLEDGE', RESILIENCE_SRC)

    def test_healthy_phase(self):
        self.assertIn('PHASE_HEALTHY', RESILIENCE_SRC)

    def test_phase_order(self):
        self.assertIn('_PHASE_ORDER', RESILIENCE_SRC)


# ── Failure Record ──────────────────────────────────────────────────

class TestFailureRecord(unittest.TestCase):
    """Verify failure record dataclass."""

    def test_class_defined(self):
        self.assertIn("class FailureRecord", RESILIENCE_SRC)

    def test_has_service_field(self):
        cls_section = RESILIENCE_SRC.split("class FailureRecord")[1].split("\nclass ")[0]
        self.assertIn("service", cls_section)

    def test_has_error_field(self):
        cls_section = RESILIENCE_SRC.split("class FailureRecord")[1].split("\nclass ")[0]
        self.assertIn("error", cls_section)

    def test_has_fix_applied_field(self):
        cls_section = RESILIENCE_SRC.split("class FailureRecord")[1].split("\nclass ")[0]
        self.assertIn("fix_applied", cls_section)


# ── Healing Engine ──────────────────────────────────────────────────

class TestHealingEngineClass(unittest.TestCase):
    """Verify HealingEngine dataclass structure."""

    def test_class_defined(self):
        self.assertIn("class HealingEngine", RESILIENCE_SRC)

    def test_has_history_limit(self):
        cls_section = RESILIENCE_SRC.split("class HealingEngine")[1].split("\n    # ──")[0]
        self.assertIn("history_limit", cls_section)

    def test_has_escalation_threshold(self):
        cls_section = RESILIENCE_SRC.split("class HealingEngine")[1].split("\n    # ──")[0]
        self.assertIn("escalation_threshold", cls_section)

    def test_has_knowledge_store(self):
        cls_section = RESILIENCE_SRC.split("class HealingEngine")[1]
        self.assertIn("_knowledge", cls_section)


# ── Phase 1: Containment ───────────────────────────────────────────

class TestContainment(unittest.TestCase):
    """Verify containment phase isolates failures."""

    def test_containment_method(self):
        self.assertIn("def _containment(", RESILIENCE_SRC)

    def test_opens_circuit_breaker(self):
        fn = RESILIENCE_SRC.split("def _containment(")[1].split("\n    # ──")[0]
        self.assertIn("record_failure", fn)

    def test_advances_to_diagnosis(self):
        fn = RESILIENCE_SRC.split("def _containment(")[1].split("\n    # ──")[0]
        self.assertIn("PHASE_DIAGNOSIS", fn)


# ── Phase 2: Diagnosis ─────────────────────────────────────────────

class TestDiagnosis(unittest.TestCase):
    """Verify diagnosis phase pattern analysis."""

    def test_diagnosis_method(self):
        self.assertIn("def _diagnosis(", RESILIENCE_SRC)

    def test_checks_knowledge_base(self):
        fn = RESILIENCE_SRC.split("def _diagnosis(")[1].split("\n    # ──")[0]
        self.assertIn("_knowledge", fn)

    def test_counts_consecutive_errors(self):
        fn = RESILIENCE_SRC.split("def _diagnosis(")[1].split("\n    # ──")[0]
        self.assertIn("consecutive", fn)

    def test_escalates_to_meta_cognitive(self):
        fn = RESILIENCE_SRC.split("def _diagnosis(")[1].split("\n    # ──")[0]
        self.assertIn("PHASE_META_COGNITIVE", fn)


# ── Phase 3: Meta-Cognitive ────────────────────────────────────────

class TestMetaCognitive(unittest.TestCase):
    """Verify meta-cognitive phase reflects on past fixes."""

    def test_meta_cognitive_method(self):
        self.assertIn("def _meta_cognitive(", RESILIENCE_SRC)

    def test_tracks_fixes_tried(self):
        fn = RESILIENCE_SRC.split("def _meta_cognitive(")[1].split("\n    # ──")[0]
        self.assertIn("fixes_tried", fn)

    def test_escalation_suggestions(self):
        fn = RESILIENCE_SRC.split("def _meta_cognitive(")[1].split("\n    # ──")[0]
        self.assertIn("restart_service", fn)
        self.assertIn("rebuild_container", fn)
        self.assertIn("alert_operator", fn)


# ── Phase 4: Knowledge ─────────────────────────────────────────────

class TestKnowledge(unittest.TestCase):
    """Verify knowledge phase records fixes."""

    def test_knowledge_method(self):
        self.assertIn("def _record_knowledge(", RESILIENCE_SRC)

    def test_stores_fix(self):
        fn = RESILIENCE_SRC.split("def _record_knowledge(")[1].split("\n    # ──")[0]
        self.assertIn("_knowledge", fn)

    def test_transitions_to_healthy(self):
        fn = RESILIENCE_SRC.split("def _record_knowledge(")[1].split("\n    # ──")[0]
        self.assertIn("PHASE_HEALTHY", fn)


# ── Main Heal Entry Point ──────────────────────────────────────────

class TestHealMethod(unittest.TestCase):
    """Verify the main heal() coroutine."""

    def test_heal_is_async(self):
        self.assertIn("async def heal(", RESILIENCE_SRC)

    def test_records_failure(self):
        fn = RESILIENCE_SRC.split("async def heal(")[1].split("\n    def ")[0]
        self.assertIn("FailureRecord", fn)
        self.assertIn("_history", fn)

    def test_history_bounded(self):
        fn = RESILIENCE_SRC.split("async def heal(")[1].split("\n    def ")[0]
        self.assertIn("history_limit", fn)

    def test_shortcut_with_fix_applied(self):
        fn = RESILIENCE_SRC.split("async def heal(")[1].split("\n    def ")[0]
        self.assertIn("fix_applied", fn)
        self.assertIn("_record_knowledge", fn)


# ── Utility Methods ─────────────────────────────────────────────────

class TestHealingUtilities(unittest.TestCase):
    """Verify reset, status, and knowledge inspection."""

    def test_reset_method(self):
        self.assertIn("def reset(", RESILIENCE_SRC)

    def test_status_method(self):
        self.assertIn("def status(", RESILIENCE_SRC)

    def test_knowledge_base_method(self):
        self.assertIn("def knowledge_base(", RESILIENCE_SRC)


# ── Live behaviour (import-based) ──────────────────────────────────

class TestHealingEngineLive(unittest.TestCase):
    """Run the healing engine to verify phase progression."""

    def test_full_phase_progression(self):
        """Heal should progress through all phases."""
        import sys
        sys.path.insert(0, str(ROOT))
        from common.resilience import HealingEngine

        healer = HealingEngine(escalation_threshold=2)
        loop = asyncio.new_event_loop()

        # Phase 1: containment
        r1 = loop.run_until_complete(healer.heal("svc-a", "timeout"))
        self.assertEqual(r1["phase"], "containment")

        # Phase 2: diagnosis
        r2 = loop.run_until_complete(healer.heal("svc-a", "timeout"))
        self.assertEqual(r2["phase"], "diagnosis")

        # Phase 3: meta-cognitive (escalated after 2 consecutive)
        r3 = loop.run_until_complete(healer.heal("svc-a", "timeout"))
        self.assertEqual(r3["phase"], "meta_cognitive")

        # Phase 4: knowledge
        r4 = loop.run_until_complete(healer.heal("svc-a", "timeout"))
        self.assertEqual(r4["phase"], "knowledge")

        loop.close()

    def test_knowledge_shortcut(self):
        """Providing fix_applied should go straight to knowledge."""
        import sys
        sys.path.insert(0, str(ROOT))
        from common.resilience import HealingEngine

        healer = HealingEngine()
        loop = asyncio.new_event_loop()
        r = loop.run_until_complete(
            healer.heal("svc-b", "oom", fix_applied="restart")
        )
        self.assertEqual(r["phase"], "knowledge")
        self.assertEqual(r["fix"], "restart")
        loop.close()

    def test_status_snapshot(self):
        import sys
        sys.path.insert(0, str(ROOT))
        from common.resilience import HealingEngine

        healer = HealingEngine()
        loop = asyncio.new_event_loop()
        loop.run_until_complete(healer.heal("svc-c", "err"))
        snap = healer.status("svc-c")
        self.assertIn("phase", snap)
        self.assertIn("failures", snap)
        loop.close()


if __name__ == "__main__":
    unittest.main()
