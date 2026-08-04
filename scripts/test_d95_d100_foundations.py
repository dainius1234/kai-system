"""Tests for D95–D100: GPU-era foundation stubs.

Verifies that:
  1. All stub classes are importable and instantiable.
  2. can_*() gates return False (Phase 0: GPU not provisioned).
  3. Stub return values have correct types and schema fields.
  4. D98 CognitiveFingerprintCollector records samples and reports progress.
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "agentic"))
sys.path.insert(0, str(Path(__file__).parent.parent / "memu-graph"))
sys.path.insert(0, str(Path(__file__).parent.parent / "common"))


def run(coro):
    # `asyncio.run` creates and closes its own loop. The previous
    # `get_event_loop().run_until_complete()` reused whatever loop the
    # thread happened to have — and FastAPI's TestClient, used by
    # suites that sort earlier, closes it on the way out. These tests
    # then failed with 'There is no current event loop', naming this
    # file rather than the one that closed the loop.
    return asyncio.run(coro)


# ── D95: DialecticalReasoner ──────────────────────────────────────────

def test_d95_import():
    from dialectic import DialecticalReasoner, DialecticalTriad
    assert DialecticalReasoner
    assert DialecticalTriad


def test_d95_can_synthesize_is_false():
    from dialectic import DialecticalReasoner
    r = DialecticalReasoner()
    assert r.can_synthesize() is False


def test_d95_stub_triad_returned():
    from dialectic import DialecticalReasoner
    r = DialecticalReasoner()
    triad = run(r.synthesize("Thesis statement here.", "Antithesis statement here."))
    assert triad.thesis == "Thesis statement here."
    assert triad.antithesis == "Antithesis statement here."
    assert "stub" in triad.resolution_level.lower()
    assert triad.confidence == 0.0


def test_d95_triad_synthesis_is_string():
    from dialectic import DialecticalReasoner
    r = DialecticalReasoner()
    triad = run(r.synthesize("A", "not A"))
    assert isinstance(triad.synthesis, str)
    assert len(triad.synthesis) > 0


# ── D96: AnalogyEngine ────────────────────────────────────────────────

def test_d96_import():
    from analogy import Analogy, AnalogyEngine
    assert AnalogyEngine
    assert Analogy


def test_d96_can_find_is_false():
    from analogy import AnalogyEngine
    e = AnalogyEngine()
    assert e.can_find() is False


def test_d96_stub_analogy_returned():
    from analogy import AnalogyEngine
    e = AnalogyEngine()
    analogy = run(e.find_analogy("army attacking fortress", "radiation destroying tumour"))
    assert analogy.source_domain == "army attacking fortress"
    assert analogy.confidence == 0.0
    assert "stub" in analogy.proposed_solution.lower() or "[STUB]" in analogy.proposed_solution


def test_d96_analogy_fields_typed():
    from analogy import Analogy
    a = Analogy(source_domain="A", target_domain="B")
    assert isinstance(a.structural_mappings, list)
    assert isinstance(a.graph_path, list)


# ── D97: ConceptBlender ───────────────────────────────────────────────

def test_d97_import():
    from concept_blend import BlendedConcept, ConceptBlender
    assert ConceptBlender
    assert BlendedConcept


def test_d97_can_blend_is_false():
    from concept_blend import ConceptBlender
    b = ConceptBlender()
    assert b.can_blend() is False


def test_d97_stub_blend_returned():
    from concept_blend import ConceptBlender
    b = ConceptBlender()
    blend = run(b.blend("fire", "flower"))
    assert blend.concept_a == "fire"
    assert blend.concept_b == "flower"
    assert blend.novelty_score == 0.0
    assert blend.confidence == 0.0


def test_d97_blend_emergent_properties_non_empty_stub():
    from concept_blend import ConceptBlender
    b = ConceptBlender()
    blend = run(b.blend("X", "Y"))
    assert len(blend.emergent_properties) > 0


# ── D98: CognitiveFingerprintCollector ───────────────────────────────

def test_d98_import():
    from cognitive_fingerprint import (
        CognitiveFingerprintCollector,
        InteractionSample,
        quick_sample,
    )
    assert CognitiveFingerprintCollector
    assert InteractionSample
    assert quick_sample


def test_d98_can_infer_is_false_below_threshold(tmp_path, monkeypatch):
    import cognitive_fingerprint as cf_mod
    monkeypatch.setattr(cf_mod, "FINGERPRINT_LOG", tmp_path / "fp.jsonl")
    from cognitive_fingerprint import CognitiveFingerprintCollector
    c = CognitiveFingerprintCollector()
    assert c.can_infer() is False


def test_d98_record_writes_sample(tmp_path, monkeypatch):
    import cognitive_fingerprint as cf_mod
    monkeypatch.setattr(cf_mod, "FINGERPRINT_LOG", tmp_path / "fp.jsonl")
    from cognitive_fingerprint import CognitiveFingerprintCollector, quick_sample
    c = CognitiveFingerprintCollector()
    c.record(quick_sample("How do I optimise this?", session_id="s1"))
    assert c.sample_count() == 1


def test_d98_multiple_samples_counted(tmp_path, monkeypatch):
    import cognitive_fingerprint as cf_mod
    fp_path = tmp_path / "fp.jsonl"
    monkeypatch.setattr(cf_mod, "FINGERPRINT_LOG", fp_path)
    from cognitive_fingerprint import CognitiveFingerprintCollector, quick_sample
    c = CognitiveFingerprintCollector()
    for i in range(5):
        c.record(quick_sample(f"query {i}"))
    assert c.sample_count() == 5


def test_d98_progress_report(tmp_path, monkeypatch):
    import cognitive_fingerprint as cf_mod
    monkeypatch.setattr(cf_mod, "FINGERPRINT_LOG", tmp_path / "fp.jsonl")
    from cognitive_fingerprint import CognitiveFingerprintCollector, quick_sample
    c = CognitiveFingerprintCollector()
    c.record(quick_sample("test"))
    p = c.progress()
    assert p["samples_collected"] == 1
    assert p["inference_threshold"] == 90
    assert p["ready_for_inference"] is False
    assert 0.0 < p["progress_pct"] < 100.0


def test_d98_quick_sample_fields():
    from cognitive_fingerprint import quick_sample
    s = quick_sample("How do I do this now?", session_id="abc", query_type="question")
    assert s.query.startswith("How do I do this")
    assert s.session_id == "abc"
    assert s.query_type == "question"
    assert s.time_horizon == "immediate"


def test_d98_infer_stub_below_threshold(tmp_path, monkeypatch):
    import cognitive_fingerprint as cf_mod
    monkeypatch.setattr(cf_mod, "FINGERPRINT_LOG", tmp_path / "fp.jsonl")
    from cognitive_fingerprint import CognitiveFingerprintCollector
    c = CognitiveFingerprintCollector()
    fp = c.infer()
    assert fp.sample_count == 0
    assert "stub" in fp.dominant_style or "pending" in fp.dominant_style


# ── D99: SyntheticExperienceGenerator ───────────────────────────────

def test_d99_import():
    from synthetic_experience import SyntheticExperienceGenerator, SyntheticScenario
    assert SyntheticExperienceGenerator
    assert SyntheticScenario


def test_d99_can_generate_is_false():
    from synthetic_experience import SyntheticExperienceGenerator
    g = SyntheticExperienceGenerator()
    assert g.can_generate() is False


def test_d99_stub_scenario_returned():
    from synthetic_experience import SyntheticExperienceGenerator
    g = SyntheticExperienceGenerator()
    scenario = run(g.generate("a world without electricity"))
    assert "stub" in scenario.narrative.lower() or "[STUB]" in scenario.narrative
    assert scenario.confidence == 0.0


def test_d99_batch_generate_caps_at_five():
    from synthetic_experience import SyntheticExperienceGenerator
    g = SyntheticExperienceGenerator()
    seeds = [f"seed {i}" for i in range(10)]
    results = run(g.batch_generate(seeds))
    assert len(results) <= 5


# ── D100: TransitiveReasoner ──────────────────────────────────────────

def test_d100_import():
    from transitive import Connection, GraphInsight, ReasoningResult, TransitiveReasoner
    assert TransitiveReasoner
    assert Connection
    assert GraphInsight
    assert ReasoningResult


def test_d100_can_reason_is_false():
    from transitive import TransitiveReasoner
    r = TransitiveReasoner()
    assert r.can_reason() is False


def test_d100_stub_reasoning_result():
    from transitive import TransitiveReasoner
    r = TransitiveReasoner()
    result = run(r.reason("what connects X to Y?"))
    assert result.query == "what connects X to Y?"
    assert len(result.insights) > 0
    assert result.insights[0].confidence == 0.0
    assert "stub" in result.insights[0].insight_type.lower()


def test_d100_shortest_path_stub_returns_empty():
    from transitive import TransitiveReasoner
    r = TransitiveReasoner()
    path = run(r.shortest_path("A", "B"))
    assert path == []


def test_d100_pagerank_stub_returns_empty():
    from transitive import TransitiveReasoner
    r = TransitiveReasoner()
    ranks = run(r.pagerank())
    assert ranks == []


def test_d100_mine_rules_stub_returns_empty():
    from transitive import TransitiveReasoner
    r = TransitiveReasoner()
    rules = run(r.mine_rules())
    assert rules == []


def test_d100_connection_fields():
    from transitive import Connection
    c = Connection(source="A", target="B", relation="causes")
    assert c.weight == 1.0
    assert c.evidence_count == 1


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
