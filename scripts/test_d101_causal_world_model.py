"""Tests for D101: Causal World Model & Counterfactual Policy Learning.

Covers:
  - CausalEdge, Policy, SimulationScenario, SimulationResult dataclasses
  - CausalGraph: add_edge, get_edge, stub query methods, can_reason()
  - WorldModelSimulator: stub methods, can_simulate()
  - PolicyMemory (in-memory stub): add_policy, get_relevant_policies, can_learn_policies()
  - CausalSurpriseDetector: check_surprise, can_detect_surprise()
  - Factory singletons: get_causal_graph, get_simulator, get_policy_memory, get_surprise_detector
  - policy_memory (JSONL-persisted): PolicyLibrary, Policy, can_distill(), store, retrieve
"""
import sys
import types
import importlib
from dataclasses import asdict
from pathlib import Path
import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _import_cwm():
    sys.path.insert(0, str(Path(__file__).parent.parent / "agentic"))
    import causal_world_model
    return causal_world_model


def _import_pm():
    sys.path.insert(0, str(Path(__file__).parent.parent / "agentic"))
    import policy_memory
    return policy_memory


# ---------------------------------------------------------------------------
# Importability
# ---------------------------------------------------------------------------

def test_causal_world_model_importable():
    cwm = _import_cwm()
    assert hasattr(cwm, "CausalEdge")
    assert hasattr(cwm, "CausalGraph")
    assert hasattr(cwm, "WorldModelSimulator")
    assert hasattr(cwm, "PolicyMemory")
    assert hasattr(cwm, "CausalSurpriseDetector")


def test_policy_memory_importable():
    pm = _import_pm()
    assert hasattr(pm, "Policy")
    assert hasattr(pm, "PolicyLibrary")
    assert hasattr(pm, "library")


# ---------------------------------------------------------------------------
# CausalEdge
# ---------------------------------------------------------------------------

def test_causal_edge_required_fields():
    cwm = _import_cwm()
    edge = cwm.CausalEdge(source="stress", target="poor_sleep", strength=0.7, confidence=0.6)
    assert edge.source == "stress"
    assert edge.target == "poor_sleep"
    assert edge.strength == 0.7
    assert edge.confidence == 0.6


def test_causal_edge_defaults():
    cwm = _import_cwm()
    edge = cwm.CausalEdge(source="A", target="B", strength=0.5, confidence=0.5)
    assert edge.temporal_lag_seconds == 0.0
    assert edge.direction == "direct"
    assert edge.context_modifiers == {}
    assert edge.source_type == "observed"
    assert edge.evidence_count == 0
    assert isinstance(edge.last_updated, str)


def test_causal_edge_serializable():
    cwm = _import_cwm()
    edge = cwm.CausalEdge(
        source="skipped_breakfast",
        target="afternoon_fatigue",
        strength=0.65,
        confidence=0.4,
        temporal_lag_seconds=14400.0,
        source_type="observed",
        evidence_count=7,
    )
    d = asdict(edge)
    assert d["source"] == "skipped_breakfast"
    assert d["temporal_lag_seconds"] == 14400.0


# ---------------------------------------------------------------------------
# Policy (in-memory stub variant)
# ---------------------------------------------------------------------------

def test_policy_defaults():
    cwm = _import_cwm()
    p = cwm.Policy(
        name="burnout-guard",
        condition="If workload exceeds 55h/week for 2+ weeks",
        action="force a recovery day",
        expected_outcome="prevent sustained performance drop",
        confidence=0.72,
    )
    assert p.version == 1
    assert p.success_rate == 0.0
    assert p.last_applied is None
    assert p.supporting_edges == []


def test_simulation_scenario_defaults():
    cwm = _import_cwm()
    sc = cwm.SimulationScenario(
        goal="avoid burnout",
        initial_state={"workload_hours": 60},
        actions=["take break", "delegate", "continue"],
    )
    assert sc.horizon_steps == 3
    assert sc.variations_per_action == 10


def test_simulation_result_confidence_default():
    cwm = _import_cwm()
    sr = cwm.SimulationResult(
        scenario_id="s001",
        action="take break",
        outcome_path=[{"step": 1, "state": "rested"}],
        final_utility=0.85,
        key_causal_edges_triggered=["overwork->fatigue"],
    )
    assert sr.confidence == 0.0


# ---------------------------------------------------------------------------
# CausalGraph
# ---------------------------------------------------------------------------

def test_causal_graph_starts_empty():
    cwm = _import_cwm()
    g = cwm.CausalGraph()
    assert g.edge_count() == 0


def test_causal_graph_add_and_get_edge():
    cwm = _import_cwm()
    g = cwm.CausalGraph()
    edge = cwm.CausalEdge(source="low_sleep", target="poor_focus", strength=0.8, confidence=0.7)
    eid = g.add_edge(edge)
    assert isinstance(eid, str)
    assert "low_sleep" in eid
    assert g.get_edge(eid) is edge


def test_causal_graph_edge_count_increments():
    cwm = _import_cwm()
    g = cwm.CausalGraph()
    for i in range(5):
        g.add_edge(cwm.CausalEdge(source=f"a{i}", target=f"b{i}", strength=0.5, confidence=0.5))
    assert g.edge_count() == 5


def test_causal_graph_get_edge_missing_returns_none():
    cwm = _import_cwm()
    g = cwm.CausalGraph()
    assert g.get_edge("nonexistent") is None


def test_causal_graph_can_reason_false():
    cwm = _import_cwm()
    g = cwm.CausalGraph()
    assert g.can_reason() is False


def test_causal_graph_query_causal_path_returns_empty():
    cwm = _import_cwm()
    g = cwm.CausalGraph()
    assert g.query_causal_path("stress", "illness") == []


def test_causal_graph_get_downstream_effects_empty():
    cwm = _import_cwm()
    g = cwm.CausalGraph()
    assert g.get_downstream_effects("stress") == []


def test_causal_graph_get_upstream_causes_empty():
    cwm = _import_cwm()
    g = cwm.CausalGraph()
    assert g.get_upstream_causes("illness") == []


def test_causal_graph_predict_outcome_returns_empty_dict():
    cwm = _import_cwm()
    g = cwm.CausalGraph()
    result = g.predict_outcome({"mood": "neutral"}, "meditate")
    assert result == {}


# ---------------------------------------------------------------------------
# WorldModelSimulator
# ---------------------------------------------------------------------------

def test_simulator_can_simulate_false():
    cwm = _import_cwm()
    sim = cwm.WorldModelSimulator(cwm.CausalGraph())
    assert sim.can_simulate() is False


def test_simulator_simulate_scenario_returns_empty():
    cwm = _import_cwm()
    sim = cwm.WorldModelSimulator(cwm.CausalGraph())
    sc = cwm.SimulationScenario(
        goal="maximize focus",
        initial_state={"sleep_hours": 6},
        actions=["sleep_more", "caffeine", "exercise"],
    )
    results = sim.simulate_scenario(sc)
    assert results == []


def test_simulator_run_background_returns_zero():
    cwm = _import_cwm()
    sim = cwm.WorldModelSimulator(cwm.CausalGraph())
    assert sim.run_background_simulations(["avoid burnout", "improve health"]) == 0


# ---------------------------------------------------------------------------
# PolicyMemory (in-memory stub inside causal_world_model)
# ---------------------------------------------------------------------------

def test_in_memory_policy_memory_add_and_retrieve():
    cwm = _import_cwm()
    pm = cwm.PolicyMemory()
    p = cwm.Policy(
        name="morning-routine",
        condition="If it's before 9am",
        action="do not schedule meetings",
        expected_outcome="deep work preserved",
        confidence=0.9,
    )
    pid = pm.add_policy(p)
    assert "morning-routine" in pid


def test_in_memory_policy_memory_get_relevant_empty():
    cwm = _import_cwm()
    pm = cwm.PolicyMemory()
    assert pm.get_relevant_policies({"context": "trading"}) == []


def test_in_memory_policy_memory_can_learn_false():
    cwm = _import_cwm()
    pm = cwm.PolicyMemory()
    assert pm.can_learn_policies() is False


def test_in_memory_policy_update_success_no_exception():
    cwm = _import_cwm()
    pm = cwm.PolicyMemory()
    pm.update_policy_success("policy:morning-routine", True)  # should not raise


# ---------------------------------------------------------------------------
# CausalSurpriseDetector
# ---------------------------------------------------------------------------

def test_surprise_detector_check_surprise_returns_none():
    cwm = _import_cwm()
    g = cwm.CausalGraph()
    det = cwm.CausalSurpriseDetector(g, threshold=0.3)
    result = det.check_surprise({"mood": "good"}, {"mood": "terrible"})
    assert result is None


def test_surprise_detector_can_detect_false():
    cwm = _import_cwm()
    g = cwm.CausalGraph()
    det = cwm.CausalSurpriseDetector(g)
    assert det.can_detect_surprise() is False


def test_surprise_detector_custom_threshold():
    cwm = _import_cwm()
    g = cwm.CausalGraph()
    det = cwm.CausalSurpriseDetector(g, threshold=0.5)
    assert det.surprise_threshold == 0.5


# ---------------------------------------------------------------------------
# Factory singletons
# ---------------------------------------------------------------------------

def test_factory_singletons_return_correct_types():
    cwm = _import_cwm()
    # reset to force fresh construction in this test scope
    cwm._causal_graph = None
    cwm._policy_memory = None
    cwm._simulator = None
    cwm._surprise_detector = None

    assert isinstance(cwm.get_causal_graph(), cwm.CausalGraph)
    assert isinstance(cwm.get_policy_memory(), cwm.PolicyMemory)
    assert isinstance(cwm.get_simulator(), cwm.WorldModelSimulator)
    assert isinstance(cwm.get_surprise_detector(), cwm.CausalSurpriseDetector)


def test_factory_singletons_are_same_object():
    cwm = _import_cwm()
    cwm._causal_graph = None
    cwm._simulator = None
    g1 = cwm.get_causal_graph()
    g2 = cwm.get_causal_graph()
    assert g1 is g2


# ---------------------------------------------------------------------------
# PolicyLibrary (JSONL-persisted, agentic/policy_memory.py)
# ---------------------------------------------------------------------------

def test_policy_library_can_distill_false():
    pm = _import_pm()
    lib = pm.PolicyLibrary()
    assert lib.can_distill() is False


def test_policy_library_policy_count_zero_when_no_log(tmp_path, monkeypatch):
    pm = _import_pm()
    monkeypatch.setattr(pm, "POLICY_LOG", tmp_path / "policies.jsonl")
    lib = pm.PolicyLibrary()
    assert lib.policy_count() == 0


def test_policy_library_store_and_count(tmp_path, monkeypatch):
    pm = _import_pm()
    monkeypatch.setattr(pm, "POLICY_LOG", tmp_path / "policies.jsonl")
    lib = pm.PolicyLibrary()
    policy = pm.Policy(
        condition="If trading window is open and conviction > 8.0",
        action="execute trade immediately",
        expected_outcome="position entered at correct price",
        domain="trading",
        confidence=0.85,
        simulation_count=47,
    )
    lib.store(policy)
    lib._count_cache = None  # force recount
    assert lib.policy_count() == 1


def test_policy_library_retrieve_relevant(tmp_path, monkeypatch):
    pm = _import_pm()
    monkeypatch.setattr(pm, "POLICY_LOG", tmp_path / "policies.jsonl")
    lib = pm.PolicyLibrary()
    lib.store(pm.Policy(
        condition="If trading conviction exceeds 8.0",
        action="execute trade",
        expected_outcome="profit",
        domain="trading",
        confidence=0.8,
    ))
    lib.store(pm.Policy(
        condition="If sleep hours below 6",
        action="block meetings before 10am",
        expected_outcome="energy restored",
        domain="health",
        confidence=0.7,
    ))
    hits = lib.retrieve_relevant("trading conviction", top_k=5)
    assert len(hits) >= 1
    assert any("trading" in h.condition.lower() for h in hits)


def test_policy_library_retrieve_empty_no_file(tmp_path, monkeypatch):
    pm = _import_pm()
    monkeypatch.setattr(pm, "POLICY_LOG", tmp_path / "nonexistent.jsonl")
    lib = pm.PolicyLibrary()
    assert lib.retrieve_relevant("anything") == []


def test_policy_library_progress_dict(tmp_path, monkeypatch):
    pm = _import_pm()
    monkeypatch.setattr(pm, "POLICY_LOG", tmp_path / "policies.jsonl")
    lib = pm.PolicyLibrary()
    prog = lib.progress()
    assert "policy_count" in prog
    assert "distillation_active" in prog
    assert prog["distillation_active"] is False


def test_policy_library_singleton_exists():
    pm = _import_pm()
    assert isinstance(pm.library, pm.PolicyLibrary)


def test_policy_library_validate_no_exception():
    pm = _import_pm()
    pm.library.validate("P12345-abcd", outcome_matched=True)  # should not raise
