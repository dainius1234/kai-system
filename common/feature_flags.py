"""Lightweight feature flags for sovereign AI services.

Flags are controlled via environment variables prefixed with ``FF_``.
Each flag defaults to OFF unless explicitly enabled.  This keeps the
system safe-by-default: new capabilities must be opted into.

Usage:
    from common.feature_flags import is_enabled, get_all_flags

    if is_enabled("DREAM_PHASE_7"):
        # agent-evolver dream phase runs
        ...

    # API: return all flag states for operator visibility
    flags = get_all_flags()

Environment:
    FF_DREAM_PHASE_7=true        # enable dream phase 7
    FF_CHECKPOINT_AUTO=true      # auto-checkpoint on recover/dream
    FF_TREE_SEARCH=false         # disable tree search temporarily
"""
from __future__ import annotations

import os
from typing import Any, Dict, List

# ── Flag registry ────────────────────────────────────────────────────
# name → (description, default)
# Add new flags here.  Environment variable is ``FF_<NAME>``.
_REGISTRY: Dict[str, tuple] = {
    "DREAM_PHASE_7":         ("Agent-Evolver insight generation during dream cycle", True),
    "CHECKPOINT_AUTO":       ("Auto-checkpoint on /recover and /dream", True),
    "TREE_SEARCH":           ("CoT tree search with conviction pruning", True),
    "PRIORITY_QUEUE":        ("Latency-sensitive priority queue", True),
    "SAGE_CRITIQUE":         ("Verifier self-critique + adversary self-review", True),
    "IMAGINATION_ENGINE":    ("P19 imagination / scenario simulation", True),
    "PROACTIVE_AGENT":       ("P21/D87: background observer — detects anomalies and writes proactive_observation memories every PROACTIVE_INTERVAL_SECONDS (default 300)", True),
    "OPERATOR_MODEL":        ("P22 operator preference learning", True),
    "NARRATIVE_IDENTITY":    ("P18 narrative identity context", True),
    "CONSCIENCE_FILTER":     ("P20 conscience value-gate on actions", True),
    "MARS_CONSOLIDATION":    ("MARS memory decay + consolidation", True),
    "SELF_ASSESSMENT":       ("P14 temporal self-model", True),
    "SECURITY_AUDIT":        ("P9 automated security self-hacking", True),
    "WAKE_INTENT_ROUTING":   ("Pre-classify chat intent via wake-intent service before routing", False),
    "GRAPH_INGEST":          ("Phase B write-side fan-out from memu-core to memu-graph", False),
    "LETTA_TASKS":           ("Delegate long-running tasks to letta-agent memory controller", False),
    "LETTA_MEMORY_SYNC":     ("Sync letta-agent archival memories back to memu-core after each run", False),
    "FINANCIAL_CONTEXT":     ("P29 inject CIS/VAT/tax summary into agentic context on finance queries", True),
    # F4/D87: master toggle for the context enrichment gather.
    # Set FF_CONTEXT_ENRICHMENT=false to run a bare /chat (LLM only, no memory/personality/
    # world-state injection) for A/B quality comparison.
    "CONTEXT_ENRICHMENT":    ("Master toggle: 14-way context gather (memory + personality + soul + world-state channels). Also gates sensory world_context injection.", True),
    # F6: self-improvement loops — off by default; activate after GPU Day validates quality
    "DREAM_ENABLED":         ("Trigger dream cycle consolidation (6-phase memory integration)", False),
    "EVOLVER_ENABLED":       ("Agent-Evolver: cluster failure patterns → proactive insights", False),
    "SAGE_SELF_REVIEW":      ("SAGE critique on all plans before execution (not just high-stakes)", False),
    # D88: 8 advanced cognition mechanisms
    "ANOMALY_DETECTION":     ("D88/M1: track rolling baselines per sensor; alert on >2σ deviation", True),
    "WORLD_MODEL_PERSISTENCE": ("D88/M4: write structured world_state JSON to memu-core each proactive cycle", True),
    "SENSORY_LEARNING":      ("D88/M5: detect recurring sensor patterns across 10 recent cycles; write sensor_pattern memories", True),
    "SKILL_HUNTER":          ("D88/M6+M8: skill-hunter service integration; reactive skill acquisition on capability gaps", True),
    "PROACTIVE_SCHEDULING":  ("D88/M7: fuse calendar events + sensor state into proactive_schedule memories", True),
    # D89: cognitive depth — FSM, teammates, foundations
    "FSM":                   ("D89: Kai Finite State Machine — IDLE/ACTIVE/FOCUSED/DEGRADED/RECOVERING state tracking", True),
    "PERSISTENT_TEAMMATES":  ("D89: named cognitive teammates (Scout, Doctor, Sage, Oracle) with per-specialty system prompts", True),
    "HOUSE_DOCTOR":          ("D89/E: House Doctor service — continuous differential diagnosis from cross-sensor correlation", True),
    "RITUAL_DISCOVERY":      ("D89/C: emergent ritual detection at ≥7/10 cycles; writes RITUALS.md proposals", True),
    "GAP_LOGGING":           ("D89/C1: log capability gaps before reactive acquisition; fire hunt only after GAP_HUNT_THRESHOLD misses", True),
    "TRUST_NEGOTIATION":     ("D89/B: autonomy request protocol — KAI can request temporary elevated authority; currently pending_approval", True),
    "PREDICTIVE_EMPATHY":    ("D89/D: emotional_context key in world model; full implementation pending emotional memory accumulation", True),
    "CURIOSITY":             ("D89/F: resource-aware curiosity idle tick; no-ops in CPU phase; activates on GPU + IDLE state", True),
    # D90: Swarm Assembly — full CognitiveFSM pipeline with real stage functions
    "SWARM":                 ("D90: CognitiveFSM swarm pipeline — Scout/Sage/Doctor/Oracle stage functions, shared SwarmContext, reputation tracking, conflict resolution", True),
    # D91: Obsidian Brain — bidirectional vault ↔ memu-core sync
    "VAULT_SYNC":            ("D91: vault-sync service enabled — file watcher, ingest, export, mapper", True),
    "VAULT_CONTEXT":         ("D91: inject vault memory snippet into world-context gather (gated separately because it adds latency)", False),
    # D92: Socratic Self-Questioning — pre-GATHER query decomposition
    "SOCRATIC":              ("D92: pre-GATHER Socratic decomposition — 3-5 questions reframe the query before Scout gathers evidence", True),
    # D93: Autonomous Hypothesis Engine — idle-cycle knowledge gap scanning
    "HYPOTHESIS_ENGINE":     ("D93: idle-cycle gap scanner — forms testable hypotheses from low-confidence memories and tests them", True),
    # D94: Temporal Projection — fan-of-futures forecasting from supported claims
    "TEMPORAL_PROJECTION":   ("D94: ForecastFan — base/optimistic/pessimistic/wild-card scenario branches from supported claims", True),
    # D95–D100: GPU-era stubs — interfaces fixed now, activated when hardware/data arrives
    "DIALECTICAL_SYNTHESIS": ("D95: Hegelian thesis/antithesis/synthesis reasoner — pending dual-model GPU", False),
    "ANALOGICAL_REASONING":  ("D96: cross-domain isomorphic pattern search — pending populated knowledge graph", False),
    "CONCEPT_BLENDING":      ("D97: two distant graph nodes → novel emergent concept — pending graph + GPU", False),
    "COGNITIVE_FINGERPRINT": ("D98: operator thinking-style model — collecting interaction samples now; inference pending 90+ samples", True),
    "SYNTHETIC_EXPERIENCE":  ("D99: fictional scenario generation during dream cycles — pending GPU", False),
    "TRANSITIVE_REASONING":  ("D100: PageRank + community detection + shortest-path on memu-graph — pending populated graph", False),
    "CAUSAL_WORLD_MODEL":    ("D101: persistent causal graph + GPU mental simulations + policy distillation — pending GPU + 30d data", False),
    "CAUSAL_SURPRISE":       ("D101: prediction-error detection — fires hypothesis cycle on divergence; requires FF_CAUSAL_WORLD_MODEL", False),
    "POLICY_MEMORY":         ("D101: auto-distillation of simulation outcomes into ranked strategies — requires FF_CAUSAL_WORLD_MODEL", False),
}


def is_enabled(flag_name: str) -> bool:
    """Check whether a feature flag is enabled.

    Reads ``FF_<FLAG_NAME>`` from the environment.  Falls back to the
    default value in the registry, or False if the flag is unknown.
    """
    env_key = f"FF_{flag_name.upper()}"
    env_val = os.environ.get(env_key)
    if env_val is not None:
        return env_val.strip().lower() in ("1", "true", "yes", "on")
    # fall back to registry default
    entry = _REGISTRY.get(flag_name.upper())
    if entry:
        return bool(entry[1])
    return False


def get_all_flags() -> List[Dict[str, Any]]:
    """Return the state of every registered flag."""
    result = []
    for name, (desc, default) in sorted(_REGISTRY.items()):
        result.append({
            "flag": name,
            "enabled": is_enabled(name),
            "default": default,
            "env_var": f"FF_{name}",
            "description": desc,
        })
    return result


def register_flag(name: str, description: str, default: bool = False) -> None:
    """Register a new flag at runtime (e.g. from a service plugin)."""
    _REGISTRY[name.upper()] = (description, default)
