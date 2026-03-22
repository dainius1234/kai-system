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
    "PROACTIVE_AGENT":       ("P21 proactive context pre-fetch", True),
    "OPERATOR_MODEL":        ("P22 operator preference learning", True),
    "NARRATIVE_IDENTITY":    ("P18 narrative identity context", True),
    "CONSCIENCE_FILTER":     ("P20 conscience value-gate on actions", True),
    "MARS_CONSOLIDATION":    ("MARS memory decay + consolidation", True),
    "SELF_ASSESSMENT":       ("P14 temporal self-model", True),
    "SECURITY_AUDIT":        ("P9 automated security self-hacking", True),
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
