"""D101: Policy Memory — distilled strategies from causal world model simulations.

Stores generalized if-then policies that emerge from repeated causal simulation.
Each policy is a compact, versioned record of what action leads to what outcome
under what conditions, with a confidence score calibrated against real-world tests.

Phase 0 (NOW):
  - store() and retrieve_relevant() work immediately.
    Manually-crafted seed policies can be recorded today.
  - can_distill() → False until WorldModelSimulator is active.
  - library singleton is ready to receive policies from any module.

Phase 1 (GPU):
  - Auto-distillation: each completed simulation cycle generates candidate policies.
  - Validation loop: policies tested against real outcomes; confidence adjusted.
  - Proactive surfacing: retrieve_relevant() called by /chat handler to prepend
    high-confidence policies to system context before each response.

Feature flag: FF_POLICY_MEMORY (default False) — activates auto-distillation
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("kai.policy_memory")

POLICY_LOG = Path("/data/policies.jsonl")
POLICY_SOURCE_TYPES = ("simulation", "observation", "human_labeled")
POLICY_DOMAINS = ("trading", "health", "work", "schedule", "communication", "general")

_POLICY_FIELDS: Optional[set] = None   # populated lazily from dataclass


def _policy_field_names() -> set:
    global _POLICY_FIELDS
    if _POLICY_FIELDS is None:
        _POLICY_FIELDS = {f.name for f in Policy.__dataclass_fields__.values()}  # type: ignore[attr-defined]
    return _POLICY_FIELDS


# ---------------------------------------------------------------------------
# Data structure
# ---------------------------------------------------------------------------

@dataclass
class Policy:
    """A distilled generalized strategy: if <condition> → do <action> → expect <outcome>."""
    condition: str              # "If X…"  — human-readable trigger description
    action: str                 # "…then Y" — recommended action
    expected_outcome: str       # what success looks like
    domain: str = "general"     # trading | health | work | schedule | communication | general
    expected_utility: float = 0.0   # 0=neutral, positive=beneficial, negative=harmful
    confidence: float = 0.0         # calibrated from simulation + real-world validation
    simulation_count: int = 0       # number of simulated runs that produced this policy
    validation_count: int = 0       # number of real-world tests against reality
    source: str = "simulation"      # simulation | observation | human_labeled
    version: int = 1
    policy_id: str = ""             # set on store() if empty
    created_at: float = field(default_factory=time.time)
    last_validated: float = 0.0


# ---------------------------------------------------------------------------
# Policy Library
# ---------------------------------------------------------------------------

class PolicyLibrary:
    """Stores, retrieves, and validates distilled causal policies.

    Phase 0: manual store/retrieve; no auto-distillation.
    Phase 1: WorldModelSimulator populates automatically; proactive surfacing
             in /chat context injection.
    """

    def __init__(self) -> None:
        self._count_cache: Optional[int] = None

    # --- capability gate --------------------------------------------------

    def can_distill(self) -> bool:
        """False until WorldModelSimulator is active and FF_POLICY_MEMORY is set."""
        try:
            from feature_flags import is_enabled
            if not is_enabled("POLICY_MEMORY"):
                return False
        except ImportError:
            pass
        return False  # GPU-era gate

    # --- write path -------------------------------------------------------

    def store(self, policy: Policy) -> None:
        """Append a policy to the persistent log.

        Works in Phase 0 — manually-crafted seed policies are valid inputs.
        """
        try:
            POLICY_LOG.parent.mkdir(parents=True, exist_ok=True)
            if not policy.policy_id:
                policy.policy_id = (
                    f"P{int(policy.created_at)}-"
                    f"{abs(hash(policy.condition)) & 0xFFFF:04x}"
                )
            with POLICY_LOG.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(asdict(policy)) + "\n")
            self._count_cache = None
            logger.debug("Policy stored: %s (domain=%s)", policy.policy_id, policy.domain)
        except Exception as exc:
            logger.debug("Could not store policy: %s", exc)

    def validate(self, policy_id: str, outcome_matched: bool) -> None:
        """Record a real-world test of a policy.

        Phase 0: logs only.
        Phase 1: updates confidence in-place and propagates to causal graph.
        """
        logger.debug(
            "Policy %s validated: outcome_matched=%s",
            policy_id,
            outcome_matched,
        )

    # --- read path --------------------------------------------------------

    def retrieve_relevant(
        self,
        context: str,
        top_k: int = 5,
        min_confidence: float = 0.0,
    ) -> List[Policy]:
        """Keyword-match policies against context string.

        Phase 0: surface heuristic — word overlap between context and condition.
        Phase 1: embedding similarity over condition + action + domain.
        """
        if not POLICY_LOG.exists():
            return []
        try:
            ctx_tokens = set(context.lower().split())
            matches: List[tuple[float, Policy]] = []
            with POLICY_LOG.open(encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                        p = Policy(**{k: v for k, v in data.items() if k in _policy_field_names()})
                        if p.confidence < min_confidence:
                            continue
                        cond_tokens = set(p.condition.lower().split())
                        overlap = len(ctx_tokens & cond_tokens)
                        if overlap > 0:
                            matches.append((overlap + p.confidence, p))
                    except Exception:
                        continue
            matches.sort(key=lambda t: t[0], reverse=True)
            return [p for _, p in matches[:top_k]]
        except Exception as exc:
            logger.debug("Policy retrieval error: %s", exc)
            return []

    def all_policies(self) -> List[Policy]:
        """Return all stored policies (use sparingly — no pagination in Phase 0)."""
        if not POLICY_LOG.exists():
            return []
        policies = []
        try:
            with POLICY_LOG.open(encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                        policies.append(
                            Policy(**{k: v for k, v in data.items() if k in _policy_field_names()})
                        )
                    except Exception:
                        continue
        except Exception as exc:
            logger.debug("Error reading policy log: %s", exc)
        return policies

    # --- introspection ----------------------------------------------------

    def policy_count(self) -> int:
        if self._count_cache is None:
            try:
                if POLICY_LOG.exists():
                    self._count_cache = sum(1 for _ in POLICY_LOG.open())
                else:
                    self._count_cache = 0
            except Exception:
                self._count_cache = 0
        return self._count_cache

    def progress(self) -> Dict[str, Any]:
        return {
            "policy_count": self.policy_count(),
            "distillation_active": self.can_distill(),
        }


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

library = PolicyLibrary()
