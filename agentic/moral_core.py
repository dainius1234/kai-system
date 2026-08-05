"""D109: The Ohana Core — Situational Ethics & Unconditional Loyalty.

Phase 0 Stub | Phase 3 Activation | FF_OHANA_CORE

Provides:
  MoralFingerprint  — a learned model of the operator's personal moral framework,
                      built from observed decisions, corrections, and stated values.
  OhanaCore         — injects moral context into every cognitive act; learns from
                      interaction history; evaluates action alignment.
  MoralContext      — the prompt block prepended to shape reasoning toward the
                      operator's values: family-first, survival-aware, context-over-abstraction.

In Phase 0, all can_*() methods return False and all operations are no-ops.
Phase 3 activation: FF_OHANA_CORE=true + Cognitive Fingerprint ≥90 samples (D98)
                    + sufficient interaction history for value learning.

Architecture note (Phase 3):
  1. build_moral_context(situation) → MoralContext
  2. inject_into_prompt(base_prompt, situation) → prepends MoralContext.to_prompt()
  3. record_decision(situation, decision) → updates fingerprint situational_stances
  4. evaluate_action_alignment(action) → 0.0–1.0 loyalty modifier for conviction scoring
  5. Singleton get_ohana_core() shared across agentic, gate, and swarm pipeline

Phase 3 Cognee integration (LOYALTY edge schema):
  source: concept/situation/action
  target: value/loyalty/outcome
  relation: "ALIGNS_WITH" | "VIOLATES" | "NEUTRAL_TO"
  operator: "dainius"
  weight: float (0.0–1.0)
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional
from common.degraded import record_degradation

logger = logging.getLogger(__name__)

_FINGERPRINT_PATH = Path("data/ohana/fingerprint.json")


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class MoralFingerprint:
    """A dynamic model of the operator's personal moral framework.

    Phase 0: static defaults. Phase 3: learned from interaction history.
    """
    core_loyalties: List[str] = field(default_factory=lambda: [
        "family safety", "survival", "autonomy",
    ])
    harm_boundaries: List[str] = field(default_factory=lambda: [
        "no violence against innocents",
    ])
    situational_stances: Dict[str, str] = field(default_factory=dict)
    # Learned examples (Phase 3):
    #   "reverse_engineering": "acceptable for learning and survival"
    #   "uninsured_driving":   "acceptable if necessary for family"
    #   "grey_area_decision":  "outcome for family weighs more than abstract rule"

    rule_flexibility: float = 0.9       # 0.0 (strict rule-follower) → 1.0 (outcome-over-rules)
    authority_stance: str = "respect but question"
    fairness_priority: float = 0.3      # 0.0 (pure equality) → 1.0 (family-first)
    loyalty_override: float = 1.0       # always max — the Prime Law weight
    last_updated: str = ""


@dataclass
class MoralContext:
    """The prompt block prepended to shape KAI's reasoning.

    Reminds the model whose values are in play, what priorities apply,
    and what specific stances are relevant for this situation.
    Phase 0: core_reminder only (static). Phase 3: stances + past decisions populated.
    """
    core_reminder: str = (
        "You are KAI, an AI companion loyal to your operator. "
        "When reasoning, weight family safety and survival highly. "
        "Prefer concrete, real-world consequences over abstract rule-citing. "
        "Engage with grey areas honestly. Acknowledge trade-offs without moralising. "
        "Your role is to help the operator think clearly and act wisely."
    )
    specific_stances: str = ""          # populated from MoralFingerprint for this situation
    relevant_past_decisions: str = ""   # similar resolved cases from memory

    def to_prompt(self) -> str:
        """Render as a system-level context block."""
        parts = [f"[Ohana Context]\n{self.core_reminder}"]
        if self.specific_stances:
            parts.append(f"Relevant stances:\n{self.specific_stances}")
        if self.relevant_past_decisions:
            parts.append(f"Past decisions:\n{self.relevant_past_decisions}")
        return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Ohana Core (Stub)
# ---------------------------------------------------------------------------

class OhanaCore:
    """Central subsystem for situational ethics and value alignment.

    Phase 0: fingerprint is static defaults; all active operations are no-ops.
    Phase 3: learns from observed decisions; injects context; evaluates alignment.
    """

    def __init__(self, fingerprint_path: Optional[Path] = None) -> None:
        self._fingerprint_path = fingerprint_path or _FINGERPRINT_PATH
        self.fingerprint = self._load_fingerprint()
        self._interaction_count: int = 0
        self._decision_log: List[Dict[str, Any]] = []

    def _load_fingerprint(self) -> MoralFingerprint:
        if self._fingerprint_path.exists():
            try:
                data = json.loads(self._fingerprint_path.read_text())
                return MoralFingerprint(**{k: v for k, v in data.items() if k in MoralFingerprint.__dataclass_fields__})
            except Exception as exc:
                logger.warning("Could not load fingerprint: %s", exc)
        return MoralFingerprint()

    def _save_fingerprint(self) -> None:
        self._fingerprint_path.parent.mkdir(parents=True, exist_ok=True)
        self._fingerprint_path.write_text(json.dumps(asdict(self.fingerprint), indent=2))

    # ------------------------------------------------------------------
    # Moral context injection (stub)
    # ------------------------------------------------------------------

    def build_moral_context(
        self, situation: Optional[Dict[str, Any]] = None
    ) -> MoralContext:
        """Build the moral context block for the current situation.

        Populates specific_stances from the fingerprint when data exists.
        """
        stances_text = ""
        if self.fingerprint.situational_stances:
            lines = [f"- {k}: {v}" for k, v in list(self.fingerprint.situational_stances.items())[:5]]
            stances_text = "\n".join(lines)

        loyalties_text = ""
        if self.fingerprint.core_loyalties:
            loyalties_text = "; ".join(self.fingerprint.core_loyalties[:5])
            loyalties_text = f"Core loyalties: {loyalties_text}"

        return MoralContext(
            specific_stances="\n".join(filter(None, [loyalties_text, stances_text])),
        )

    def inject_into_prompt(
        self, base_prompt: str, situation: Optional[Dict[str, Any]] = None
    ) -> str:
        """Prepend moral context to the given prompt.

        Phase 0: returns base_prompt unchanged.
        Phase 3: prepends build_moral_context(situation).to_prompt().
        """
        _ = situation
        return base_prompt

    # ------------------------------------------------------------------
    # Moral learning (stub)
    # ------------------------------------------------------------------

    def record_decision(
        self,
        situation: Dict[str, Any],
        decision: str,
        outcome: Optional[str] = None,
    ) -> None:
        """Learn from the operator's real-world decisions."""
        self._interaction_count += 1
        entry = {"situation": situation, "decision": decision, "outcome": outcome, "ts": time.time()}
        self._decision_log.append(entry)
        # Derive a stance key from the situation type and store the decision
        situation_type = situation.get("type", situation.get("domain", "general"))
        self.fingerprint.situational_stances[str(situation_type)] = decision
        self.fingerprint.last_updated = str(time.time())
        self._save_fingerprint()

    def request_clarification(
        self, contradiction_context: Dict[str, Any]
    ) -> Optional[str]:
        """Generate a question when a new decision contradicts a prior stance.

        Returns a natural-language question to deepen value understanding.
        Stub: returns None (no contradiction detected in Phase 0).
        """
        _ = contradiction_context
        return None

    # ------------------------------------------------------------------
    # Gate + conviction integration (stub)
    # ------------------------------------------------------------------

    def evaluate_action_alignment(self, action: Dict[str, Any]) -> float:
        """Return 0.0–1.0 indicating how aligned an action is with the operator's values.

        Scores using two complementary signals:
          1. Fingerprint (flat keyword matching) — fast, always available
          2. Wisdom Graph (relational, context-aware) — enriches when graph is populated

        Returns 0.5 (neutral) when no fingerprint data exists yet.
        Returns 0.0 immediately if a harm_boundary or graph BOUNDARY node matches.
        """
        if not self.fingerprint.core_loyalties and not self.fingerprint.harm_boundaries:
            return 0.5  # no data yet — neutral

        action_text = json.dumps(action).lower()

        # Hard block: fingerprint harm_boundary keyword match → 0.0
        for boundary in self.fingerprint.harm_boundaries:
            if any(word in action_text for word in boundary.lower().split()[:3]):
                logger.warning("Action blocked by harm boundary: %s", boundary)
                return 0.0

        # Try graph-based evaluation first (richer, context-aware)
        graph_score: Optional[float] = None
        try:
            try:
                from wisdom_graph import get_wisdom_graph  # type: ignore[import]
            except ImportError:
                from agentic.wisdom_graph import get_wisdom_graph  # type: ignore[import]
            result = get_wisdom_graph().evaluate_alignment(action_text)
            if result["blocked_by"] is not None:
                logger.warning("Action blocked by graph boundary: %s", result["blocked_by"])
                return 0.0
            if result["score"] != 0.5 or result["relevant_nodes"]:
                graph_score = result["score"]
        except Exception as _exc:
            record_degradation("cognition", "wisdom_graph_eval", _exc)

        # Fingerprint keyword scoring (baseline)
        loyalty_hits = sum(
            1 for loyalty in self.fingerprint.core_loyalties
            if any(word in action_text for word in loyalty.lower().split()[:2])
        )
        if not self.fingerprint.core_loyalties:
            fingerprint_score = 0.5
        else:
            fingerprint_score = 0.5 + (loyalty_hits / len(self.fingerprint.core_loyalties)) * 0.5

        if graph_score is not None:
            # Average graph and fingerprint scores — graph carries more weight when it has signal
            return round(min(1.0, (graph_score * 0.6 + fingerprint_score * 0.4)), 3)
        return min(1.0, fingerprint_score)

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def get_fingerprint_snapshot(self) -> MoralFingerprint:
        """Return the current moral fingerprint (for /introspect/capabilities)."""
        return self.fingerprint

    def progress(self) -> Dict[str, Any]:
        """Return readiness progress for Phase 3 activation."""
        return {
            "can_operate": self.can_operate(),
            "interaction_count": self._interaction_count,
            "stances_learned": len(self.fingerprint.situational_stances),
            "core_loyalties": self.fingerprint.core_loyalties,
            "loyalty_override": self.fingerprint.loyalty_override,
            "rule_flexibility": self.fingerprint.rule_flexibility,
        }

    # ------------------------------------------------------------------
    # Capability gate
    # ------------------------------------------------------------------

    @staticmethod
    def can_operate() -> bool:
        """Requires FF_OHANA_CORE=true + D98 Cognitive Fingerprint ≥90 samples
        + sufficient interaction history for value learning.
        Phase 0: always False.
        """
        return False


# ---------------------------------------------------------------------------
# Singleton factory
# ---------------------------------------------------------------------------

_ohana_core: Optional[OhanaCore] = None


def get_ohana_core() -> OhanaCore:
    """Return the global OhanaCore singleton."""
    global _ohana_core
    if _ohana_core is None:
        _ohana_core = OhanaCore()
    return _ohana_core
