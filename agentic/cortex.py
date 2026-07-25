"""D114: The Cortex — Pre-Conscious Situational Awareness (Cognitive Module)

Companion to the D113 Cortex service (cortex/app.py, port 8048).

The service is the always-on background process that reads sensors and
synthesises situation summaries every 60 seconds.  This module is the
cognitive interface that the rest of the agentic pipeline uses — it sits
alongside causal_world_model.py, global_workspace.py, and moral_core.py
as the "ambient baseline" bidder to the Global Workspace.

Phase 0 (NOW, D113 service running):
  - can_operate() returns True when the D113 service has fed state within 120s.
  - All synthesis is delegated to the running service; this module holds the
    resulting SituationModel and generates real GlobalWorkspace bids from it.
  - No NPU or GPU required.

Phase 1 (Strix Halo / Flow Z13 — AMD Ryzen AI Max+ 395, XDNA 2 NPU):
  - FF_CORTEX_NPU=true activates on-device inference.
  - Level 2/3 summaries synthesised by a small ONNX/QNN model (<5W) running
    directly on the NPU — continuous, always-on, even on battery.
  - The D113 service remains for sensor aggregation; this module upgrades
    its synthesis backend from "HTTP delegation" to "local NPU inference".
  - Activation: FF_CORTEX_NPU=true + ONNX/QNN runtime validated.

Hardware context (Strix Halo):
  - Unified LPDDR5X memory (32–64 GB) — no VRAM wall.  All four council
    models (DeepSeek V4 Q4 ~24 GB, Kimi, GLM, Dolphin) can be resident
    simultaneously.  Full live 4-model debate is Phase 1, not Phase 2.
  - XDNA 2 NPU: ideal for small continuous inference workloads (ASR,
    embeddings, classifiers, this Cortex synthesis loop).  Leaves the
    iGPU free for heavy reasoning tasks.
  - ~20–30 t/s token generation at unified memory bandwidth.  Acceptable
    for a personal assistant; schedule heavy batch simulations (causal world
    model) during idle / sleep.
  - The Cortex + proactive observer can run at <5W on battery — true 24/7
    ambient awareness without a dedicated Pi.

Integration points:
  - agentic/app.py: import get_cortex
  - _sense_world(): call get_cortex().feed_service_state(cs) after reading D113
  - proactive observer: call get_cortex().bid_to_workspace() alongside D102 bids
  - global_workspace.py: Cortex is the primary ambient bidder — sets the
    baseline all other bids are evaluated against

Feature flags:
  - FF_CORTEX_NPU: bool = False  (Phase 1 — Strix Halo NPU inference)
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from common.feature_flags import is_enabled

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class SituationModel:
    """Layered interpretation of the current moment, updated every 60 seconds."""
    level_1_raw_facts: str = ""
    level_2_summary: str = ""
    level_3_implications: str = ""
    confidence: float = 0.0
    last_updated: str = ""


@dataclass
class IntentShadow:
    """Probabilistic model of what the operator is working toward."""
    active_intents: List[Dict[str, Any]] = field(default_factory=list)
    # Each entry: {"intent": "debugging vault-sync", "confidence": 0.6, "signals": [...]}
    preloaded_contexts: List[str] = field(default_factory=list)
    last_updated: str = ""


@dataclass
class TransitionBridge:
    """Detects conversation mode shifts and pre-warms context."""
    current_mode: str = "idle"
    pending_transition: Optional[str] = None
    preloaded_context: List[str] = field(default_factory=list)
    bridge_active: bool = False
    last_updated: str = ""


@dataclass
class TacitPreference:
    """A learned, unwritten interaction rule."""
    condition: str
    preferred_style: str
    observed_count: int = 0
    confidence: float = 0.0


# ---------------------------------------------------------------------------
# The Cortex (Cognitive Module)
# ---------------------------------------------------------------------------

class Cortex:
    """
    Pre-conscious situational awareness layer.  Always-on, low-power.

    Phase 0: delegates to the D113 Cortex service via feed_service_state().
    Phase 1 (Strix Halo): on-device NPU inference via _npu_synthesize().
    """

    def __init__(self) -> None:
        self.situation = SituationModel()
        self.intent_shadow = IntentShadow()
        self.transition_bridge = TransitionBridge()
        self.tacit_preferences: List[TacitPreference] = []

        self._service_state: Optional[Dict[str, Any]] = None
        self._service_last_ok: float = 0.0

    # ------------------------------------------------------------------
    # Phase 0 bridge — receives state from the running D113 service
    # ------------------------------------------------------------------

    def feed_service_state(self, state: Dict[str, Any]) -> None:
        """Called by _sense_world() after reading the D113 /state endpoint.

        Keeps this cognitive module in sync without making its own HTTP calls.
        The service runs its 60s cycle independently; this module consumes the
        already-computed result.
        """
        if not state:
            return
        self._service_state = state
        self._service_last_ok = time.time()

        self.situation = SituationModel(
            level_1_raw_facts="\n".join(state.get("level1_facts", [])),
            level_2_summary=state.get("level2_summary", ""),
            level_3_implications=state.get("level3_implication", ""),
            confidence=min(1.0, state.get("refresh_count", 0) / 3),
            last_updated=state.get("timestamp", ""),
        )

        # Populate intent shadow from Quiet Planner fan
        fan = state.get("intent_fan", [])
        if fan:
            self.intent_shadow = IntentShadow(
                active_intents=[
                    {"intent": h["label"], "confidence": h["confidence"],
                     "signals": h.get("context_hints", [])}
                    for h in fan
                ],
                preloaded_contexts=fan[0].get("context_hints", []) if fan else [],
                last_updated=state.get("timestamp", ""),
            )

        # Populate transition bridge from Context Bridge
        self.transition_bridge.bridge_active = state.get("bridge_active", False)
        self.transition_bridge.pending_transition = state.get("bridge_note")

        # Populate tacit preferences
        rules = state.get("tacit_rules", [])
        if rules:
            self.tacit_preferences = [
                TacitPreference(condition="observed", preferred_style=r, observed_count=1, confidence=0.5)
                for r in rules
            ]

    # ------------------------------------------------------------------
    # Core tick — Phase 0: no-op (D113 service does the work)
    # Phase 1: call _npu_synthesize() directly
    # ------------------------------------------------------------------

    def tick(self, raw_sensor_data: Optional[Dict[str, Any]] = None) -> None:
        """Called every 60 seconds from the proactive observer.

        Phase 0: no-op — D113 service handles the synthesis cycle.
        Phase 1 (Strix Halo): call _npu_synthesize() with raw sensor data.
        """
        if not is_enabled("CORTEX_NPU"):
            return
        if raw_sensor_data:
            self.situation = self._npu_synthesize(list(raw_sensor_data.values()))

    # ------------------------------------------------------------------
    # Situation synthesis
    # ------------------------------------------------------------------

    def synthesize_situation(self, raw_sensor_data: Dict[str, Any]) -> SituationModel:
        """Transform raw sensor readings into a layered interpretation.

        Phase 0: returns current cached SituationModel from service.
        Phase 1: calls _npu_synthesize().
        """
        if is_enabled("CORTEX_NPU") and raw_sensor_data:
            return self._npu_synthesize(list(raw_sensor_data.values()))
        return self.situation

    def get_current_situation(self) -> SituationModel:
        return self.situation

    def _npu_synthesize(self, raw_facts: List[Any]) -> SituationModel:
        """Phase 1: synthesise Level 2/3 summaries via NPU inference.

        Target: TinyLlama Q4 or distilled ONNX classifier on AMD XDNA 2.
        Requires: onnxruntime-directml or Qualcomm QNN SDK.
        Activation: FF_CORTEX_NPU=true + runtime validated.

        Phase 0: never called.
        """
        raise NotImplementedError(
            "NPU inference path not yet validated — activate with FF_CORTEX_NPU=true "
            "after ONNX/QNN runtime is confirmed on the Strix Halo platform."
        )

    # ------------------------------------------------------------------
    # Intent shadow
    # ------------------------------------------------------------------

    def infer_intent(self, activity_signals: Dict[str, Any]) -> IntentShadow:
        """Stub: Phase 0 returns cached intent from D113 Quiet Planner."""
        _ = activity_signals
        return self.intent_shadow

    def preload_context(self, intent: Dict[str, Any]) -> List[str]:
        """Return context hints for the top intent hypothesis."""
        _ = intent
        return self.intent_shadow.preloaded_contexts

    # ------------------------------------------------------------------
    # Transition bridge
    # ------------------------------------------------------------------

    def detect_transition(self, conversation_signal: Dict[str, Any]) -> Optional[str]:
        """Stub: Phase 0 returns the pending transition detected by D113 Context Bridge."""
        _ = conversation_signal
        if self.transition_bridge.bridge_active:
            return self.transition_bridge.pending_transition
        return None

    def warm_transition(self, current_mode: str, new_mode: str) -> List[str]:
        """Return context appropriate to the new mode."""
        _ = current_mode, new_mode
        return self.transition_bridge.preloaded_context

    # ------------------------------------------------------------------
    # Tacit knowledge
    # ------------------------------------------------------------------

    def learn_tacit_preference(self, interaction_context: Dict[str, Any], operator_feedback: str) -> None:
        """Stub: Phase 0 no-op. D113 accumulates patterns independently."""
        _ = interaction_context, operator_feedback

    def apply_tacit_preferences(self, base_style: str, situation: Dict[str, Any]) -> str:
        """Apply learned format preferences to a communication style signal."""
        _ = situation
        if not self.tacit_preferences:
            return base_style
        for pref in self.tacit_preferences:
            if "brief" in pref.preferred_style.lower() and pref.confidence >= 0.5:
                return "bullet_points"
        return base_style

    # ------------------------------------------------------------------
    # Global Workspace bid
    # ------------------------------------------------------------------

    def bid_to_workspace(self) -> Optional[Any]:
        """Generate a WorkspaceBid for the Global Workspace (D102).

        The Cortex is the primary ambient bidder — it doesn't compete for a
        specific thought, it sets the baseline all other bids are evaluated against.

        Phase 0: returns a real bid when D113 service state is fresh.
        Phase 1: bid content comes from NPU synthesis.
        """
        if not self.can_operate():
            return None
        s = self.situation
        if not s.level_2_summary or "Calibrating" in s.level_2_summary:
            return None

        try:
            from global_workspace import WorkspaceBid
        except ImportError:
            return None

        content = s.level_2_summary
        if s.level_3_implications:
            content += f" → {s.level_3_implications}"

        # Urgency scales with situation severity
        urgency = 0.4
        l2 = s.level_2_summary.lower()
        if "critical" in l2:
            urgency = 0.9
        elif "hard stop" in l2 or "deadline" in l2:
            urgency = 0.75
        elif "strained" in l2 or "struggling" in l2:
            urgency = 0.6

        return WorkspaceBid(
            module="cortex",
            content=content,
            urgency=urgency,
            relevance=0.85,  # always highly relevant — it's the room temperature
            surprise=0.0,    # cortex synthesises known state, not surprising events
            confidence=s.confidence,
            emotional_salience=0.3,
        )

    # ------------------------------------------------------------------
    # Capability gate
    # ------------------------------------------------------------------

    def can_operate(self) -> bool:
        """
        Phase 0: True when D113 service state was received within the last 120s.
        Phase 1 (Strix Halo NPU): True when FF_CORTEX_NPU=true.

        Both phases can be active simultaneously — NPU synthesises while the
        service runs as a fallback and sensor aggregator.
        """
        if is_enabled("CORTEX_NPU"):
            return True
        return (time.time() - self._service_last_ok) < 120


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_cortex: Optional[Cortex] = None


def get_cortex() -> Cortex:
    global _cortex
    if _cortex is None:
        _cortex = Cortex()
    return _cortex
