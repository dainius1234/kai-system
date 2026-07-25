"""D101: Causal World Model & Counterfactual Policy Learning.

This module provides:
  - CausalGraph       — typed causal edges on top of the knowledge graph
  - WorldModelSimulator — GPU-driven mental simulations
  - PolicyMemory      — distilled strategies from simulations (in-memory stub;
                        see also agentic/policy_memory.py for JSONL-persisted version)
  - CausalSurpriseDetector — prediction-error-based model refinement

Phase 0 (NOW):
  - All can_*() methods return False. Interfaces are frozen.
  - CausalGraph stores edges in-memory so modules can begin recording
    causal observations immediately.
  - Factory functions return shared singletons (thread-safe reads in Phase 0).

Phase 1 (GPU + data thresholds):
  - CausalGraph backed by Cognee/Kuzu CAUSES edge type.
  - WorldModelSimulator runs N=50 scenario variants per idle GPU cycle.
  - PolicyMemory distils simulation outcomes into ranked strategies.
  - CausalSurpriseDetector wired into proactive observer loop.

Activation conditions (all must be true):
  - FF_CAUSAL_WORLD_MODEL = True
  - GPU available (RTX 5080)
  - Cognitive fingerprint collected (90+ samples — D98)
  - Knowledge graph ≥1000 nodes
  - Historical sensor data ≥30 days

Feature flag: FF_CAUSAL_WORLD_MODEL (default False)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class CausalEdge:
    """A probabilistic cause-effect relationship between two concepts."""
    source: str                         # Node ID or concept name (cause)
    target: str                         # Node ID or concept name (effect)
    strength: float                     # 0.0–1.0, probabilistic weight
    confidence: float                   # calibrated confidence (0.0–1.0)
    temporal_lag_seconds: float = 0.0   # typical delay from cause to effect
    direction: str = "direct"           # "direct" or "inverse"
    context_modifiers: Dict[str, Any] = field(default_factory=dict)
    source_type: str = "observed"       # observed | simulated | inferred | user_stated
    evidence_count: int = 0
    last_updated: str = ""              # ISO timestamp


@dataclass
class Policy:
    """A distilled strategy: if <condition> → do <action> → expect <outcome>."""
    name: str
    condition: str
    action: str
    expected_outcome: str
    confidence: float                   # 0.0–1.0
    evidence_type: str = "simulated"    # simulated | observed | analogical
    supporting_edges: List[str] = field(default_factory=list)
    success_rate: float = 0.0
    last_applied: Optional[str] = None
    version: int = 1


@dataclass
class SimulationScenario:
    """A scenario to evaluate via mental simulation."""
    goal: str
    initial_state: Dict[str, Any]
    actions: List[str]
    horizon_steps: int = 3
    variations_per_action: int = 10


@dataclass
class SimulationResult:
    """Output of a single simulation run."""
    scenario_id: str
    action: str
    outcome_path: List[Dict[str, Any]]
    final_utility: float
    key_causal_edges_triggered: List[str]
    confidence: float = 0.0


# ---------------------------------------------------------------------------
# Causal Graph (Stub)
# ---------------------------------------------------------------------------

class CausalGraph:
    """In-memory causal edge store.

    Phase 0: edges stored in dict, not yet persisted to Cognee/Kuzu.
    Phase 3: edges written as typed CAUSES relationships in the knowledge graph.

    Cognee schema extension (Phase 3):
        (:Concept)-[:CAUSES {
            strength: float, confidence: float,
            temporal_lag_seconds: float, direction: string,
            context_modifiers: json, source_type: string,
            evidence_count: int
        }]->(:Concept)
    """

    def __init__(self) -> None:
        self._edges: Dict[str, CausalEdge] = {}

    # --- edge CRUD --------------------------------------------------------

    def add_edge(self, edge: CausalEdge) -> str:
        """Store a causal edge. Returns edge ID."""
        edge_id = f"causal:{edge.source}->{edge.target}"
        self._edges[edge_id] = edge
        logger.debug("CausalEdge added (stub): %s", edge_id)
        return edge_id

    def get_edge(self, edge_id: str) -> Optional[CausalEdge]:
        return self._edges.get(edge_id)

    # --- queries (stub returns) -------------------------------------------

    def query_causal_path(self, source: str, target: str) -> List[CausalEdge]:
        """Find chain(s) of causal edges from source to target."""
        return []   # Phase 3: graph traversal

    def get_downstream_effects(self, cause: str) -> List[CausalEdge]:
        """All effects of a given cause."""
        return []   # Phase 3: Cognee query

    def get_upstream_causes(self, effect: str) -> List[CausalEdge]:
        """All causes of a given effect."""
        return []   # Phase 3: Cognee query

    def predict_outcome(
        self,
        current_state: Dict[str, Any],
        action: str,
    ) -> Dict[str, Any]:
        """Predict next state given an action, using causal edges."""
        return {}   # Phase 3: probabilistic forward inference

    # --- capability gate --------------------------------------------------

    @staticmethod
    def can_reason() -> bool:
        """Requires GPU + ≥1000 graph nodes + 30 days historical data.

        In Phase 0, always returns False.
        """
        try:
            from feature_flags import is_enabled
            if not is_enabled("CAUSAL_WORLD_MODEL"):
                return False
        except ImportError:
            pass
        return False  # GPU-era gate

    # --- introspection ----------------------------------------------------

    def edge_count(self) -> int:
        return len(self._edges)


# ---------------------------------------------------------------------------
# World Model Simulator (Stub)
# ---------------------------------------------------------------------------

class WorldModelSimulator:
    """Runs counterfactual simulations using the causal graph.

    Phase 3: GPU-accelerated Monte Carlo; N=50 action variants per cycle,
    scored by expected utility (weighted by cognitive fingerprint values).
    Top insights stored as simulated_experience memories.
    """

    def __init__(self, causal_graph: CausalGraph) -> None:
        self.graph = causal_graph

    def simulate_scenario(self, scenario: SimulationScenario) -> List[SimulationResult]:
        """Run a batch of simulations for a given scenario."""
        return []   # Phase 3: GPU simulation body

    def run_background_simulations(self, active_goals: List[str]) -> int:
        """Run simulations during idle GPU cycles.

        Returns number of results generated (Phase 0: always 0).
        """
        return 0

    @staticmethod
    def can_simulate() -> bool:
        """Requires GPU + causal graph ready + cognitive fingerprint."""
        try:
            from feature_flags import is_enabled
            if not is_enabled("CAUSAL_WORLD_MODEL"):
                return False
        except ImportError:
            pass
        return False  # GPU-era gate


# ---------------------------------------------------------------------------
# Policy Memory (Stub — in-memory)
# ---------------------------------------------------------------------------

class PolicyMemory:
    """Stores and retrieves learned policies (in-memory stub).

    For JSONL-persisted, production-grade Phase 0 policy storage see
    agentic/policy_memory.py (PolicyLibrary).

    Phase 3: distilled automatically from WorldModelSimulator output;
    persisted to PostgreSQL with versioning and success-rate tracking.
    """

    def __init__(self) -> None:
        self._policies: Dict[str, Policy] = {}

    def add_policy(self, policy: Policy) -> str:
        """Store a policy. Returns policy ID."""
        pid = f"policy:{policy.name}"
        self._policies[pid] = policy
        logger.debug("Policy stored (stub): %s", pid)
        return pid

    def get_relevant_policies(self, context: Dict[str, Any]) -> List[Policy]:
        """Retrieve policies applicable to the given context."""
        return []   # Phase 3: embedding similarity over condition + action + domain

    def update_policy_success(self, policy_id: str, success: bool) -> None:
        """Update success rate after real-world application."""

    @staticmethod
    def can_learn_policies() -> bool:
        """Requires simulation data + sufficient historical outcomes."""
        try:
            from feature_flags import is_enabled
            if not is_enabled("CAUSAL_WORLD_MODEL"):
                return False
        except ImportError:
            pass
        return False  # GPU-era gate


# ---------------------------------------------------------------------------
# Causal Surprise Detector (Stub)
# ---------------------------------------------------------------------------

class CausalSurpriseDetector:
    """Compares world model predictions against actual observations.

    When divergence exceeds surprise_threshold, returns a description of
    the gap and triggers edge-strength recalibration and hypothesis formation.

    Phase 3: wired into the proactive observer; divergence fires a
    HypothesisEngine cycle with the surprise as the seed topic.
    """

    def __init__(
        self,
        causal_graph: CausalGraph,
        threshold: float = 0.3,
    ) -> None:
        self.graph = causal_graph
        self.surprise_threshold = threshold

    def check_surprise(
        self,
        predicted_state: Dict[str, Any],
        actual_state: Dict[str, Any],
    ) -> Optional[str]:
        """Return surprise description if divergence ≥ threshold, else None."""
        return None  # Phase 3: cosine distance or probability difference

    @staticmethod
    def can_detect_surprise() -> bool:
        """Requires causal graph + prediction infrastructure + GPU."""
        try:
            from feature_flags import is_enabled
            if not is_enabled("CAUSAL_SURPRISE"):
                return False
        except ImportError:
            pass
        return False  # GPU-era gate


# ---------------------------------------------------------------------------
# Singletons / factory functions
# ---------------------------------------------------------------------------

_causal_graph: Optional[CausalGraph] = None
_policy_memory: Optional[PolicyMemory] = None
_simulator: Optional[WorldModelSimulator] = None
_surprise_detector: Optional[CausalSurpriseDetector] = None


def get_causal_graph() -> CausalGraph:
    global _causal_graph
    if _causal_graph is None:
        _causal_graph = CausalGraph()
    return _causal_graph


def get_policy_memory() -> PolicyMemory:
    global _policy_memory
    if _policy_memory is None:
        _policy_memory = PolicyMemory()
    return _policy_memory


def get_simulator() -> WorldModelSimulator:
    global _simulator
    if _simulator is None:
        _simulator = WorldModelSimulator(get_causal_graph())
    return _simulator


def get_surprise_detector() -> CausalSurpriseDetector:
    global _surprise_detector
    if _surprise_detector is None:
        _surprise_detector = CausalSurpriseDetector(get_causal_graph())
    return _surprise_detector
