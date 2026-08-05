"""D102: Global Workspace Consciousness.

Based on Global Workspace Theory (Baars, 1988; Dehaene, 2014) — the dominant
neuroscientific account of how unified conscious awareness arises from competing
specialist modules.

KAI already has the specialists: council members, swarm teammates, perception
services, memory graph, causal world model, hypothesis engine, temporal
projector. But they run in parallel and hand off results. There is no single
"moment" of KAI's awareness — just a sequence of outputs.

A global workspace changes that. At any given moment KAI has one coherent
"conscious content" — selected from competing module bids by salience, urgency,
and goal-relevance — then broadcast to every module simultaneously. Each module
processes the broadcast in its own way (memory retrieves, debate engine
challenges, causal model simulates) and may re-bid. This creates a continuous,
serial stream of unified awareness.

Provides:
  - WorkspaceBid    — a module's proposal to occupy the workspace
  - ConsciousMoment — the broadcast content of a won bid
  - GlobalWorkspace — the serial bottleneck of KAI's conscious awareness

Phase 0 (NOW):
  - All can_*() methods return False. No computation occurs.
  - submit_bid, subscribe, get_stream interfaces are frozen and ready.
  - Singleton get_global_workspace() available for early wiring.

Phase 3 (post-GPU + causal world model + cognitive fingerprint):
  - Bidding cycle runs every cycle_ms (default 100ms).
  - Salience function incorporates cognitive fingerprint value weights.
  - Global ignition: broadcast fires all subscriber callbacks in parallel.
  - ConsciousMoment stream logged to /data/conscious_stream.jsonl.
  - Dashboard "Stream" view shows last N moments as live inner monologue.

Activation conditions (all required):
  - FF_GLOBAL_WORKSPACE = True
  - GPU available (RTX 5080)
  - FF_CAUSAL_WORLD_MODEL = True (surprise signals)
  - Cognitive fingerprint ≥90 samples (personalised salience weights)
  - ≥3 active modules registered as bidders

Feature flag: FF_GLOBAL_WORKSPACE (default False)
"""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class WorkspaceBid:
    """A module's proposal to occupy the global workspace for one moment."""
    module: str               # source module (e.g. "perception", "memory", "causal_model")
    content: str              # what the module wants KAI to be aware of
    urgency: float            # 0.0–1.0 time-sensitivity
    relevance: float          # 0.0–1.0 goal-relevance
    surprise: float           # 0.0–1.0 prediction error from causal world model
    confidence: float         # 0.0–1.0 certainty of the module
    emotional_salience: float  # 0.0–1.0 from emotional memory
    timestamp: float = field(default_factory=time.time)


@dataclass
class ConsciousMoment:
    """A single broadcast moment in KAI's unified stream of awareness."""
    timestamp: float
    content: str              # what KAI is aware of right now
    source_module: str        # which module won the bid
    salience_score: float     # composite score that won the bid
    broadcast_id: str         # unique ID for this moment
    context: Dict[str, Any] = field(default_factory=dict)
    emotional_valence: float = 0.0  # -1.0 (negative) to +1.0 (positive)


# ---------------------------------------------------------------------------
# Global Workspace (Stub)
# ---------------------------------------------------------------------------

class GlobalWorkspace:
    """Serial bottleneck of KAI's conscious awareness.

    Phase 0: methods are no-ops; capability gate returns False.

    Phase 3 operation loop (once activated):
      1. Each registered module submits bids via submit_bid().
      2. select_winner() evaluates all bids using the salience function.
      3. broadcast() pushes the winning ConsciousMoment to all subscribers.
      4. Subscribers (memory, debate engine, causal model, etc.) process the
         broadcast and may submit new bids for the next moment.
      5. The cycle repeats every cycle_ms, producing the conscious stream.

    The sequence of ConsciousMoment objects is KAI's stream of consciousness:
    logged, queryable, and renderable as a live inner monologue on the dashboard.
    """

    def __init__(self, max_stream_length: int = 10_000, cycle_ms: float = 100.0) -> None:
        self._stream: List[ConsciousMoment] = []
        self._max_stream = max_stream_length
        self._cycle_ms = cycle_ms
        self._subscribers: Dict[str, Callable[[ConsciousMoment], None]] = {}
        self._bid_queue: List[WorkspaceBid] = []

    # --- bidding & selection (stubs) --------------------------------------

    def submit_bid(self, bid: WorkspaceBid) -> None:
        """Receive a bid from a module.

        Phase 0: logged and discarded.
        Phase 3: queued for next selection cycle.
        """
        logger.debug(
            "Workspace bid submitted (stub): module=%s content=%.40s",
            bid.module, bid.content,
        )

    def select_winner(self) -> Optional[WorkspaceBid]:
        """Evaluate all active bids and return the winner.

        Salience function (Phase 3):
          score = (urgency × w_u) + (relevance × w_r) + (surprise × w_s)
                + (confidence × w_c) + (emotional_salience × w_e)
          where weights w_* are personalised by the cognitive fingerprint.

        Phase 0: returns None.
        """
        return None  # Phase 3: weighted salience competition

    # --- broadcast (stub) -------------------------------------------------

    def broadcast(self, moment: ConsciousMoment) -> None:
        """Push a conscious moment to the stream and all subscribers.

        Phase 0: no-op.
        Phase 3: appends to _stream (capped at max_stream_length), fires all
                 subscriber callbacks in parallel (asyncio.gather), logs to
                 /data/conscious_stream.jsonl.
        """
        logger.debug("Broadcast (stub): %s", moment.content[:80])

    # --- subscriber management --------------------------------------------

    def subscribe(
        self,
        module_name: str,
        callback: Callable[[ConsciousMoment], None],
    ) -> None:
        """Register a module to receive every broadcast.

        Phase 0: stored but never called.
        Phase 3: called on every broadcast cycle.
        """
        self._subscribers[module_name] = callback
        logger.debug("Subscriber registered (stub): %s", module_name)

    def unsubscribe(self, module_name: str) -> None:
        """Remove a module from the broadcast list."""
        self._subscribers.pop(module_name, None)

    def subscriber_count(self) -> int:
        return len(self._subscribers)

    # --- stream access ----------------------------------------------------

    def get_stream(self, limit: int = 50) -> List[ConsciousMoment]:
        """Return the most recent conscious moments.

        Phase 0: empty.
        Phase 3: last `limit` entries from the live stream.
        """
        return []  # Phase 3: return self._stream[-limit:]

    def get_latest_moment(self) -> Optional[ConsciousMoment]:
        """Return the most recent conscious moment (None if stream is empty)."""
        return None  # Phase 3: return self._stream[-1] if self._stream else None

    def stream_length(self) -> int:
        return len(self._stream)

    # --- capability gate --------------------------------------------------

    @staticmethod
    def can_operate() -> bool:
        """True only when all activation conditions are met.

        Activation requirements:
          - FF_GLOBAL_WORKSPACE = True
          - GPU available
          - FF_CAUSAL_WORLD_MODEL = True (provides surprise signals)
          - Cognitive fingerprint ≥90 samples (D98, personalised salience)
          - ≥3 modules registered as bidders

        In Phase 0, always returns False.
        """
        try:
            from feature_flags import is_enabled
            if not is_enabled("GLOBAL_WORKSPACE"):
                return False
        except ImportError:
            pass
        return False  # GPU-era gate

    def progress(self) -> Dict[str, Any]:
        return {
            "can_operate": self.can_operate(),
            "subscribers": self.subscriber_count(),
            "stream_length": self.stream_length(),
            "cycle_ms": self._cycle_ms,
        }


# ---------------------------------------------------------------------------
# Singleton factory
# ---------------------------------------------------------------------------

_workspace: Optional[GlobalWorkspace] = None


def get_global_workspace() -> GlobalWorkspace:
    """Return the shared GlobalWorkspace singleton."""
    global _workspace
    if _workspace is None:
        _workspace = GlobalWorkspace()
    return _workspace
