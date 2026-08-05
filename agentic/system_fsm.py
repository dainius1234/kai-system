"""D89: Kai Finite State Machine — operational state management.

States:
    IDLE       — no active user session; sensors polling in background
    ACTIVE     — user interaction in progress
    FOCUSED    — PUB/WORK mode; minimal interruptions
    DEGRADED   — ≥1 critical service unreachable
    RECOVERING — auto-heal attempt in progress

Transitions are defined in _TRANSITIONS.  Undefined (state, event) pairs
are silently ignored — the FSM stays in its current state.
"""
from __future__ import annotations

import asyncio
import logging
from enum import Enum
from typing import List, Optional, Tuple

logger = logging.getLogger("kai.fsm")


class KaiState(str, Enum):
    IDLE = "idle"
    ACTIVE = "active"
    FOCUSED = "focused"
    DEGRADED = "degraded"
    RECOVERING = "recovering"


class KaiEvent(str, Enum):
    USER_MESSAGE = "user_message"
    SESSION_END = "session_end"
    SERVICE_DOWN = "service_down"
    SERVICE_RESTORED = "service_restored"
    ANOMALY_CRITICAL = "anomaly_critical"
    HEAL_STARTED = "heal_started"
    HEAL_COMPLETE = "heal_complete"
    FOCUS_ENTER = "focus_enter"
    FOCUS_EXIT = "focus_exit"


# (current_state, event) → next_state
_TRANSITIONS: dict[tuple[KaiState, KaiEvent], KaiState] = {
    (KaiState.IDLE, KaiEvent.USER_MESSAGE): KaiState.ACTIVE,
    (KaiState.ACTIVE, KaiEvent.SESSION_END): KaiState.IDLE,
    (KaiState.IDLE, KaiEvent.SERVICE_DOWN): KaiState.DEGRADED,
    (KaiState.ACTIVE, KaiEvent.SERVICE_DOWN): KaiState.DEGRADED,
    (KaiState.FOCUSED, KaiEvent.SERVICE_DOWN): KaiState.DEGRADED,
    (KaiState.DEGRADED, KaiEvent.HEAL_STARTED): KaiState.RECOVERING,
    (KaiState.RECOVERING, KaiEvent.HEAL_COMPLETE): KaiState.IDLE,
    (KaiState.RECOVERING, KaiEvent.SERVICE_RESTORED): KaiState.IDLE,
    (KaiState.DEGRADED, KaiEvent.SERVICE_RESTORED): KaiState.IDLE,
    (KaiState.IDLE, KaiEvent.ANOMALY_CRITICAL): KaiState.DEGRADED,
    (KaiState.ACTIVE, KaiEvent.ANOMALY_CRITICAL): KaiState.DEGRADED,
    (KaiState.IDLE, KaiEvent.FOCUS_ENTER): KaiState.FOCUSED,
    (KaiState.ACTIVE, KaiEvent.FOCUS_ENTER): KaiState.FOCUSED,
    (KaiState.FOCUSED, KaiEvent.FOCUS_EXIT): KaiState.IDLE,
    # Stay in FOCUSED during messages — PUB mode does not break focus
    (KaiState.FOCUSED, KaiEvent.USER_MESSAGE): KaiState.FOCUSED,
}


class KaiFSM:
    """Thread-safe finite state machine for Kai's operational state."""

    def __init__(self) -> None:
        self._state = KaiState.IDLE
        self._lock = asyncio.Lock()
        self._history: List[Tuple[str, KaiState, KaiState]] = []

    @property
    def state(self) -> KaiState:
        return self._state

    async def fire(self, event: KaiEvent) -> Optional[KaiState]:
        """Fire an event; return new state if a transition occurred, else None."""
        async with self._lock:
            next_state = _TRANSITIONS.get((self._state, event))
            if next_state is None:
                return None
            prev = self._state
            self._state = next_state
            self._history.append((event.value, prev, next_state))
            if len(self._history) > 100:
                self._history = self._history[-100:]
            logger.info("FSM %s --[%s]--> %s", prev.value, event.value, next_state.value)
            return next_state

    def snapshot(self) -> dict:
        return {
            "state": self._state.value,
            "recent_transitions": [
                {"event": e, "from": f.value, "to": t.value}
                for e, f, t in self._history[-10:]
            ],
        }


_fsm = KaiFSM()


async def fire(event: KaiEvent) -> Optional[KaiState]:
    return await _fsm.fire(event)


def current_state() -> KaiState:
    return _fsm.state


def fsm_snapshot() -> dict:
    return _fsm.snapshot()
