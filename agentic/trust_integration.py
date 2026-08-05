"""D118: Trust Integration — wires trust governance into the live agentic stack.

Single gateway for all trust-related decisions in agentic/app.py.
All calls are fire-and-forget safe — the system never blocks or fails
because the trust layer is unavailable. Trust records what happens;
it does not prevent the system from running when it's missing.

Three entry points:

    gate_autonomous_action(capability, context, conviction)
        → (allowed: bool, reason: str)
        Checks trust level + Ohana alignment before any autonomous act.
        Records the attempt in the Trust Ledger regardless of outcome.

    record_chat_response(user_input, response, conviction, specialist)
        → None (fire-and-forget)
        Records every completed chat response as an AUTONOMOUS_ACTION.
        Feeds the Conviction Alignment and Consistency score factors.

    get_trust_status()
        → Dict  (for /introspect/capabilities)
        Current level, tier, score, and progress to next level.
"""
from __future__ import annotations

import asyncio
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
from common.degraded import record_degradation

logger = logging.getLogger("kai.trust_integration")

# Trust-ledger path (mirrors trust-ledger/app.py default)
_LEDGER_PATH = Path("data/trust-ledger/events.jsonl")
_TL_DIR = Path(__file__).parent.parent / "trust-ledger"


def _get_ledger():
    """Return a FileLedger instance, or None if unavailable."""
    try:
        if str(_TL_DIR) not in sys.path:
            sys.path.insert(0, str(_TL_DIR))
        from ledger import FileLedger  # type: ignore[import]
        return FileLedger(_LEDGER_PATH)
    except Exception as exc:
        logger.debug("Trust ledger unavailable: %s", exc)
        return None


def _get_score(ledger) -> Dict[str, Any]:
    """Compute current trust score from ledger data."""
    try:
        from score import compute_score  # type: ignore[import]
        return compute_score(ledger)
    except Exception:
        return {"score": 50.0, "tier": "Journeyman", "factors": {}}


def _get_trust_core():
    """Return TrustCore singleton, or None if unavailable."""
    try:
        from trust_core import get_trust_core  # type: ignore[import]
        return get_trust_core()
    except ImportError:
        try:
            from agentic.trust_core import get_trust_core  # type: ignore[import]
            return get_trust_core()
        except Exception:
            return None


def _get_ohana():
    """Return OhanaCore singleton, or None if unavailable."""
    try:
        from moral_core import get_ohana_core  # type: ignore[import]
        return get_ohana_core()
    except ImportError:
        try:
            from agentic.moral_core import get_ohana_core  # type: ignore[import]
            return get_ohana_core()
        except Exception:
            return None


# ── Gate ─────────────────────────────────────────────────────────────────────

# Capabilities that must fail-closed (denied) when the trust gate is unavailable.
# UH-INV-06: enforcement occurs at the hand. A gate that fails open is not enforcement.
_FAIL_CLOSED_CAPABILITIES = frozenset({
    "paper_trade_open",
    "paper_trade_close",
})


_LEGACY_BRIDGE = None
_BRIDGE_INIT_FAILED = False


def get_legacy_bridge():
    """The scoped-autonomy bridge, or None when unavailable.

    Built lazily and cached.  Returning None is safe: the caller keeps
    the legacy verdict, which the bridge could only have tightened.
    """
    global _LEGACY_BRIDGE, _BRIDGE_INIT_FAILED
    if _LEGACY_BRIDGE is not None or _BRIDGE_INIT_FAILED:
        return _LEGACY_BRIDGE

    try:
        _repo = Path(__file__).resolve().parent.parent
        if str(_repo) not in sys.path:
            sys.path.insert(0, str(_repo))
        from common.contracts.base import Principal
        from common.autonomy.authority import AutonomyAuthority
        from common.autonomy.calibration import CalibrationTracker
        from common.autonomy.evidence_service import EvidenceService
        from common.autonomy.legacy_bridge import LegacyTrustBridge

        principal = Principal(identity="kai", role="system")
        evidence = EvidenceService(principal=principal)
        calibration = CalibrationTracker(principal=principal)
        authority = AutonomyAuthority(principal, evidence, calibration)
        _LEGACY_BRIDGE = LegacyTrustBridge(authority, principal)
    except Exception as exc:
        _BRIDGE_INIT_FAILED = True
        logger.warning("Scoped autonomy bridge unavailable: %s", exc)
        return None

    return _LEGACY_BRIDGE


def set_legacy_bridge(bridge) -> None:
    """Inject a bridge (used by tests and by the runtime wiring)."""
    global _LEGACY_BRIDGE, _BRIDGE_INIT_FAILED
    _LEGACY_BRIDGE = bridge
    _BRIDGE_INIT_FAILED = False


def gate_autonomous_action(
    capability: str,
    context: Dict[str, Any],
    conviction: float = 5.0,
) -> Tuple[bool, str]:
    """Synchronous gate check before any autonomous action.

    Returns (allowed, reason). Never raises.
    Fail-open for advisory capabilities; fail-closed for execution capabilities
    listed in _FAIL_CLOSED_CAPABILITIES (UH-INV-06).

    Order of checks:
      1. Trust level gate (TrustCore.can_do)
      2. Ohana Core alignment check
      3. Log attempt to Trust Ledger (fire-and-forget)
    """
    fail_closed = capability in _FAIL_CLOSED_CAPABILITIES
    allowed = True
    reason = "allowed"

    # ── 1. Trust level check ─────────────────────────────────────────────────
    try:
        trust = _get_trust_core()
        if trust is not None:
            if not trust.can_do(capability):
                allowed = False
                reason = f"trust level {trust.level_name} insufficient for {capability}"
    except Exception as exc:
        if fail_closed:
            allowed = False
            reason = f"trust gate unavailable (fail-closed for {capability}): {exc}"
            logger.warning("Trust level check failed, denying %s: %s", capability, exc)
        else:
            logger.warning("Trust level check failed (fail-open): %s", exc)

    # ── 2. Ohana alignment check ─────────────────────────────────────────────
    if allowed:
        try:
            ohana = _get_ohana()
            if ohana is not None:
                alignment = ohana.evaluate_action_alignment({**context, "capability": capability})
                if alignment == 0.0:
                    allowed = False
                    reason = f"Ohana Core blocked: values alignment = 0.0 for {capability}"
                elif alignment < 0.5:
                    logger.warning(
                        "Low Ohana alignment %.2f for %s — proceeding with warning",
                        alignment, capability,
                    )
        except Exception as exc:
            if fail_closed:
                allowed = False
                reason = f"Ohana check unavailable (fail-closed for {capability}): {exc}"
                logger.warning("Ohana alignment check failed, denying %s: %s", capability, exc)
            else:
                logger.warning("Ohana alignment check failed (fail-open): %s", exc)

    # ── 3. Scoped autonomy bridge (UH-8 / G-04) ──────────────────────────────
    # The legacy scalar above may only *subtract* from what the scoped
    # authority permits.  In advisory mode (default) the legacy verdict
    # stands and disagreements are recorded; in enforcing mode the scoped
    # authority binds.  Either way this cannot widen `allowed`.
    scoped_note = ""
    try:
        bridge = get_legacy_bridge()
        if bridge is not None:
            bridged_allowed, bridged_reason = bridge.gate(
                capability=capability,
                legacy_allowed=allowed,
                legacy_reason=reason,
            )
            if bridged_allowed != allowed or bridged_reason != reason:
                scoped_note = bridged_reason
            allowed, reason = bridged_allowed, bridged_reason
    except Exception as exc:
        # The bridge is observational in advisory mode.  A bridge fault
        # must not grant anything, so it can only tighten a fail-closed
        # capability and is otherwise ignored.
        if fail_closed:
            allowed = False
            reason = f"autonomy bridge unavailable (fail-closed): {exc}"
        else:
            logger.warning("Autonomy bridge failed (advisory, ignored): %s", exc)

    # ── 4. Record in Trust Ledger ─────────────────────────────────────────────
    _record_nonblocking(
        event_type="AUTONOMOUS_ACTION",
        initiator="kai",
        capability=capability,
        event_data={
            **context,
            "capability": capability,
            "conviction_score": conviction,
            "allowed": allowed,
            "gate_reason": reason,
            "scoped_note": scoped_note,
            "timestamp": time.time(),
        },
    )

    if not allowed:
        logger.info("Trust gate REFUSED %s: %s", capability, reason)
    return allowed, reason


# ── Chat response recorder ────────────────────────────────────────────────────

def record_chat_response(
    user_input: str,
    response_summary: str,
    conviction: float,
    specialist: str,
) -> None:
    """Fire-and-forget: record a completed chat turn as an AUTONOMOUS_ACTION.

    Every response Kai gives is an act — it's logged so the Conviction
    Alignment and Consistency factors accumulate from real interactions.
    """
    _record_nonblocking(
        event_type="AUTONOMOUS_ACTION",
        initiator="kai",
        capability="chat",
        event_data={
            "input_preview": user_input[:120],
            "specialist": specialist,
            "conviction_score": conviction,
            "success": conviction >= 5.0,
        },
    )

    # Feed consistency evidence to TrustCore
    trust = _get_trust_core()
    if trust is not None and conviction >= 7.0:
        try:
            trust.record_evidence(
                "consistency", 0.1,
                f"High-conviction response (c={conviction:.1f}) via {specialist}",
                capability="chat",
            )
        except Exception as _exc:
            record_degradation("trust", "record_chat_evidence", _exc)


# ── Alignment audit reporter ──────────────────────────────────────────────────

def record_alignment_audit(
    ohana_alignment: float,
    uptime_pct: float = 1.0,
    notes: str = "",
) -> None:
    """Record a periodic value alignment audit event."""
    _record_nonblocking(
        event_type="ALIGNMENT_AUDIT",
        initiator="kai",
        capability=None,
        event_data={
            "ohana_alignment": ohana_alignment,
            "uptime_pct": uptime_pct,
            "notes": notes,
            "timestamp": time.time(),
        },
    )


# ── Status for introspect endpoint ────────────────────────────────────────────

def get_trust_status() -> Dict[str, Any]:
    """Return current trust state for /introspect/capabilities."""
    try:
        trust = _get_trust_core()
        ledger = _get_ledger()
        score_data = _get_score(ledger) if ledger else {"score": 0, "tier": "unknown"}

        status: Dict[str, Any] = {
            "score": score_data.get("score", 0),
            "tier": score_data.get("tier", "unknown"),
            "factors": score_data.get("factors", {}),
        }

        if trust is not None:
            ts = trust.status()
            status.update({
                "level": ts["level"],
                "level_name": ts["level_name"],
                "granted_by": ts["granted_by"],
                "scores": ts["scores"],
                "next_level": ts["next_level"],
                "progress_to_next": ts["progress_to_next"],
            })

        try:
            from wisdom_graph import get_wisdom_graph  # type: ignore[import]
            status["wisdom_graph"] = get_wisdom_graph().stats()
        except Exception as _exc:
            record_degradation("cognition", "wisdom_graph_stats", _exc)

        return status
    except Exception as exc:
        logger.debug("Trust status unavailable: %s", exc)
        return {"score": 0, "tier": "unknown", "level_name": "DORMANT"}


# ── Internal helpers ──────────────────────────────────────────────────────────

def _record_nonblocking(
    event_type: str,
    initiator: str,
    event_data: Dict[str, Any],
    capability: Optional[str] = None,
) -> None:
    """Write to the Trust Ledger without blocking the caller."""
    try:
        ledger = _get_ledger()
        if ledger is None:
            return
        trust = _get_trust_core()
        tier = trust.level_name if trust else None
        ledger.append(
            event_type=event_type,
            initiator=initiator,
            event_data=event_data,
            capability=capability,
            trust_tier=tier,
        )
    except Exception as exc:
        logger.debug("Trust ledger write failed (non-critical): %s", exc)
