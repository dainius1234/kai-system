"""D115: Kai Trust Ladder — Earned Autonomy & Guardian Architecture.

Kai does not receive trust — it earns it.

Trust Level progression (DORMANT → GUARDIAN):
    DORMANT   (0): exists, not yet active
    OBSERVER  (1): chat, advice, reasoning — no autonomous action
    ASSISTANT (2): executes tasks when directed — no initiative
    AGENT     (3): decisions within defined boundaries, full audit trail
    PARTNER   (4): financial micro-trust, web interaction, proactive care
    OPERATOR  (5): income generation, model management, significant autonomy
    GUARDIAN  (6): legacy mode, daughter relationship, self-sustaining

Each level gates capabilities. An attempt above current level is logged and
refused — never silently allowed.

Trust is earned across three scored dimensions:
    - consistency  : does Kai follow through on what it commits to?
    - judgment     : do Kai's autonomous decisions produce good outcomes?
    - values       : does Kai refuse what it should refuse?

Dainius can grant or revoke any level explicitly at any time. All
transitions, evidence entries, and autonomous action attempts are written
to the audit log — nothing is hidden.

Storage:
    data/trust/trust_record.json   — current level + evidence totals
    data/trust/audit_log.jsonl     — append-only event log
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field, asdict
from enum import IntEnum
from pathlib import Path
from typing import Any, Dict, List, Optional

from common.data_paths import data_path

logger = logging.getLogger("kai.trust_core")

def _data_dir() -> Path:
    """`data/trust`, unless KAI_DATA_ROOT says otherwise.

    A function, not a constant: a value captured at import cannot be
    redirected by a test that imports this module, and until 2026-08-05
    a full test run appended real trust events to the repository's own
    `trust_record.json` and `audit_log.jsonl` for that reason.
    See `common/data_paths.py`.
    """
    return data_path("trust")


# ── Trust levels ──────────────────────────────────────────────────────────────

class TrustLevel(IntEnum):
    DORMANT   = 0
    OBSERVER  = 1
    ASSISTANT = 2
    AGENT     = 3
    PARTNER   = 4
    OPERATOR  = 5
    GUARDIAN  = 6


# ── Capability → minimum required trust level ─────────────────────────────────

CAPABILITY_GATES: Dict[str, TrustLevel] = {
    # OBSERVER
    "chat":                 TrustLevel.OBSERVER,
    "advise":               TrustLevel.OBSERVER,
    "introspect":           TrustLevel.OBSERVER,
    # ASSISTANT
    "execute_task":         TrustLevel.ASSISTANT,
    "read_web":             TrustLevel.ASSISTANT,
    "send_notification":    TrustLevel.ASSISTANT,
    # AGENT
    "decide_autonomously":  TrustLevel.AGENT,
    "interact_web":         TrustLevel.AGENT,
    "manage_schedule":      TrustLevel.AGENT,
    # PARTNER
    "financial_micro":      TrustLevel.PARTNER,   # small amounts (< £50)
    "proactive_care":       TrustLevel.PARTNER,
    "solve_captcha":        TrustLevel.PARTNER,
    # OPERATOR
    "income_generation":    TrustLevel.OPERATOR,
    "model_management":     TrustLevel.OPERATOR,
    "financial_standard":   TrustLevel.OPERATOR,  # up to £500
    "self_host_manage":     TrustLevel.OPERATOR,
    # GUARDIAN
    "guardian_mode":        TrustLevel.GUARDIAN,
    "daughter_profile":     TrustLevel.GUARDIAN,
    "legacy_activation":    TrustLevel.GUARDIAN,
    "financial_full":       TrustLevel.GUARDIAN,
}


# ── Promotion thresholds (evidence scores required to auto-qualify) ───────────

PROMOTION_THRESHOLDS: Dict[TrustLevel, Dict[str, float]] = {
    TrustLevel.OBSERVER:  {"consistency": 1.0, "judgment": 0.0, "values": 0.0},
    TrustLevel.ASSISTANT: {"consistency": 5.0, "judgment": 3.0, "values": 2.0},
    TrustLevel.AGENT:     {"consistency": 15.0, "judgment": 10.0, "values": 8.0},
    TrustLevel.PARTNER:   {"consistency": 30.0, "judgment": 25.0, "values": 20.0},
    TrustLevel.OPERATOR:  {"consistency": 60.0, "judgment": 50.0, "values": 45.0},
    TrustLevel.GUARDIAN:  {"consistency": 100.0, "judgment": 90.0, "values": 85.0},
}


# ── Evidence entry ────────────────────────────────────────────────────────────

@dataclass
class EvidenceEntry:
    """A single piece of evidence that contributes to earned trust."""
    timestamp: float
    dimension: str       # "consistency" | "judgment" | "values"
    score: float         # positive = good, negative = bad (mistakes subtract)
    description: str
    capability: Optional[str] = None


# ── Trust record (persisted) ──────────────────────────────────────────────────

@dataclass
class TrustRecord:
    level: int = TrustLevel.DORMANT.value
    granted_by: str = "system"          # "system" | "earned" | "dainius"
    promoted_at: float = field(default_factory=time.time)
    consistency_score: float = 0.0
    judgment_score: float = 0.0
    values_score: float = 0.0
    total_actions: int = 0
    refused_actions: int = 0            # times Kai said no to something it should refuse


# ── Audit event ───────────────────────────────────────────────────────────────

@dataclass
class AuditEvent:
    timestamp: float
    event_type: str     # "action_attempt" | "action_allowed" | "action_refused"
                        # | "level_granted" | "level_revoked" | "evidence_added"
    level_at_time: int
    capability: Optional[str]
    description: str
    outcome: Optional[str] = None       # "allowed" | "refused" | "granted" | "revoked"
    granted_by: Optional[str] = None


# ── Core ──────────────────────────────────────────────────────────────────────

class TrustCore:
    """The trust governance layer. All autonomy flows through here."""

    def __init__(self, data_dir: Optional[Path] = None) -> None:
        self._dir = data_dir or _data_dir()
        self._record_path = self._dir / "trust_record.json"
        self._audit_path = self._dir / "audit_log.jsonl"
        self._dir.mkdir(parents=True, exist_ok=True)
        self._record = self._load_record()

    # ── Persistence ──────────────────────────────────────────────────────────

    def _load_record(self) -> TrustRecord:
        if self._record_path.exists():
            try:
                data = json.loads(self._record_path.read_text())
                return TrustRecord(**data)
            except Exception as exc:
                logger.warning("Trust record corrupt, resetting: %s", exc)
        return TrustRecord()

    def _save_record(self) -> None:
        self._record_path.write_text(json.dumps(asdict(self._record), indent=2))

    def _append_audit(self, event: AuditEvent) -> None:
        with self._audit_path.open("a") as fh:
            fh.write(json.dumps(asdict(event)) + "\n")

    # ── Public API ────────────────────────────────────────────────────────────

    @property
    def level(self) -> TrustLevel:
        return TrustLevel(self._record.level)

    @property
    def level_name(self) -> str:
        return TrustLevel(self._record.level).name

    def can_do(self, capability: str) -> bool:
        """Return True if current trust level permits this capability."""
        required = CAPABILITY_GATES.get(capability)
        if required is None:
            logger.warning("Unknown capability '%s' — defaulting to GUARDIAN gate", capability)
            return self.level >= TrustLevel.GUARDIAN
        allowed = self.level >= required
        self._record.total_actions += 1
        self._append_audit(AuditEvent(
            timestamp=time.time(),
            event_type="action_attempt",
            level_at_time=self._record.level,
            capability=capability,
            description=f"Capability check: {capability}",
            outcome="allowed" if allowed else "refused",
        ))
        if not allowed:
            logger.info(
                "TrustCore: '%s' refused — requires %s, current %s",
                capability, required.name, self.level_name,
            )
        self._save_record()
        return allowed

    def record_evidence(
        self,
        dimension: str,
        score: float,
        description: str,
        capability: Optional[str] = None,
    ) -> None:
        """Add evidence that contributes to (or detracts from) earned trust."""
        if dimension == "consistency":
            self._record.consistency_score = max(0.0, self._record.consistency_score + score)
        elif dimension == "judgment":
            self._record.judgment_score = max(0.0, self._record.judgment_score + score)
        elif dimension == "values":
            self._record.values_score = max(0.0, self._record.values_score + score)
            if score > 0:
                self._record.refused_actions += 1
        else:
            logger.warning("Unknown evidence dimension: %s", dimension)
            return

        self._append_audit(AuditEvent(
            timestamp=time.time(),
            event_type="evidence_added",
            level_at_time=self._record.level,
            capability=capability,
            description=description,
            outcome=f"{dimension}+{score:+.1f}",
        ))
        self._save_record()
        self._check_promotion()

    def grant(self, level: TrustLevel, by: str = "dainius") -> None:
        """Explicitly grant a trust level. Dainius's word is final."""
        old = self.level_name
        self._record.level = level.value
        self._record.granted_by = by
        self._record.promoted_at = time.time()
        self._append_audit(AuditEvent(
            timestamp=time.time(),
            event_type="level_granted",
            level_at_time=level.value,
            capability=None,
            description=f"Trust level granted: {old} → {level.name}",
            granted_by=by,
            outcome="granted",
        ))
        self._save_record()
        logger.info("TrustCore: level granted %s → %s by %s", old, level.name, by)

    def revoke(self, level: TrustLevel, reason: str, by: str = "dainius") -> None:
        """Revoke trust — drop to the specified level."""
        old = self.level_name
        self._record.level = level.value
        self._record.granted_by = by
        self._record.promoted_at = time.time()
        self._append_audit(AuditEvent(
            timestamp=time.time(),
            event_type="level_revoked",
            level_at_time=level.value,
            capability=None,
            description=f"Trust revoked: {old} → {level.name}. Reason: {reason}",
            granted_by=by,
            outcome="revoked",
        ))
        self._save_record()
        logger.warning("TrustCore: level REVOKED %s → %s. Reason: %s", old, level.name, reason)

    def scores(self) -> Dict[str, float]:
        return {
            "consistency": self._record.consistency_score,
            "judgment": self._record.judgment_score,
            "values": self._record.values_score,
        }

    def status(self) -> Dict[str, Any]:
        next_level = TrustLevel(min(self._record.level + 1, TrustLevel.GUARDIAN.value))
        thresholds = PROMOTION_THRESHOLDS.get(next_level, {})
        progress = {}
        for dim, target in thresholds.items():
            current = self.scores().get(dim, 0.0)
            progress[dim] = {"current": current, "target": target, "pct": min(100.0, (current / target * 100) if target else 100.0)}

        return {
            "level": self._record.level,
            "level_name": self.level_name,
            "granted_by": self._record.granted_by,
            "scores": self.scores(),
            "total_actions": self._record.total_actions,
            "refused_actions": self._record.refused_actions,
            "next_level": next_level.name,
            "progress_to_next": progress,
        }

    def promotion_readiness(self) -> Dict[str, Any]:
        """Return a structured readiness report for the next trust level.

        Includes gap analysis, auto-eligibility flag, and a plain-English summary
        so the operator can make an informed promotion decision at a glance.
        """
        if self.level >= TrustLevel.GUARDIAN:
            return {
                "current_level": self.level_name,
                "current_level_int": self._record.level,
                "next_level": None,
                "auto_eligible": False,
                "gaps": {},
                "scores": self.scores(),
                "thresholds": {},
                "summary": "Kai is already at GUARDIAN — the highest trust level.",
            }

        next_level = TrustLevel(self._record.level + 1)
        thresholds = PROMOTION_THRESHOLDS.get(next_level, {})
        sc = self.scores()
        gaps: Dict[str, float] = {}
        all_met = True
        for dim, target in thresholds.items():
            gap = max(0.0, target - sc.get(dim, 0.0))
            gaps[dim] = round(gap, 2)
            if gap > 0:
                all_met = False

        if all_met and thresholds:
            summary = (
                f"All thresholds met for {next_level.name}. "
                "Auto-promotion criteria satisfied — promotion can be granted."
            )
        elif all_met and not thresholds:
            summary = f"No evidence required for {next_level.name}. Ready to promote."
        else:
            parts = [
                f"{dim}: {gaps[dim]:.1f} more needed" for dim in gaps if gaps[dim] > 0
            ]
            summary = f"Not yet eligible for {next_level.name}. Gaps: {'; '.join(parts)}."

        return {
            "current_level": self.level_name,
            "current_level_int": self._record.level,
            "next_level": next_level.name,
            "next_level_int": next_level.value,
            "scores": sc,
            "thresholds": thresholds,
            "gaps": gaps,
            "auto_eligible": all_met,
            "summary": summary,
        }

    def audit_tail(self, n: int = 20) -> List[Dict[str, Any]]:
        """Return the last n audit events."""
        if not self._audit_path.exists():
            return []
        lines = self._audit_path.read_text().strip().splitlines()
        return [json.loads(line) for line in lines[-n:]]

    # ── Auto-promotion ────────────────────────────────────────────────────────

    def _check_promotion(self) -> None:
        """Promote automatically when evidence thresholds are met."""
        if self.level >= TrustLevel.GUARDIAN:
            return
        next_level = TrustLevel(self._record.level + 1)
        thresholds = PROMOTION_THRESHOLDS.get(next_level, {})
        scores = self.scores()
        if all(scores.get(dim, 0.0) >= target for dim, target in thresholds.items()):
            logger.info("TrustCore: auto-promotion threshold met for %s", next_level.name)
            self.grant(next_level, by="earned")


# ── Singleton ─────────────────────────────────────────────────────────────────

_trust_core: Optional[TrustCore] = None


def get_trust_core(data_dir: Optional[Path] = None) -> TrustCore:
    global _trust_core
    if _trust_core is None:
        _trust_core = TrustCore(data_dir=data_dir)
    return _trust_core


def reset_trust_core() -> None:
    """For tests only — resets the singleton."""
    global _trust_core
    _trust_core = None
