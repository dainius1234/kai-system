"""Bridge from the legacy scalar TrustLevel to scoped autonomy grants.

Closes UH tracker gap G-04.  Two authority systems coexisted: the old
``TrustLevel`` scalar (one number, applied everywhere, never expiring)
and UH-8's scoped grants.  Two authorities is worse than either alone —
whichever is more permissive wins by accident.

This bridge unifies them under one rule: **the legacy scalar may only
deny, never grant.**  A capability is permitted only if the new
authority permits it; the legacy check can then subtract, never add.
Whatever the old system says, it cannot widen what the new one allows.

Migration runs in two modes, mirroring the actuator migration discipline:

  - ``advisory`` (default) — the new authority is consulted and any
    disagreement is recorded, but the legacy decision stands.  This is
    the shadow phase: it shows what enforcement *would* do without
    changing behaviour.
  - ``enforcing`` — the new authority's decision binds.  Enabled with
    ``KAI_AUTONOMY_ENFORCE=true``.

Disagreements are counted either way, so the decision to switch modes
rests on observed data rather than optimism.
"""
from __future__ import annotations

import logging
import os
from typing import Dict, List, Optional, Tuple

from common.contracts.base import Principal
from common.contracts.autonomy import AutonomyLevel
from common.autonomy.authority import AutonomyAuthority

logger = logging.getLogger("kai.autonomy.legacy_bridge")

ENFORCE_ENV = "KAI_AUTONOMY_ENFORCE"


# Legacy capability name → (capability, domain) in the scoped model.
# A capability absent from this map has no scoped equivalent yet and is
# treated as un-migrated.
CAPABILITY_DOMAINS: Dict[str, Tuple[str, str]] = {
    "paper_trade_open": ("paper_trade_open", "trading"),
    "paper_trade_close": ("paper_trade_close", "trading"),
    "auto_trade": ("auto_trade", "trading"),
    "web_scout": ("web_fetch", "research"),
    "web_search": ("web_search", "research"),
    "model_council": ("model_council", "reasoning"),
    "chat": ("chat", "conversation"),
    "advise": ("advise", "conversation"),
}


def enforcing() -> bool:
    return os.getenv(ENFORCE_ENV, "false").lower() in {"1", "true", "yes"}


class Disagreement:
    __slots__ = ("capability", "domain", "legacy_allowed",
                 "scoped_allowed", "scoped_reason")

    def __init__(
        self,
        capability: str,
        domain: str,
        legacy_allowed: bool,
        scoped_allowed: bool,
        scoped_reason: str,
    ) -> None:
        self.capability = capability
        self.domain = domain
        self.legacy_allowed = legacy_allowed
        self.scoped_allowed = scoped_allowed
        self.scoped_reason = scoped_reason

    def as_dict(self) -> Dict[str, object]:
        return {
            "capability": self.capability,
            "domain": self.domain,
            "legacy_allowed": self.legacy_allowed,
            "scoped_allowed": self.scoped_allowed,
            "scoped_reason": self.scoped_reason,
        }


class LegacyTrustBridge:
    """Unifies the legacy scalar and scoped grants under one decision.

    Parameters:
        authority: the scoped autonomy authority
        principal: owning principal
    """

    def __init__(
        self,
        authority: AutonomyAuthority,
        principal: Principal,
    ) -> None:
        self._authority = authority
        self._principal = principal
        self._disagreements: List[Disagreement] = []
        self._consulted = 0

    # ── Scoped decision ─────────────────────────────────────────────

    def scoped_decision(
        self,
        capability: str,
        domain: Optional[str] = None,
    ) -> Tuple[bool, str]:
        """What the scoped authority alone would decide."""
        mapped = CAPABILITY_DOMAINS.get(capability)
        if mapped is None and domain is None:
            return False, (
                f"'{capability}' has no scoped mapping — un-migrated "
                f"capability, no grant can authorise it"
            )

        scoped_capability, scoped_domain = (
            mapped if mapped is not None else (capability, domain)
        )
        if domain is not None:
            scoped_domain = domain

        grants = self._authority.active_grants(
            capability=scoped_capability, domain=scoped_domain
        )
        if not grants:
            return False, (
                f"no active grant for '{scoped_capability}' in "
                f"'{scoped_domain}' (effective level A0_NONE)"
            )

        level = max(g.level for g in grants)
        return True, f"granted at {level.name}"

    # ── Combined decision ───────────────────────────────────────────

    def gate(
        self,
        capability: str,
        legacy_allowed: bool,
        legacy_reason: str,
        domain: Optional[str] = None,
    ) -> Tuple[bool, str]:
        """Combine the legacy verdict with the scoped one.

        The legacy scalar can only subtract.  In enforcing mode a
        capability needs *both* to permit it; in advisory mode the legacy
        verdict stands and the disagreement is recorded.
        """
        self._consulted += 1
        scoped_allowed, scoped_reason = self.scoped_decision(capability, domain)

        if scoped_allowed != legacy_allowed:
            mapped = CAPABILITY_DOMAINS.get(capability)
            self._disagreements.append(Disagreement(
                capability=capability,
                domain=(domain or (mapped[1] if mapped else "unmapped")),
                legacy_allowed=legacy_allowed,
                scoped_allowed=scoped_allowed,
                scoped_reason=scoped_reason,
            ))

        if not enforcing():
            if scoped_allowed != legacy_allowed:
                logger.info(
                    "AUTONOMY SHADOW: '%s' legacy=%s scoped=%s (%s) — "
                    "legacy stands (advisory mode)",
                    capability, legacy_allowed, scoped_allowed, scoped_reason,
                )
            return legacy_allowed, legacy_reason

        # Enforcing: both must permit.  Legacy subtracts only.
        if not scoped_allowed:
            return False, f"scoped authority denies: {scoped_reason}"
        if not legacy_allowed:
            return False, f"legacy gate denies: {legacy_reason}"
        return True, f"permitted ({scoped_reason})"

    # ── Migration telemetry ─────────────────────────────────────────

    @property
    def disagreements(self) -> List[Disagreement]:
        return list(self._disagreements)

    @property
    def consulted(self) -> int:
        return self._consulted

    def migration_report(self) -> Dict[str, object]:
        """Evidence for whether enforcing mode is safe to switch on."""
        would_break = [
            d for d in self._disagreements
            if d.legacy_allowed and not d.scoped_allowed
        ]
        would_tighten = [
            d for d in self._disagreements
            if not d.legacy_allowed and d.scoped_allowed
        ]
        return {
            "mode": "enforcing" if enforcing() else "advisory",
            "consulted": self._consulted,
            "disagreements": len(self._disagreements),
            "would_be_denied_under_enforcement": len(would_break),
            "legacy_stricter_than_scoped": len(would_tighten),
            "unmapped_capabilities": sorted({
                d.capability for d in self._disagreements
                if d.capability not in CAPABILITY_DOMAINS
            }),
            "ready_to_enforce": not would_break and self._consulted > 0,
        }
