"""Tier migration driver — advances actuators in ascending risk order.

Closes the driver half of UH tracker gap G-01.  Advancing an actuator by
hand through four states across 33 entries is where mistakes get made, so
this encodes the roadmap's ordering rules once:

  - a tier may not start until every lower tier is fully migrated;
  - an actuator with a live legacy path cannot reach VERIFIED;
  - an actuator with no dispatch handler cannot reach ACTIVE, because
    "migrated" would then mean nothing actually routes through it.

The last rule is the one that keeps this honest.  The registry alone
would happily mark a handler-less actuator ACTIVE; this driver refuses,
so a green migration report means traffic can genuinely flow.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from common.contracts.base import Principal
from common.actuator_registry.registry import (
    ActuatorDispatchError,
    ActuatorRegistry,
    MigrationStatus,
    MigrationTier,
)


class MigrationError(Exception):
    pass


class MigrationResult:
    __slots__ = ("tier", "migrated", "skipped", "blocked", "errors")

    def __init__(self, tier: MigrationTier) -> None:
        self.tier = tier
        self.migrated: List[str] = []
        self.skipped: List[str] = []
        self.blocked: List[str] = []
        self.errors: Dict[str, str] = {}

    @property
    def ok(self) -> bool:
        return not self.blocked and not self.errors

    def as_dict(self) -> Dict[str, Any]:
        return {
            "tier": self.tier.value,
            "migrated": sorted(self.migrated),
            "skipped": sorted(self.skipped),
            "blocked": sorted(self.blocked),
            "errors": dict(self.errors),
            "ok": self.ok,
        }


def lower_tiers_complete(
    registry: ActuatorRegistry, tier: MigrationTier
) -> List[str]:
    """Identities in lower tiers that are not yet migrated."""
    outstanding: List[str] = []
    for entry in registry.list_actuators():
        if entry.migration_tier.value >= tier.value:
            continue
        if entry.migration_status not in (
            MigrationStatus.VERIFIED, MigrationStatus.ACTIVE,
            MigrationStatus.DISABLED,
        ):
            outstanding.append(entry.identity)
    return sorted(outstanding)


def migrate_tier(
    registry: ActuatorRegistry,
    tier: MigrationTier,
    principal: Principal,
    require_handler: bool = True,
    activate: bool = True,
) -> MigrationResult:
    """Advance every actuator in one tier.

    Parameters:
        require_handler: refuse to activate an actuator with no dispatch
            handler.  Leaving this on is what makes a migrated actuator
            mean something.
        activate: advance to ACTIVE.  Set False to stop at VERIFIED when
            a supervised soak is wanted first.
    """
    result = MigrationResult(tier)

    outstanding = lower_tiers_complete(registry, tier)
    if outstanding:
        raise MigrationError(
            f"cannot migrate tier {tier.value}: lower tiers still have "
            f"un-migrated actuators: {', '.join(outstanding)}"
        )

    for entry in registry.list_actuators(migration_tier=tier):
        identity = entry.identity

        if entry.migration_status in (
            MigrationStatus.ACTIVE, MigrationStatus.DISABLED
        ):
            result.skipped.append(identity)
            continue

        if require_handler and registry.handler_for(identity) is None:
            result.blocked.append(identity)
            result.errors[identity] = "no dispatch handler attached"
            continue

        try:
            if entry.migration_status == MigrationStatus.LEGACY:
                registry.advance_migration(
                    identity, MigrationStatus.MIGRATING, principal
                )

            if not entry.legacy_disabled:
                registry.disable_legacy_path(identity, principal)

            if entry.migration_status == MigrationStatus.MIGRATING:
                registry.advance_migration(
                    identity, MigrationStatus.VERIFIED, principal
                )

            if activate and entry.migration_status == MigrationStatus.VERIFIED:
                registry.advance_migration(
                    identity, MigrationStatus.ACTIVE, principal
                )

            result.migrated.append(identity)
        except ActuatorDispatchError as exc:
            result.blocked.append(identity)
            result.errors[identity] = str(exc)

    return result


def migration_plan(registry: ActuatorRegistry) -> Dict[str, Any]:
    """What can be migrated now, and what is waiting on what."""
    candidates = registry.next_migration_candidates()
    if not candidates:
        return {
            "complete": True,
            "next_tier": None,
            "candidates": [],
            "blocked_on": [],
        }

    tier = candidates[0].migration_tier
    return {
        "complete": False,
        "next_tier": tier.value,
        "candidates": sorted(c.identity for c in candidates),
        "blocked_on": lower_tiers_complete(registry, tier),
        "missing_handlers": sorted(
            c.identity for c in candidates
            if registry.handler_for(c.identity) is None
        ),
    }
