"""Actuator registry — typed, capability-gated actuator dispatch.

Each actuator is:
  - registered with identity, risk tier, and migration status
  - capability-gated (no execution without a valid consumed capability)
  - audited (every dispatch produces an ActuatorReceipt)
  - migration-tracked (legacy → migrating → verified → active)

The registry enforces the UH-7 migration ordering:
  1. read-only data retrieval
  2. isolated local/test operations
  3. document and browser reads
  4. notifications/draft creation
  5. file mutations
  6. calendar/external messages
  7. recovery/admin operations
  8. financial/destructive/public/self-modifying operations last
"""
from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set

from common.contracts.base import (
    Principal,
    Provenance,
    RiskTier,
)
from common.contracts.action import (
    ActionCapability,
    ActuatorReceipt,
)


class MigrationStatus(str, Enum):
    LEGACY = "legacy"
    MIGRATING = "migrating"
    VERIFIED = "verified"
    ACTIVE = "active"
    DISABLED = "disabled"


class MigrationTier(int, Enum):
    READ_ONLY = 1
    LOCAL_TEST = 2
    DOCUMENT_READ = 3
    NOTIFICATION = 4
    FILE_MUTATION = 5
    EXTERNAL_MESSAGE = 6
    RECOVERY_ADMIN = 7
    FINANCIAL_DESTRUCTIVE = 8


class ActuatorRegistration:
    __slots__ = (
        "identity", "display_name", "description",
        "risk_tier", "migration_tier", "migration_status",
        "action_types", "reversible",
        "legacy_path", "legacy_disabled", "legacy_disabled_at",
        "registered_at",
    )

    def __init__(
        self,
        identity: str,
        display_name: str,
        description: str,
        risk_tier: RiskTier,
        migration_tier: MigrationTier,
        action_types: List[str],
        reversible: bool = False,
        legacy_path: Optional[str] = None,
    ) -> None:
        if not identity or not identity.strip():
            raise ValueError("actuator identity must not be empty")
        if not action_types:
            raise ValueError("actuator must support at least one action_type")

        self.identity = identity
        self.display_name = display_name
        self.description = description
        self.risk_tier = risk_tier
        self.migration_tier = migration_tier
        self.migration_status = MigrationStatus.LEGACY
        self.action_types = list(action_types)
        self.reversible = reversible
        self.legacy_path = legacy_path
        self.legacy_disabled = legacy_path is None
        self.legacy_disabled_at: Optional[datetime] = None
        self.registered_at = datetime.now(timezone.utc)


class ActuatorDispatchError(Exception):
    pass


class ActuatorRegistry:
    """Registry of all actuators with capability-gated dispatch.

    No actuator can execute without:
      1. Being registered in the registry
      2. Having migration_status in (VERIFIED, ACTIVE)
      3. Receiving a valid, consumed ActionCapability targeting this actuator
    """

    def __init__(self, principal: Principal) -> None:
        self._principal = principal
        self._actuators: Dict[str, ActuatorRegistration] = {}
        self._action_type_map: Dict[str, str] = {}
        self._handlers: Dict[str, Callable[..., Dict[str, Any]]] = {}
        self._dispatch_log: List[ActuatorReceipt] = []

    def register(
        self,
        registration: ActuatorRegistration,
        handler: Optional[Callable[..., Dict[str, Any]]] = None,
    ) -> None:
        if registration.identity in self._actuators:
            raise ActuatorDispatchError(
                f"actuator already registered: {registration.identity}"
            )

        for action_type in registration.action_types:
            if action_type in self._action_type_map:
                existing = self._action_type_map[action_type]
                raise ActuatorDispatchError(
                    f"action_type '{action_type}' already claimed by '{existing}'"
                )

        self._actuators[registration.identity] = registration
        for action_type in registration.action_types:
            self._action_type_map[action_type] = registration.identity

        if handler is not None:
            self._handlers[registration.identity] = handler

    def set_handler(
        self,
        actuator_identity: str,
        handler: Callable[..., Dict[str, Any]],
    ) -> None:
        if actuator_identity not in self._actuators:
            raise ActuatorDispatchError(
                f"actuator not registered: {actuator_identity}"
            )
        self._handlers[actuator_identity] = handler

    def advance_migration(
        self,
        actuator_identity: str,
        to_status: MigrationStatus,
        principal: Principal,
    ) -> None:
        reg = self._actuators.get(actuator_identity)
        if reg is None:
            raise ActuatorDispatchError(
                f"actuator not registered: {actuator_identity}"
            )

        valid_transitions = {
            MigrationStatus.LEGACY: {MigrationStatus.MIGRATING, MigrationStatus.DISABLED},
            MigrationStatus.MIGRATING: {MigrationStatus.VERIFIED, MigrationStatus.DISABLED},
            MigrationStatus.VERIFIED: {MigrationStatus.ACTIVE, MigrationStatus.DISABLED},
            MigrationStatus.ACTIVE: {MigrationStatus.DISABLED},
            MigrationStatus.DISABLED: set(),
        }

        allowed = valid_transitions.get(reg.migration_status, set())
        if to_status not in allowed:
            raise ActuatorDispatchError(
                f"invalid migration transition: {reg.migration_status.value} → {to_status.value}"
            )

        # UH-7 invariant: the old path is disabled before the new path is
        # marked verified.  Retaining both paths is an explicitly rejected
        # anti-pattern.
        if to_status == MigrationStatus.VERIFIED and not reg.legacy_disabled:
            raise ActuatorDispatchError(
                f"cannot verify '{actuator_identity}': legacy path "
                f"'{reg.legacy_path}' is still enabled"
            )

        reg.migration_status = to_status

    def disable_legacy_path(
        self,
        actuator_identity: str,
        principal: Principal,
    ) -> None:
        """Disable an actuator's legacy direct path.

        Must happen before the actuator can be marked VERIFIED.
        """
        reg = self._actuators.get(actuator_identity)
        if reg is None:
            raise ActuatorDispatchError(
                f"actuator not registered: {actuator_identity}"
            )

        reg.legacy_disabled = True
        reg.legacy_disabled_at = datetime.now(timezone.utc)

    def dispatch(
        self,
        capability: ActionCapability,
        actuator_identity: str,
        action_type: str,
        workflow_id: str,
        principal: Principal,
    ) -> ActuatorReceipt:
        reg = self._actuators.get(actuator_identity)
        if reg is None:
            raise ActuatorDispatchError(
                f"actuator not registered: {actuator_identity}"
            )

        if reg.migration_status not in (MigrationStatus.VERIFIED, MigrationStatus.ACTIVE):
            raise ActuatorDispatchError(
                f"actuator '{actuator_identity}' not ready: status={reg.migration_status.value}"
            )

        if action_type not in reg.action_types:
            raise ActuatorDispatchError(
                f"actuator '{actuator_identity}' does not support action_type '{action_type}'"
            )

        if not capability.used:
            raise ActuatorDispatchError(
                "capability must be consumed before dispatch"
            )

        expected_actuator = capability.provenance.source.split(":", 1)[-1]
        if expected_actuator != actuator_identity:
            raise ActuatorDispatchError(
                f"capability audience mismatch: for '{expected_actuator}', not '{actuator_identity}'"
            )

        if capability.capability_type != action_type:
            raise ActuatorDispatchError(
                f"capability type mismatch: '{capability.capability_type}' vs '{action_type}'"
            )

        handler = self._handlers.get(actuator_identity)
        now = datetime.now(timezone.utc)

        if handler is not None:
            result = handler(capability.parameters, action_type)
        else:
            result = {
                "action_type": action_type,
                "parameters": capability.parameters,
                "simulated": True,
            }

        receipt = ActuatorReceipt(
            capability_id=capability.id,
            workflow_id=workflow_id,
            actuator=actuator_identity,
            action_taken=action_type,
            result=result,
            side_effects=[],
            reversible=reg.reversible,
            executed_at=now,
            principal=principal,
            purpose="actuator_dispatch",
            provenance=Provenance(
                source=f"actuator_registry:{actuator_identity}",
                upstream_ids=[capability.id, workflow_id],
            ),
        )

        self._dispatch_log.append(receipt)
        return receipt

    def get(self, actuator_identity: str) -> Optional[ActuatorRegistration]:
        return self._actuators.get(actuator_identity)

    def get_by_action_type(self, action_type: str) -> Optional[ActuatorRegistration]:
        identity = self._action_type_map.get(action_type)
        if identity is None:
            return None
        return self._actuators.get(identity)

    def list_actuators(
        self,
        migration_status: Optional[MigrationStatus] = None,
        migration_tier: Optional[MigrationTier] = None,
    ) -> List[ActuatorRegistration]:
        results = list(self._actuators.values())
        if migration_status is not None:
            results = [r for r in results if r.migration_status == migration_status]
        if migration_tier is not None:
            results = [r for r in results if r.migration_tier == migration_tier]
        return sorted(results, key=lambda r: (r.migration_tier.value, r.identity))

    def migration_report(self) -> Dict[str, Any]:
        by_tier: Dict[int, List[Dict[str, str]]] = {}
        for reg in sorted(self._actuators.values(), key=lambda r: (r.migration_tier.value, r.identity)):
            tier = reg.migration_tier.value
            if tier not in by_tier:
                by_tier[tier] = []
            by_tier[tier].append({
                "identity": reg.identity,
                "display_name": reg.display_name,
                "status": reg.migration_status.value,
                "risk_tier": reg.risk_tier.value,
                "legacy_path": reg.legacy_path,
                "legacy_disabled": reg.legacy_disabled,
            })

        total = len(self._actuators)
        migrated = sum(
            1 for r in self._actuators.values()
            if r.migration_status in (MigrationStatus.VERIFIED, MigrationStatus.ACTIVE)
        )
        legacy_open = sorted(
            r.identity for r in self._actuators.values()
            if not r.legacy_disabled
        )

        return {
            "total_actuators": total,
            "migrated": migrated,
            "remaining": total - migrated,
            "legacy_paths_open": legacy_open,
            "by_tier": by_tier,
        }

    def next_migration_candidates(self) -> List[ActuatorRegistration]:
        """Actuators eligible to migrate next, in roadmap risk order.

        UH-7 migrates by ascending risk: an actuator is only a candidate
        once every lower tier is fully migrated.
        """
        unmigrated = [
            r for r in self._actuators.values()
            if r.migration_status in (MigrationStatus.LEGACY, MigrationStatus.MIGRATING)
        ]
        if not unmigrated:
            return []

        lowest_tier = min(r.migration_tier.value for r in unmigrated)
        return sorted(
            (r for r in unmigrated if r.migration_tier.value == lowest_tier),
            key=lambda r: r.identity,
        )

    @property
    def dispatch_log(self) -> List[ActuatorReceipt]:
        return list(self._dispatch_log)
