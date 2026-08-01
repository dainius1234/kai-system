"""UH-7: Actuator registry and risk-ordered migration.

Components:
  - registry: capability-gated actuator dispatch with migration tracking
  - catalog:  the full inventory of side-effecting surfaces, tiered by risk
"""

from common.actuator_registry.registry import (
    ActuatorDispatchError,
    ActuatorRegistration,
    ActuatorRegistry,
    MigrationStatus,
    MigrationTier,
)
from common.actuator_registry.catalog import ALL_ACTUATORS, build_catalog

__all__ = [
    "ActuatorDispatchError",
    "ActuatorRegistration",
    "ActuatorRegistry",
    "MigrationStatus",
    "MigrationTier",
    "ALL_ACTUATORS",
    "build_catalog",
]
