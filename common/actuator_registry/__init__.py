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
from common.actuator_registry.handlers import (
    READ_ONLY_ENDPOINTS,
    HandlerError,
    attach_read_handlers,
    build_read_handler,
)
from common.actuator_registry.migration import (
    MigrationError,
    MigrationResult,
    migrate_tier,
    migration_plan,
)

__all__ = [
    "ActuatorDispatchError",
    "ActuatorRegistration",
    "ActuatorRegistry",
    "MigrationStatus",
    "MigrationTier",
    "ALL_ACTUATORS",
    "build_catalog",
    "READ_ONLY_ENDPOINTS",
    "HandlerError",
    "attach_read_handlers",
    "build_read_handler",
    "MigrationError",
    "MigrationResult",
    "migrate_tier",
    "migration_plan",
]
