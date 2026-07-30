"""UH-3: Scoped world state — deterministic reducers, immutable snapshots.

Components:
  - reducers:       pure functions converting PerceptionEvents to Claims/Evidence
  - snapshot_store: immutable snapshot production with retention, conflict
                    detection, supersession, and principal-scoped views
"""

from common.world_state.reducers import (
    REDUCER_REVISION,
    ReducerOutput,
    ReducerRegistry,
)
from common.world_state.snapshot_store import SnapshotStore

__all__ = [
    "REDUCER_REVISION",
    "ReducerOutput",
    "ReducerRegistry",
    "SnapshotStore",
]
