"""UH-2: Perception spine — validated event ingress in shadow mode.

Components:
  - journal:   append-only durable event log with replay support
  - ingress:   validated intake with dedup, staleness, principal isolation
  - adapters:  convert raw sensor HTTP responses to PerceptionEvents
  - shadow:    background runner that polls sensors without side effects
"""

from common.perception_spine.journal import EventJournal, JournalEntry
from common.perception_spine.ingress import (
    IngressResult,
    IngressVerdict,
    PerceptionIngress,
)
from common.perception_spine.adapters import ADAPTER_REGISTRY
from common.perception_spine.shadow import ShadowPerceptionRunner

__all__ = [
    "EventJournal",
    "JournalEntry",
    "IngressResult",
    "IngressVerdict",
    "PerceptionIngress",
    "ADAPTER_REGISTRY",
    "ShadowPerceptionRunner",
]
