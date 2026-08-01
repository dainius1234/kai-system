"""Subject-scoped erasure across every data layer.

Closes roadmap §16.30: end-to-end deletion across source events, views,
proposals, audit-allowed references and learning derivatives.
"""

from common.erasure.coordinator import (
    ErasureCoordinator,
    ErasureError,
    content_digest,
)
from common.erasure.handlers import build_full_coordinator

__all__ = [
    "ErasureCoordinator",
    "ErasureError",
    "content_digest",
    "build_full_coordinator",
]
