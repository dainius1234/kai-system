"""Bidder registry — authenticated proposal specialists.

Bidders are the only entities that may submit proposals.  Each bidder:

  - has a unique identity and declared independence group
  - declares its expertise domain
  - is validated at registration (duplicate/stub bidders rejected)
  - cannot issue capabilities or trigger execution

The registry enforces that:
  - duplicate identities are rejected
  - correlated bidders (same independence group) are tracked
  - stub bidders (no expertise) are rejected
  - a minimum quorum of independent bidders is required for consensus
"""
from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Dict, List, Optional, Set

from pydantic import BaseModel, Field


class BidderStatus(str, Enum):
    ACTIVE = "active"
    SUSPENDED = "suspended"
    REVOKED = "revoked"


class BidderRegistration(BaseModel):
    model_config = {"extra": "forbid"}

    identity: str
    display_name: str
    expertise_domain: str
    independence_group: str
    registered_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    status: BidderStatus = BidderStatus.ACTIVE


class BidderRegistry:
    """Registry of authenticated proposal bidders.

    Parameters:
        min_independent_groups: minimum number of distinct independence
            groups required for a qualifying consensus.
    """

    def __init__(self, min_independent_groups: int = 2) -> None:
        self._bidders: Dict[str, BidderRegistration] = {}
        self._min_independent = min_independent_groups

    def register(self, registration: BidderRegistration) -> str:
        if registration.identity in self._bidders:
            raise ValueError(
                f"duplicate bidder identity: {registration.identity}"
            )
        if not registration.expertise_domain.strip():
            raise ValueError("stub bidder rejected: empty expertise_domain")
        if not registration.independence_group.strip():
            raise ValueError("stub bidder rejected: empty independence_group")

        self._bidders[registration.identity] = registration
        return registration.identity

    def get(self, identity: str) -> Optional[BidderRegistration]:
        return self._bidders.get(identity)

    def is_registered(self, identity: str) -> bool:
        b = self._bidders.get(identity)
        return b is not None and b.status == BidderStatus.ACTIVE

    def active_bidders(self) -> List[BidderRegistration]:
        return [
            b for b in self._bidders.values()
            if b.status == BidderStatus.ACTIVE
        ]

    def independence_groups(self) -> Set[str]:
        return {
            b.independence_group for b in self._bidders.values()
            if b.status == BidderStatus.ACTIVE
        }

    def has_qualifying_diversity(self) -> bool:
        return len(self.independence_groups()) >= self._min_independent

    def correlated_bidders(self, identity: str) -> List[str]:
        bidder = self._bidders.get(identity)
        if bidder is None:
            return []
        return [
            b.identity for b in self._bidders.values()
            if b.independence_group == bidder.independence_group
            and b.identity != identity
            and b.status == BidderStatus.ACTIVE
        ]

    def suspend(self, identity: str) -> None:
        if identity in self._bidders:
            self._bidders[identity].status = BidderStatus.SUSPENDED

    def revoke(self, identity: str) -> None:
        if identity in self._bidders:
            self._bidders[identity].status = BidderStatus.REVOKED

    def count(self) -> int:
        return len(self.active_bidders())
