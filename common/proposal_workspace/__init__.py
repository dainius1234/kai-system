"""UH-4: Proposal-only workspace — deliberation without execution.

Components:
  - bidder:    registered authenticated bidder registry with independence
               group tracking and duplicate/stub rejection
  - workspace: proposal intake, evidence/contradiction checking, deterministic
               envelope production — no capability issuance, no execution
"""

from common.proposal_workspace.bidder import (
    BidderRegistration,
    BidderRegistry,
    BidderStatus,
)
from common.proposal_workspace.workspace import (
    EvidenceGap,
    ProposalEnvelope,
    ProposalSubmission,
    ProposalWorkspace,
    WorkspaceStatus,
)

__all__ = [
    "BidderRegistration",
    "BidderRegistry",
    "BidderStatus",
    "EvidenceGap",
    "ProposalEnvelope",
    "ProposalSubmission",
    "ProposalWorkspace",
    "WorkspaceStatus",
]
