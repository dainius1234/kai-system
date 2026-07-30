"""UH-5: Policy, human approval and capability bridge.

Components:
  - policy_engine:  risk classification and policy-as-code evaluation
  - approval:       digest-bound, single-use, nonce-protected approval gate
  - capability:     audience-bound, single-use capability tokens with
                    revocation and expiry
"""

from common.policy_bridge.policy_engine import (
    POLICY_VERSION,
    PolicyEngine,
    PolicyEvaluation,
)
from common.policy_bridge.approval import ApprovalError, ApprovalGate
from common.policy_bridge.capability import CapabilityBridge, CapabilityError

__all__ = [
    "POLICY_VERSION",
    "PolicyEngine",
    "PolicyEvaluation",
    "ApprovalError",
    "ApprovalGate",
    "CapabilityBridge",
    "CapabilityError",
]
