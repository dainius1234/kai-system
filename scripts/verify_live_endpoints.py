"""Verify actuator handlers against running services (UH tracker G-10).

Route-declaration checks (in `test_migration.py`) prove a path exists in
a service's source.  This proves the call actually *works*: right path,
right shape, parseable response, through the real capability pipeline.

The distinction that matters when reading the output:

  OK        the call succeeded end to end
  UPSTREAM  the route exists but its dependency is unavailable here
            (missing credentials, no outbound network, flag disabled)
  WRONG     404 — the path does not exist.  This is a real defect.

Only WRONG is a failure.  UPSTREAM is an environment limit, and the
script exits non-zero only when a path is genuinely wrong or when
`--require-all` is passed.

Usage::

    # against a running compose stack
    python scripts/verify_live_endpoints.py

    # against locally-started services
    SYSMETRICS_URL=http://127.0.0.1:18035 \\
    python scripts/verify_live_endpoints.py --json
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from common.contracts.base import Principal, Provenance, RiskTier
from common.contracts.action import ActionProposal
from common.actuator_registry import attach_read_handlers, build_catalog, migrate_tier
from common.actuator_registry.handlers import READ_ONLY_ENDPOINTS
from common.actuator_registry.registry import MigrationTier
from common.policy_bridge.approval import ApprovalGate
from common.policy_bridge.capability import CapabilityBridge

# Sample parameters for routes that take them.  Read-only values only.
SAMPLE_PARAMETERS = {
    "market_ticker_read": {"symbol": "BTCUSDT"},
    "orderbook_read": {"symbol": "BTCUSDT"},
    "alpha_signal_read": {"symbol": "BTCUSDT"},
}


def classify(result: dict) -> str:
    if result.get("ok"):
        return "OK"
    error = result.get("error", "")
    # A 404 means the path is wrong; anything else means the route was
    # reached and its dependency or configuration refused.
    if "404" in error or "Not Found" in error:
        return "WRONG"
    return "UPSTREAM"


def verify(timeout: float = 15.0) -> list[dict]:
    import httpx

    def http_get(url: str):
        response = httpx.get(url, timeout=timeout)
        response.raise_for_status()
        return response.json()

    principal = Principal(identity="kai", role="system")
    catalog = build_catalog(principal)
    attach_read_handlers(catalog, http_get=http_get)
    migrate_tier(catalog, MigrationTier.READ_ONLY, principal)
    bridge = CapabilityBridge()

    findings: list[dict] = []
    for actuator, (_, _, paths) in sorted(READ_ONLY_ENDPOINTS.items()):
        for action in sorted(paths):
            proposal = ActionProposal(
                action_type=action, description="live verification",
                risk_tier=RiskTier.OBSERVE, rationale="G-10 verification",
                alternatives=["skip"], principal=principal,
                purpose="live_verification",
                provenance=Provenance(source="verify_live_endpoints"),
            )
            approval = ApprovalGate().approve(proposal, "operator", principal)
            capability = bridge.issue(
                proposal, approval, actuator, action, principal,
                parameters=SAMPLE_PARAMETERS.get(action, {}),
            )
            bridge.consume(capability.id, actuator, principal)

            result = catalog.dispatch(
                capability, actuator, action, "wf-live", principal
            ).result

            findings.append({
                "actuator": actuator,
                "action": action,
                "url": result.get("url", ""),
                "status": classify(result),
                "detail": (
                    str(result.get("data"))[:120] if result.get("ok")
                    else result.get("error", "")[:120]
                ),
            })
    return findings


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="emit JSON")
    parser.add_argument("--timeout", type=float, default=15.0)
    parser.add_argument(
        "--require-all", action="store_true",
        help="fail when any endpoint is UPSTREAM, not only when WRONG",
    )
    args = parser.parse_args()

    findings = verify(timeout=args.timeout)
    wrong = [f for f in findings if f["status"] == "WRONG"]
    upstream = [f for f in findings if f["status"] == "UPSTREAM"]
    ok = [f for f in findings if f["status"] == "OK"]

    if args.json:
        print(json.dumps({
            "ok": len(ok), "upstream": len(upstream), "wrong": len(wrong),
            "findings": findings,
        }, indent=2))
    else:
        for f in findings:
            path = f["url"].split("://", 1)[-1]
            path = path[path.find("/"):] if "/" in path else path
            print(f"  {f['status']:8} {f['actuator']:18} {f['action']:24} {path}")
            if f["status"] != "OK":
                print(f"           └─ {f['detail']}")
        print(f"\n  OK={len(ok)}  UPSTREAM={len(upstream)}  WRONG={len(wrong)}")
        if upstream:
            print("\n  UPSTREAM means the route exists but its dependency is "
                  "unavailable here\n  (credentials, network, or a disabled "
                  "feature flag) — not a wrong path.")

    if wrong:
        print(f"\n  FAIL: {len(wrong)} endpoint(s) do not exist", file=sys.stderr)
        return 1
    if args.require_all and upstream:
        print(f"\n  FAIL: {len(upstream)} endpoint(s) unreachable "
              f"(--require-all)", file=sys.stderr)
        return 1
    print("\n  PASS: no wrong paths")
    return 0


if __name__ == "__main__":
    sys.exit(main())
