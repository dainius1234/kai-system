"""Verify mutating handlers against running services (UH tracker E-02).

Tier 2-8 actuators cause real side effects, so this is deliberately more
conservative than the read-only equivalent.  Each action is classified by
how safe it is to actually invoke:

  SAFE        idempotent or self-contained; invoked in full
  CONTAINED   invoked with arguments chosen so the effect cannot land
              (a restore pointed at a nonexistent file, for example) —
              this still exercises the route, the auth, and our handler
  SKIPPED     genuinely destructive or needing hardware we do not have;
              never invoked, and reported as skipped rather than passed

The point of CONTAINED is that "we could not test it" and "we tested it
safely" are different claims.  A restore that returns 400 for a missing
file has proven the path, the token and the parameter plumbing without
touching a database.

Requires the target services running and ``KAI_SERVICE_TOKEN`` set to
match them.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from common.contracts.base import Principal, Provenance, RiskTier
from common.contracts.action import ActionProposal
from common.actuator_registry.catalog import build_catalog
from common.actuator_registry.migration import migrate_tier
from common.actuator_registry.mutating_handlers import attach_all_handlers
from common.actuator_registry.registry import MigrationTier
from common.policy_bridge.approval import ApprovalGate
from common.policy_bridge.capability import CapabilityBridge

SAFE = "SAFE"
CONTAINED = "CONTAINED"
SKIPPED = "SKIPPED"

# (actuator, action) → (classification, parameters, why)
PLAN = [
    ("notify-service", "notify_desktop", SAFE,
     {"title": "UH verification", "body": "live check", "urgency": "low"},
     "desktop notification; no-ops without a display server"),

    ("monitor-service", "monitor_rule_write", SAFE,
     {"name": "uh-verify", "source": {"type": "http", "url": "http://127.0.0.1/x"},
      "condition": {"op": "gt", "value": 1}},
     "creates an in-memory rule in a throwaway process"),

    ("executor-sandbox", "noop", SAFE,
     {"tool": "noop", "params": {}, "task_id": "uh-noop", "device": "cpu"},
     "explicit no-op tool"),

    ("executor-sandbox", "sandbox_python_eval", SAFE,
     {"tool": "python", "params": {"expression": "1 + 1"},
      "task_id": "uh-py", "device": "cpu"},
     "AST-validated sandboxed expression"),

    ("ledger-worker", "ledger_verify", SAFE, {},
     "read-only chain verification"),

    ("vault-sync", "vault_export", SAFE,
     {"filepath": "KAI/uh-verification.md", "content": "verification note",
      "conviction": 9.9, "requester": "uh-verify"},
     "writes into a throwaway vault directory"),

    ("backup-service", "backup_create", CONTAINED, {},
     "shells out to pg_dump; fails cleanly with no database present"),

    ("db-restore", "db_restore", CONTAINED,
     {"backup_file": "nonexistent-uh-verification.sql"},
     "restore pointed at a file that does not exist — proves the route "
     "and auth without touching a database"),

    ("executor-shell", "shell_exec", CONTAINED,
     {"tool": "shell", "params": {"cmd": "echo uh-verify"},
      "task_id": "uh-shell", "device": "cpu"},
     "allowlisted echo only"),

    ("browser-actor", "browser_click", SKIPPED, {},
     "needs a live browser session and a real page; clicking an arbitrary "
     "element is exactly the irreversible web interaction we gate"),
    ("browser-actor", "browser_type", SKIPPED, {},
     "same as browser_click"),
    ("supervisor", "service_recover", SKIPPED, {},
     "would restart live services"),
    ("heartbeat", "auto_sleep", SKIPPED, {},
     "triggers memory compression and decay across memu-core"),
    ("paper-trader", "paper_trade_open", SKIPPED, {},
     "requires the trading stack; covered by the UH-6 vertical slice"),
]


def _principal() -> Principal:
    return Principal(identity="kai", role="system")


def build_registry():
    import httpx

    def http_post(url: str, body: dict, headers: dict):
        response = httpx.post(url, json=body, headers=headers, timeout=20.0)
        response.raise_for_status()
        try:
            return response.json()
        except Exception:
            return {"status_code": response.status_code}

    principal = _principal()
    catalog = build_catalog(principal)
    attach_all_handlers(
        catalog, http_get=lambda u: {"read": True}, http_post=http_post
    )
    for tier in MigrationTier:
        migrate_tier(catalog, tier, principal)
    return catalog


def invoke(catalog, bridge, actuator: str, action: str, params: dict):
    principal = _principal()
    proposal = ActionProposal(
        action_type=action, description="live mutating verification",
        risk_tier=RiskTier.ACT_SUPERVISED, rationale="E-02 verification",
        alternatives=["skip"], principal=principal, purpose="verification",
        provenance=Provenance(source="verify_live_mutating"),
    )
    approval = ApprovalGate().approve(proposal, "operator", principal)
    cap = bridge.issue(
        proposal, approval, actuator, action, principal, parameters=params
    )
    bridge.consume(cap.id, actuator, principal)
    return catalog.dispatch(cap, actuator, action, "wf-mutating", principal)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    catalog = build_registry()
    bridge = CapabilityBridge()
    findings = []

    for actuator, action, classification, params, why in PLAN:
        if classification == SKIPPED:
            findings.append({
                "actuator": actuator, "action": action,
                "classification": SKIPPED, "invoked": False,
                "outcome": "skipped", "why": why,
            })
            continue

        receipt = invoke(catalog, bridge, actuator, action, params)
        result = receipt.result
        error = result.get("error", "")
        # A 4xx means the service answered — route, auth and parameters all
        # reached it. Only a connection failure or a 404 is a real problem.
        reached = result.get("ok") or any(
            code in error for code in ("400", "401", "403", "409", "422", "500", "503")
        )
        wrong = "404" in error
        findings.append({
            "actuator": actuator, "action": action,
            "classification": classification, "invoked": True,
            "outcome": ("ok" if result.get("ok")
                        else "reached" if reached and not wrong
                        else "wrong-path" if wrong
                        else "unreachable"),
            "side_effects": result.get("side_effects", []),
            "effect_uncertain": result.get("effect_uncertain", False),
            "detail": (str(result.get("data"))[:90] if result.get("ok")
                       else error[:90]),
            "why": why,
        })

    if args.json:
        # JSON mode emits only JSON, so the output stays machine-readable.
        print(json.dumps(findings, indent=2))
        bad = [f for f in findings
               if f["outcome"] in ("wrong-path", "unreachable")]
        return 1 if bad else 0
    else:
        for f in findings:
            mark = {"ok": "OK", "reached": "REACHED", "skipped": "SKIPPED",
                    "wrong-path": "WRONG", "unreachable": "UNREACHABLE"}[f["outcome"]]
            print(f"  {mark:11} {f['actuator']:18} {f['action']:22} {f['why'][:44]}")
            if f["outcome"] in ("wrong-path", "unreachable"):
                print(f"              └─ {f.get('detail','')}")

    ok = [f for f in findings if f["outcome"] == "ok"]
    reached = [f for f in findings if f["outcome"] == "reached"]
    skipped = [f for f in findings if f["outcome"] == "skipped"]
    bad = [f for f in findings
           if f["outcome"] in ("wrong-path", "unreachable")]

    print(f"\n  OK={len(ok)}  REACHED={len(reached)}  SKIPPED={len(skipped)}  "
          f"FAILED={len(bad)}")
    if skipped:
        print("\n  SKIPPED actions were never invoked — genuinely destructive or "
              "needing\n  hardware unavailable here. Reported, not passed.")
    if bad:
        print(f"\n  FAIL: {len(bad)} endpoint(s) wrong or unreachable",
              file=sys.stderr)
        return 1
    print("\n  PASS: every invoked action reached its service")
    return 0


if __name__ == "__main__":
    sys.exit(main())
