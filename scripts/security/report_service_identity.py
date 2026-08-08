#!/usr/bin/env python3
"""Where membership authentication is used, and where identity is needed.

The defect class
----------------

    SHARED AUTHENTICATION PROVES MEMBERSHIP, NOT IDENTITY.

`KAI_SERVICE_TOKEN` is one secret shared by every service in every
profile, and `common/service_auth.check_token()` returns
`(ok, status, detail)` — no principal. So a protected endpoint learns
*a caller holds the service token*. It cannot learn **which** caller,
and any holder can act as any other.

That is sufficient for some endpoints and structurally insufficient for
others, so this counts both rather than condemning all of them.

Classification
--------------

    A  membership is enough — the endpoint behaves identically whoever
       calls, and nothing downstream is attributed to the caller
    B  identity is material — authority, provenance, permissions, rate
       limits, source type, memory subject, ownership or audit
       attribution depend on WHO called

**The A/B split is a judgement, and it is declared here rather than
inferred**, because no static rule can read intent. `_CLASS` states the
verdict and the reason for every protected operation, so a reader can
disagree with a specific line instead of with a number. Anything not
listed is reported as **UNCLASSIFIED**, never silently assumed A —
absence must not read as safety.

Exit 0 always: this is a report.
"""
from __future__ import annotations

import ast
import sys
from pathlib import Path
from typing import Dict, List, Tuple

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO))

from scripts.security.gate_inputs import inspected  # noqa: E402

_MARKER = "require_service_auth"


def _protected_ops(text: str) -> List[Tuple[str, int]]:
    """Every real `require_service_auth("op")` CALL in a module.

    Parsed rather than grepped. The first version of this used a regex,
    and its very first run reported a tenth service — `common/db_restore`
    — which does not exist: it matched the *usage example inside
    `common/service_auth.py`'s own docstring*. A scope larger than
    reality reports failure over things that are right, and it did so
    here on an instrument built to measure exactly that.

    An unparseable module raises, so a syntax error surfaces as UNKNOWN
    rather than as an absence of endpoints.
    """
    out: List[Tuple[str, int]] = []
    for node in ast.walk(ast.parse(text)):
        if not isinstance(node, ast.Call):
            continue
        fn = node.func
        name = getattr(fn, "id", None) or getattr(fn, "attr", None)
        if name != _MARKER or not node.args:
            continue
        arg = node.args[0]
        if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
            out.append((arg.value, node.lineno))
    return out

#: operation -> (class, reason). Judgement, stated so it can be argued
#: with. B where the caller's identity changes what the system should
#: record or permit; A where it genuinely does not.
_CLASS: Dict[str, Tuple[str, str]] = {
    # ── B: identity is material ──────────────────────────────────────
    "cortex_observe_turn": ("B", "the turn's provenance IS the caller; "
                                 "R3 requires receiver-derived identity"),
    "tool_execute": ("B", "who requested an execution is the primary "
                          "authority question in the system"),
    "executor_recover": ("B", "recovery is state-changing and audited"),
    "subject_erasure": ("B", "erasure is per-subject; who asked is the "
                             "audit record"),
    "checkpoint_restore": ("B", "destructive; attribution required"),
    "checkpoint_delete": ("B", "destructive; attribution required"),
    "postgres_restore": ("B", "overwrites the live database"),
    "paper_trade_slice": ("B", "trades are attributed to an originator"),
    "vault_ingest": ("B", "ingested content carries provenance"),
    "vault_export": ("B", "who exported vault data is an audit fact"),
    "monitor_rule_create": ("B", "rules have owners"),
    "monitor_rule_update": ("B", "rules have owners"),
    "monitor_rule_delete": ("B", "rules have owners"),
    "monitor_rule_enable": ("B", "changes what the system watches"),
    "monitor_rule_disable": ("B", "SILENCES an alert — the most "
                                  "attribution-sensitive act in monitoring"),
    "browser_run": ("B", "arbitrary browser automation, attributed"),
    "browser_navigate": ("B", "outbound action taken on Kai's behalf"),
    "browser_click": ("B", "outbound action taken on Kai's behalf"),
    "browser_type": ("B", "outbound action, may enter credentials"),
    "telegram_alert": ("B", "speaks to the operator AS Kai; who caused "
                            "the message is material"),
    "backup_full": ("B", "produces an artefact attributed to a caller"),
    "backup_postgres": ("B", "produces an artefact attributed to a caller"),
    "backup_redis": ("B", "produces an artefact attributed to a caller"),
    "backup_memory": ("B", "produces an artefact attributed to a caller"),
    "backup_ledger": ("B", "produces an artefact attributed to a caller"),

    # ── A: membership is genuinely enough ────────────────────────────
    "cortex_state_read": ("A", "read-only; returns the same state to any "
                               "authorised internal caller"),
    "browser_scrape": ("A", "read-only fetch, no state change"),
    "browser_screenshot": ("A", "read-only capture"),
    "browser_search": ("A", "read-only query"),
    "monitor_rule_check": ("A", "evaluates rules, changes nothing"),
    "monitor_alerts_clear": ("A", "clears a local display buffer"),
    "desktop_notify": ("A", "shows a notification; no authority follows"),
    "notify_dismiss_one": ("A", "dismisses a local notification"),
    "notify_dismiss_all": ("A", "dismisses local notifications"),
}


def audit(root: Path = None) -> Tuple[List[str], int, Dict[str, int]]:
    """Return (rows, protected endpoints inspected, class counts)."""
    root = root or REPO
    skip = {"_archive", ".venv", "__pycache__", ".git", "scripts", "output"}

    found: List[Tuple[str, str, int]] = []
    unparsed: List[str] = []
    for path in sorted(root.rglob("*.py")):
        rel = str(path.relative_to(root))
        if any(part in skip for part in path.relative_to(root).parts):
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        if _MARKER not in text:
            continue
        try:
            ops = _protected_ops(text)
        except SyntaxError:
            # I-1: a module this scan cannot read is unmeasured, not clean.
            unparsed.append(rel)
            continue
        for op, num in ops:
            found.append((rel, op, num))

    if not found:
        # I-1: finding nothing is a broken scan, not a clean system —
        # this mechanism is known to be in use.
        return (["no protected endpoints found — the scan is broken, "
                 "not the system"], 0, {})

    rows: List[str] = []
    counts: Dict[str, int] = {}
    services = set()
    for rel in unparsed:
        rows.append(f"  {rel}: could not be parsed — its endpoints are "
                    f"UNKNOWN, not absent")
    rows.append(f"  {'service':<18}{'operation':<24}{'class':<14}why")
    for rel, op, num in sorted(found, key=lambda x: (x[0], x[1], x[2])):
        service = rel.split("/")[0]
        services.add(service)
        klass, why = _CLASS.get(op, ("UNCLASSIFIED",
                                     "not classified — must not be read "
                                     "as safe"))
        counts[klass] = counts.get(klass, 0) + 1
        rows.append(f"  {service:<18}{op:<24}{klass:<14}{why}  ({rel}:{num})")

    counts["_services"] = len(services)
    return rows, len(found), counts


def main() -> int:
    rows, n, counts = audit()
    print(inspected(n, "endpoint(s) protected by shared-token auth",
                    f"across {counts.get('_services', 0)} service(s)"))
    print()
    for row in rows:
        print(row)
    print()
    a, b = counts.get("A", 0), counts.get("B", 0)
    u = counts.get("UNCLASSIFIED", 0)
    print(f"  A (membership is enough) ....... {a}")
    print(f"  B (identity is material) ....... {b}")
    if u:
        print(f"  UNCLASSIFIED ................... {u}  <- treat as B until judged")
    print()
    print(f"  {b} of {n} protected endpoints rest on a mechanism that cannot")
    print("  distinguish one caller from another. Every service holding the")
    print("  shared token can act as any other service on all of them.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
