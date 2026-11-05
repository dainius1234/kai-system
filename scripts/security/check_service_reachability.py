#!/usr/bin/env python3
"""A service told to wait for a peer, and given its URL, must be able to reach it.

Docker's embedded DNS only resolves names on networks a container has
joined. Two services on disjoint networks cannot see each other at all —
the call fails at name resolution, before any connection is attempted.

The rule, and why it is this narrow
-----------------------------------

Intent has to be declared **twice** before this reports anything:

1. the service `depends_on` the target — "do not start me until it is
   there", and
2. an environment value holds a URL naming the target — "here is where
   to find it".

Both, and no shared network. That is not a preference; it is a
contradiction the compose file states about itself.

The narrowing was earned, not designed. Measured 2026-08-07:

| rule | findings | of |
|---|---|---|
| any env URL naming a service on no shared network | 36 | 130 |
| any `depends_on` target on no shared network | 14 | 90 |
| **both, on no shared network** | **5** | **51** |

The broad forms are wrong, and instructively so. `dashboard` holds URLs
for fourteen optional services it degrades without — flagging those would
report failure over a design. And `heartbeat depends_on memu-core` while
actually calling `memu-core-introspect`, which it *can* reach: a
`depends_on` can legitimately mean "wait for readiness" rather than "I
will call you". Only the intersection is decidable.

Reported, not enforced — and why
--------------------------------

All five findings need a **network topology decision**, and network zones
are deliberate policy here with their own gate (`check_network_zones`).
Widening a service's network membership to silence this check would
change the security topology to make a report go quiet, which is the
wrong way round. So this counts and names them, and stays a report until
someone decides.

Exit 0 always. `kind=REPORT` in the registry says so out loud, rather
than leaving its absence from `policy-check` to be inferred.
"""
from __future__ import annotations

import ast
import re
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO))

from scripts.security.gate_inputs import compose_files, inspected  # noqa: E402

_URL = re.compile(r"https?://([a-z0-9][a-z0-9._-]*)")

#: httpx / requests verbs. A name reaching one of these is a real call.
_REQUEST_METHODS = {"post", "get", "put", "delete", "patch", "request",
                    "stream"}


def _environment(cfg: dict) -> Dict[str, str]:
    """Compose accepts a mapping or a `KEY=value` list. Handle both."""
    env = (cfg or {}).get("environment") or {}
    if isinstance(env, list):
        out = {}
        for item in env:
            if isinstance(item, str) and "=" in item:
                k, v = item.split("=", 1)
                out[k] = v
        return out
    return {k: v for k, v in env.items() if isinstance(v, str)}


def _depends(cfg: dict) -> Set[str]:
    dep = (cfg or {}).get("depends_on") or {}
    return set(dep.keys()) if isinstance(dep, dict) else set(dep)


def _source_dir(cfg: dict) -> Path:
    """The service's source tree, derived from its own `build:` stanza."""
    build = (cfg or {}).get("build")
    if isinstance(build, dict) and build.get("dockerfile"):
        return REPO / Path(build["dockerfile"]).parent
    return None


class _EnvUse(ast.NodeVisitor):
    """Which `os.getenv` names are bound, and which names reach a request."""

    def __init__(self) -> None:
        self.bound: Dict[str, str] = {}     # local variable -> env name
        self.in_request: Set[str] = set()   # names appearing in an HTTP call

    def visit_Assign(self, node: ast.Assign) -> None:
        for sub in ast.walk(node.value):
            if (isinstance(sub, ast.Call)
                    and getattr(sub.func, "attr", "") == "getenv"
                    and sub.args and isinstance(sub.args[0], ast.Constant)):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        self.bound[target.id] = sub.args[0].value
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        if getattr(node.func, "attr", "") in _REQUEST_METHODS:
            for sub in ast.walk(node):
                if isinstance(sub, ast.Name):
                    self.in_request.add(sub.id)
        self.generic_visit(node)


def code_confidence(cfg: dict, env_name: str) -> str:
    """How strongly the service's own code confirms it makes this call.

    Three states, and the middle one is the point:

      confirmed  the env var is bound to a name and that name reaches an
                 HTTP call — the strongest evidence the call is real
      indirect   the env name appears in the source, but not via that
                 pattern
      absent     the env name does not appear in the source at all

    **This classifies. It does not filter**, and that distinction is the
    whole design. The obvious sharpening — report only `confirmed` — was
    tested on 2026-08-07 and would have SILENTLY DROPPED A REAL FINDING:

        supervisor/app.py:42
            SERVICES = [
                {"name": "heartbeat",
                 "url": os.getenv("HEARTBEAT_URL", "http://heartbeat:8010")},
                ...
            ]
        supervisor/app.py:94
            base = svc["url"]

    The variable is never bound; the value goes straight into a dict
    literal and is consumed by subscript later. `supervisor` cannot reach
    `heartbeat` and that is a genuine defect, and a filter on `confirmed`
    would have reported it as unwired and moved on.

    In a check adjacent to security, a false negative costs more than a
    false positive: a false positive is argued about, a false negative is
    never seen. So the extra signal is added to every finding and removes
    none of them.
    """
    src = _source_dir(cfg)
    if src is None or not src.is_dir():
        return "indirect"       # cannot see the source; claim nothing
    seen_anywhere = False
    for py in sorted(src.rglob("*.py")):
        try:
            text = py.read_text(encoding="utf-8", errors="replace")
            tree = ast.parse(text)
        except Exception:
            continue
        if env_name in text:
            seen_anywhere = True
        use = _EnvUse()
        use.visit(tree)
        for var, name in use.bound.items():
            if name == env_name and var in use.in_request:
                return "confirmed"
    return "indirect" if seen_anywhere else "absent"


def audit(root: Path = None) -> Tuple[List[str], int, int]:
    """Return (findings, doubly-declared edges inspected, compose files)."""
    import yaml

    root = root or REPO
    files = compose_files(root)
    if not files:
        # I-1: nothing inspected is not a clean bill of health.
        return ([f"{root}: no compose files found — this check inspected "
                 f"nothing and must not report success"], 0, 0)

    findings: List[str] = []
    checked = 0
    for path in files:
        try:
            doc = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        except Exception:
            continue        # `check_ci_tolerations` owns unparseable files
        services = doc.get("services") or {}
        nets = {n: set((c or {}).get("networks") or [])
                for n, c in services.items()}

        for name, cfg in services.items():
            urls: Dict[str, str] = {}
            for key, value in _environment(cfg).items():
                match = _URL.search(value)
                if match and match.group(1) in services:
                    urls[match.group(1)] = key

            for target in _depends(cfg) & set(urls):
                checked += 1
                if nets.get(name, set()) & nets.get(target, set()):
                    continue
                confidence = code_confidence(cfg, urls[target])
                _note = {
                    "confirmed": "the code binds this variable and passes it "
                                 "to an HTTP call, so the call is real",
                    "indirect": "the variable name appears in the source but "
                                "not bound-then-called (it may go straight "
                                "into a data structure — see supervisor), so "
                                "verify by hand",
                    "absent": "the variable name does not appear in this "
                              "service's source AT ALL — configuration with "
                              "no consumer, a different defect from an "
                              "unreachable peer",
                }[confidence]
                findings.append(
                    f"[{confidence}] {_note}. "
                    f"{path.name}: `{name}` depends on `{target}` and holds "
                    f"its address in {urls[target]}, but they share no "
                    f"network — `{name}` is on "
                    f"{sorted(nets.get(name, set())) or ['(none)']}, "
                    f"`{target}` on "
                    f"{sorted(nets.get(target, set())) or ['(none)']}. "
                    f"Docker's DNS only resolves names on a joined network, "
                    f"so the call fails at name resolution.")
    return findings, checked, len(files)


def main() -> int:
    findings, checked, files = audit()

    print(inspected(checked, "edge(s) where a service both depends on a "
                             "peer and holds its URL",
                    f"across {files} compose file(s)"))
    print()

    if findings:
        print(f"REPORTED: {len(findings)} peer(s) named twice and reachable "
              f"never:\n")
        for line in findings:
            print(f"  - {line}")
        print("\n  Not failed, deliberately. Every one of these needs a "
              "network topology\n  decision, and network zones are policy "
              "with their own gate. Widening a\n  service's networks to "
              "silence this report would change the security\n  topology "
              "to quieten a check, which is the wrong way round.")
        print("\n  Note `executor` separately: it is given MEMU_URL and told "
              "to wait for\n  memu-core, and `executor/app.py` contains ZERO "
              "occurrences of MEMU.\n  That config is either an unfinished "
              "intent or dead weight — and deleting\n  it to tidy up would "
              "erase the only surviving evidence of which.")
        return 0

    print(f"PASS: every doubly-declared peer is reachable ({checked} "
          f"inspected).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
