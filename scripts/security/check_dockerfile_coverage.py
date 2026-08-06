#!/usr/bin/env python3
"""Every Dockerfile is built by something, or says why it is not.

On 2026-08-05 one character — `--start_period` for `--start-period` in
`document-parser/Dockerfile` — stopped every image build in
`core-tests.yml` and the thirteen steps behind it. The typo was not
subtle. What made it survive was that **nothing had ever parsed that
file**, and the reason for that was an asymmetry nobody had measured:

    Dockerfiles in the tree                          52
    referenced by docker-compose.full.yml            30   <- all CI built
    in another profile but not in full.yml           19   <- incl. the typo
    referenced by no profile at all                   3

**Twenty-two of fifty-two Dockerfiles, 42%, were never built by CI.**
`core-tests.yml` had one build step and it built `full.yml`, whose name
suggests a superset and is in fact the smaller set.

The build step is split now, so `minimal` and `full` are both built and
49 of the 52 are covered. This gate is what stops the gap reopening: it
compares the tree against every profile and fails on a Dockerfile that
no profile references and no one has declared.

`build:` is what counts, not `image:`. A service pulling a pre-built
image does not exercise a Dockerfile in this tree.

This gate originally *declared* three orphans — `trust-ledger`,
`orchestrator`, `sandboxes/shell` — with a reason, an owner and a review
date, because whether they were dead code or missing profile entries is
a decision about the system's shape and not a thing to guess at in a
gate. On 2026-08-05 the operator made that decision: dead code unless
proven otherwise. Verified before deleting, one at a time:

  orchestrator/       app.py said "DEPRECATED … memu-core is the real
                      orchestrator" in its own first line, 31 lines, one
                      /health route. Reached by nothing but two
                      hand-written lists. Directory removed.
  trust-ledger/       `app.py` is a FastAPI service; `agentic` imports
                      `FileLedger` from `ledger.py` **in-process** via
                      sys.path and never calls the HTTP API, and
                      `scripts/test_trust_ledger.py` imports `ledger`
                      and `score` only. Service removed; `ledger.py` and
                      `score.py` kept — they are load-bearing.
  sandboxes/shell/    Dockerfile removed; `app.py` **kept**. It is not
                      dead: `scripts/test_shell_sandbox.py` is 177 lines
                      exercising its allowlist gating. No profile
                      deploys the container, so the container went and
                      the tested module stayed.

`DECLARED` is empty now and that is the honest state, not an oversight.
It stays in place because the *mechanism* — a dated, owned toleration —
is what stops the next orphan being silently skipped.

Exit 0 = every Dockerfile is built or declared.  Exit 1 = one is neither.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Set, Tuple

REPO = Path(__file__).resolve().parent.parent.parent

sys.path.insert(0, str(REPO))

from scripts.security.gate_inputs import inspected, require  # noqa: E402

PROFILES = (
    "docker-compose.full.yml",
    "docker-compose.minimal.yml",
    "docker-compose.sovereign.yml",
)

_EXCLUDED = {".git", ".venv", "venv", "_archive", "node_modules",
             "__pycache__", "site-packages"}


@dataclass(frozen=True)
class Unbuilt:
    """A Dockerfile no profile builds, and the reason that is tolerated."""
    path: str
    reason: str
    owner: str
    review_by: str


#: Deliberately empty. The operator resolved all three original entries
#: on 2026-08-05 by deleting the code (see the module docstring). Adding
#: an entry here is how a *new* orphan gets tolerated — with a reason,
#: an owner and a date — rather than skipped.
DECLARED: Tuple[Unbuilt, ...] = ()


def tree_dockerfiles() -> Set[str]:
    """Every Dockerfile, from a walk rather than a list.

    Derived for the reason this gate exists: a hand-written list would
    have omitted `document-parser` exactly as `full.yml` did.
    """
    out: Set[str] = set()
    for path in REPO.rglob("Dockerfile*"):
        if any(part in _EXCLUDED for part in path.parts):
            continue
        if path.is_file():
            out.add(str(path.relative_to(REPO)))
    return out


def profile_dockerfiles(doc: dict) -> Dict[str, str]:
    """service -> the Dockerfile it builds, for every `build:` spelling."""
    out: Dict[str, str] = {}
    for name, cfg in sorted((doc.get("services") or {}).items()):
        build = (cfg or {}).get("build")
        if not build:
            continue                       # `image:` builds nothing here
        if isinstance(build, str):
            out[name] = f"{build.strip('./')}/Dockerfile"
            continue
        dockerfile = str(build.get("dockerfile") or "").strip("./")
        context = str(build.get("context") or "").strip("./")
        if dockerfile:
            out[name] = dockerfile
        elif context:
            out[name] = f"{context}/Dockerfile"
        else:
            out[name] = "Dockerfile"
    return out


def coverage_findings(built: Set[str], tree: Set[str],
                      declared: Set[str]) -> List[str]:
    """The three rules, as a function of three sets.

    Split out from `audit()` so the rules can be driven by synthetic
    sets. They used to be exercised only against the real tree and the
    real `DECLARED`, which was fine while `DECLARED` held three entries
    and became vacuous the moment the operator resolved all three: two
    tests started looping over an empty tuple and asserting nothing.
    The assertion ratchet caught the drop (21 → 17) — which is exactly
    what it is for.
    """
    findings: List[str] = []

    for spelling in sorted(built - tree):
        # I-1: a profile naming a Dockerfile that is not there is a
        # finding, not a file to quietly skip.
        findings.append(
            f"{spelling}: referenced by a compose profile but not present "
            f"in the tree")

    for orphan in sorted(tree - built - declared):
        findings.append(
            f"{orphan}: no compose profile builds it, so nothing parses it. "
            f"A parse error here is invisible until somebody builds it by "
            f"hand — which is exactly how document-parser's "
            f"`--start_period` survived. Add it to a profile, or declare "
            f"it in DECLARED with a reason, an owner and a review date.")

    for stale in sorted(declared & built):
        # A declaration that stopped being true is noise that teaches
        # people to ignore the list.
        findings.append(
            f"{stale}: declared as unbuilt but a profile now builds it — "
            f"remove the declaration")

    return findings


def audit() -> Tuple[List[str], int, int]:
    """Return (findings, Dockerfiles inspected, profiles read)."""
    import yaml

    paths = require(PROFILES)
    built: Set[str] = set()
    for path in paths:
        doc = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        built |= set(profile_dockerfiles(doc).values())

    tree = tree_dockerfiles()
    findings = coverage_findings(built, tree, {d.path for d in DECLARED})
    return findings, len(tree), len(paths)


def main() -> int:
    findings, count, profiles = audit()

    print(inspected(count, "Dockerfile(s)",
                    f"against {profiles} compose profiles"))
    if DECLARED:
        print(f"\n  Declared unbuilt ({len(DECLARED)}) — reported so the "
              f"debt is dated, not hidden:")
        for entry in DECLARED:
            print(f"    ~ {entry.path}")
            print(f"        owner={entry.owner}  review by {entry.review_by}")
    print()

    if findings:
        print(f"FAIL: {len(findings)} coverage problem(s):\n")
        for line in findings:
            print(f"  - {line}")
        print("\n  22 of 52 Dockerfiles were never built by CI until "
              "2026-08-05, and\n  one of them held a parse error that "
              "stopped thirteen steps. A file\n  nothing builds is a file "
              "nothing checks.")
        return 1
    print(f"PASS: every Dockerfile is built by a profile or declared "
          f"({len(DECLARED)} declared).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
