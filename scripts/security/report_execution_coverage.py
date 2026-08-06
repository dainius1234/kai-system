#!/usr/bin/env python3
"""Which services CI has never started — derived, not remembered.

Ten defects were found on 2026-08-06 and every single one lived in code
that had **never executed**:

    a withdrawn image tag         nothing pulled it
    `--start_period`              nothing parsed that Dockerfile
    four broken COPY contexts     nothing built those images
    a missing DB_PASSWORD         nothing brought the stack up
    `_pool_lock` unbound          nothing reached the writer branch
    `socket` unbound              nothing reached the write
    /data owned by root           nothing wrote to the volume
    ten stale COPY lines          nothing started the container
    python-multipart × 2          nothing imported the app
    `status == "ok"`              nothing called the endpoint

Not one was code that used to work and broke. That is a different kind
of risk from "the system is buggy", and it means the remaining exposure
sits wherever execution still has not reached — not wherever the code
looks worst.

So this measures that surface instead of guessing at it.

**A report, not a gate.** It fails nothing. What counts as acceptable
coverage is a decision about how much CI time the operator wants to
spend, and a gate that turns red on a number nobody has chosen is a
gate people learn to ignore. It prints, so the number is visible and
moves in a direction somebody chose.

Both sides are derived:

  * the services, from every `docker-compose*.yml` at the root;
  * what CI starts, from the `up -d` lines in `core-tests.yml` —
    including the *named subsets*, because two of the three steps bring
    up three services each rather than a whole profile. Reading those
    as "the profile is covered" would overstate coverage by nineteen
    services, which is precisely the class of error this exists to
    surface.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple

REPO = Path(__file__).resolve().parent.parent.parent

sys.path.insert(0, str(REPO))

from scripts.security.gate_inputs import (  # noqa: E402
    compose_files as _compose_files, inspected)

WORKFLOW = ".github/workflows/core-tests.yml"

_UP = re.compile(r"docker\s+compose\s+-f\s+(\S+)\s+up\s+(-d\s*)?(?P<tail>[^\n|;]*)")


def compose_files(root: Path = None) -> List[Path]:
    """Every compose profile — see `gate_inputs.compose_files`."""
    return _compose_files(root or REPO)


def started_by_ci(root: Path = None,
                  unresolved: List[str] = None) -> Set[str]:
    """Service names CI actually starts, read from its `up -d` lines.

    A bare `up -d` starts every service without a `profiles:` gate. An
    `up -d a b c` starts exactly those three. Conflating the two is how
    a coverage number becomes a comfortable fiction.

    `unresolved`, if given, collects any compose file the workflow names
    that is not in the tree. The meta-check flagged the first draft for
    skipping those silently (I-1), and it was right to: skipping makes
    this survey cover less than it claims while still printing a
    confident number. It is *reported* rather than fatal because this is
    a report — but it is never invisible.
    """
    import yaml

    root = root or REPO
    path = root / WORKFLOW
    if not path.exists():
        return set()
    text = path.read_text(encoding="utf-8")
    started: Set[str] = set()
    for match in _UP.finditer(text):
        profile = match.group(1)
        tail = match.group("tail") or ""
        candidate = root / profile
        if not candidate.exists():
            if unresolved is not None:
                unresolved.append(
                    f"{profile}: named by an `up` line in {WORKFLOW} but "
                    f"not in the tree — its services could not be counted, "
                    f"so this survey is incomplete")
            continue
        doc = yaml.safe_load(candidate.read_text(encoding="utf-8")) or {}
        defined = set(doc.get("services") or {})

        # Only tokens the profile actually defines count as service
        # names. The first draft of this took every non-flag token, and
        # the minimal bring-up is
        #
        #     docker compose -f … up -d --build \
        #       2>&1 | tee /tmp/bringup.log
        #
        # so it read the trailing `\` as a service, concluded the step
        # named one service, and reported 3 services covered instead of
        # 15. That is the *exact* defect `check_compose_env` was given
        # fail-closed handling for earlier today — a line continuation
        # parsed as a name — reproduced by me in a new file hours later.
        # Matching against the profile's own services is what makes the
        # question unambiguous.
        named = [t for t in tail.split() if t in defined]
        if named:
            started.update(named)
            continue
        # No service named: a bare `up` starts everything the profile
        # does not gate behind `profiles:`.
        started.update(
            name for name, cfg in (doc.get("services") or {}).items()
            if not (cfg or {}).get("profiles"))
    return started


def services(root: Path = None) -> Dict[str, Dict[str, Set[str]]]:
    """name -> {built, default_in, gated_in} across every profile."""
    import yaml

    root = root or REPO
    out: Dict[str, Dict[str, Set[str]]] = {}
    for path in compose_files(root):
        doc = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        tag = path.name.split(".")[1] if "." in path.name else path.name
        for name, cfg in (doc.get("services") or {}).items():
            cfg = cfg or {}
            entry = out.setdefault(
                name, {"built": set(), "default_in": set(), "gated_in": set()})
            if cfg.get("build"):
                entry["built"].add(tag)
            (entry["gated_in"] if cfg.get("profiles")
             else entry["default_in"]).add(tag)
    return out


def survey(root: Path = None,
           unresolved: List[str] = None
           ) -> Tuple[List[str], List[str], int, int]:
    """(never-run defaults, never-run opt-ins, built total, started total)."""
    all_services = services(root)
    started = started_by_ci(root, unresolved)
    built = {n for n, v in all_services.items() if v["built"]}
    never = built - started
    defaults = sorted(n for n in never if all_services[n]["default_in"])
    optin = sorted(n for n in never if not all_services[n]["default_in"])
    return defaults, optin, len(built), len(built & started)


def main() -> int:
    unresolved: List[str] = []
    defaults, optin, built, started = survey(unresolved=unresolved)

    for line in unresolved:
        print(f"  INCOMPLETE — {line}")

    print(inspected(built, "service(s) with a Dockerfile",
                    f"against the `up -d` lines in {WORKFLOW}"))
    print(f"\n  started by CI at least once ...... {started}")
    print(f"  never started by CI .............. {built - started}")

    if defaults:
        print(f"\n  NEVER RUN, yet starts BY DEFAULT in a shipped profile "
              f"({len(defaults)}) —\n  these boot on a real deployment with "
              f"nothing having exercised them:")
        for name in defaults:
            print(f"    ! {name}")
    if optin:
        print(f"\n  NEVER RUN, opt-in everywhere ({len(optin)}) — gated "
              f"behind `profiles:`,\n  so a deployment only gets them by "
              f"asking:")
        for name in optin:
            print(f"    ~ {name}")

    print(f"\n  Reported, never failed: what coverage is enough is a "
          f"decision about CI\n  time, and a gate red on a number nobody "
          f"chose is a gate people ignore.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
