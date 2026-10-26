#!/usr/bin/env python3
"""A `depends_on` must say what it is waiting for.

The defect
----------

Compose accepts two forms::

    depends_on:                     # (a) a bare list
      - postgres

    depends_on:                     # (b) a mapping with a condition
      postgres:
        condition: service_healthy

Form (a) waits for the container to be **created**. Not started, not
listening, not healthy — created. So a service that opens a database
connection in its module body starts against a Postgres that has not
finished initialising, dies, and — with `restart: unless-stopped` —
comes back and dies again, which reads in the log as a flapping service
rather than as an ordering bug.

`docker-compose.full.yml` had sixteen of these. They were fixed on
2026-08-07 in `e47622b` and the register entry was closed. Eleven more
survived, in the same tree, on the same day:

    full.yml 0 | minimal.yml 1 | sovereign.yml 10

Nothing found them because nothing was looking: thirty gate scripts and
not one mentioned `depends_on` readiness. The fix's scope was one file;
the class's scope was the tree. That is this programme's own finding —
*a scope smaller than the claim* — turned on a fix rather than a check.

What is enforced
----------------

**Every `depends_on` is a mapping with a valid, explicit condition.** A
bare list is a finding. So is a mapping entry with no `condition:`, and
so is a condition compose does not accept — compose rejects that one
too, but it does so during `up`, on a runner, minutes in.

`service_started` is permitted. It means the same as a bare list, but it
means it *on purpose*, and the count of them is printed on every run so
it cannot quietly grow.

What is reported and NOT enforced
---------------------------------

**`service_started` against a target that declares a healthcheck.**
The tempting rule is "the readiness signal exists, so use it" — and it
is wrong as a hard gate. `docker-compose.minimal.yml` has four, all on
`dashboard`:

    tts-service, notify-service, document-parser, agentic

Blocking a UI's start on a slow optional dependency is a legitimate
design decision, and `service_healthy` against a service that never
reaches healthy blocks the entire bring-up. That profile is green in CI.
A gate that failed here would report failure over something that is
right — the inverted corollary of this programme's finding, which has
already cost it 100 findings for 1 real and 69 findings on a tree that
was correct.

So these are listed, counted, and left to a human. If the count grows,
the growth is visible in the run output.

Also deliberately not checked: whether a service *without* a healthcheck
ought to have one. Unanswerable from the compose file.

Calibration
-----------

Run against `docker-compose.full.yml` as it stood at `e47622b~1`, the
enforced clause reports 16 services / 32 edges — exactly the known
defect count.

    git show e47622b~1:docker-compose.full.yml > /tmp/f.yml

Exit 0 = every dependency states what it waits for.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, NamedTuple

REPO = Path(__file__).resolve().parent.parent.parent

sys.path.insert(0, str(REPO))

from scripts.security.gate_inputs import compose_files, inspected  # noqa: E402

#: Conditions compose understands.
_VALID = {"service_started", "service_healthy",
          "service_completed_successfully"}


class Report(NamedTuple):
    findings: List[str]     # enforced — exit 1
    advisories: List[str]   # reported — a human decides
    edges: int              # denominator
    started: int            # explicit `service_started`, any target


def audit(root: Path = None) -> Report:
    import yaml

    root = root or REPO
    files = compose_files(root)
    if not files:
        # I-1: an empty universe is not a clean one.
        return Report([f"{root}: no docker-compose*.yml found — this gate "
                       f"inspected nothing and must not report success"],
                      [], 0, 0)

    findings: List[str] = []
    advisories: List[str] = []
    edges = 0
    started = 0

    for path in files:
        try:
            doc = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        except Exception:
            continue        # `check_ci_tolerations` owns unparseable files
        services: Dict = doc.get("services") or {}
        for name, cfg in services.items():
            deps = (cfg or {}).get("depends_on")
            if not deps:
                continue

            if isinstance(deps, list):
                edges += len(deps)
                findings.append(
                    f"{path.name}: `{name}` depends_on {list(deps)} as a "
                    f"bare list, which waits for those containers to be "
                    f"*created* and not for them to be ready. Give each one "
                    f"an explicit `condition:`.")
                continue

            for target, spec in (deps or {}).items():
                edges += 1
                condition = (spec or {}).get("condition")
                if condition is None:
                    findings.append(
                        f"{path.name}: `{name}` -> `{target}` declares no "
                        f"`condition:`, so it waits for creation only.")
                    continue
                if condition not in _VALID:
                    findings.append(
                        f"{path.name}: `{name}` -> `{target}` uses condition "
                        f"`{condition}`, which compose does not accept. "
                        f"Valid: {', '.join(sorted(_VALID))}.")
                    continue
                if condition != "service_started":
                    continue
                started += 1

                target_cfg = services.get(target)
                if target_cfg is None:
                    # Cross-profile reference or a typo. `check_compose_drift`
                    # owns target existence; not this gate's subject.
                    continue
                if (target_cfg.get("healthcheck") or {}).get("test"):
                    advisories.append(
                        f"{path.name}: `{name}` -> `{target}` waits with "
                        f"`service_started` although `{target}` declares a "
                        f"healthcheck")

    return Report(findings, advisories, edges, started)


def main() -> int:
    report = audit()

    print(inspected(report.edges, "depends_on edge(s)",
                    f"{report.started} explicitly `service_started`"))
    print()

    if report.advisories:
        print(f"  Reported, not enforced — {len(report.advisories)} "
              f"dependency(ies) decline an available readiness signal:")
        for line in report.advisories:
            print(f"    ~ {line}")
        print("    Legitimate when the dependency is optional and the "
              "dependent degrades.\n    Not decidable from the compose "
              "file, so it is a human's call, not this\n    gate's. The "
              "count is printed so it cannot grow unseen.\n")

    if report.findings:
        print(f"FAIL: {len(report.findings)} dependency declaration(s) do "
              f"not say what they wait for:\n")
        for line in report.findings:
            print(f"  - {line}")
        print("\n  A bare `depends_on` list waits for container CREATION, "
              "not readiness.\n  Sixteen of these were fixed in "
              "docker-compose.full.yml and the class\n  was called closed; "
              "eleven survived in minimal.yml and sovereign.yml\n  because "
              "nothing measured the class across the tree.")
        return 1

    print(f"PASS: every dependency states what it waits for "
          f"({report.edges} edge(s) inspected, "
          f"{report.started} `service_started`).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
