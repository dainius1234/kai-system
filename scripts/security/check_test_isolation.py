"""No test file may change the interpreter for the files that follow it.

On 2026-08-04 the repo-wide pytest run — the one job that executes this
repository's ~4,200 tests — was aborting during collection. Not failing:
aborting, before a single test ran, on every run since at least 27 July.
One line did most of it:

    sys.modules["common"] = types.ModuleType("common")     # test_cortex.py

Five other files, all sorting after `test_cortex`, failed to import
`common.<anything>`. Every one of them passed alone. The failures named
the innocent files; nothing named the cause.

That is the shape this gate exists to stop. It is the inverse of the
self-consuming guard the rest of this directory watches for. A
self-consuming guard stops checking and looks like a pass. This stops
checking and looks like a *failure somewhere else*, which is worse,
because the obvious response is to go and change the file that is not
broken.

**What it measures, and why it measures it this way.** The first detector
written for this imported each test file in a subprocess and asked what
it left behind in `sys.modules`. It found seven offenders and it was
wrong, because the worst one it could not see —
`test_cognitive_mechanisms.py` replacing `fastapi` with a two-attribute
stub — does that from `setup_method`, per test, long after import. An
import-time probe cannot observe a run-time edit.

So this reads the run itself. `isolation_plugin.py` hooks
`pytest_runtest_protocol` and diffs `sys.modules`, `os.environ` and
`sys.path` across file boundaries in the real session, which is the
mechanism that actually decides the answer.

**Two different rules, because the two kinds of leak are not the same.**

  - `replaced` — a name that pointed at a real module and now points at
    something else. `fastapi` becoming a stub. This is at **zero and
    enforced**: any occurrence fails, no baseline, no ratchet. Reaching
    zero obliges keeping it, the same ratchet shape as the assertion
    floors.
  - `added` and `env_set` — a stub under a previously free name, or an
    environment variable left set. Damaging only when something else
    wants that name or reads that variable, so these ratchet down from a
    recorded baseline rather than failing outright. `BINANCE_API_KEY`,
    `INTERSERVICE_HMAC_SECRET` and four `FF_*` flags are among them, and
    they are debt, not decoration.

Exit 0 = nothing replaced, and the declared leakage has not grown.
Exit 1 = something replaced, leakage grew, or the report is unreadable.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple

REPO = Path(__file__).resolve().parent.parent.parent
BASELINE = Path(__file__).resolve().parent / "test_isolation_baseline.json"

sys.path.insert(0, str(REPO))

from scripts.security.gate_inputs import inspected, require  # noqa: E402

_PYTEST = [
    sys.executable, "-m", "pytest",
    "--ignore=_archive", "--ignore=.venv",
    "--ignore=scripts/test_dashboard.py", "--ignore=scripts/test_dashboard_ui.py",
    "-q", "-p", "no:cacheprovider", "-p", "scripts.security.isolation_plugin",
    "--tb=no",
]


def run_and_report(destination: Path) -> int:
    """Run the suite with the plugin attached; return pytest's status."""
    env = {**os.environ,
           "PYTHONPATH": ".",
           "MEMU_ALLOW_FAKE_EMBEDDINGS": "true",
           "KAI_ISOLATION_REPORT": str(destination)}
    result = subprocess.run(_PYTEST, cwd=str(REPO), env=env,
                            capture_output=True, text=True)
    return result.returncode


def compare(report: Dict[str, dict],
            baseline: dict) -> Tuple[List[str], List[str], Dict[str, int]]:
    """Return (replacements, grown, totals).

    A file in the report but not the baseline counts as growth even when
    its counts are small: an undeclared leak is exactly the thing that
    was invisible for a week, and declaring it is one edit.
    """
    declared = baseline.get("leaky_files", {})
    replacements: List[str] = []
    grown: List[str] = []
    totals = {"replaced": 0, "added": 0, "env_set": 0}

    for path, finding in sorted(report.items()):
        rel = os.path.relpath(path, str(REPO))
        for line in finding.get("replaced", []):
            replacements.append(f"{rel}: {line}")
        totals["replaced"] += len(finding.get("replaced", []))
        totals["added"] += len(finding.get("added", []))
        totals["env_set"] += len(finding.get("env_set", []))

        known = declared.get(rel)
        if known is None:
            if finding.get("added") or finding.get("env_set"):
                grown.append(f"{rel}: not declared "
                             f"({len(finding.get('added', []))} added, "
                             f"{len(finding.get('env_set', []))} env)")
            continue
        for kind in ("added", "env_set"):
            now, before = len(finding.get(kind, [])), known.get(kind, 0)
            if now > before:
                grown.append(f"{rel}: {kind} {before} -> {now}")

    return replacements, grown, totals


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--from-report", type=Path,
                        help="read a report the test run already produced")
    parser.add_argument("--write-baseline", action="store_true",
                        help="record current leakage (refuses to record a replacement)")
    args = parser.parse_args()

    require(("scripts/security/isolation_plugin.py",
             "scripts/security/test_isolation_baseline.json"))

    if args.from_report:
        if not args.from_report.exists():
            print(f"REFUSED: no report at {args.from_report}.")
            print("  A missing report is not an absence of leaks. Produce one with")
            print("  KAI_ISOLATION_REPORT=<path> pytest -p scripts.security.isolation_plugin")
            return 1
        report = json.loads(args.from_report.read_text(encoding="utf-8"))
    else:
        destination = REPO / ".isolation-report.json"
        run_and_report(destination)
        if not destination.exists():
            print("REFUSED: the test run produced no isolation report.")
            print("  The plugin did not load, or the run died before sessionfinish.")
            return 1
        report = json.loads(destination.read_text(encoding="utf-8"))

    baseline = json.loads(BASELINE.read_text(encoding="utf-8"))
    replacements, grown, totals = compare(report, baseline)

    print(inspected(len(report), "test files that alter global state",
                    f"{totals['replaced']} replaced, {totals['added']} added, "
                    f"{totals['env_set']} env"))

    # An empty report against a baseline that names seven files is not a
    # clean repository — it is a run that did not happen. CI proved this on
    # 2026-08-04: the suite aborted during collection, the plugin wrote an
    # empty report, and this gate printed "WARNING: zero inputs inspected —
    # this is not a pass" and then **passed**.
    #
    # That is boundary blindness in the file written to prevent boundary
    # blindness. The warning was doing the work of a rule, and a warning is
    # not a rule. `--write-baseline` is exempt: recording zero is how the
    # last declared leak would legitimately be removed.
    declared = baseline.get("leaky_files", {})
    if declared and not report and not args.write_baseline:
        print(f"\nFAIL: the report is empty, but {len(declared)} file(s) are "
              f"declared as leaking.\n")
        print("  Either every declared leak was fixed in one commit — in which")
        print("  case run `make test-isolation-baseline` and say so — or the")
        print("  run this report came from never executed. The second is far")
        print("  more likely, and reads identically to the first.")
        return 1

    if args.write_baseline:
        if replacements:
            print("\nREFUSED: a replacement cannot be baselined.")
            for line in replacements:
                print(f"  - {line}")
            return 1
        files = {}
        for path, finding in sorted(report.items()):
            rel = os.path.relpath(path, str(REPO))
            if finding.get("added") or finding.get("env_set"):
                files[rel] = {"env_set": len(finding.get("env_set", [])),
                              "added": len(finding.get("added", []))}
        BASELINE.write_text(json.dumps({
            "note": baseline.get("note", ""),
            "replaced_allowed": 0,
            "leaky_files": files,
            "total_env_set": totals["env_set"],
            "total_added": totals["added"],
        }, indent=2) + "\n", encoding="utf-8")
        print(f"\nBaseline recorded: {len(files)} files.")
        return 0

    if replacements:
        print(f"\nFAIL: {len(replacements)} real module(s) replaced and left replaced:\n")
        for line in replacements:
            print(f"  - {line}")
        print("\n  A test that swaps a real module for a stub and does not put it")
        print("  back has edited the interpreter for every file collected after")
        print("  it. The failure will appear in those files, not this one.")
        print("  Scope it: `with stubbed({...}): spec.loader.exec_module(mod)`")
        print("  — see scripts/module_stubs.py.")
        return 1

    if grown:
        print(f"\nFAIL: cross-file leakage grew in {len(grown)} place(s):\n")
        for line in grown:
            print(f"  - {line}")
        print("\n  These ratchet down, never up. If the leak is genuinely")
        print("  necessary, record it with `make test-isolation-baseline` in a")
        print("  commit that says why.")
        return 1

    print("\nPASS: no real module is left replaced, and no declared leak grew.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
