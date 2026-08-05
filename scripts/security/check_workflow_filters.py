#!/usr/bin/env python3
"""Every `jq` filter embedded in a workflow must compile.

`drift-detector.yml` failed **all 15 of its scheduled runs** from
2026-04-27 to 2026-08-05 on this line:

    --jq '[.[] | select(.title | startswith(\\"Weekly drift report (\\"))]
          | sort_by(.updatedAt) | reverse | .[0].number // empty'

Inside a single-quoted shell argument, `\\"` is a literal backslash and
quote. jq receives it verbatim and refuses:

    jq: error: syntax error, unexpected INVALID_CHARACTER
        (Unix shell quoting issues?)

With `set -euo pipefail` above it, that killed the job.

**Nothing in this repository could see it.** The file is valid YAML, so
`check_ci_tolerations.unparseable()` passed it. The script is valid
bash — `bash -n` accepts it — so a shell lint would pass it too. The
defect lives one level further in, in a string that only jq ever parses,
and it stayed there for three and a half months because a *scheduled*
workflow has no author watching the result.

That last part is the reason this gate exists rather than a one-line
fix. A push workflow that breaks gets noticed by whoever pushed. A
scheduled one fails into an empty room.

The check is syntax, not behaviour: each filter is compiled against
`null` and only `syntax error` / `compile error` counts. A filter that
compiles and then legitimately fails on unrepresentative input is not a
finding — inventing input for every filter would be a false-positive
machine, which is the defect this repository works hardest to avoid.

Fails closed. If `jq` is not installed, that is reported rather than
treated as "nothing to check": a gate that cannot run has not passed.

Exit 0 = every filter compiles.  Exit 1 = one does not, or jq is absent.
"""
from __future__ import annotations

import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import List, Tuple

REPO = Path(__file__).resolve().parent.parent.parent
WORKFLOWS = REPO / ".github" / "workflows"

sys.path.insert(0, str(REPO))

from scripts.security.gate_inputs import inspected, require  # noqa: E402

# `--jq '...'` (gh) and `jq '...'` (standalone). Single-quoted only:
# a double-quoted filter is subject to shell expansion and cannot be
# compiled without knowing the variables, so it is out of scope and
# said so rather than silently skipped.
_SINGLE = re.compile(r"(?:--jq|(?<![\w-])jq)\s+'([^']+)'")
_DOUBLE = re.compile(r"(?:--jq|(?<![\w-])jq)\s+\"")


def filters_in(text: str) -> List[str]:
    return _SINGLE.findall(text)


def compiles(filter_text: str) -> Tuple[bool, str]:
    """(ok, detail). Only a syntax/compile error is a failure."""
    try:
        proc = subprocess.run(
            ["jq", filter_text], input="null", capture_output=True,
            text=True, timeout=20)
    except FileNotFoundError:
        return False, "jq is not installed, so this filter was not checked"
    except subprocess.TimeoutExpired:
        return False, "jq did not finish"
    stderr = proc.stderr or ""
    if "syntax error" in stderr or "compile error" in stderr:
        first = stderr.strip().splitlines()[0]
        return False, first[:200]
    return True, "compiles"


def audit() -> Tuple[List[str], int, int]:
    """Return (findings, filters checked, workflows inspected)."""
    findings: List[str] = []
    if shutil.which("jq") is None:
        # I-1. Absence of the tool is absence of the answer.
        return (["jq is not installed — no filter could be checked, and an "
                 "unchecked filter is not a passing one"], 0, 0)

    paths = sorted(WORKFLOWS.glob("*.yml"))
    checked = 0
    for path in paths:
        text = path.read_text(encoding="utf-8")
        for line_no, line in enumerate(text.splitlines(), 1):
            if _DOUBLE.search(line):
                findings.append(
                    f"{path.name}:{line_no}: a double-quoted jq filter is "
                    f"subject to shell expansion and cannot be compiled here. "
                    f"Use single quotes so it can be checked.")
        for filter_text in filters_in(text):
            checked += 1
            ok, detail = compiles(filter_text)
            if not ok:
                findings.append(
                    f"{path.name}: {detail}\n      filter: "
                    f"{filter_text[:120]}")
    return findings, checked, len(paths)


def main() -> int:
    require((".github/workflows",))
    findings, checked, workflows = audit()

    print(inspected(checked, "embedded jq filter(s)",
                    f"across {workflows} workflows"))
    print()
    if findings:
        print(f"FAIL: {len(findings)} jq problem(s):\n")
        for line in findings:
            print(f"  - {line}")
        print("\n  A workflow can be valid YAML and valid bash and still die "
              "on a\n  filter only jq ever parses. `drift-detector.yml` failed "
              "every one\n  of its scheduled runs for three and a half months "
              "on exactly this.")
        return 1
    if checked == 0:
        # I-2: nothing found must not read the same as everything passing.
        print("PASS: no embedded jq filters found — nothing to check.")
        return 0
    print(f"PASS: all {checked} embedded jq filter(s) compile.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
