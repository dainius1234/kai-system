#!/usr/bin/env python3
"""A compose bring-up that warned did not fully succeed.

The rule this enforces is the operator's third directive applied to
compose output:

> **Nothing repeats unexplained.** A recurring signal is fixed, made to
> fail, or declared — with a name against it and a date on it.

Why it is a script and not two lines of shell
---------------------------------------------

Both bring-up steps carried this, written out separately:

    if grep -q 'variable is not set' /tmp/bringup.log; then …

Two copies of a rule is the list-beside-the-thing pattern in shell form,
and it had already cost a run. On 2026-08-06 the full profile failed
with

    level=warning msg="secret file kai-system_db_password does not exist"
    Container kai-system-backup-service-1  Error response from daemon:
      invalid mount config for type "bind": bind source path does not
      exist: …/runtime-secrets/db_password

and the guard said nothing, because it is named *"a compose variable was
substituted blank"* and that is genuinely all it looked for. The class is
**compose warned that something it needed was missing**; the guard's
scope was one member of that class. The same finding, in its sixteenth
venue: a check whose scope was smaller than the thing it was standing in
for.

So this matches the class and declares the exceptions, rather than
matching one member and hoping.

Two warning shapes, because compose emits both
----------------------------------------------

    WARN[0000] The "DB_PASSWORD" variable is not set. Defaulting to …
    time="…" level=warning msg="secret file … does not exist"

Nothing guarantees a third shape will not appear, which is why the
allowlist is of *benign* warnings rather than the match being a list of
known-bad ones. An unrecognised warning fails: that is I-1 applied to a
log — an unfamiliar signal is not evidence of health.

Usage:  assert_clean_bringup.py <logfile> [<logfile> …]
Exit 0 = the bring-up printed no warning that was not declared benign.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import List, Tuple

#: Warnings that are genuinely nothing, each with the reason it is
#: nothing. Declared in one place (I-4) so a second bring-up step cannot
#: hold a different opinion about the same line.
#:
#: Deliberately short. Every entry here is a signal somebody has decided
#: to read past forever, and the bar for that is the same bar as a CI
#: toleration: it needs a reason, not a shrug.
BENIGN: Tuple[Tuple[str, str], ...] = (
    ("Node.js 20 is deprecated",
     "GitHub's runner deprecation notice, not compose output at all. "
     "Owned by GitHub's action versions, tracked separately."),
)

#: Both shapes compose uses. Kept as patterns for the *class*, not for
#: individual messages, so a new kind of missing thing is caught by the
#: rule that already exists rather than by one written after it bites.
_WARNING = re.compile(r"^\s*WARN\[|level=warning|^\s*WARNING:", re.MULTILINE)


def warnings_in(text: str) -> List[str]:
    """Every warning line, minus the ones declared benign."""
    out = []
    for line in text.splitlines():
        if not _WARNING.search(line):
            continue
        if any(needle in line for needle, _ in BENIGN):
            continue
        out.append(line.strip())
    return out


def audit(paths: List[Path]) -> Tuple[List[str], int, int]:
    """Return (warnings, lines read, files read).

    A file that does not exist is a finding, not a skip. The step that
    was supposed to write it either did not run or redirected somewhere
    else, and in both cases this has verified nothing — which is the
    boundary blindness the whole programme is about.
    """
    findings: List[str] = []
    lines = 0
    read = 0
    for path in paths:
        if not path.exists():
            findings.append(
                f"{path}: the log this was asked to inspect does not "
                f"exist. The bring-up step did not write it, so this "
                f"check has verified nothing and must not report success.")
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        count = len(text.splitlines())
        lines += count
        read += 1
        if count == 0:
            # I-1 again, caught in this file's own calibration: the first
            # draft printed "WARNING: the log was empty" and returned 0,
            # so a bring-up that produced no output at all was reported
            # as a clean bring-up. `docker compose up` always prints a
            # line per container; an empty log means the redirect did not
            # capture, and a check reading nothing has verified nothing.
            findings.append(
                f"{path.name}: empty. `docker compose up` prints a line "
                f"per container, so no output means the redirect did not "
                f"capture — this check read nothing and cannot report a "
                f"clean bring-up.")
            continue
        findings.extend(f"{path.name}: {w}" for w in warnings_in(text))
    return findings, lines, read


def main(argv: List[str]) -> int:
    if not argv:
        print("usage: assert_clean_bringup.py <logfile> [<logfile> …]")
        return 2
    paths = [Path(a) for a in argv]
    findings, lines, read = audit(paths)

    print(f"  inspected: {lines} line(s) of bring-up output "
          f"(across {read}/{len(paths)} log file(s))")
    if lines == 0 and read:
        print("  WARNING: the log was empty — this is not a pass.")

    if findings:
        print(f"\n::error::The bring-up printed {len(findings)} warning(s) "
              f"nobody has accounted for\n")
        for line in findings:
            print(f"  - {line}")
        print("\n  A bring-up that warns did not fully succeed. Either the "
              "thing it\n  warned about is missing and should be provided, "
              "or the warning is\n  genuinely nothing and belongs in BENIGN "
              "with the reason why.\n\n  On 2026-08-06 `secret file "
              "kai-system_db_password does not exist`\n  printed as a "
              "warning and the profile then failed to start a container\n  "
              "on the very same missing file.")
        return 1

    print("\nPASS: the bring-up printed no unaccounted warning.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
