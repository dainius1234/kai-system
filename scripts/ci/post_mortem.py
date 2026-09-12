#!/usr/bin/env python3
"""Print the captured step logs — but only the ones that have something.

The defect
----------

The post-mortem printed thirteen sections in a fixed order, whether or
not they had content, and the image build was **first**. When the build
fails, every step after it is skipped, so twelve of those sections say
"(did not run)" — and the GitHub Actions log API serves a **fixed byte
window from the end** (measured: identical 15,780 characters for two
different `tail_lines`). The twelve empty sections plus the teardown
chatter after them pushed the one section that mattered out of the
window.

Observed on run 712, 2026-08-07: `memu-core` was missing from the
"images built" list and step 45 had taken 10.5 minutes instead of 2.3,
but the build's own output could not be retrieved from the run at all.
The diagnostic had defeated itself by printing noise.

`dump_unhealthy.py` already applies the remedy to container dumps —
print only what is wrong. The post-mortem never got the same treatment,
which is the "fix has a denominator" rule missing a member again.

What this does
--------------

Prints a section only when its log file exists **and has content**,
names the empty ones in a single line rather than a section each, and
reports the denominator so a silent post-mortem is visibly a
post-mortem that found nothing rather than one that ran nothing.

Order is deliberate: build logs last. A build failure aborts everything
downstream, so when the build is the only section with content it lands
at the very bottom of the window, which is the part guaranteed to be
served.

Exit 0 always — this is a diagnostic. It must not add a second failure
on top of the one being diagnosed.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Tuple

#: (label, path, how many trailing lines are worth printing)
#:
#: Ordered least-to-most likely to be the actual cause. Everything that
#: only runs *after* a successful build comes first; the build itself is
#: last, because if it failed nothing else ran and its output is the
#: whole answer.
SECTIONS: List[Tuple[str, str, int]] = [
    ("restart-persistence", "/tmp/restart-persistence.log", 25),
    ("kill-isolation", "/tmp/kill-isolation.log", 15),
    ("live smoke", "/tmp/live-smoke.log", 25),
    ("full profile live smoke", "/tmp/full-smoke.log", 30),
    ("full profile bring-up", "/tmp/full-bringup.log", 20),
    ("full profile: containers Docker says are wrong",
     "/tmp/full-logs.log", 45),
    ("sovereign boot", "/tmp/sovereign-boot.log", 30),
    ("sovereign: containers Docker says are wrong",
     "/tmp/sovereign-logs.log", 45),
    ("minimal bring-up", "/tmp/bringup.log", 20),
    ("minimal: containers Docker says are wrong",
     "/tmp/minimal-logs.log", 30),
    ("the full image build", "/tmp/build-full.log", 8),
    ("the minimal image build", "/tmp/build-minimal.log", 40),
]


def section_body(path: str, lines: int) -> str | None:
    """The last *lines* lines of *path*, or None if there is nothing.

    None covers three cases that must not be told apart here — the file
    is absent, empty, or only whitespace — because all three mean the
    same thing to a reader: this step produced no output worth showing.
    Written positively (`if text.strip()`) rather than as a bare
    existence guard, so absence is a decision rather than a skip.
    """
    p = Path(path)
    if not p.is_file():
        return None
    try:
        text = p.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    if not text.strip():
        return None
    return "\n".join(text.splitlines()[-lines:])


def main(sections=None) -> int:
    sections = sections or SECTIONS
    printed: List[str] = []
    empty: List[str] = []

    rendered: List[str] = []
    for label, path, lines in sections:
        body = section_body(path, lines)
        if body is None:
            empty.append(label)
            continue
        printed.append(label)
        rendered.append(f"── {label} (its real output) ──\n{body}")

    print(f"  inspected: {len(sections)} captured step log(s); "
          f"{len(printed)} had output, {len(empty)} were empty or absent")
    if empty:
        # One line, not a section each. Twelve "(did not run)" headers is
        # what evicted the real output from the log window.
        print(f"  no output (step did not run, or ran silently): "
              f"{', '.join(empty)}")
    print()

    if not printed:
        # I-1: nothing found is not a clean bill of health. A post-mortem
        # that prints nothing must say so, or it reads as "no problem".
        print("  NOTHING WAS CAPTURED. Every step log is absent or empty, "
              "which means\n  the job failed before any instrumented step "
              "ran — look at the step list\n  for the first red step, not "
              "at this output.")
        return 0

    for block in rendered:
        print(block)

    return 0


if __name__ == "__main__":
    sys.exit(main())
