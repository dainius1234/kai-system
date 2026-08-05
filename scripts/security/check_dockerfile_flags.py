#!/usr/bin/env python3
"""A Dockerfile instruction flag is spelled with hyphens, never underscores.

`core-tests.yml` reached its build step and died on this:

    target document-parser: failed to solve: dockerfile parse error on
    line 20: unknown flag: --start_period (did you mean start-period?)

    HEALTHCHECK --interval=30s --timeout=5s --start_period=20s --retries=3

One character. It stopped every image build, which stopped the bring-up,
the live smoke, the kill-isolation test, the restart-persistence test,
the memu-graph cycle and the sovereign boot — thirteen steps that then
reported nothing at all.

**It was invisible for a specific, structural reason.** `document-parser`
is one of nineteen services declared in `docker-compose.minimal.yml` and
**not** in `docker-compose.full.yml`, and the only build step in CI built
`full.yml`. Nothing ever built this image, so nothing ever parsed this
Dockerfile. The typo could have sat there indefinitely.

Docker's own parser catches it — but only at build time, twenty minutes
into a CI run, on a runner. This catches it in milliseconds, locally,
before the push.

The rule is exact rather than a list of known flags: **every flag Docker
accepts on an instruction uses hyphens.** `--platform`, `--from`,
`--chown`, `--chmod`, `--mount`, `--interval`, `--timeout`,
`--start-period`, `--start-interval`, `--retries`, `--link`,
`--parents`, `--exclude` — not one contains an underscore. Measured
before adopting: of the 35 period-flags in this repository, 34 were
already correct, so the rule costs nothing and catches the one that was
not.

A list of valid flag names would have been the other option, and it is
the shape this repository keeps finding broken: a hand-written list that
is true of what someone remembered and false of everything Docker adds
later.

Exit 0 = every flag is hyphenated.  Exit 1 = one is not.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import List, Tuple

REPO = Path(__file__).resolve().parent.parent.parent

sys.path.insert(0, str(REPO))

from scripts.security.gate_inputs import inspected  # noqa: E402

_EXCLUDED = {".git", ".venv", "venv", "_archive", "node_modules",
             "__pycache__", "site-packages"}

#: Instructions that take flags. A `--flag` anywhere else on a line (in a
#: RUN command, say) belongs to the program being run, not to Docker, and
#: `pip install --no-cache-dir` is not this gate's business.
_FLAGGED = ("ADD", "COPY", "FROM", "HEALTHCHECK", "RUN")

_FLAG = re.compile(r"--([A-Za-z][A-Za-z0-9_-]*)")


def dockerfiles() -> List[Path]:
    """Every Dockerfile in the tree, found by walking rather than listed.

    Derived from the tree for the same reason the compose gates are: a
    hand-written list would have missed `document-parser` exactly as the
    `full.yml` build did.
    """
    out = []
    for path in REPO.rglob("Dockerfile*"):
        if any(part in _EXCLUDED for part in path.parts):
            continue
        if path.is_file():
            out.append(path)
    return sorted(out)


def findings_in(text: str, name: str) -> List[str]:
    out: List[str] = []
    for line_no, line in enumerate(text.splitlines(), 1):
        stripped = line.strip()
        instruction = stripped.split(" ", 1)[0].upper() if stripped else ""
        if instruction not in _FLAGGED:
            continue
        # Only the flags before the sub-command belong to Docker.
        head = stripped.split(" CMD ")[0] if instruction == "HEALTHCHECK" \
            else stripped
        for flag in _FLAG.findall(head):
            if "_" not in flag:
                continue
            out.append(
                f"{name}:{line_no}: {instruction} --{flag} — Docker "
                f"instruction flags use hyphens, so this is rejected at "
                f"build time with 'unknown flag'. Did you mean "
                f"--{flag.replace('_', '-')}?")
    return out


def audit() -> Tuple[List[str], int, int]:
    """Return (findings, flags inspected, Dockerfiles read)."""
    findings: List[str] = []
    paths = dockerfiles()
    if not paths:
        # I-1. No Dockerfiles found means the walk is wrong, not that the
        # repository is clean.
        return (["no Dockerfiles found — this gate inspected nothing, "
                 "which is not a pass"], 0, 0)
    flags = 0
    for path in paths:
        try:
            text = path.read_text(encoding="utf-8")
        except OSError as exc:
            findings.append(f"{path.name}: unreadable ({exc})")
            continue
        for line in text.splitlines():
            stripped = line.strip()
            if stripped.split(" ", 1)[0].upper() in _FLAGGED:
                flags += len(_FLAG.findall(stripped))
        findings.extend(findings_in(text, str(path.relative_to(REPO))))
    return findings, flags, len(paths)


def main() -> int:
    findings, flags, files = audit()

    print(inspected(flags, "instruction flag(s)",
                    f"across {files} Dockerfiles"))
    print()
    if findings:
        print(f"FAIL: {len(findings)} misspelled flag(s):\n")
        for line in findings:
            print(f"  - {line}")
        print("\n  Docker rejects these at build time, which in this "
              "repository means\n  twenty minutes into a CI run — and only if "
              "something builds that\n  image at all. `document-parser` is in "
              "the minimal profile and not\n  the full one, so nothing ever "
              "built it and nothing ever parsed it.")
        return 1
    print(f"PASS: all {flags} instruction flag(s) across {files} "
          f"Dockerfiles are hyphenated.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
