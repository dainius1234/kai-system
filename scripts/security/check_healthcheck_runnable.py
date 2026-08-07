#!/usr/bin/env python3
"""A healthcheck must invoke a binary the image actually contains.

The defect
----------

`docker-compose.sovereign.yml` declared, for ten services:

    test: ["CMD-SHELL", "wget -qO- http://localhost:8000/health || exit 1"]

The images are `python:3.11-slim`. **No Dockerfile in this repository
installs `wget` or `curl`.** So the check could never succeed: `wget:
not found` exits non-zero, the container is marked unhealthy forever,
and every dependent service waits on a readiness signal that will never
arrive.

Found on 2026-08-07, when the sovereign profile finally got far enough
to run. Its two core services were *fine* —

    sovereign-memu-core | INFO: Application startup complete.
    sovereign-memu-core | INFO: Uvicorn running on http://0.0.0.0:8001
    sovereign-tool-gate | INFO: Application startup complete.

— and the step still failed after exactly 180s, the wait timeout. The
services were healthy the whole time; the instrument measuring them was
broken. That is this programme's subject exactly, moved into the
healthcheck: **a probe that reports failure over something that is
right.**

And the shape is familiar. The same two services in `minimal` and
`full`, and the `HEALTHCHECK` in their own Dockerfiles, all use

    python -c "import urllib.request; urllib.request.urlopen(...)"

Three profiles, one fact, one copy different — and the different one had
never run.

Scope
-----

Only services this repository **builds**. A service running somebody
else's image (`postgres`, `redis`) is entitled to use the binaries that
image ships — `pg_isready` and `redis-cli` are correct there and absent
here, and flagging them would be a finding against something that works.

`provided` is derived: the interpreters a `python:*` base guarantees,
plus POSIX shell builtins, plus anything a Dockerfile `apt-get install`s.
Nothing is hard-coded as forbidden; `wget` is reported because nothing
provides it, not because it is on a list.

Exit 0 = every healthcheck can run in the image it checks.
"""
from __future__ import annotations

import re
import shlex
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple

REPO = Path(__file__).resolve().parent.parent.parent

sys.path.insert(0, str(REPO))

from scripts.security.gate_inputs import (  # noqa: E402
    built_services, compose_files, inspected)

#: Always present in a POSIX shell, so never a missing binary.
_SHELL_BUILTINS = {"sh", "bash", "test", "[", "echo", "exit", "true",
                   "false", "cd", "set", "printf", ":"}

_APT = re.compile(r"apt-get\s+install[^\n]*", re.M)


def provided_by(dockerfile: Path):
    """Executables the image can be relied on to have, or None.

    `None` means the Dockerfile could not be read, so the question is
    unanswerable. The first draft returned the shell builtins in that
    case, which is wrong in the *over*-reporting direction: every
    `python` healthcheck would have been reported as calling a missing
    binary. Guessing generously would be wrong the other way. A gate
    that cannot see the image says so — `audit` turns it into a finding
    naming the file, not a verdict about the healthcheck.
    """
    # Positive condition on purpose. Written as `if not
    # dockerfile.exists(): return None` this behaves identically — the
    # caller turns None into a finding — but it matches the shape the
    # meta-check scans for, and a reader has to follow the return value
    # to another function to see that absence is refused rather than
    # skipped. Structure that states the rule beats structure that
    # merely implements it.
    if not dockerfile.is_file():
        return None
    out = set(_SHELL_BUILTINS)
    text = dockerfile.read_text(encoding="utf-8")
    base = re.search(r"^FROM\s+(\S+)", text, re.M)
    if base and "python" in base.group(1):
        out |= {"python", "python3", "pip", "pip3"}
    for line in _APT.findall(text):
        for token in line.split():
            if token.startswith("-") or token in {"apt-get", "install"}:
                continue
            out.add(token)
    return out


def commands_in(test) -> List[str]:
    """The executables a healthcheck `test:` actually invokes."""
    if not test:
        return []
    parts = list(test) if isinstance(test, list) else [str(test)]
    if parts and parts[0] in {"CMD", "CMD-SHELL", "NONE"}:
        parts = parts[1:]
    body = " ".join(parts)
    try:
        tokens = shlex.split(body)
    except ValueError:
        tokens = body.split()

    # **Tokenise before splitting on separators, not after.** The first
    # draft did `re.split(r"...;...", body)` and the healthcheck it was
    # meant to bless is
    #
    #     python -c "import urllib.request; urllib.request.urlopen(...)"
    #
    # whose `;` is *inside the quoted Python*. That split the string
    # mid-argument and reported `urllib.request.urlopen(...)"` as a
    # missing binary — 69 findings against 49 images, on a tree that
    # was already correct.
    #
    # A gate with false positives sends people to break working code and
    # buries the true finding, which is the defect this file exists to
    # catch. `shlex` knows what a quote is; a regex over the raw string
    # does not.
    separators = {"||", "&&", ";", "|"}
    out: List[str] = []
    expect_command = True
    for token in tokens:
        if token in separators:
            expect_command = True
            continue
        if not expect_command:
            continue
        expect_command = False
        if "=" in token and not token.startswith("/"):
            expect_command = True       # VAR=value prefix; command follows
            continue
        out.append(Path(token).name)
    return out


def audit(root: Path = None) -> Tuple[List[str], int, int]:
    """Return (findings, healthchecks inspected, built services seen)."""
    import yaml

    root = root or REPO
    images = built_services(root)
    if not images:
        # I-1: nothing discovered is not a clean bill of health.
        return ([f"{root}: no built services found — this gate inspected "
                 f"nothing and must not report success"], 0, 0)

    cache: Dict[str, Set[str]] = {}
    findings: List[str] = []
    checked = 0
    for path in compose_files(root):
        try:
            doc = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        except Exception:
            continue        # `check_ci_tolerations` owns unparseable files
        for name, cfg in (doc.get("services") or {}).items():
            test = ((cfg or {}).get("healthcheck") or {}).get("test")
            if not test or name not in images:
                continue    # not ours to hold to this rule
            checked += 1
            dockerfile = images[name]
            if name not in cache:
                cache[name] = provided_by(dockerfile)
            if cache[name] is None:
                findings.append(
                    f"{path.name}: `{name}` declares a healthcheck but its "
                    f"Dockerfile ({dockerfile}) could not be read, so what "
                    f"the image provides is unknown. Nothing was checked "
                    f"here, and an unread image is not a clean one.")
                continue
            for command in commands_in(test):
                if command in cache[name]:
                    continue
                findings.append(
                    f"{path.name}: `{name}` healthchecks with `{command}`, "
                    f"which {dockerfile.name} does not provide. The check "
                    f"can never pass, so the container is unhealthy forever "
                    f"and everything waiting on it waits for a signal that "
                    f"will not come — while the service itself is fine.")
    return findings, checked, len(images)


def main() -> int:
    findings, checked, images = audit()

    print(inspected(checked, "healthcheck(s) on services we build",
                    f"across {images} built service(s)"))
    print()

    if findings:
        print(f"FAIL: {len(findings)} healthcheck(s) cannot run:\n")
        for line in findings:
            print(f"  - {line}")
        print("\n  Ten sovereign healthchecks called `wget` against "
              "python:3.11-slim\n  images that never installed it. memu-core "
              "and tool-gate were both\n  logging 'Application startup "
              "complete' while compose reported them\n  unhealthy for 180 "
              "seconds and failed the boot.")
        return 1

    print(f"PASS: every healthcheck can run in the image it checks "
          f"({checked} inspected).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
