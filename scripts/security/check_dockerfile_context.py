#!/usr/bin/env python3
"""Every `COPY` source must exist in the build context that profile declares.

A Dockerfile's `COPY` paths are relative to the **build context**, not to
the Dockerfile. The same Dockerfile is therefore correct under one
profile and broken under another, and this repository had both.

Found on 2026-08-05, once `core-tests.yml` finally built the minimal
profile:

    target notify-service: failed to solve: failed to compute cache key:
    failed to calculate checksum of ref ...: "/requirements.txt": not found

Measured across all three profiles rather than fixed one at a time:

    docker-compose.full.yml        30 builds,  0 broken COPY
    docker-compose.minimal.yml     32 builds, 10 broken COPY
    docker-compose.sovereign.yml   12 builds, 39 broken COPY

`full.yml` was the only profile CI ever built, and it was the only one
that worked. Thirty services use `context: .` with root-relative paths
(`COPY tool-gate/app.py`). **Every build service in the sovereign
profile used `build: ./tool-gate`**, which makes the context the service
directory, so `COPY tool-gate/requirements.txt` looked for
`tool-gate/tool-gate/requirements.txt`.

**The sovereign profile — the one the architecture is named after — could
not build a single one of its services.** It appeared to work in CI
because the sovereign boot step runs `up -d` *without* `--build`, so it
silently reused images the earlier `full.yml` build had produced under
the same compose project name. A step that verifies the sovereign
profile was verifying images from a different profile.

Three Dockerfiles were worse than context-mismatched: `COPY ../../common`
escapes the context, which Docker rejects under **any** context. Those
images could not build in any configuration at all.

Three rules, all exact:

  - a `COPY` source must exist under the declared context;
  - a `COPY` source must not begin with `../`, ever;
  - a `COPY` source must not be excluded by `.dockerignore`.

The third exists because the second fix created the risk. With ~50
services building from `context: .` and no `.dockerignore`, every build
shipped the whole repository — 63 MB of `.git` included — to the daemon.
Adding one is a large win and a new way to break every build at once: an
excluded path fails at `COPY`, not at parse, so it would surface exactly
where this class always surfaces, twenty minutes into a run. Checked
here instead, against all 110 COPY sources.

Globs are skipped rather than guessed at — resolving them needs the same
matching Docker does, and a wrong answer here would report a defect
against a working build.

Exit 0 = every COPY resolves.  Exit 1 = one does not.
"""
from __future__ import annotations

import fnmatch
import sys
from pathlib import Path
from typing import List, Tuple

REPO = Path(__file__).resolve().parent.parent.parent

sys.path.insert(0, str(REPO))

from scripts.security.gate_inputs import inspected, require  # noqa: E402

PROFILES = (
    "docker-compose.full.yml",
    "docker-compose.minimal.yml",
    "docker-compose.sovereign.yml",
)


def build_target(cfg: dict) -> Tuple[str, str] | None:
    """(context, dockerfile) for a service, in every spelling used here."""
    build = (cfg or {}).get("build")
    if not build:
        return None
    if isinstance(build, str):
        directory = build.strip("./") or "."
        return directory, f"{directory}/Dockerfile"
    context = str(build.get("context") or ".").strip("./") or "."
    dockerfile = str(build.get("dockerfile") or "").strip("./")
    return context, (dockerfile or f"{context}/Dockerfile")


def copy_sources(text: str) -> List[Tuple[int, str]]:
    """(line number, source) for every COPY, excluding --flags and the target."""
    out: List[Tuple[int, str]] = []
    for line_no, line in enumerate(text.splitlines(), 1):
        stripped = line.strip()
        if not stripped.upper().startswith("COPY "):
            continue
        parts = [p for p in stripped.split()[1:] if not p.startswith("--")]
        if len(parts) < 2:
            continue                       # malformed or a JSON-form COPY
        for source in parts[:-1]:           # the last token is the target
            out.append((line_no, source))
    return out


def dockerignore_patterns(base: Path) -> Tuple[List[str], set]:
    """(exclusions, negations) from `.dockerignore`, or empty if absent.

    Absence is not a finding — a repository may legitimately have none.
    It only becomes one when a COPY source collides with a pattern.
    """
    path = base / ".dockerignore"
    if not path.exists():
        return [], set()
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()
             if line.strip() and not line.startswith("#")]
    return ([p for p in lines if not p.startswith("!")],
            {p[1:] for p in lines if p.startswith("!")})


def excluded_by(source: str, excludes: List[str], negations: set) -> str:
    """The pattern that would drop this source from the context, or ''."""
    if source in negations:
        return ""
    head = source.rstrip("/").split("/")[0]
    for pattern in excludes:
        if (head == pattern or fnmatch.fnmatch(source, pattern)
                or fnmatch.fnmatch(head, pattern)):
            return pattern
    return ""


def audit(root: Path = None) -> Tuple[List[str], int, int]:
    """Return (findings, COPY sources checked, build definitions read).

    `root` exists so this can be driven against a synthetic tree. Without
    it the tests would have to patch a module global that `require()`
    does not consult — which is exactly what they did first, and every
    synthetic assertion silently read the real repository instead.
    """
    import yaml

    findings: List[str] = []
    base = root or REPO
    paths = ([base / name for name in PROFILES if (base / name).exists()]
             if root else require(PROFILES))
    checked = 0
    builds = 0
    unreadable: List[str] = []
    excludes, negations = dockerignore_patterns(base)
    for path in paths:
        doc = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        for service, cfg in sorted((doc.get("services") or {}).items()):
            target = build_target(cfg)
            if target is None:
                continue
            context, dockerfile = target
            dockerfile_path = base / dockerfile
            if not dockerfile_path.exists():
                # Recorded and printed, not skipped. `check_dockerfile_
                # coverage` owns the *finding* for a dangling reference,
                # but a build this gate could not read must still be
                # subtracted from what it claims to have checked —
                # otherwise the denominator says it inspected something
                # it never opened. Flagged by the meta-check's I-1 scan
                # the first time this ran, and the scanner was right.
                unreadable.append(f"{path.name}: {service} -> {dockerfile}")
                continue
            builds += 1
            ctx_root = base if context == "." else base / context
            for line_no, source in copy_sources(
                    dockerfile_path.read_text(encoding="utf-8")):
                if "*" in source or "?" in source:
                    continue
                checked += 1
                if source.startswith("../"):
                    findings.append(
                        f"{path.name}: {service} — {dockerfile}:{line_no} "
                        f"COPY {source} escapes the build context. Docker "
                        f"rejects this under any context, so this image "
                        f"cannot build at all.")
                elif context == "." and excluded_by(source, excludes, negations):
                    findings.append(
                        f"{path.name}: {service} — {dockerfile}:{line_no} "
                        f"COPY {source} is excluded from the context by "
                        f"`.dockerignore` pattern "
                        f"'{excluded_by(source, excludes, negations)}'. The "
                        f"file exists; the build will still fail at COPY, "
                        f"because Docker never receives it.")
                elif not (ctx_root / source).exists():
                    findings.append(
                        f"{path.name}: {service} — {dockerfile}:{line_no} "
                        f"COPY {source} does not exist under the declared "
                        f"context '{context}'. The same Dockerfile may be "
                        f"correct in another profile; COPY is relative to "
                        f"the context, not to the Dockerfile.")
    if unreadable:
        findings.append(
            f"{len(unreadable)} build definition(s) name a Dockerfile "
            f"that is not in the tree, so their COPY sources were "
            f"never checked: {', '.join(sorted(unreadable)[:4])}"
            + (' ...' if len(unreadable) > 4 else ''))
    return findings, checked, builds


def main() -> int:
    findings, checked, builds = audit()

    print(inspected(checked, "COPY source(s)",
                    f"across {builds} build definitions in {len(PROFILES)} "
                    f"profiles"))
    print()
    if findings:
        print(f"FAIL: {len(findings)} unresolvable COPY source(s):\n")
        for line in findings:
            print(f"  - {line}")
        print("\n  COPY is relative to the build CONTEXT. On 2026-08-05 the "
              "sovereign\n  profile could not build one of its nine services "
              "for this reason,\n  and looked healthy because its boot step "
              "ran `up -d` without\n  `--build` and reused images another "
              "profile had made.")
        return 1
    print(f"PASS: all {checked} COPY source(s) resolve in every profile "
          f"that builds them.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
