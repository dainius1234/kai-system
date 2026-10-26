#!/usr/bin/env python3
"""A declared prerequisite must be IN FORCE where the service is started.

The doctrine this encodes, in one line:

> **Nothing is true because it was true last time.** A condition written
> in a compose file is a record of an intention. It becomes protection
> only at the moment something honours it.

THE FAILURE THAT EARNED IT
==========================

`docker-compose.full.yml` declares:

    memu-graph:
      depends_on:
        ollama-pull:
          condition: service_completed_successfully

That is exactly right, and it was in the tree the whole time. Stage-1
attempt 2 (run 31906667051) then started `memu-graph` with `--no-deps`,
which tells Compose to ignore it. The model pull had been alive 0.49
seconds when the first request went out; all ten replays returned HTTP
404 and the run went green over an experiment that measured nothing.

So this was never a missing gate. **It was a declared gate, bypassed at
the call site** — which is worse, because the declaration reads as
protection to everyone who greps for it.

`check_depends_on_readiness.py` already proves every `depends_on`
*declares* a condition. Its own founding lesson was *"the fix's scope was
one file; the class's scope was the tree."* It then inherited that shape
one level up: its scope is the DECLARATION, and the class's scope is the
EXECUTION. This file is the missing half.

WHY THIS DOES NOT BAN `--no-deps`
=================================

Two of the three bypasses measured when this was written are **correct**.
The Stage-1 output preflight and the model-readiness probe deliberately
must not start the stack; that is the point of them. A rule that banned
the flag would be a scope wider than reality — the inverted form of the
same defect, which reports failure over things that are right (R5).

So the rule is about KNOWING, not forbidding, and it borrows the shape
that already works here (`check_ci_tolerations.py`):

> A site that bypasses a declared condition must say **which** condition
> it bypasses and **what compensates for it**, with an owner and a review
> date. An undeclared bypass fails.

UNRESOLVED IS NOT CLEAN
=======================

Some invocations name their compose file through a shell variable
(`-f "$COMPOSE_FILE"`). The declarations behind those cannot be resolved
by reading the tree, so they are reported **UNRESOLVED** and counted
separately. They are never folded into the clean total.

That distinction is in here because the first draft of this measurement
printed "none declared" for exactly those sites — unknown wearing the
clothes of clean, twenty minutes after the principle was written down.

Exit 0 = every bypass is declared, and nothing is unresolved.
Exit 1 = an undeclared bypass, or an unresolvable site.
"""
from __future__ import annotations

import glob
import pathlib
import re
import sys
from dataclasses import dataclass

import yaml

REPO = pathlib.Path(__file__).resolve().parent.parent.parent

# Conditions that make a real promise about readiness. `service_started`
# is deliberately included: it promises less, but it still promises.
REAL_CONDITIONS = ("service_healthy", "service_completed_successfully",
                   "service_started")

# Where an execution site can live. Derived by glob, not listed by hand.
SITE_GLOBS = (".github/workflows/*.yml", ".github/workflows/*.yaml",
              "scripts/**/*.sh", "Makefile")


@dataclass(frozen=True)
class Bypass:
    """A `--no-deps` site that knowingly skips a declared condition."""
    file: str
    service: str
    dependency: str
    reason: str
    owner: str
    review_by: str


# ── declared bypasses ────────────────────────────────────────────────
#
# Each entry says which condition is skipped and what stands in its
# place. "Nothing stands in its place" is a legitimate entry only if the
# skipped condition genuinely does not apply to what the site does.
DECLARED: tuple[Bypass, ...] = (
    Bypass(
        file=".github/workflows/stage1-replay.yml",
        service="memu-graph",
        dependency="ollama-pull",
        reason="The output-write preflight sends nothing to a model: it "
               "opens a container-local path, writes, reads back and "
               "deletes. A completed model pull is not a prerequisite "
               "for that, and starting the stack to prove a filesystem "
               "permission would cost minutes to measure something "
               "provable in milliseconds. COMPENSATION: none needed — "
               "the skipped condition does not bear on the claim.",
        owner="orion",
        review_by="2027-01-01",
    ),
    Bypass(
        file=".github/workflows/stage1-replay.yml",
        service="memu-graph",
        dependency="ollama-pull",
        reason="The model-readiness probe must run BEFORE the condition "
               "it is checking can be assumed, so it cannot wait on it. "
               "COMPENSATION: the probe is itself the enforcement — it "
               "queries the server's own inventory for an exact match "
               "with runtime.model and refuses if absent, which is a "
               "stronger claim than `the pull container exited 0`.",
        owner="orion",
        review_by="2027-01-01",
    ),
    Bypass(
        file=".github/workflows/stage1-replay.yml",
        service="memu-graph",
        dependency="ollama-pull",
        reason="The replay must not start the rest of the stack, which "
               "would change the conditions the capture was taken under. "
               "COMPENSATION: the condition is restored EXPLICITLY as "
               "two preceding steps — the pull runs in the foreground "
               "under `set -e` so its exit status gates the chain, and "
               "check_model_ready.py then proves the exact model is "
               "present. This is the bypass that cost run 31906667051 "
               "ten empty executions when nothing compensated for it.",
        owner="orion",
        review_by="2027-01-01",
    ),
)


def all_services(root: pathlib.Path = REPO) -> set[str]:
    """Every service name in every compose file.

    Resolving a site's service against only the services that HAVE
    conditions would make a service with none unresolvable rather than
    trivially clean -- a scope smaller than the check's name, which is
    the defect this whole file exists to catch. Derived from the tree.
    """
    names: set[str] = set()
    for cf in sorted(glob.glob(str(root / "docker-compose*.yml"))):
        try:
            doc = yaml.safe_load(pathlib.Path(cf).read_text())
        except Exception:  # noqa: BLE001
            continue
        names |= set(((doc or {}).get("services") or {}))
    return names


_ASSIGN = re.compile(r'^\s*([A-Z_][A-Z0-9_]*)="?([^"\n$]+?)"?\s*$', re.M)


def literals_in(text: str) -> dict[str, str]:
    """Simple `VAR="literal"` assignments, so `-f "$COMPOSE_FILE"` resolves.

    Only unconditional literals with no interpolation. A value built from
    another variable stays UNRESOLVED rather than being guessed at.
    """
    return {m.group(1): m.group(2).strip() for m in _ASSIGN.finditer(text)}


def declarations(root: pathlib.Path = REPO) -> dict[str, dict[str, dict[str, str]]]:
    """compose file -> service -> {dependency: condition}, from the tree."""
    out: dict[str, dict[str, dict[str, str]]] = {}
    for cf in sorted(glob.glob(str(root / "docker-compose*.yml"))):
        rel = str(pathlib.Path(cf).relative_to(root))
        try:
            doc = yaml.safe_load(pathlib.Path(cf).read_text())
        except Exception as exc:  # noqa: BLE001 — a datum, not a crash
            out[rel] = {"__unreadable__": {"error": f"{type(exc).__name__}"}}
            continue
        svcs: dict[str, dict[str, str]] = {}
        for svc, body in ((doc or {}).get("services") or {}).items():
            dep = (body or {}).get("depends_on")
            if isinstance(dep, dict):
                conds = {k: (v or {}).get("condition")
                         for k, v in dep.items() if isinstance(v, dict)}
                real = {k: v for k, v in conds.items() if v in REAL_CONDITIONS}
                if real:
                    svcs[svc] = real
        out[rel] = svcs
    return out


_INVOCATION = re.compile(r"docker\s+compose\b([^\n]*?)\brun\b([^\n]*)")


def sites(known_services: set[str], root: pathlib.Path = REPO) -> list[dict]:
    """Every `docker compose run` invocation, with what it names."""
    found: list[dict] = []
    paths: list[pathlib.Path] = []
    for pattern in SITE_GLOBS:
        paths += [pathlib.Path(p) for p in
                  glob.glob(str(root / pattern), recursive=True)]
    for path in sorted(set(paths)):
        raw = path.read_text(errors="replace")
        env = literals_in(raw)
        # Join backslash continuations: a multi-line invocation is one call.
        text = re.sub(r"\\\s*\n\s*", " ", raw)
        for m in _INVOCATION.finditer(text):
            head, tail = m.group(1), m.group(2)
            if "--no-deps" not in tail:
                continue
            compose = None
            fm = re.search(r"-f\s+(\S+)", head)
            if fm:
                compose = fm.group(1).strip("\"'")
                # `-f "$COMPOSE_FILE"` where the file is assigned a plain
                # literal in the same script is resolvable. Anything that
                # is not stays unknown.
                var = re.fullmatch(r"\$\{?([A-Z_][A-Z0-9_]*)\}?", compose)
                if var:
                    compose = env.get(var.group(1), compose)
            service = next((t for t in tail.split() if t in known_services),
                           None)
            found.append({
                "file": str(path.relative_to(root)),
                "line": text[:m.start()].count("\n") + 1,
                "compose": compose,
                "service": service,
            })
    return found


def main(root: pathlib.Path = REPO,
         declared: tuple[Bypass, ...] = DECLARED) -> int:
    decls = declarations(root)
    known = all_services(root)
    all_sites = sites(known, root)

    print("DECLARED PREREQUISITES MUST BE IN FORCE")
    print("=" * 64)
    total_conditions = sum(len(c) for f in decls.values() for c in f.values())
    print(f"  compose file(s)          : {len(decls)}")
    print(f"  service(s) with a real depends_on condition: "
          f"{sum(len(f) for f in decls.values())}")
    print(f"  declared condition(s)    : {total_conditions}")
    print(f"  --no-deps execution site(s): {len(all_sites)}")
    print()

    undeclared: list[dict] = []
    unresolved: list[dict] = []
    accounted = 0

    for site in all_sites:
        compose = site["compose"]
        # A compose file named through a shell variable cannot be resolved
        # from the tree. That is UNKNOWN, and unknown is not clean.
        if compose is None or "$" in compose or not (root / compose).is_file():
            unresolved.append(site)
            continue
        if site["service"] is None:
            unresolved.append(site)
            continue
        skipped = decls.get(compose, {}).get(site["service"], {})
        if not skipped:
            accounted += 1          # nothing declared, nothing bypassed
            continue
        for dependency in skipped:
            match = [d for d in declared
                     if d.file == site["file"] and d.service == site["service"]
                     and d.dependency == dependency]
            if match:
                accounted += 1
            else:
                undeclared.append(dict(site, dependency=dependency,
                                       condition=skipped[dependency]))

    if unresolved:
        print(f"UNRESOLVED — {len(unresolved)} site(s) whose declarations "
              f"cannot be read from the tree:")
        for s in unresolved:
            why = ("compose file is a variable" if s["compose"]
                   and "$" in s["compose"] else
                   "no compose file named" if not s["compose"] else
                   "service not resolvable")
            print(f"  - {s['file']}:{s['line']}  ({why}: "
                  f"{s['compose']!r}, service {s['service']!r})")
        print("    Unknown is NOT clean. Resolve the file, or declare the")
        print("    site so the bypass is at least visible.")
        print()

    if undeclared:
        print(f"UNDECLARED BYPASS — {len(undeclared)} declared condition(s) "
              f"skipped with nothing said about it:")
        for s in undeclared:
            print(f"  - {s['file']}:{s['line']}  {s['service']} "
                  f"--no-deps skips {s['dependency']}: {s['condition']}")
        print()
        print("  A declaration that is bypassed reads as protection while")
        print("  not being in force. Say which condition is skipped and")
        print("  what compensates for it — reason, owner, review date.")
        print()

    print(f"  inspected: {len(all_sites)} bypass site(s) against "
          f"{total_conditions} declared condition(s)")
    if undeclared or unresolved:
        print(f"FAIL: {len(undeclared)} undeclared, {len(unresolved)} "
              f"unresolved.")
        return 1
    print(f"PASS: every bypass is declared ({accounted} accounted for, "
          f"{len(declared)} declaration(s) on file).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
