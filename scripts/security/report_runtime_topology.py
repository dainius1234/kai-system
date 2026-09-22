#!/usr/bin/env python3
"""Runtime topology: what is DEFINED, what is GATED, what actually STARTS.

Task #41's denominator, derived from the tree rather than remembered. The
task carried the figure 26 for weeks; re-deriving it produced 34, and the
gap was never a counting error — it was that nobody had asked the tree.

SIX DISTINCTIONS, KEPT APART ON PURPOSE
=======================================

Collapsing any adjacent pair produces a confident wrong answer, and this
programme has now paid for two of them:

    defined                    a `services:` key exists somewhere
    profile-gated              declared with `profiles:` in every file
                               that defines it
    profile-set-enabled        some invocation selects its profile via
                               `--profile` / `COMPOSE_PROFILES`
    individually startable      some invocation names it explicitly --
                               which DOES start a gated service, because
                               Compose enables a service's own profiles
                               when it is targeted by name
    runtime-proven             something has actually executed inside it
    expected by a live caller  an ACTIVE service references its URL

`memu-graph` is why rows 3 and 4 are separate: it is gated
(`introspection`) and `core-tests.yml` names it, so it starts without any
profile ever being selected. A first version of this analysis reported it
as "on the dangerous list but not gated" — a false positive produced by
conflating *gated* with *never started*, caught before it reached the
record.

RUNTIME-PROVEN IS NOT DERIVABLE HERE, AND SAYS SO
=================================================

There is no machine-readable evidence record in this repository, so this
report cannot compute that column and does not pretend to. It prints
UNKNOWN and names the gap. That absence is itself the finding registered
as task #51 — evidence lives in prose, so it cannot be joined to anything.

WHY THE SECURITY GATE MATTERS TO A TOPOLOGY REPORT
==================================================

`check_default_profiles.py` requires consequential services to be behind
a profile: only the contained core may start under a bare
`docker compose up`. So "never started" is very often the CORRECT and
INTENDED state, not a defect. A report that counted never-started
services without that distinction would read as 34 broken components when
most are deliberately isolated.

This also surfaces that gate's own scope: its `DANGEROUS_SERVICES` is a
hand-written tuple of names beside the thing, and the tree is the
authority on what is gated. Reported, never repaired here.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO))

from scripts.security.gate_inputs import inspected, require  # noqa: E402

try:
    import yaml
except ImportError:                                          # pragma: no cover
    print("ERROR: PyYAML required.", file=sys.stderr)
    sys.exit(2)

COMPOSE_FILES = (
    "docker-compose.full.yml",
    "docker-compose.minimal.yml",
    "docker-compose.sovereign.yml",
)

#: Where a repo-defined path could start something. Derived by glob, not
#: enumerated: a hand-written list here would be the same defect the
#: report exists to describe.
def invocation_sources(root: Path) -> List[Path]:
    out = [p for p in (root / "Makefile",) if p.exists()]
    out += sorted((root / ".github" / "workflows").glob("*.y*ml"))
    out += sorted((root / "scripts").rglob("*.sh"))
    return out


# Anything may sit between `docker compose` and `up` — `-f FILE`, and
# `--profile NAME`, which is the very flag this report exists to look
# for. An earlier version anchored `up` directly after the optional
# `-f FILE` and therefore could not see
# `docker compose -f X --profile recovery up -d` AT ALL: the one form
# that would disprove its headline finding was the one form it could not
# parse. Found by the known-positive in scripts/test_runtime_topology.py,
# which is the entire reason that case exists.
_UP = re.compile(
    r"docker compose\s+(?P<pre>[^|>&\n]*?)\bup\b\s*(?P<rest>[^\n|>&]*)"
)
_FILE = re.compile(r"-f\s+(\S+)")
_PROFILE_ENV = re.compile(r"COMPOSE_PROFILES=([\w,*]+)")
_PROFILE_FLAG = re.compile(r"--profile[= ](\S+)")
_URL = re.compile(r"https?://([a-z0-9][a-z0-9-]{1,40}):(\d{2,5})")
_TARGET = re.compile(r'"([a-z0-9][a-z0-9-]{1,40}):(\d{2,5})"')


def load(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def definitions(root: Path, files) -> Tuple[Dict[str, Dict], Dict[str, Dict]]:
    """(service -> {file: [profiles]}, file -> {service: [depends_on]})."""
    defined: Dict[str, Dict] = {}
    deps: Dict[str, Dict] = {}
    for p in files:
        services = load(p).get("services") or {}
        deps[p.name] = {n: list((s or {}).get("depends_on") or [])
                        for n, s in services.items()}
        for name, spec in services.items():
            spec = spec or {}
            defined.setdefault(name, {})[p.name] = list(spec.get("profiles") or [])
    return defined, deps


def _closure(deps: Dict[str, List[str]], seed: Set[str]) -> Set[str]:
    out: Set[str] = set()
    stack = list(seed)
    while stack:
        n = stack.pop()
        if n in out:
            continue
        out.add(n)
        stack.extend(deps.get(n, []))
    return out


def invocations(root: Path, files) -> List[Dict]:
    """Every `docker compose … up`, parsed. Comments start nothing."""
    known = {p.name for p in files}
    found: List[Dict] = []
    for src in invocation_sources(root):
        try:
            text = src.read_text(encoding="utf-8")
        except OSError:
            continue
        for line in text.splitlines():
            if line.lstrip().startswith("#"):
                continue
            m = _UP.search(line)
            if not m:
                continue
            pre = m.group("pre")
            fmatch = _FILE.search(pre)
            fname = Path(fmatch.group(1) if fmatch else "docker-compose.yml").name
            if fname not in known:
                continue
            names = [
                w for w in m.group("rest").split()
                if not w.startswith(("-", "$")) and w not in {"\\", "2>&1", "|", "tee"}
            ]
            # `--profile` may appear on either side of `up`; the env form
            # only ever precedes the command.
            profiles = (set(_PROFILE_ENV.findall(line))
                        | set(_PROFILE_FLAG.findall(pre))
                        | set(_PROFILE_FLAG.findall(m.group("rest"))))
            found.append({
                "source": str(src.relative_to(root)),
                "file": fname,
                "named": names,
                "profiles": sorted(profiles),
            })
    return found


def started_by(root: Path, files, invs) -> Tuple[Dict[str, Set[str]], Dict[str, Set[str]]]:
    """(service -> invocation sources, service -> how it was reached).

    `how` is `named` or `default-profile` or `profile-set`, which is the
    distinction the whole report exists to preserve.
    """
    _, deps = definitions(root, files)
    by_name = {p.name: p for p in files}
    started: Dict[str, Set[str]] = {}
    how: Dict[str, Set[str]] = {}
    for inv in invs:
        services = load(by_name[inv["file"]]).get("services") or {}
        d = deps[inv["file"]]
        selected = set(inv["profiles"])
        if inv["named"]:
            seed = set(inv["named"]) & set(services)
            reached = _closure(d, seed)
            for n in reached:
                how.setdefault(n, set()).add("named" if n in seed else "dependency")
        else:
            seed = set()
            for n, s in services.items():
                prof = set((s or {}).get("profiles") or [])
                if not prof:
                    seed.add(n)
                    how.setdefault(n, set()).add("default-profile")
                elif "*" in selected or (prof & selected):
                    seed.add(n)
                    how.setdefault(n, set()).add("profile-set")
            reached = _closure(d, seed)
            for n in reached - seed:
                how.setdefault(n, set()).add("dependency")
        for n in reached:
            started.setdefault(n, set()).add(inv["source"])
    return started, how


def expecters(root: Path, defined) -> Dict[str, Set[str]]:
    """service -> files that reference its URL, excluding self-reference."""
    sources = [p for p in root.rglob("*.py") if "__pycache__" not in p.parts]
    sources += [p for p in (root / "prometheus.yml",) if p.exists()]
    out: Dict[str, Set[str]] = {}
    for p in sources:
        try:
            text = p.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        rel = str(p.relative_to(root))
        for pattern in (_URL, _TARGET):
            for host, _port in pattern.findall(text):
                if host in defined and not rel.startswith(f"{host}/"):
                    out.setdefault(host, set()).add(rel)
    return out


def owner_of(path: str, defined) -> str | None:
    top = path.split("/")[0]
    if top in defined:
        return top
    return "prometheus" if path == "prometheus.yml" and "prometheus" in defined else None


def gate_scope(defined) -> Tuple[Set[str], Set[str], Set[str]]:
    """(hand-written list, gated-everywhere, gated-but-unlisted).

    The P0 containment gate keeps its population as a tuple of names; the
    tree is the authority on what is actually gated. Surfaced here rather
    than repaired, because deciding whether a newly-gated service is
    *consequential* is a judgement and not a derivation.
    """
    from scripts.security.check_default_profiles import DANGEROUS_SERVICES
    gated = {n for n, files in defined.items() if all(files.values())}
    return set(DANGEROUS_SERVICES), gated, gated - set(DANGEROUS_SERVICES)


def survey(root: Path = REPO):
    files = [root / f for f in COMPOSE_FILES]
    defined, _ = definitions(root, files)
    invs = invocations(root, files)
    started, how = started_by(root, files, invs)
    expected = expecters(root, defined)

    rows = []
    for name in sorted(defined):
        per_file = defined[name]
        gated_everywhere = all(per_file.values())
        live = sorted(
            p for p in expected.get(name, ())
            if (o := owner_of(p, defined)) and o in started
        )
        rows.append({
            "service": name,
            "profiles": sorted({pr for pl in per_file.values() for pr in pl}),
            "gated_everywhere": gated_everywhere,
            "started": name in started,
            "how": sorted(how.get(name, ())),
            "expected_by": sorted(expected.get(name, ())),
            "live_expecters": live,
            # EVIDENCE STATUS, NOT HISTORY. "UNKNOWN" means this
            # instrument cannot establish whether anything has executed
            # inside the service -- there is no machine-readable evidence
            # record to join to. It does NOT mean "never executed", and
            # it must not decay into that reading: several of these have
            # demonstrably run, in CI and by hand, and the evidence lives
            # in prose nobody can query.
            "runtime_proven": "UNKNOWN (evidence status: no machine-readable record to join)",
        })
    return rows, invs, defined


def main() -> int:
    require(COMPOSE_FILES)
    rows, invs, defined = survey()

    definitions_count = sum(len(load(REPO / f).get("services") or {})
                            for f in COMPOSE_FILES)
    print(inspected(definitions_count, "service definition(s)",
                    f"across {len(COMPOSE_FILES)} compose files"))

    started = [r for r in rows if r["started"]]
    gated = [r for r in rows if r["gated_everywhere"]]
    never = [r for r in rows if not r["started"]]
    orphaned = [r for r in never if r["live_expecters"]]
    profile_sets = sorted({p for i in invs for p in i["profiles"]})

    print(f"\n  DEFINED                       {len(rows)}")
    print(f"  PROFILE-GATED everywhere      {len(gated)}")
    print(f"  STARTED by a repo path        {len(started)}")
    print(f"  NEVER-STARTED                 {len(never)}")
    print(f"  ...expected by a LIVE caller  {len(orphaned)}")
    print(f"  RUNTIME-PROVEN                UNKNOWN for all {len(rows)} — this is "
          f"an EVIDENCE STATUS,\n"
          f"                                not a history. It means no "
          f"machine-readable record\n"
          f"                                exists to join to (task #51), NOT "
          f"that these never ran.")

    print(f"\n  `compose up` invocations found: {len(invs)}")
    print(f"  profile SETS selected by any of them: "
          f"{profile_sets if profile_sets else 'NONE'}")
    if not profile_sets:
        print("\n  No repo-defined path has exercised the intended explicit")
        print("  profile activation mechanism AS A PROFILE SET. Stated this")
        print("  narrowly on purpose: individually naming a gated service DOES")
        print("  start it, because Compose enables a service's own profiles")
        print("  when it is targeted by name. The services below prove it.")
        named_gated = [r["service"] for r in rows
                       if r["gated_everywhere"] and "named" in r["how"]]
        print(f"  gated services started by being NAMED: "
              f"{named_gated if named_gated else '(none)'}")

    listed, gated_names, unlisted = gate_scope(defined)
    print(f"\n  P0 containment gate scope: {len(listed)} names declared in "
          f"DANGEROUS_SERVICES, {len(gated_names)} services gated in the tree")
    if unlisted:
        print(f"  gated but NOT watched by that gate: {sorted(unlisted)}")
        print("  No violation today — they ARE gated. The gate simply could "
              "not\n  notice if they stopped being. Denominator drift, "
              "reported not repaired.")

    print("\n  NEVER-STARTED services with a LIVE caller "
          "(the default-core dependency gap):")
    for r in sorted(orphaned, key=lambda r: (-len(r["live_expecters"]), r["service"])):
        print(f"    {r['service']:22} profiles={','.join(r['profiles']) or '-':18} "
              f"live callers: {', '.join(r['live_expecters'])}")

    print("\n  A never-started service is very often CORRECT: the P0 "
          "containment model\n  requires consequential services to be "
          "profile-gated, so absence under a\n  bare `docker compose up` is "
          "the intended state, not a defect.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
