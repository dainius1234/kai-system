#!/usr/bin/env python3
"""The full UH-2 perception intake surface, with a verdict per source.

Why this is a script and not a note
-----------------------------------

The intake denominator has moved three times in one afternoon, each time
because the previous count was a subset that looked like a total:

    "2 of 7"    the sources ShadowPerceptionRunner happens to poll
    "2 of 11"   the sources ADAPTER_REGISTRY declares
    this file   every source the system intends, including those with
                no adapter at all — the MISSING class, which neither
                earlier count could see because both started from a
                list of things that already existed

A number that keeps growing when someone looks harder is not a
measurement, it is a guess with a total. This derives the population
from the tree every time it runs, prints the denominator, and can be
re-run after any change.

The verdicts
------------

    WORKING      every step of the path proven
    BROKEN       the intended path exists and cannot function
    DORMANT      code exists, nothing runs it
    MISSING      an intended capability with no acquisition path at all
    SUPERSEDED   a legacy capability that should be retired
    UNKNOWN      insufficient evidence — never inferred as WORKING

A source counts as WORKING only if the whole chain holds: the service
exists, is defined in the profile, sits on a network its owner can
reach, has an adapter, has a downstream reducer, and is actually polled.
It must never count as WORKING because a URL, an adapter, a compose
variable or an in-process test event exists — under that weaker bar all
eleven registered sources would pass, and two do.

Exit 0 always: this is a report. `kind=REPORT` in the registry.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO))

from scripts.security.gate_inputs import compose_files, inspected  # noqa: E402

#: The profiles compose itself uses to group perception services. This
#: is the denominator, and it is DERIVED — `sensors` and `watchers` are
#: the project's own declaration of what observes the world.
#:
#: The first version of this rule was "any service on sensor-net or
#: observability-net", which reported `redis`, `grafana`, `prometheus`
#: and `dashboard` as missing perception sources. Being *on* the
#: observability plane does not make a service an observer of the world;
#: it makes it something the observability plane carries. That rule
#: produced 49 MISSING, almost all of them nonsense, and it is the
#: false-positive direction this programme keeps paying for.
_PERCEPTION_PROFILES = {"sensors", "watchers"}

#: Readers that observe the OUTSIDE world and therefore sit on egress
#: rather than a sensor plane, so the profile rule above cannot see them.
#: Identified by already having an adapter — which is itself the
#: project's evidence that somebody intended them as a source.
_ADAPTER_ONLY_SOURCES = {"email-reader", "news-feed", "telegram-bot",
                         "weather-service", "calendar-service"}


def candidate_sources(services):
    """Every service the project itself declares as perception.

    Derived from `profiles:` membership plus the egress readers that
    already have an adapter. Nothing is named here that the tree does
    not already group.
    """
    out = set()
    for name, cfg in services.items():
        if set((cfg or {}).get("profiles") or []) & _PERCEPTION_PROFILES:
            out.add(name)
    return out | (_ADAPTER_ONLY_SOURCES & set(services))


def _service_for(source: str) -> str:
    """The compose service an adapter key refers to."""
    return {
        "weather": "weather-service", "calendar": "calendar-service",
        "docker": "docker-watcher", "git": "git-watcher",
        "system": "sysmetrics", "screen": "screen-watcher",
        "clipboard": "clipboard-service", "email": "email-reader",
        "news": "news-feed", "telegram": "telegram-bot", "market": None,
    }.get(source, source)


def audit(root: Path = None) -> Tuple[List[str], int, Dict[str, int]]:
    """Return (rows, sources inspected, verdict counts)."""
    import yaml

    root = root or REPO
    files = compose_files(root)
    if not files:
        # I-1: nothing inspected is not a clean bill of health.
        return (["no compose files found — this report inspected nothing "
                 "and must not be read as a clean surface"], 0, {})

    try:
        from common.perception_spine.adapters import ADAPTER_REGISTRY
        from common.world_state.reducers import REDUCER_MAP
        import common.perception_spine.shadow as shadow
    except Exception as exc:                       # pragma: no cover
        return ([f"UH-2 modules could not be imported ({exc}) — the "
                 f"surface is UNKNOWN, not empty"], 0, {})

    endpoints: Dict[str, tuple] = {}
    for value in vars(shadow).values():
        if (isinstance(value, dict) and value
                and all(isinstance(v, tuple) for v in value.values())):
            endpoints = value
            break

    # Which event types have a real reducer, as opposed to falling
    # through to reduce_generic. A generic fallback is not coverage: it
    # is the reducer equivalent of `except: pass`.
    adapter_src = (REPO / "common/perception_spine/adapters.py").read_text(
        encoding="utf-8")
    emitted: Dict[str, str] = {}
    for fn, etype in re.findall(
            r"def adapt_(\w+)\(.*?event_type=\"([a-z_.]+)\"",
            adapter_src, re.S):
        emitted[fn] = etype
    reduced: Set[str] = set(REDUCER_MAP)

    rows: List[str] = []
    counts: Dict[str, int] = {}
    inspected_n = 0

    for path in files:
        try:
            doc = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        except Exception:
            continue
        services = doc.get("services") or {}
        if not services:
            continue
        owner_nets = set((services.get("agentic") or {}).get("networks") or [])
        candidates = candidate_sources(services)
        covered_services = {
            _service_for(s) for s in ADAPTER_REGISTRY if _service_for(s)}

        rows.append(f"\n  ── {path.name} — owner `agentic` on "
                    f"{sorted(owner_nets) or ['(absent)']} ──")
        rows.append(f"  {'source':<22}{'verdict':<12}why")

        # 1. every registered adapter
        for source in sorted(ADAPTER_REGISTRY):
            inspected_n += 1
            svc = _service_for(source)
            etype = emitted.get(source if source != "system" else
                                "system_metrics", "")
            has_reducer = etype in reduced

            if svc is None:
                verdict, why = "SUPERSEDED", ("in-process source, adapted "
                                              "directly by paper_trade_slice "
                                              "— not intake")
            elif svc not in services:
                verdict, why = "MISSING", f"{svc} is not defined in this profile"
            elif source not in endpoints:
                verdict, why = "DORMANT", ("adapter registered, no polling "
                                           "entry — nothing invokes it")
            elif not (owner_nets & set((services[svc] or {}).get("networks") or [])):
                verdict, why = "BROKEN", (
                    f"owner cannot reach {svc} on "
                    f"{sorted(set((services[svc] or {}).get('networks') or []))}")
            elif not has_reducer:
                verdict, why = "BROKEN", (f"no reducer for `{etype}` — falls "
                                          f"through to reduce_generic")
            else:
                verdict, why = "WORKING", "full path proven"
            counts[verdict] = counts.get(verdict, 0) + 1
            rows.append(f"  {source:<22}{verdict:<12}{why}")

        # 2. observers with no adapter at all — the MISSING class that
        #    an adapter-derived denominator cannot see
        for svc in sorted(candidates - covered_services):
            inspected_n += 1
            counts["MISSING"] = counts.get("MISSING", 0) + 1
            rows.append(f"  {svc:<22}{'MISSING':<12}observes, but no adapter "
                        f"and no event contract")

    return rows, inspected_n, counts


def main() -> int:
    rows, n, counts = audit()
    print(inspected(n, "perception source(s) across all profiles",
                    "derived from ADAPTER_REGISTRY plus every service compose "
                    "itself groups under a `sensors` or `watchers` profile"))
    for row in rows:
        print(row)
    print()
    if counts:
        total = sum(counts.values())
        order = ["WORKING", "BROKEN", "DORMANT", "MISSING", "SUPERSEDED",
                 "UNKNOWN"]
        summary = "  ".join(f"{k} {counts.get(k, 0)}" for k in order)
        print(f"  {summary}   (total {total})")
        working = counts.get("WORKING", 0)
        print(f"\n  {working} of {total} source-profile pairs have a proven "
              f"path. A source is not\n  WORKING because a URL, an adapter, a "
              f"compose variable or an in-process\n  test event exists — under "
              f"that bar every one of them would pass.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
