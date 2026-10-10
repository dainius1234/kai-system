#!/usr/bin/env python3
"""Calibration for the default-core degradation report.

The report answers: when a profile-gated dependency is absent — which is
the INTENDED state, not an outage — what does the live core do?

Two things need calibrating, and they fail in opposite directions:

* the **live probes** must actually observe a bounded, explicit failure
  rather than reporting one because nothing was tried;
* the **fallback classifier** must not over-report. Its first version
  recognised only ``status: unavailable`` and therefore called
  ``{'ok': False}`` a silent fallback — 22 false positives out of 53,
  and a gate that over-reports sends people to "fix" correct code. That
  is the failure mode this repository works hardest to avoid, so both
  directions are asserted here.
"""
from __future__ import annotations

import ast
import asyncio
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from scripts.security import report_degradation_tolerance as dt  # noqa: E402

passed = 0
failed = 0
EXPECTED_SCENARIOS = 12
executed: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    global passed, failed
    if condition:
        passed += 1
    else:
        failed += 1
        print(f"  FAIL: {name}" + (f" — {detail}" if detail else ""))


def scenario(name: str) -> None:
    executed.append(name)


def shape(src: str):
    return dt._fallback_shape(ast.parse(src, mode="eval").body)


# ── the classifier, both directions ──────────────────────────────────

def test_an_empty_container_is_a_silent_fallback() -> None:
    """Known-positive. `{"entries": []}` is indistinguishable from a
    backend that answered and had nothing."""
    scenario("empty container is silent")
    for src in ('{"entries": []}', '{"positions": []}', '{}',
                '{"count": 0, "rules": []}', '[]',
                '{"content": "", "id": None}'):
        verdict, _ = shape(src)
        check(f"{src} -> SILENT", verdict == dt.SILENT, f"got {verdict}")


def test_an_explicit_marker_is_not_a_silent_fallback() -> None:
    """Known-negative, in every spelling the tree actually uses. If any
    of these regresses to SILENT the report starts accusing correct
    code."""
    scenario("explicit marker is bounded")
    for src in ('{"status": "unavailable"}', '{"status": "down"}',
                '{"status": "degraded"}', '{"ok": False}',
                '{"success": False}', '{"error": "backend absent"}',
                '{"status": "unavailable", "content": ""}',
                'None'):
        verdict, _ = shape(src)
        check(f"{src} -> BOUNDED", verdict == dt.BOUNDED, f"got {verdict}")


def test_a_truthy_ok_is_not_treated_as_a_failure_marker() -> None:
    """`{"ok": True}` is not a failure marker — it is the misleading
    case, and must not be laundered into BOUNDED by the key alone."""
    scenario("truthy ok is not a marker")
    verdict, _ = shape('{"ok": True, "entries": []}')
    check("classified as SILENT, not BOUNDED", verdict == dt.SILENT, f"got {verdict}")


# ── the live probes ──────────────────────────────────────────────────

def test_every_mechanism_is_bounded_against_a_refused_connection() -> None:
    scenario("refused is bounded")
    results = asyncio.run(dt.measure())
    refused = [r for r in results if r["absence"] == "REFUSED"]
    expected = len(dt.MECHANISMS) + 1          # +1 for the breaker probe
    check("every mechanism probed", len(refused) == expected,
          f"{len(refused)} vs {expected}")
    check("none blocked", all(r["verdict"] != dt.BLOCKED for r in refused),
          str([r for r in refused if r["verdict"] == dt.BLOCKED]))


def test_every_mechanism_is_bounded_against_a_blackhole() -> None:
    """The load-bearing one, twice over.

    A refused connection returns fast whether or not anyone set a
    timeout, so only a socket that accepts and never answers can show a
    missing bound. And the `elapsed > 0.1` assertion is what caught the
    instrument reading its own state: `resilient_call`'s breakers are
    keyed by hostname, every probe here targets 127.0.0.1, and the
    REFUSED run's failures left the circuit open — so BLACKHOLE returned
    a fallback in 0.0s having never connected. "Fast" and "correct" are
    different observations, and only the second assertion could tell
    them apart."""
    scenario("blackhole is bounded")
    results = asyncio.run(dt.measure())
    hole = [r for r in results if r["absence"] == "BLACKHOLE"]
    expected = len(dt.MECHANISMS) + 1
    check("every mechanism probed", len(hole) == expected,
          f"{len(hole)} vs {expected}")
    for r in hole:
        check(f"{r['mechanism']} bounded", r["verdict"] != dt.BLOCKED, str(r))
        check(f"{r['mechanism']} actually waited", (r["elapsed"] or 0) > 0.1,
              f"{r['elapsed']}s — did it really try?")


def test_the_blackhole_really_accepts_and_never_answers() -> None:
    """Calibrating the calibrator: if the blackhole refused instead of
    hanging, every timeout assertion above would pass vacuously."""
    scenario("blackhole is a blackhole")
    import socket
    with dt.Blackhole() as hole:
        s = socket.socket()
        s.settimeout(2.0)
        try:
            s.connect(("127.0.0.1", hole.port))
            connected = True
        except OSError:
            connected = False
        finally:
            s.close()
    check("the connection is accepted", connected,
          "it refused — the timeout cases would be vacuous")


def test_a_closed_port_really_refuses() -> None:
    scenario("closed port refuses")
    import socket
    port = dt.closed_port()
    s = socket.socket()
    s.settimeout(2.0)
    try:
        s.connect(("127.0.0.1", port))
        refused = False
    except OSError:
        refused = True
    finally:
        s.close()
    check("the connection is refused", refused, f"port {port} accepted")


# ── the denominator ──────────────────────────────────────────────────

def test_the_edge_denominator_comes_from_the_topology_report() -> None:
    """Not recomputed here. Two implementations of one count are two
    things to keep in step, and they will not stay in step."""
    scenario("denominator is imported")
    import scripts.security.report_runtime_topology as topo
    rows, _, _ = topo.survey()
    expected = sum(len(r["live_expecters"]) for r in rows if not r["started"])
    check("edges match the topology report", len(dt.edges()) == expected,
          f"{len(dt.edges())} vs {expected}")
    check("edges exceed services — a caller-dependency PAIR is the unit",
          len(dt.edges()) > len({e["dependency"] for e in dt.edges()}),
          "if these are equal the unit has silently become the service")


def test_call_sites_are_restricted_to_gated_dependencies() -> None:
    scenario("call sites are gated-only")
    import scripts.security.report_runtime_topology as topo
    rows, _, _ = topo.survey()
    gated = {r["service"] for r in rows if not r["started"] and r["live_expecters"]}
    sites = dt.call_sites(gated)
    check("some are found", len(sites) > 0, str(len(sites)))
    check("every one targets a gated service",
          all(s["dependency"] in gated for s in sites),
          str([s for s in sites if s["dependency"] not in gated][:3]))
    started = {r["service"] for r in rows if r["started"]}
    check("none targets a service that IS started",
          not [s for s in sites if s["dependency"] in started],
          str([s for s in sites if s["dependency"] in started][:3]))


def test_the_real_tree_reports_both_classes() -> None:
    """A report that could only say SILENT, or only say BOUNDED, would be
    reporting on itself."""
    scenario("both classes present")
    import scripts.security.report_runtime_topology as topo
    rows, _, _ = topo.survey()
    gated = {r["service"] for r in rows if not r["started"] and r["live_expecters"]}
    verdicts = {s["verdict"] for s in dt.call_sites(gated)}
    check("SILENT_FALLBACK occurs", dt.SILENT in verdicts, str(verdicts))
    check("BOUNDED_DEGRADATION occurs", dt.BOUNDED in verdicts, str(verdicts))


def test_an_edge_takes_its_worst_call_site_outcome() -> None:
    """One silent substitution is enough to make the edge silent — the
    caller only has to be misled once. A max() that behaved like a
    majority vote would launder ten bad sites behind one good one."""
    scenario("edge takes worst outcome")
    import scripts.security.report_runtime_topology as topo
    rows, _, _ = topo.survey()
    gated = {r["service"] for r in rows if not r["started"] and r["live_expecters"]}
    sites = dt.call_sites(gated)
    edge_rows = dt.classify_edges(gated)
    by_edge = {(e["caller"], e["dependency"]): e for e in edge_rows}
    mixed = 0
    for site in sites:
        if site["verdict"] != dt.SILENT:
            continue
        edge = by_edge.get((site["caller"], site["dependency"]))
        check(f"{site['caller']}->{site['dependency']} inherits SILENT",
              edge is not None and edge["verdict"] == dt.SILENT,
              str(edge))
        mixed += 1
    check("at least one silent site exists to test with", mixed > 0, "")


def test_every_edge_is_classified_and_the_axes_are_separate() -> None:
    scenario("all edges classified")
    import scripts.security.report_runtime_topology as topo
    rows, _, _ = topo.survey()
    gated = {r["service"] for r in rows if not r["started"] and r["live_expecters"]}
    edge_rows = dt.classify_edges(gated)
    check("every edge has a verdict", len(edge_rows) == len(dt.edges()),
          f"{len(edge_rows)} vs {len(dt.edges())}")
    check("no verdict is blank",
          all(r["verdict"] for r in edge_rows), "")
    check("transport, substitution and visibility are all recorded",
          all(r["transport"] and r["substitution"] and r["visibility"]
              for r in edge_rows), "")
    check("verdicts come from the declared vocabulary",
          {r["verdict"] for r in edge_rows} <= {
              dt.BOUNDED, dt.SILENT, dt.MISLEADING, dt.BLOCKED,
              dt.RETRY_STORM, dt.CRASH, dt.UNKNOWN},
          str({r["verdict"] for r in edge_rows}))


def run_all() -> None:
    test_an_empty_container_is_a_silent_fallback()
    test_an_explicit_marker_is_not_a_silent_fallback()
    test_a_truthy_ok_is_not_treated_as_a_failure_marker()
    test_every_mechanism_is_bounded_against_a_refused_connection()
    test_every_mechanism_is_bounded_against_a_blackhole()
    test_the_blackhole_really_accepts_and_never_answers()
    test_a_closed_port_really_refuses()
    test_the_edge_denominator_comes_from_the_topology_report()
    test_call_sites_are_restricted_to_gated_dependencies()
    test_the_real_tree_reports_both_classes()
    test_an_edge_takes_its_worst_call_site_outcome()
    test_every_edge_is_classified_and_the_axes_are_separate()

    check(f"all {EXPECTED_SCENARIOS} scenarios ran",
          len(executed) == EXPECTED_SCENARIOS, f"{len(executed)}: {executed}")
    check("no scenario ran twice", len(set(executed)) == len(executed), str(executed))


if __name__ == "__main__":
    run_all()
    print(f"\n{'='*60}")
    print(f"Degradation Tolerance Tests: {passed} passed, {failed} failed")
    if failed:
        print("EXIT GATE: FAIL")
        sys.exit(1)
    print("EXIT GATE: PASS")
