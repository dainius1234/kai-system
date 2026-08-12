#!/usr/bin/env python3
"""What the default core does when a profile-gated dependency is absent.

#41 defect class B. Profiles-off is the **intended** security posture:
`check_default_profiles.py` requires consequential services to be gated,
so `docker compose up` starts only the contained core. Absence is
therefore the *normal* state, not an outage — and what the live callers
do about it has never been measured.

THE DENOMINATOR IS 25 EDGES; THE INSTRUMENT MEASURES MECHANISMS
===============================================================

25 live caller -> gated dependency edges exist (derived, not listed --
`report_runtime_topology` owns that number and this module imports it).
They do not contain 25 different behaviours. They contain four call
mechanisms, and instrumenting the mechanism 25 times would multiply
effort without multiplying evidence.

So: the full 25-edge denominator is reported, every edge is attributed to
the mechanism it uses, and each MECHANISM is exercised for real against a
dependency that genuinely is not there.

WHAT "FOR REAL" MEANS HERE, AND WHAT IT DOES NOT
================================================

Each mechanism runs its actual production code path against two kinds of
absent dependency:

  REFUSED    a closed port on 127.0.0.1 -- connection refused at once,
             which is what Docker's DNS//connect gives you for a service
             that was never started on a shared network
  BLACKHOLE  a socket that accepts and then never answers, which is what
             a hung or half-started dependency gives you

The second exists because they are different failures and a timeout that
is missing only shows up against the second. A refused connection
returns quickly whether or not anyone set a timeout.

This measures the CALLER'S LOGIC, honestly and without Docker. It does
not measure container DNS resolution, compose network behaviour, or what
the dashboard UI renders. Those stay UNKNOWN and are labelled UNKNOWN.

THE VERDICTS
============

    BOUNDED_DEGRADATION  bounded wait, explicit unavailable/degraded
                         result, caller unharmed. The acceptable outcome.
    MISLEADING_HEALTHY   the missing capability reads as present
    SILENT_FALLBACK      a substitute answer is returned with nothing
                         marking it as a substitute
    BLOCKED              no bound on the wait
    RETRY_STORM          repeated attempts without a ceiling
    CRASH                the exception escapes the caller
    UNKNOWN              not established here

MISLEADING_HEALTHY and SILENT_FALLBACK rank above CRASH deliberately. A
crash is loud and someone fixes it. A capability that reports itself
working while absent is the failure this whole programme exists to catch.
"""
from __future__ import annotations

import asyncio
import socket
import sys
import time
from contextlib import closing
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO))

from scripts.security.gate_inputs import inspected, require  # noqa: E402
from scripts.security import report_runtime_topology as topo  # noqa: E402

BOUNDED = "BOUNDED_DEGRADATION"
MISLEADING = "MISLEADING_HEALTHY"
SILENT = "SILENT_FALLBACK"
BLOCKED = "BLOCKED"
RETRY_STORM = "RETRY_STORM"
CRASH = "CRASH"
UNKNOWN = "UNKNOWN"

#: A wait longer than this, for a dependency that is simply not there, is
#: not a bound anyone would call bounded. Deliberately generous: the
#: point is to catch *no* ceiling, not to police tuning.
BOUND_SECONDS = 30.0

#: How the four mechanisms are recognised in a caller's source. Derived
#: by matching against the caller file, never by listing which service
#: uses what -- that list would drift the moment a caller is edited.
MECHANISMS: Tuple[Tuple[str, str, str], ...] = (
    ("resilient_call", "resilient_call(",
     "retry + circuit breaker + fallback (common/resilience.py)"),
    ("pooled_client", "pooled_client(",
     "shared pool, per-call timeout, caller-owned try/except"),
    ("raw_httpx", "httpx.AsyncClient",
     "a client constructed at the call site"),
    ("node_probe", "async def probe(",
     "dashboard's concurrent health poll over NODES"),
)

#: Not a call mechanism — a property OF one. Reported alongside so the
#: retry-storm question has an answer rather than an assumption.
BREAKER_PROBE = ("circuit_breaker", "ten consecutive calls to one absent target")


def closed_port() -> int:
    """A port nothing listens on: bind, read the number, release it."""
    with closing(socket.socket()) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class Blackhole:
    """Accepts the connection and never answers. A hung dependency."""

    def __init__(self) -> None:
        self._sock = socket.socket()
        self._sock.bind(("127.0.0.1", 0))
        self._sock.listen(8)
        self.port = self._sock.getsockname()[1]
        self._held: List[socket.socket] = []

    def __enter__(self) -> "Blackhole":
        return self

    def __exit__(self, *exc: Any) -> None:
        for c in self._held:
            c.close()
        self._sock.close()


def mechanisms_used(caller_file: str) -> List[str]:
    """Which mechanisms a caller's source actually uses."""
    path = REPO / caller_file
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return []
    return [name for name, needle, _ in MECHANISMS if needle in text]


def edges() -> List[Dict[str, str]]:
    """Every live caller -> gated, never-started dependency edge.

    Imported from the topology report rather than recomputed, so the two
    denominators cannot drift apart. A second implementation of the same
    count is a second thing to keep in step.
    """
    rows, _, _ = topo.survey()
    out: List[Dict[str, str]] = []
    for row in rows:
        if row["started"]:
            continue
        for caller in row["live_expecters"]:
            out.append({
                "caller": caller,
                "dependency": row["service"],
                "profiles": ",".join(row["profiles"]) or "-",
                "mechanisms": ",".join(mechanisms_used(caller)) or "(none matched)",
            })
    return out


# ── the CALL SITES, which is where the dangerous classes live ───────
#
# The mechanism is only half the question. `resilient_call` returns
# whatever `fallback=` the call site handed it, so a safe mechanism plus
# `fallback={"entries": [], "count": 0}` produces an EMPTY SUCCESS: a
# result indistinguishable from the backend having answered and having
# nothing to report. That is SILENT_FALLBACK, and no amount of care in
# the mechanism can detect it.

import ast  # noqa: E402
import re  # noqa: E402

#: `http://<service>:<port>` inside a default-URL literal.
_URL_LITERAL = re.compile(r"https?://([a-z0-9][a-z0-9-]{1,40}):\d{2,5}")

_FALLBACK_CALLS = {"_proxy_get", "_proxy_post", "resilient_call"}
#: Markers that make a substitute visibly a substitute.
_EXPLICIT = {"unavailable", "down", "degraded", "error", "absent"}


def _url_service(node: ast.AST, url_consts: Dict[str, str]) -> str:
    """Best-effort: which service this call's URL targets."""
    for sub in ast.walk(node):
        if isinstance(sub, ast.Name) and sub.id in url_consts:
            return url_consts[sub.id]
    return ""


def _fallback_shape(node: ast.AST) -> Tuple[str, str]:
    """(classification, rendering) for a `fallback=` expression."""
    try:
        text = ast.unparse(node)
    except Exception:                                          # noqa: BLE001
        text = "<unparseable>"
    if isinstance(node, ast.Dict):
        # A FAILURE MARKER IS A FAILURE MARKER WHATEVER IT IS SPELLED.
        # A first version recognised only `status: unavailable` and
        # therefore reported `{'ok': False}` as a silent fallback --- an
        # over-report, and a gate that over-reports sends people to
        # "fix" correct code. `ok=False`, `success=False` and an `error`
        # key all say the same thing out loud.
        for k, v in zip(node.keys, node.values):
            if not isinstance(k, ast.Constant):
                continue
            key = str(k.value).lower()
            if (key == "status" and isinstance(v, ast.Constant)
                    and str(v.value).lower() in _EXPLICIT):
                return BOUNDED, text
            if key in {"ok", "success", "available", "healthy"} and \
                    isinstance(v, ast.Constant) and v.value is False:
                return BOUNDED, text
            if key in {"error", "unavailable", "degraded"}:
                return BOUNDED, text
        # Nothing marks it. What remains is empty containers, zeroes and
        # nulls: a result the caller cannot distinguish from a backend
        # that answered and had nothing to say.
        return SILENT, text
    if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
        return SILENT, text
    if isinstance(node, ast.Constant) and node.value is None:
        return BOUNDED, text
    return UNKNOWN, text


def call_sites(gated: Set[str]) -> List[Dict[str, str]]:
    """Every fallback-bearing call whose URL targets a GATED service."""
    out: List[Dict[str, str]] = []
    for rel in sorted({e["caller"] for e in edges()}):
        path = REPO / rel
        if path.suffix != ".py":
            # prometheus.yml and friends carry no call sites. Not an
            # absent input -- a different KIND of input, which is a
            # statement about the file rather than about its existence.
            continue
        # I-1. THIS WAS `or not path.exists(): continue`, and the
        # instrumentation gate refused the commit over it. The caller
        # list comes from the topology report; a name in that list with
        # no file on disk means the two disagree, and skipping it would
        # report "0 silent fallbacks" for a file nobody could read. An
        # absent input is a failure to certify, never a clean bill.
        if not path.exists():
            raise SystemExit(
                f"REFUSING: {rel} is named as a live caller by "
                f"report_runtime_topology but does not exist. The "
                f"topology and this report disagree; neither can be "
                f"trusted until that is resolved.")
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except SyntaxError as exc:
            raise SystemExit(
                f"REFUSING: {rel} does not parse ({exc}). A caller whose "
                f"source cannot be read has UNKNOWN fallback behaviour, "
                f"which is not the same as having none.")
        # NAME -> service, read from `X_URL = backend_url(..., "http://svc:port")`
        url_consts: Dict[str, str] = {}
        for node in ast.walk(tree):
            if not isinstance(node, ast.Assign):
                continue
            for target in node.targets:
                if not isinstance(target, ast.Name):
                    continue
                for sub in ast.walk(node.value):
                    if isinstance(sub, ast.Constant) and isinstance(sub.value, str):
                        m = _URL_LITERAL.search(sub.value)
                        if m:
                            url_consts[target.id] = m.group(1)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            fn = node.func
            fname = fn.id if isinstance(fn, ast.Name) else getattr(fn, "attr", "")
            if fname not in _FALLBACK_CALLS:
                continue
            fb = next((kw.value for kw in node.keywords if kw.arg == "fallback"), None)
            if fb is None:
                continue
            service = _url_service(node, url_consts)
            if service not in gated:
                continue
            verdict, text = _fallback_shape(fb)
            out.append({"caller": rel, "line": str(node.lineno),
                        "dependency": service, "fallback": text,
                        "verdict": verdict})
    return out


def classify_edges(gated: Set[str]) -> List[Dict[str, str]]:
    """One verdict per caller -> dependency EDGE, on three separate axes.

    The axes are kept apart because collapsing them is the defect:

      transport   did the dependency answer?          (no -- it is absent)
      substitution did the caller put something in
                  its place?
      visibility  can the caller or the user tell the
                  substitute from a real answer?

    A healthy process, a bounded wait and a substituted value can all be
    true at once while the third axis fails. That is SILENT_FALLBACK, and
    it is the only outcome here where the system is confidently wrong
    rather than merely unavailable: `fallback=[]` and
    `fallback={"entries": [], "count": 0}` are SHAPE-IDENTICAL to a
    successful empty answer and carry no provenance, so "the profile is
    off" and "there are genuinely zero results" arrive as the same bytes.

    An edge takes its WORST call-site outcome. One silent substitution is
    enough to make the edge silent, because the caller only has to be
    misled once.
    """
    sites = call_sites(gated)
    by_edge: Dict[Tuple[str, str], List[Dict[str, str]]] = {}
    for site in sites:
        by_edge.setdefault((site["caller"], site["dependency"]), []).append(site)

    out: List[Dict[str, str]] = []
    for edge in edges():
        key = (edge["caller"], edge["dependency"])
        mine = by_edge.get(key, [])
        mechanisms = edge["mechanisms"]
        if any(m["verdict"] == SILENT for m in mine):
            verdict = SILENT
            visibility = "NO — shape-identical to a successful empty answer"
            substitution = f"{sum(1 for m in mine if m['verdict'] == SILENT)} of {len(mine)} call site(s)"
        elif mine:
            verdict = BOUNDED
            visibility = "yes — every substitute carries a failure marker"
            substitution = f"{len(mine)} call site(s), all marked"
        elif mechanisms and mechanisms != "(none matched)":
            # No fallback-bearing call site: the mechanism propagates the
            # failure and the call site must handle it. Bounded at the
            # mechanism, unknown at the call site -- stated as both.
            verdict = BOUNDED
            visibility = "propagated — the call site's handling is UNKNOWN"
            substitution = "none declared"
        else:
            verdict = UNKNOWN
            visibility = "UNKNOWN"
            substitution = "UNKNOWN"
        out.append({
            "caller": edge["caller"], "dependency": edge["dependency"],
            "profiles": edge["profiles"],
            "transport": "no answer — dependency intentionally absent",
            "substitution": substitution,
            "visibility": visibility,
            "verdict": verdict,
        })
    return out


# ── the live probes ─────────────────────────────────────────────────

async def _probe_resilient(url: str) -> Dict[str, Any]:
    """`resilient_call` against an absent dependency.

    THE BREAKERS ARE RESET FIRST, and that is not tidiness. They live in
    a module-level dict keyed by HOSTNAME, and every probe here targets
    `127.0.0.1` — so the REFUSED run's failures opened the circuit and
    the BLACKHOLE run afterwards returned its fallback in 0.0s WITHOUT
    EVER CONNECTING. The measurement looked like a fast, clean
    degradation and was actually the instrument reading its own previous
    result.

    In production this contamination does not arise: each service has its
    own hostname and therefore its own breaker. It arises here purely
    because the harness collapses them onto one address. Caught by
    `test_every_mechanism_is_bounded_against_a_blackhole` asserting the
    probe actually WAITED, rather than only that it returned quickly —
    "fast" and "correct" are not the same observation.
    """
    from common.resilience import resilient_call, _breakers
    _breakers.clear()
    started = time.monotonic()
    result = await resilient_call(
        "GET", url, timeout=2.0, retries=2, backoff=0.3,
        fallback={"status": "unavailable"},
    )
    return {"elapsed": time.monotonic() - started, "result": result,
            "raised": None}


async def _probe_pooled(url: str) -> Dict[str, Any]:
    """The bare shape the 48/38 call sites use: pooled client, timeout,
    caller's own try/except. Reproduced rather than imported, because
    each call site owns its own handler and there is no single function
    to call."""
    from common.http_hygiene import pooled_client
    started = time.monotonic()
    raised = None
    result: Any = None
    try:
        async with pooled_client(timeout=2.0) as client:
            resp = await client.get(url, timeout=2.0)
            result = resp.json()
    except Exception as exc:                                  # noqa: BLE001
        raised = type(exc).__name__
    return {"elapsed": time.monotonic() - started, "result": result,
            "raised": raised}


async def _probe_breaker(url: str) -> Dict[str, Any]:
    """Does repeated failure stop retrying, or storm?

    Answers the RETRY_STORM question directly: call the same absent
    target ten times and see whether the mechanism keeps dialling. The
    breaker is reset first so the count starts from a known state.
    """
    from common.resilience import resilient_call, _breakers
    _breakers.clear()
    started = time.monotonic()
    for _ in range(10):
        await resilient_call("GET", url, timeout=1.0, retries=2, backoff=0.1,
                             fallback={"status": "unavailable"})
    elapsed = time.monotonic() - started
    name = url.split("//")[-1].split("/")[0].split(":")[0]
    opened = name in _breakers and not _breakers[name].allow()
    return {"elapsed": elapsed, "raised": None,
            "result": {"status": "unavailable" if opened else "still-dialling"}}


async def _probe_node(url: str) -> Dict[str, Any]:
    """Dashboard's own `probe()` body, which classifies rather than
    raises. This is the mechanism behind every row of the node grid."""
    from common.http_hygiene import pooled_client
    started = time.monotonic()
    try:
        async with pooled_client() as client:
            resp = await client.get(url, timeout=2.0)
            resp.raise_for_status()
            entry = {"status": "up", "details": resp.json()}
    except Exception as exc:                                  # noqa: BLE001
        entry = {"status": "down", "error": str(exc)}
    return {"elapsed": time.monotonic() - started, "result": entry,
            "raised": None}


PROBES = {
    "resilient_call": _probe_resilient,
    "pooled_client": _probe_pooled,
    "raw_httpx": _probe_pooled,      # same shape; the pool is the only difference
    "node_probe": _probe_node,
    "circuit_breaker": _probe_breaker,
}


def classify(observation: Dict[str, Any]) -> Tuple[str, str]:
    """(verdict, why). Never guesses: anything unrecognised is UNKNOWN."""
    if observation["elapsed"] > BOUND_SECONDS:
        return BLOCKED, f"waited {observation['elapsed']:.1f}s with no ceiling"
    result = observation["result"]
    raised = observation["raised"]

    if raised:
        # The exception reached the mechanism's edge. Whether the CALL
        # SITE catches it is a per-site property this cannot see, so this
        # is reported as the mechanism propagating, not as a crash.
        return BOUNDED, (f"raised {raised} in {observation['elapsed']:.1f}s — "
                         f"bounded, and the call site must handle it")
    if isinstance(result, dict):
        marker = str(result.get("status", "")).lower()
        if marker in {"unavailable", "down", "degraded", "error"}:
            return BOUNDED, (f"explicit {marker!r} in "
                             f"{observation['elapsed']:.1f}s")
        if marker in {"ok", "up", "healthy", "running"}:
            return MISLEADING, f"absent dependency reported {marker!r}"
        if result:
            return SILENT, f"returned {result!r} with nothing marking it a substitute"
    if result is None:
        return BOUNDED, (f"returned None in {observation['elapsed']:.1f}s — "
                         "absence is representable, but the call site must "
                         "distinguish it from an empty success")
    return UNKNOWN, f"unrecognised result {result!r}"


async def measure() -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    refused = f"http://127.0.0.1:{closed_port()}/health"
    with Blackhole() as hole:
        hanging = f"http://127.0.0.1:{hole.port}/health"
        for name in [m[0] for m in MECHANISMS] + [BREAKER_PROBE[0]]:
            for label, url in (("REFUSED", refused), ("BLACKHOLE", hanging)):
                try:
                    obs = await PROBES[name](url)
                except Exception as exc:                      # noqa: BLE001
                    out.append({"mechanism": name, "absence": label,
                                "verdict": CRASH,
                                "why": f"{type(exc).__name__} escaped the mechanism",
                                "elapsed": None})
                    continue
                verdict, why = classify(obs)
                out.append({"mechanism": name, "absence": label,
                            "verdict": verdict, "why": why,
                            "elapsed": round(obs["elapsed"], 2)})
    return out


def main() -> int:
    require(topo.COMPOSE_FILES)
    all_edges = edges()
    print(inspected(len(all_edges), "live caller -> absent-dependency edge(s)",
                    f"across {len({e['dependency'] for e in all_edges})} "
                    f"gated services and "
                    f"{len({e['caller'] for e in all_edges})} live callers"))

    used: Dict[str, int] = {}
    for e in all_edges:
        for m in e["mechanisms"].split(","):
            used[m] = used.get(m, 0) + 1
    print("\n  EDGES BY MECHANISM (the reduction that makes this affordable):")
    for m, n in sorted(used.items(), key=lambda kv: -kv[1]):
        print(f"    {m:22} {n:3} edge(s)")

    gated = {e["dependency"] for e in all_edges}
    sites = call_sites(gated)
    print(f"\n  CALL SITES targeting a GATED service with an explicit "
          f"fallback: {len(sites)}")
    silent = [s for s in sites if s["verdict"] == SILENT]
    for s_ in sorted(sites, key=lambda r: (r["verdict"] != SILENT, r["caller"])):
        print(f"    {s_['verdict']:20} {s_['caller']}:{s_['line']:<5} "
              f"-> {s_['dependency']:20} fallback={s_['fallback'][:60]}")
    if silent:
        print(f"\n  {len(silent)} of {len(sites)} return an EMPTY SUCCESS for an "
              f"absent gated\n  dependency — a result the caller cannot tell "
              f"from 'the backend\n  answered and there is nothing'. This is the "
              f"dangerous class.")

    print("\n  PER-EDGE VERDICT — the full denominator, three axes kept apart:")
    edge_rows = classify_edges(gated)
    tally: Dict[str, int] = {}
    for r in edge_rows:
        tally[r["verdict"]] = tally.get(r["verdict"], 0) + 1
    for verdict in (SILENT, MISLEADING, BLOCKED, RETRY_STORM, CRASH,
                    BOUNDED, UNKNOWN):
        n = tally.get(verdict, 0)
        if n:
            print(f"    {verdict:22} {n:3} of {len(edge_rows)} edge(s)")
    print()
    for r in sorted(edge_rows, key=lambda r: (r["verdict"] != SILENT,
                                              r["caller"], r["dependency"])):
        print(f"    {r['verdict']:20} {r['caller']:24} -> "
              f"{r['dependency']:20} [{r['profiles']}]")
        print(f"      transport={r['transport']}")
        print(f"      substitution={r['substitution']}")
        print(f"      distinguishable={r['visibility']}")

    print("\n  MECHANISM BEHAVIOUR AGAINST A DEPENDENCY THAT IS NOT THERE:")
    results = asyncio.run(measure())
    for r in results:
        el = f"{r['elapsed']}s" if r["elapsed"] is not None else "n/a"
        print(f"    {r['mechanism']:16} {r['absence']:10} {r['verdict']:20} "
              f"{el:>7}  {r['why']}")

    bad = [r for r in results if r["verdict"] in
           {MISLEADING, SILENT, BLOCKED, RETRY_STORM, CRASH}]
    print(f"\n  {len(results)} mechanism/absence observation(s); "
          f"{len(bad)} in a dangerous class")
    if bad:
        for r in bad:
            print(f"    ! {r['mechanism']} / {r['absence']}: {r['verdict']}")

    print("\n  NOT MEASURED HERE, and therefore UNKNOWN: container DNS "
          "resolution,\n  compose network behaviour, what the dashboard UI "
          "renders, and whether\n  each individual CALL SITE handles what its "
          "mechanism propagates. This\n  measures the caller's logic, not the "
          "deployed system.")
    print("\n  A profile-gated dependency being absent is the INTENDED state. "
          "An\n  explicit 'unavailable' is correct architecture; a capability "
          "that reads\n  as working while absent is not.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
