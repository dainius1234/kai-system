"""Revalidate all 96 Dashboard audit findings against current code (Wave 1).

The `KAI-DASH-*` findings were captured at commit `7adab8d`. P0 containment
and the Unified Hunter work have changed the tree since, so some findings
are already remediated and some are not. Building against the finding list
as written would mean fixing what is already fixed and missing what moved.

Each of the 96 findings is revalidated against the tree as it is now and
reported as one of:

  LIVE        the condition the finding describes still holds
  REMEDIATED  the condition no longer holds
  PARTIAL     materially reduced but not closed
  MANUAL      not statically decidable; needs human review

MANUAL is not a pass. Neither is REMEDIATED: programme Rule 7 says a
finding closes only through an evidence-backed register action, so this
tool reports state — it does not close anything.

Every one of the 96 findings must be accounted for. If any is missing from
the table, the run reports a GAP and exits non-zero. That self-audit exists
because the architecture-rules gate once silently omitted 6 of its 15 rules
while reporting a clean pass; a coverage table that does not check its own
coverage is not evidence.

Exit codes:
  0  every finding accounted for (whatever their status)
  1  coverage gap, or --gate given while findings are still LIVE
"""
from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from pathlib import Path
from typing import Callable, Dict, List, NamedTuple, Optional, Tuple

REPO = Path(__file__).resolve().parent.parent.parent
DASHBOARD = REPO / "dashboard" / "app.py"

LIVE = "LIVE"
REMEDIATED = "REMEDIATED"
PARTIAL = "PARTIAL"
MANUAL = "MANUAL"

TOTAL_DASH_FINDINGS = 96

# ── The anchor pre-scan (KAI-GATE-005) ───────────────────────────────
#
# Pointed at an empty tree, this tracker reported **`REMEDIATED=52`**.
# Not one of those 52 checks is wrong in isolation: *"the dashboard never
# reads broker credentials"* is correctly true when there is no
# dashboard. The defect is that the check does not know it is making a
# far weaker claim than it appears to.
#
# The operator named this **category confusion**, and separated it from
# boundary blindness deliberately — different mechanism, different fix.
# Boundary blindness is *"I cannot tell correct from absent"*, and
# `gate_inputs.require()` fixes it. Category confusion is *"a check for
# the absence of something bad passes because everything is absent"*, and
# fail-closed inputs do nothing for it: the file can be present and still
# not contain the routes a check believes it is judging.
#
# Their design, taken as-is: **one pre-scan, not 96 per-check anchors.**
# It establishes that the tree is real before any verdict is rendered. It
# does not convert every absence check into a positive assertion — "no
# route reads broker credentials" still means "among the routes we found,
# none did" — but now it is known that routes *were* found.
#
# What it deliberately does not catch: one specific route being absent
# from the scan. That needs a per-check anchor, and is worth adding
# surgically if a check is ever found systematically missing its subject.
# Recorded as a known limit rather than left implied.
# Verified present, not guessed. My first draft named `require_principal`
# and `PUBLIC_ROUTES`; neither exists — the real names are
# `require_dashboard_auth` and `Scope`. The pre-scan caught that on its
# first run, against itself, which is the argument for having it.
ANCHOR_SYMBOLS = (
    "require_dashboard_auth",  # Track A's per-route identity dependency
    "Scope.",                  # the scope taxonomy every route declares
    "safe_symbol",             # Track G input validation
    "backend_url",             # Track H backend destination validation
    "degraded_response",       # Track D failure semantics
)
ANCHOR_MIN_ROUTES = 100       # 185 today; a tenth of that is not this file


class AnchorFailure(NamedTuple):
    """Why the tree could not be judged. Kinds fail differently on purpose.

    The operator's nuance: *"you don't want 'file not found' to be the
    same exit code as 'DashboardAuth not in tree'."* They are different
    problems — one is a moved file, the other is a file that is no longer
    what this tracker believes it is judging — and a single exit code
    would send whoever is debugging in the wrong direction.
    """
    kind: str      # "absent" | "unrecognisable"
    detail: str
    exit_code: int


def anchor_scan() -> Optional[AnchorFailure]:
    """Prove the tree is real before any finding is judged."""
    if not DASHBOARD.exists():
        return AnchorFailure(
            "absent", f"{_rel(DASHBOARD)} does not exist", 2)

    source = DASHBOARD.read_text(encoding="utf-8")
    missing = [s for s in ANCHOR_SYMBOLS if s not in source]
    if missing:
        return AnchorFailure(
            "unrecognisable",
            f"present, but missing {missing} — this is not the file "
            f"these 96 findings were written against", 3)

    found = len(_routes())
    if found < ANCHOR_MIN_ROUTES:
        return AnchorFailure(
            "unrecognisable",
            f"only {found} routes parsed, expected at least "
            f"{ANCHOR_MIN_ROUTES} — the scan is not seeing the app", 3)
    return None

Result = Tuple[str, str]


# ── Source inspection primitives ─────────────────────────────────────

class Route(NamedTuple):
    method: str
    path: str
    authed: bool
    lineno: int
    handler: str


_CACHE: Dict[str, object] = {}


def _text() -> str:
    if "text" not in _CACHE:
        _CACHE["text"] = DASHBOARD.read_text(encoding="utf-8")
    return _CACHE["text"]  # type: ignore[return-value]


def _tree() -> Optional[ast.Module]:
    if "tree" not in _CACHE:
        try:
            _CACHE["tree"] = ast.parse(_text())
        except (SyntaxError, OSError):
            _CACHE["tree"] = None
    return _CACHE["tree"]  # type: ignore[return-value]


def _routes() -> List[Route]:
    if "routes" in _CACHE:
        return _CACHE["routes"]  # type: ignore[return-value]
    tree = _tree()
    found: List[Route] = []
    if tree is not None:
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            for dec in node.decorator_list:
                if not (isinstance(dec, ast.Call)
                        and isinstance(dec.func, ast.Attribute)):
                    continue
                if dec.func.attr not in {"get", "post", "put", "delete", "patch"}:
                    continue
                if not dec.args or not isinstance(dec.args[0], ast.Constant):
                    continue
                found.append(Route(
                    method=dec.func.attr,
                    path=dec.args[0].value,
                    authed=_handler_is_authenticated(node),
                    lineno=dec.lineno,
                    handler=node.name,
                ))
    _CACHE["routes"] = found
    return found


AUTH_MARKERS = ("require_dashboard_auth", "require_service_auth", "DashboardPrincipal")


def _handler_is_authenticated(node) -> bool:
    """A route is authenticated if it declares an auth dependency.

    Checked on the decorator (``dependencies=[...]``) and on the handler
    signature (a ``Depends(require_dashboard_auth)`` parameter). A global
    middleware would not satisfy this deliberately — per-route declaration
    is what makes the authority of each route reviewable.
    """
    for dec in node.decorator_list:
        if any(m in ast.unparse(dec) for m in AUTH_MARKERS):
            return True
    try:
        sig = ast.unparse(node.args)
    except Exception:
        return False
    return any(m in sig for m in AUTH_MARKERS)


def _handler_src(name: str) -> str:
    tree = _tree()
    if tree is None:
        return ""
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            return ast.unparse(node)
    return ""


def _route_set(*specs: Tuple[str, str]) -> List[Route]:
    wanted = set(specs)
    return [r for r in _routes() if (r.method, r.path) in wanted]


def _routes_matching(pattern: str, methods: Optional[set] = None) -> List[Route]:
    rx = re.compile(pattern)
    return [r for r in _routes()
            if rx.search(r.path) and (methods is None or r.method in methods)]


MUTATING = {"post", "put", "delete", "patch"}

# Routes that may legitimately serve without a principal: container
# liveness probes, and the HTML shells the browser must load *before* it
# can authenticate. Nothing mutating, and nothing that reads private
# state, may ever appear here.
PUBLIC_ROUTES = frozenset({
    ("get", "/health"), ("get", "/metrics"),
    ("get", "/ui"), ("get", "/app"), ("get", "/chat"), ("get", "/thinking"),
})


def _unauthed(routes: List[Route]) -> List[str]:
    return [f"{r.method.upper()} {r.path}" for r in routes if not r.authed]


# ── Generic check builders ───────────────────────────────────────────

def route_auth(*specs: Tuple[str, str]) -> Callable[[], Result]:
    """Finding is LIVE while any of the named routes lacks inbound auth."""
    def check() -> Result:
        present = _route_set(*specs)
        if not present:
            return REMEDIATED, "route(s) no longer exist: " + ", ".join(
                f"{m.upper()} {p}" for m, p in specs)
        missing = [f"{m.upper()} {p}" for m, p in specs
                   if not any(r.method == m and r.path == p for r in present)]
        bad = _unauthed(present)
        if bad:
            detail = f"{len(bad)} unauthenticated: {', '.join(bad[:3])}"
            if len(bad) > 3:
                detail += f" (+{len(bad) - 3} more)"
            return LIVE, detail
        note = f" (absent: {', '.join(missing)})" if missing else ""
        return REMEDIATED, f"all {len(present)} route(s) authenticated{note}"
    return check


def prefix_auth(pattern: str, methods: Optional[set] = None) -> Callable[[], Result]:
    """Finding is LIVE while any route matching the pattern lacks auth."""
    def check() -> Result:
        present = _routes_matching(pattern, methods)
        if not present:
            return REMEDIATED, f"no routes match {pattern!r}"
        bad = _unauthed(present)
        if bad:
            detail = f"{len(bad)}/{len(present)} unauthenticated: {', '.join(bad[:3])}"
            if len(bad) > 3:
                detail += f" (+{len(bad) - 3} more)"
            return LIVE, detail
        return REMEDIATED, f"all {len(present)} matching route(s) authenticated"
    return check


def source_marker(marker: str, live_msg: str, fixed_msg: str,
                  present_means_live: bool = True) -> Callable[[], Result]:
    """Finding tracked by the presence or absence of a source marker."""
    def check() -> Result:
        found = marker in _text()
        if found == present_means_live:
            return LIVE, live_msg
        return REMEDIATED, fixed_msg
    return check


def manual(reason: str) -> Callable[[], Result]:
    """Not statically decidable. Records why, so the gap stays visible."""
    def check() -> Result:
        return MANUAL, reason
    check.is_manual = True  # type: ignore[attr-defined]
    return check


# ── Track A — inbound identity ───────────────────────────────────────

def dash_001() -> Result:
    """Open privileged gateway."""
    published_publicly = False
    binds = []
    for compose in sorted(REPO.glob("docker-compose*.yml")):
        for line in compose.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if stripped.startswith('- "') and ":8080:8080" in stripped:
                binds.append(f"{compose.name}:{stripped}")
                if not stripped.startswith('- "127.0.0.1:'):
                    published_publicly = True

    routes = _routes()
    open_routes = [r for r in routes if not r.authed]
    # An unauthenticated route is only acceptable if it is on the declared
    # public list. The list is held here rather than read from the app so
    # that widening it is a deliberate edit to the checker, not a silent
    # side effect of editing the app.
    unexpected = [f"{r.method.upper()} {r.path}" for r in open_routes
                  if (r.method, r.path) not in PUBLIC_ROUTES]
    mutating_open = [f"{r.method.upper()} {r.path}" for r in open_routes
                     if r.method in MUTATING]

    if mutating_open:
        return LIVE, (f"{len(mutating_open)} mutating route(s) unauthenticated: "
                      f"{', '.join(mutating_open[:3])}")
    if published_publicly and unexpected:
        return LIVE, (f"published beyond loopback and {len(unexpected)} "
                      f"undeclared route(s) unauthenticated")
    if unexpected:
        return PARTIAL, (f"loopback-bound (P0 containment) but {len(unexpected)} "
                         f"route(s) outside the public list have no auth: "
                         f"{', '.join(unexpected[:3])}")
    if published_publicly:
        return PARTIAL, "all routes authenticated but still published beyond loopback"
    return REMEDIATED, (
        f"loopback-bound; {len(routes) - len(open_routes)} of {len(routes)} routes "
        f"authenticated, the other {len(open_routes)} are declared public "
        f"(liveness and HTML shells, none mutating)"
    )


def dash_002() -> Result:
    """Anonymous callers use the server-held bearer token to change mode."""
    text = _text()
    if "DASHBOARD_GATE_TOKEN" in text or "GATE_TOKEN" in text:
        return LIVE, "dashboard still holds a Tool Gate bearer token"
    src = _handler_src("api_set_mode")
    if not src:
        return REMEDIATED, "no gate token held; /api/mode removed"
    if "TOOL_GATE" in src or "_proxy_post" in src:
        return LIVE, "/api/mode still calls Tool Gate"
    return REMEDIATED, "no gate token held; /api/mode is display-state only"


def dash_011() -> Result:
    """No verified principal, role or session ownership."""
    if any(m in _text() for m in AUTH_MARKERS):
        return REMEDIATED, "inbound principal model present"
    return LIVE, "no inbound principal, role or session model"


def dash_012() -> Result:
    """Backend token carries no delegation evidence."""
    text = _text()
    if "DASHBOARD_GATE_TOKEN" in text or "GATE_TOKEN" in text:
        return LIVE, "static server-held backend token, no actor/reason binding"
    if any(m in text for m in AUTH_MARKERS):
        return REMEDIATED, "no static backend token; inbound principal established"
    return PARTIAL, (
        "the specific static Tool Gate token is gone, but with no inbound "
        "principal there is still nothing to delegate from — recheck when "
        "backend credentials are reintroduced"
    )


def dash_018() -> Result:
    """No route-specific scopes or least-privilege backend credentials.

    A scope model that assigns the *same* scope everywhere is not least
    privilege — it is a single shared authority wearing a scope's name.
    So this checks the distribution, not merely the presence.
    """
    scopes = _declared_scopes()
    if not scopes:
        return LIVE, ("no route-specific authorisation scopes; "
                      "every route has full backend reach")
    if len(scopes) == 1:
        return LIVE, (f"every route declares the same scope "
                      f"({next(iter(scopes))}); that is not least privilege")

    unscoped = [f"{r.method.upper()} {r.path}" for r in _routes()
                if r.authed and not _route_scope(r)]
    if unscoped:
        return PARTIAL, (f"{len(scopes)} distinct scopes, but {len(unscoped)} "
                         f"authenticated route(s) declare none, e.g. {unscoped[0]}")
    return REMEDIATED, (f"{len(scopes)} distinct scopes declared across routes: "
                        f"{', '.join(sorted(scopes))}")


_SCOPE_RX = re.compile(r"Scope\.([A-Z_]+)")


def _route_scope(route: Route) -> Optional[str]:
    """The scope a route declares, if any."""
    tree = _tree()
    if tree is None:
        return None
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if node.name != route.handler:
            continue
        for dec in node.decorator_list:
            match = _SCOPE_RX.search(ast.unparse(dec))
            if match:
                return match.group(1)
        try:
            match = _SCOPE_RX.search(ast.unparse(node.args))
        except Exception:
            return None
        if match:
            return match.group(1)
    return None


def _declared_scopes() -> set:
    return {s for s in (_route_scope(r) for r in _routes()) if s}


# ── Track D — failure semantics ──────────────────────────────────────

def dash_013() -> Result:
    """Mode-sync failure returns 200 and permits mode divergence."""
    src = _handler_src("api_set_mode")
    if not src:
        return REMEDIATED, "/api/mode removed"
    if "TOOL_GATE" in src or "_proxy_post" in src:
        return LIVE, "/api/mode still syncs to Tool Gate; check failure status"
    return REMEDIATED, (
        "premise removed with DASH-002: /api/mode no longer syncs, so there "
        "is no sync failure to mask (invalid mode raises 400)"
    )


# Ways a failure path can answer without claiming success. A bare
# ``return {...}`` is not one of them: FastAPI serialises it as 200.
NON_SUCCESS_MARKERS = ("raise", "status_code", "degraded_response",
                       "JSONResponse", "HTTPException", "_sse_error")


def _swallowing_handlers() -> List[str]:
    """Route handlers whose except-path returns a 200 body."""
    tree = _tree()
    if tree is None:
        return []
    offenders = []
    routed = {r.handler for r in _routes()}
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        # Only routes matter: a helper returning a dict is not an HTTP 200.
        if node.name not in routed:
            continue
        for handler in ast.walk(node):
            if not isinstance(handler, ast.ExceptHandler):
                continue
            body = ast.unparse(handler)
            if "return" not in body and "yield" not in body:
                continue
            if any(m in body for m in NON_SUCCESS_MARKERS):
                continue
            offenders.append(node.name)
            break
    return sorted(set(offenders))


def dash_016() -> Result:
    """Backend failures returned as HTTP-200 fallback objects."""
    if _tree() is None:
        return MANUAL, "dashboard source did not parse"
    offenders = _swallowing_handlers()
    if offenders:
        return LIVE, (f"{len(offenders)} route handler(s) swallow backend failure "
                      f"into a 200 body: {', '.join(offenders[:4])}")
    return REMEDIATED, (
        "every route's failure path answers with a non-success status"
    )


def dash_061() -> Result:
    """Every successful HTTP response classified as a healthy node."""
    src = _handler_src("fetch_status")
    if not src:
        return MANUAL, "fetch_status() not found"
    if '"status": "ok"' in src or "'status': 'ok'" in src:
        # This condition was computed and then discarded — `pass`, and
        # the verdict below fired regardless. So a handler that *does*
        # read the backend's own status would still have been reported
        # LIVE, purely for containing the literal `"status": "ok"`
        # somewhere. Found by the I-5 inert-rule detector; wiring it up
        # is what the shape of the code always intended.
        reads_backend_status = (
            "json()" in src and "get(" in src
            and "status" in src.split("raise_for_status")[-1]
        )
        if not reads_backend_status:
            return LIVE, ("fetch_status() marks any non-raising response "
                          "'ok' without reading backend status")
    return REMEDIATED, "node health derived from backend-reported status"


def dash_063() -> Result:
    """Go/no-go counts all ledger entries, not recent approved successes.

    Remediated either by measuring the right thing, or by reporting the
    metric as unavailable. Substituting a total count and calling it
    proof is the defect; declining to measure is not.
    """
    src = _handler_src("build_go_no_go_report")
    if not src:
        return MANUAL, "build_go_no_go_report() not found"
    declared_unavailable = "unavailable_metric" in src
    uses_total_as_proof = re.search(
        r"(minimum_gate_decisions|NO_GO_GRACE_REQUESTS)", src)
    if uses_total_as_proof and not declared_unavailable:
        return LIVE, "uses total ledger count as proof metric"
    if declared_unavailable:
        return REMEDIATED, (
            "the proof metric is declared unavailable rather than "
            "substituted with a total count"
        )
    return REMEDIATED, "proof metric no longer a raw total count"


def dash_064() -> Result:
    """Go/no-go uses dashboard caller error ratio, not system reliability.

    The dashboard's own error ratio may still be *reported* — it is a
    real number about a real thing. The defect is it being the value the
    GO/NO_GO decision turns on.
    """
    src = _handler_src("build_go_no_go_report")
    if not src:
        return MANUAL, "build_go_no_go_report() not found"
    decides_on_fleet = "fleet_unhealthy_ratio" in src or "healthy_nodes" in src
    # Does a caller-error comparison still gate the decision?
    gates_on_caller_error = re.search(
        r"error_ratio\s*>\s*MAX_ERROR_RATIO", src)
    if gates_on_caller_error and not decides_on_fleet:
        return LIVE, "reliability judged by the dashboard's own HTTP error ratio"
    if decides_on_fleet:
        return REMEDIATED, "the decision turns on observed fleet health"
    return REMEDIATED, "reliability metric sourced from execution/fleet data"


def dash_065() -> Result:
    """Backup status reports a fresh healthy timestamp without a backup."""
    src = _handler_src("api_backup_status")
    if not src:
        return REMEDIATED, "backup status route removed"
    if "service healthy" in src and "utcnow" in src:
        return LIVE, "synthesises 'now (service healthy)' from a liveness probe only"
    return REMEDIATED, "backup status reflects a verified backup"


def dash_066() -> Result:
    """Corrections API fabricates the current time for aggregate counters."""
    src = _handler_src("api_corrections")
    if not src:
        return REMEDIATED, "corrections route removed"
    if "utcnow" in src and "timestamp" in src:
        return LIVE, "stamps aggregate counters with now() as if they were events"
    return REMEDIATED, "correction timestamps come from the source records"


def dash_067() -> Result:
    """Backend outages silently become empty lists or neutral data."""
    tree = _tree()
    if tree is None:
        return MANUAL, "dashboard source did not parse"
    empties = 0
    for node in ast.walk(tree):
        if not isinstance(node, ast.ExceptHandler):
            continue
        body = ast.unparse(node)
        if any(m in body for m in NON_SUCCESS_MARKERS):
            continue
        if re.search(r"return \{[^}]*: (\[\]|\{\}|0|None)", body):
            empties += 1
    if empties:
        return LIVE, f"{empties} exception paths return empty/neutral data as if authoritative"
    return REMEDIATED, "outages surface as degraded state, not absence"


def dash_080() -> Result:
    """NO_GO returned with HTTP 200 rather than a machine-enforcing status."""
    src = _handler_src("go_no_go")
    if not src:
        return MANUAL, "go_no_go route not found"
    if "status_code" in src or "JSONResponse" in src or "HTTPException" in src:
        return REMEDIATED, "NO_GO carries a distinguishing status"
    return LIVE, "NO_GO returned as a plain 200 body; nothing downstream can enforce it"


def dash_082() -> Result:
    """Nudge and correction outages silently represented as no data."""
    live = []
    for name in ("api_nudges", "api_corrections"):
        src = _handler_src(name)
        if src and re.search(r"except[^:]*:.*return \{[^}]*: \[\]", src, re.S):
            live.append(name)
    if live:
        return LIVE, f"outage indistinguishable from empty in: {', '.join(live)}"
    return REMEDIATED, "nudge/correction outages are distinguishable from empty"


def dash_083() -> Result:
    """Synthetic correction/backup timestamps use naive UTC strings."""
    hits = _text().count("datetime.utcnow()")
    if hits:
        return LIVE, f"{hits} uses of naive datetime.utcnow()"
    return REMEDIATED, "no naive utcnow() timestamps"


# ── Track E/G — bounds and disclosure ────────────────────────────────

def dash_017() -> Result:
    """Raw JSON proxy bodies have no byte, nesting or field-count limits."""
    text = _text()
    unbounded = text.count("await request.json()")
    if unbounded:
        return LIVE, f"{unbounded} raw request.json() read(s) with no bounds"
    if "bounded_json" not in text:
        return PARTIAL, "no bounded reader is in use"
    return REMEDIATED, (
        f"{text.count('await bounded_json(request)')} body read(s) bounded "
        f"by bytes, depth and key count"
    )


def dash_056() -> Result:
    """No global rate limit, concurrency cap or caller quota."""
    text = _text()
    if any(m in text for m in ("RateLimit", "rate_limit", "Semaphore", "quota")):
        return REMEDIATED, "gateway workload controls present"
    return LIVE, "no rate limit, concurrency cap or caller quota on the gateway"


def dash_069() -> Result:
    """Health exposes Tool Gate URL, policy version and policy hash."""
    src = _handler_src("health")
    if not src:
        return MANUAL, "health route not found"
    leaks = [k for k in ("tool_gate_url", "policy_version", "policy_hash") if k in src]
    if leaks:
        return LIVE, f"/health discloses {', '.join(leaks)}"
    return REMEDIATED, "/health discloses no internal topology"


def dash_088() -> Result:
    """HTML/static responses set no CSP, frame or referrer protections."""
    text = _text()
    headers = [h for h in ("Content-Security-Policy", "X-Frame-Options",
                           "Referrer-Policy") if h in text]
    if not headers:
        return LIVE, "no CSP, frame-ancestors or referrer policy on HTML responses"
    if len(headers) < 3:
        return PARTIAL, f"only {', '.join(headers)} set"
    return REMEDIATED, "browser security headers set on HTML responses"


def dash_057() -> Result:
    """Node health checks run sequentially."""
    src = _handler_src("fetch_status")
    if not src:
        return MANUAL, "fetch_status() not found"
    if "gather" in src or "TaskGroup" in src:
        return REMEDIATED, "node probes run concurrently"
    return LIVE, "probes NODES sequentially; worst case is the sum of all timeouts"


def dash_074() -> Result:
    """Many proxy routes create a new HTTP client per request."""
    text = _text()
    per_request = text.count("async with httpx.AsyncClient(")
    if per_request:
        return LIVE, f"{per_request} per-request httpx.AsyncClient() constructions"
    if "pooled_client" not in text:
        return PARTIAL, "no per-request clients, but no shared pool either"
    borrowed = text.count("async with pooled_client(")
    return REMEDIATED, (
        f"{borrowed} call site(s) borrow one pooled client; connections "
        f"are reused rather than rebuilt per request"
    )


def dash_087() -> Result:
    """Unified app shell reads the complete HTML file synchronously."""
    src = _handler_src("app_shell")
    if not src:
        return MANUAL, "app_shell route not found"
    if "read_text" in src or "open(" in src:
        return LIVE, "reads the app shell from disk on every request"
    return REMEDIATED, "app shell served from a cached read"


def dash_096() -> Result:
    """Audit logging optional and records only method/path/status."""
    text = _text()
    if 'AUDIT_REQUIRED", "false"' in text or "AUDIT_REQUIRED', 'false'" in text:
        return LIVE, "audit stream defaults to not-required"
    if "AuditStream" not in text:
        return MANUAL, "no AuditStream found; verify auditing exists"
    if "_audit_actor" not in text:
        return PARTIAL, "audit required by default but records no actor"
    src = _handler_src("_audit_actor")
    # The actor must come from the credential, not from a header a caller
    # can set to whatever it likes.
    if "authorization" not in src.lower():
        return PARTIAL, "an actor is recorded but not derived from the credential"
    if "sha256" not in src:
        return PARTIAL, "the actor appears to be logged without digesting"
    return REMEDIATED, (
        "audit is required by default and records a credential-derived "
        "actor digest, never the credential itself"
    )


def _identity_call_sites() -> List[Tuple[str, str]]:
    """(handler, snippet) for every backend call passing a user identity."""
    tree = _tree()
    if tree is None:
        return []
    sites = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        src = ast.unparse(node)
        if "user_id" in src:
            sites.append((node.name, src))
    return sites


def dash_023() -> Result:
    """Endpoints hard-code the global `keeper` identity.

    Checked by behaviour rather than by the string "keeper": a literal
    identity passed to a backend is the defect, whatever it is named.
    """
    hard_coded = []
    for handler, src in _identity_call_sites():
        for match in re.finditer(r"'user_id':\s*'([^']+)'", src):
            hard_coded.append(f"{handler} sends user_id={match.group(1)!r}")
    if hard_coded:
        return LIVE, "; ".join(hard_coded[:3])
    if not _identity_call_sites():
        return MANUAL, "no user_id is sent at all; verify memory reads are scoped"
    return REMEDIATED, (
        f"{len(_identity_call_sites())} handler(s) pass the caller's identity, "
        f"none hard-coded"
    )


def dash_d02() -> Result:
    """`/memory/retrieve` requires user_id; omitting it returned 422.

    Found while fixing KAI-DASH-023. The search path of `/api/memories`
    called the backend without a required parameter, so memory search had
    never worked from the dashboard — it answered 422 every time.
    """
    for handler, src in _identity_call_sites():
        if handler != "api_memories":
            continue
        if "memory/retrieve" in src and "user_id" in src:
            return REMEDIATED, "api_memories passes the required user_id"
        return LIVE, "api_memories calls /memory/retrieve without required user_id"
    src = _handler_src("api_memories")
    if not src:
        return REMEDIATED, "api_memories no longer exists"
    if "memory/retrieve" not in src:
        return REMEDIATED, "api_memories no longer calls /memory/retrieve"
    return LIVE, "api_memories calls /memory/retrieve without required user_id"


def dash_053() -> Result:
    """Chat request bodies are unbounded."""
    src = _handler_src("api_chat_proxy")
    if not src:
        return REMEDIATED, "chat proxy removed"
    if "bounded_json" in src:
        return REMEDIATED, "chat body goes through the shared bounded reader"
    return LIVE, "chat body is read without a size bound"


def dash_054() -> Result:
    """Chat proxy streams backend 4xx/5xx bodies without checking status."""
    src = _handler_src("api_chat_proxy")
    if not src:
        return REMEDIATED, "chat proxy removed"
    if "status_code" not in src:
        return LIVE, "backend status is never inspected before streaming"
    # The check must come before the body is forwarded, not after.
    guard = re.search(r"resp\.status_code\s*>=?\s*[45]\d\d", src)
    if not guard:
        return PARTIAL, "status_code appears but no >=400 guard was found"
    stream_at = src.find("aiter_bytes")
    if stream_at != -1 and guard.start() > stream_at:
        return LIVE, "the status guard runs after streaming has begun"
    return REMEDIATED, "backend status is validated before any chunk is forwarded"


def dash_055() -> Result:
    """Chat connection exceptions are sent to the browser as diagnostics."""
    src = _handler_src("api_chat_proxy")
    if not src:
        return REMEDIATED, "chat proxy removed"
    for handler in ast.walk(ast.parse(src)):
        if not isinstance(handler, ast.ExceptHandler):
            continue
        name = handler.name
        if not name:
            continue
        body = ast.unparse(handler)
        # The exception may be logged; it must not be yielded or returned.
        for stmt in handler.body:
            emitted = ast.unparse(stmt)
            if not isinstance(stmt, (ast.Expr, ast.Return)):
                continue
            if ("yield" in emitted or "return" in emitted) and name in emitted:
                return LIVE, f"exception text reaches the client: {emitted[:80]}"
    return REMEDIATED, "internal exception detail is logged, not returned"


def forced_media_type(handler: str, label: str) -> Callable[[], Result]:
    """The response type must derive from the backend, not a constant.

    Checked on the ``media_type=`` argument specifically. Searching the
    handler for the literal string would now match the allow-list of
    recognised types, which is the opposite of the defect.
    """
    def check() -> Result:
        src = _handler_src(handler)
        if not src:
            return REMEDIATED, f"{label} proxy removed"
        forced = []
        for node in ast.walk(ast.parse(src)):
            if not isinstance(node, ast.Call):
                continue
            for kw in node.keywords:
                if kw.arg != "media_type":
                    continue
                if isinstance(kw.value, ast.Constant):
                    forced.append(str(kw.value.value))
        if forced:
            return LIVE, (f"{label} response type is forced to "
                          f"{', '.join(forced)} regardless of the backend")
        return REMEDIATED, f"{label} media type derives from the backend response"
    return check


def _upload_handlers() -> List[str]:
    tree = _tree()
    if tree is None:
        return []
    found = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        try:
            sig = ast.unparse(node.args)
        except Exception:
            continue
        if "UploadFile" in sig:
            found.append(node.name)
    return found


def unbounded_upload(*handlers: str) -> Callable[[], Result]:
    """An upload limit enforced after the read is not a limit.

    Reading the whole body and then checking its length still buffers
    whatever was sent, so this requires the bounded reader specifically —
    a `len(...)` guard after `await file.read()` does not count.
    """
    def check() -> Result:
        offenders = []
        for name in handlers:
            src = _handler_src(name)
            if not src:
                continue
            if "bounded_upload" in src:
                continue
            if "await file.read()" in src or ".read()" in src:
                offenders.append(f"{name} reads the whole upload first")
            else:
                offenders.append(f"{name} has no size bound")
        if offenders:
            return LIVE, "; ".join(offenders[:3])
        return REMEDIATED, (
            f"{len(handlers)} upload handler(s) refuse during the read"
        )
    return check


def unbounded_binary_response(handler: str, label: str) -> Callable[[], Result]:
    """A proxy that materialises any backend response inherits its size."""
    def check() -> Result:
        src = _handler_src(handler)
        if not src:
            return REMEDIATED, f"{label} route removed"
        if "bounded_response" in src:
            return REMEDIATED, f"{label} response is size-bounded before forwarding"
        if "resp.content" in src or "aread()" in src:
            return LIVE, f"{label} forwards the backend response unbounded"
        return MANUAL, f"{label} response handling needs review"
    return check


def dash_093() -> Result:
    """List/search limits accept negative or extreme values."""
    tree = _tree()
    if tree is None:
        return MANUAL, "dashboard source did not parse"
    unconstrained = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for arg, default in zip(
            node.args.args[-len(node.args.defaults):] if node.args.defaults else [],
            node.args.defaults,
        ):
            if arg.arg not in {"top_k", "limit", "since", "days", "offset"}:
                continue
            rendered = ast.unparse(default)
            if "Query(" not in rendered:
                unconstrained.append(f"{node.name}({arg.arg}={rendered})")
    if unconstrained:
        return LIVE, (f"{len(unconstrained)} unconstrained limit param(s): "
                      f"{', '.join(unconstrained[:3])}")
    return REMEDIATED, "every list/search limit declares a safe range"


def dash_084() -> Result:
    """SSE keepalive timestamps are naive UTC strings."""
    src = _handler_src("sse_events")
    if not src:
        return REMEDIATED, "event stream removed"
    if "utcnow" in src:
        return LIVE, "SSE heartbeat uses naive datetime.utcnow()"
    if "heartbeat" not in src:
        return MANUAL, "no heartbeat found in sse_events(); verify by hand"
    if "timezone.utc" not in src:
        return PARTIAL, "heartbeat present but no timezone-aware stamp found"
    return REMEDIATED, "SSE heartbeat carries a timezone-aware timestamp"


def dash_051() -> Result:
    """Uploaded filenames are forwarded without canonicalisation."""
    text = _text()
    raw = re.findall(r"file\.filename or ", text)
    if raw:
        return LIVE, f"{len(raw)} filename(s) forwarded without canonicalisation"
    if "safe_filename" not in text:
        return LIVE, "no filename canonicalisation before forwarding"
    forwarded = re.findall(r'files=\{"file": \(([^,]+),', text)
    unsafe = [f for f in forwarded if "safe_filename" not in f]
    if unsafe:
        return LIVE, f"raw filename forwarded: {', '.join(unsafe[:3])}"
    return REMEDIATED, (
        f"{len(forwarded)} forwarded filename(s) canonicalised"
    )


def dash_050() -> Result:
    """Upload routing trusts filename extension and caller content type."""
    src = _handler_src("api_upload")
    if not src:
        return REMEDIATED, "upload route removed"
    if "safe_filename" not in src:
        return LIVE, "routing trusts an uncanonicalised filename"
    # The extension still selects the backend; that is a routing hint, not
    # a trust decision, provided the name it comes from is sanitised and
    # unknown types are refused.
    if "415" not in src:
        return PARTIAL, "unknown extensions are not refused"
    return REMEDIATED, (
        "routing uses a canonicalised name and refuses unknown types"
    )


def dash_052() -> Result:
    """Proxy error details disclose internal URLs and transport diagnostics."""
    text = _text()
    leaks = re.findall(r"detail=f[\"']([^\"']*)\{exc\}", text)
    if leaks:
        return LIVE, f"{len(leaks)} error detail(s) interpolate the exception"
    if "client_error" not in text:
        return MANUAL, "no exception interpolation found; verify by hand"
    return REMEDIATED, (
        f"{text.count('client_error(')} error path(s) log the cause and "
        f"return a safe message"
    )


def dash_091() -> Result:
    """Audio and vision proxies trust caller-supplied media types."""
    text = _text()
    trusted = re.findall(r"file\.content_type or ", text)
    if trusted:
        return LIVE, (f"{len(trusted)} proxy call(s) forward the caller's "
                      f"declared content type unchecked")
    return REMEDIATED, "caller media types are not forwarded unchecked"


def dash_043() -> Result:
    """One malformed event payload terminates the client stream."""
    src = _handler_src("sse_events")
    if not src:
        return REMEDIATED, "event stream removed"
    if "json.loads" not in src and "_json.loads" not in src:
        return MANUAL, "no event parsing found in sse_events()"
    # The parse must be individually guarded, not merely inside the
    # outer try that also owns the loop.
    if "ValueError" in src or "JSONDecodeError" in src:
        return REMEDIATED, "a malformed event is dropped, not fatal to the stream"
    return LIVE, "an unparseable event payload ends the subscriber's stream"


def dash_058() -> Result:
    """Root status duplicates fleet probes."""
    text = _text()
    if "_status_cache" in text or "lru_cache" in _handler_src("fetch_status"):
        return REMEDIATED, "repeated status calls in one cycle reuse one fan-out"
    calls = text.count("await fetch_status()")
    if calls > 1:
        return LIVE, f"{calls} independent fan-outs per request cycle"
    return REMEDIATED, "a single fan-out per request"


def dash_060() -> Result:
    """Readiness calls the full root fan-out."""
    src = _handler_src("readiness")
    if not src:
        return REMEDIATED, "readiness route removed"
    if "await index()" in src:
        return LIVE, "readiness rebuilds the entire root payload"
    if "fetch_status" not in src:
        return PARTIAL, "readiness no longer calls index(), but checks nothing"
    return REMEDIATED, "readiness performs a bounded core-node check"


def dash_062() -> Result:
    """Go/no-go accepts fallback zero counts as readiness evidence."""
    src = _handler_src("index")
    if not src:
        return MANUAL, "index() not found"
    if re.search(r"(ledger_size|memory_count)\s*>=\s*0", src):
        return LIVE, "a fallback zero count satisfies the readiness test"
    if "is not None" not in src:
        return PARTIAL, "readiness no longer uses >= 0, but observation is unclear"
    return REMEDIATED, "readiness requires the counts to have been observed"


def dash_059() -> Result:
    """The built-in UI polls the expensive root fan-out too often."""
    page = REPO / "dashboard" / "static" / "app.html"
    try:
        text = page.read_text(encoding="utf-8")
    except OSError:
        return MANUAL, "app.html not readable"
    match = re.search(r"setInterval\(\s*refreshDashboard\s*,\s*(\d+)\s*\)", text)
    if not match:
        return MANUAL, "no refreshDashboard interval found"
    interval = int(match.group(1))
    if interval < 10_000:
        return LIVE, f"root fan-out polled every {interval}ms per browser"
    return REMEDIATED, f"root fan-out polled every {interval}ms per browser"


def dash_068() -> Result:
    """Root response exposes internal topology and policy state."""
    src = _handler_src("index")
    if not src:
        return MANUAL, "index() not found"
    leaked = [k for k in ("tool_gate_url", "policy_version", "policy_hash")
              if f'"{k}"' in src]
    if not leaked:
        return REMEDIATED, "root payload carries no topology or policy state"
    if "EXPOSE_TOPOLOGY" in src:
        return REMEDIATED, (
            f"{', '.join(leaked)} only present behind an explicit flag"
        )
    return LIVE, f"root payload always discloses {', '.join(leaked)}"


def dash_077() -> Result:
    """Root returns complete nested backend health rather than minimised."""
    src = _handler_src("index")
    if not src:
        return MANUAL, "index() not found"
    if '"node_status": statuses' in src:
        return LIVE, "every backend's full health document is forwarded"
    if "details" not in src:
        return PARTIAL, "node_status is transformed but the shape is unclear"
    return REMEDIATED, "per-node detail is stripped from the root payload"


def dash_072() -> Result:
    """Backend URLs are environment-controlled without validation."""
    text = _text()
    raw = re.findall(r'os\.getenv\("[A-Z0-9_]*URL"', text)
    if raw:
        return LIVE, f"{len(raw)} backend URL(s) read without validation"
    if "backend_url" not in text:
        return MANUAL, "no backend URLs found; verify by hand"
    return REMEDIATED, (
        f"{text.count('backend_url(')} backend URL(s) validated for scheme "
        f"and host at import"
    )


def dash_078() -> Result:
    """Go/no-go thresholds parsed without safe-range validation."""
    text = _text()
    if re.search(r"(int|float)\(os\.getenv\(\"(NO_GO_GRACE_REQUESTS|MAX_ERROR_RATIO)", text):
        return LIVE, "thresholds parsed with a bare int()/float()"
    if "_bounded_setting" not in text:
        return MANUAL, "threshold parsing not found"
    return REMEDIATED, "thresholds are range-checked at import and fail loudly"


def dash_079() -> Result:
    """Malformed backend numeric fields can raise during conversion."""
    src = _handler_src("build_go_no_go_report") + _handler_src("index")
    if not src:
        return MANUAL, "go/no-go and index not found"
    bare = re.findall(r"(?<!_as_number\()\b(?:int|float)\(\s*\w+(?:_stats|metrics)\.get", src)
    if bare:
        return LIVE, f"{len(bare)} bare numeric conversion(s) on backend fields"
    if "_as_number" not in src:
        return PARTIAL, "no bare conversions found, but no coercion helper either"
    return REMEDIATED, "backend numerics are coerced, degrading one field not the response"


def dash_086() -> Result:
    """`_publish_event()` is fire-and-forget with silent delivery loss."""
    src = _handler_src("_publish_event")
    if not src:
        return REMEDIATED, "event publishing removed"
    if "logger.debug" in src and "return" not in src:
        return LIVE, "publish failures are swallowed at debug level"
    if "-> bool" not in src and "return True" not in src:
        return PARTIAL, "publish failures are logged but delivery is not reported"
    return REMEDIATED, "delivery is reported and failures are counted"


def dash_094() -> Result:
    """Path parameters are interpolated into backend URLs unvalidated."""
    text = _text()
    if "safe_symbol" not in text:
        return LIVE, "no path-parameter validation before URL interpolation"
    tree = _tree()
    if tree is None:
        return MANUAL, "dashboard source did not parse"
    unguarded = []
    for route in _routes():
        if "{" not in route.path:
            continue
        src = _handler_src(route.handler)
        if not src:
            continue
        # Only routes that interpolate the parameter into a URL matter.
        if not re.search(r"f\"\{[A-Z_]+URL\}[^\"]*\{", src):
            continue
        if "safe_symbol" not in src:
            unguarded.append(f"{route.method.upper()} {route.path}")
    if unguarded:
        return LIVE, f"unvalidated path interpolation: {', '.join(unguarded[:3])}"
    return REMEDIATED, "path parameters are validated before URL interpolation"


def dash_095() -> Result:
    """Broker-watch symbol and threshold validation permits malformed rules."""
    src = _handler_src("api_broker_watch")
    if not src:
        return REMEDIATED, "broker-watch route removed"
    if "safe_symbol" not in src:
        return LIVE, "the symbol reaches an outbound URL unvalidated"
    if "_as_number" not in src:
        return LIVE, "the threshold is never validated as a finite number"
    return REMEDIATED, "symbol and threshold are both validated"


def _resilience_src() -> str:
    try:
        return (REPO / "common" / "resilience.py").read_text(encoding="utf-8")
    except OSError:
        return ""


def dash_014() -> Result:
    """Retried POSTs can duplicate non-idempotent operations."""
    src = _resilience_src()
    if not src:
        return MANUAL, "common/resilience.py not readable"
    if "idempotent" not in src:
        return LIVE, "resilient_call retries every method, including POST"
    if "attempts = retries if idempotent else 1" not in src:
        return PARTIAL, "an idempotent flag exists but does not gate retries"
    return REMEDIATED, "retries are limited to idempotent methods unless declared"


def dash_015() -> Result:
    """Every 4xx response counted as a successful dependency call."""
    src = _resilience_src()
    if not src:
        return MANUAL, "common/resilience.py not readable"
    # Look at what actually guards record_success(), not at any mention
    # of a status code: this module's own docstring names the old test,
    # and the 4xx branch legitimately compares against 500.
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return MANUAL, "common/resilience.py did not parse"
    for node in ast.walk(tree):
        if not isinstance(node, ast.If):
            continue
        body = ast.unparse(node.body)
        if "record_success" not in body:
            continue
        test = ast.unparse(node.test)
        if "500" in test:
            return LIVE, f"success is recorded when {test}"
        if "300" in test and "200" in test:
            return REMEDIATED, (
                f"success requires {test}; 4xx returns without retry"
            )
        return PARTIAL, f"success is guarded by an unclear test: {test}"
    return MANUAL, "no record_success() guard found"


def dash_075() -> Result:
    """A new client is built for every retry attempt."""
    src = _resilience_src()
    if not src:
        return MANUAL, "common/resilience.py not readable"
    body = src[src.find("async def resilient_call"):]
    body = body[:body.find("\ndef ")] if "\ndef " in body else body
    loop_at = body.find("for attempt in range")
    client_at = body.find("pooled_client(")
    if "httpx.AsyncClient(" in body:
        return LIVE, "a client is constructed inside resilient_call"
    if client_at == -1:
        return MANUAL, "no client construction found"
    if loop_at != -1 and client_at > loop_at:
        return LIVE, "the client is created inside the retry loop"
    return REMEDIATED, "one pooled client is reused across every attempt"


def dash_076() -> Result:
    """Backend response bodies are not byte-bounded."""
    src = _resilience_src()
    if not src:
        return MANUAL, "common/resilience.py not readable"
    if "max_response_bytes" not in src:
        return LIVE, "resilient_call parses any response size"
    return REMEDIATED, "backend responses are size-bounded before parsing"


def dash_044() -> Result:
    """Event channels have no per-subscriber filtering."""
    src = _handler_src("sse_events")
    if not src:
        return REMEDIATED, "event stream removed"
    if "DashboardPrincipal" not in src:
        return LIVE, "the stream has no subscriber identity to filter on"
    if "_event_visible_to" not in src:
        return LIVE, "every subscriber receives every event"
    return REMEDIATED, "events carrying a subject reach only that subject"


def dash_070() -> Result:
    """Health always reports running and is not a readiness contract."""
    health = _handler_src("health")
    ready = _handler_src("readiness")
    if not health:
        return MANUAL, "health route not found"
    if not ready:
        return LIVE, "there is no readiness endpoint distinct from health"
    # Health must be cheap and unconditional; readiness must actually check.
    if "fetch_status" in health or "await" in health.replace("async def", ""):
        return PARTIAL, "/health performs work; it should be liveness only"
    if "fetch_status" not in ready:
        return PARTIAL, "/readiness exists but checks nothing"
    return REMEDIATED, (
        "/health is liveness only; /readiness checks core nodes and 503s"
    )


def dash_071() -> Result:
    """The node inventory can drift from deployed services."""
    text = _text()
    tree = _tree()
    if tree is None:
        return MANUAL, "dashboard source did not parse"
    if "NODES" not in text:
        return MANUAL, "no NODES map found"

    # Every node URL must come from the validated reader, so a service
    # that is renamed or removed fails at import rather than silently
    # probing a host that no longer exists.
    node_block = text[text.find("NODES: Dict[str, str] = {"):]
    node_block = node_block[:node_block.find("\n}")]
    if not node_block:
        return MANUAL, "could not isolate the NODES map"
    raw = re.findall(r'os\.getenv\(', node_block)
    if raw:
        return LIVE, f"{len(raw)} node URL(s) bypass validation and can drift"
    if "backend_url(" not in node_block:
        return PARTIAL, "node URLs are neither validated nor obviously static"
    return REMEDIATED, (
        "every node address is validated at import, so a bad or removed "
        "backend fails the container start"
    )


def dash_085() -> Result:
    """Redis clients are constructed per publisher rather than reused."""
    text = _text()
    constructions = text.count("aioredis.from_url(")
    if constructions > 2:
        return LIVE, f"{constructions} Redis client construction sites"
    if "_REDIS_CLIENT" not in text:
        return PARTIAL, "few construction sites but no lifecycle-managed client"
    return REMEDIATED, "one lazily-created Redis client, closed on shutdown"


def dash_092() -> Result:
    """Binary responses are fully materialised before forwarding."""
    text = _text()
    if "bounded_response" not in text:
        return LIVE, "binary responses are forwarded without a size bound"
    unbounded = re.findall(r"content=resp\.content(?!\s*\))", text)
    bare = text.count("content=resp.content,")
    if bare:
        return LIVE, f"{bare} binary response(s) forwarded unbounded"
    return REMEDIATED, (
        "binary responses are size-checked before being materialised into "
        "a reply"
    )


def dash_d03() -> Result:
    """The unauthenticated escape hatch bypassed authorisation entirely.

    `KAI_ALLOW_UNAUTHENTICATED=true` is meant to skip *authentication*
    for local development. It returned a principal and status 200 before
    the scope check ran, so setting one development flag granted more
    authority than any configured credential could — including identity
    rewrite. Found by a test written to assert the opposite.
    """
    path = REPO / "common" / "dashboard_auth.py"
    try:
        src = path.read_text(encoding="utf-8")
    except OSError:
        return MANUAL, "common/dashboard_auth.py not readable"
    if "unauthenticated_allowed" not in src:
        return REMEDIATED, "no escape hatch exists"
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return MANUAL, "dashboard_auth.py did not parse"
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if node.name != "authenticate":
            continue
        body = ast.unparse(node)
        head = body[:body.find("if not authorization")] if "if not authorization" in body else body
        if "local-development" not in head:
            return MANUAL, "escape-hatch branch not located"
        if ".may(" not in head:
            return LIVE, "the escape hatch returns before the scope check"
        return REMEDIATED, "the escape hatch is still subject to scope checks"
    return MANUAL, "authenticate() not found"


def dash_d04() -> Result:
    """A single default credential carried keeper authority.

    Five scopes and three roles are decorative if the default deployment
    hands out the top one: one leaked token would carry authority to
    rewrite SOUL.md and drive the browser agent.
    """
    path = REPO / "common" / "dashboard_auth.py"
    try:
        src = path.read_text(encoding="utf-8")
    except OSError:
        return MANUAL, "common/dashboard_auth.py not readable"
    if "ROLE_ENV, Role.KEEPER.value" in src:
        return LIVE, "the default role is keeper"
    if "ROLE_ENV, Role.OPERATOR.value" not in src:
        return MANUAL, "default role could not be determined"

    offenders = []
    for name in ("docker-compose.full.yml", "docker-compose.minimal.yml",
                 "docker-compose.sovereign.yml"):
        compose = REPO / name
        if not compose.exists():
            continue
        if "KAI_DASHBOARD_ROLE:-keeper" in compose.read_text(encoding="utf-8"):
            offenders.append(name)
    setup = REPO / "scripts" / "setup_service_token.sh"
    if setup.exists() and "KAI_DASHBOARD_ROLE=%s\\n' \"keeper\"" in setup.read_text(encoding="utf-8"):
        offenders.append("setup_service_token.sh")
    if offenders:
        return LIVE, f"keeper is still the default in: {', '.join(offenders)}"
    return REMEDIATED, "the default is operator; keeper must be asked for by name"


# ── The finding table ────────────────────────────────────────────────

class Finding(NamedTuple):
    severity: str
    track: str
    title: str
    check: Callable[[], Result]


C, H, M = "CRITICAL", "HIGH", "MEDIUM"

# Route groups reused across findings.
_BROWSER = ("/api/browser/", MUTATING)
_MONITOR = ("/api/monitor/", MUTATING)

FINDINGS: Dict[str, Finding] = {
    # ── Track A — inbound identity (foundation) ──
    "KAI-DASH-001": Finding(C, "A", "Open privileged gateway", dash_001),
    "KAI-DASH-002": Finding(C, "A", "Server-token confused deputy", dash_002),
    "KAI-DASH-011": Finding(H, "A", "No principal model", dash_011),
    "KAI-DASH-012": Finding(H, "A", "Token use lacks delegation evidence", dash_012),
    "KAI-DASH-018": Finding(H, "A", "No backend least privilege", dash_018),

    # ── Track B — authority on mutating routes ──
    "KAI-DASH-003": Finding(C, "B", "SOUL identity rewrite proxy",
                            route_auth(("post", "/api/soul"))),
    "KAI-DASH-004": Finding(C, "B", "Agent-registry rewrite proxy",
                            route_auth(("post", "/api/agents-registry"))),
    "KAI-DASH-005": Finding(C, "B", "Open private-context chat",
                            route_auth(("post", "/api/chat"))),
    "KAI-DASH-006": Finding(C, "B", "Moral-model mutation gateway",
                            prefix_auth(r"^/api/(values|conscience|loyalty|gratitude)/", MUTATING)),
    "KAI-DASH-007": Finding(C, "B", "External-message scheduling gateway",
                            prefix_auth(r"^/api/(reminders|schedule|briefing)", MUTATING)),
    "KAI-DASH-008": Finding(C, "B", "Browser automation gateway",
                            prefix_auth(*_BROWSER)),
    "KAI-DASH-009": Finding(C, "B", "Monitoring/action-rule gateway",
                            prefix_auth(*_MONITOR)),
    "KAI-DASH-010": Finding(C, "B", "File-watcher path gateway",
                            route_auth(("post", "/api/files/watch"))),
    "KAI-DASH-014": Finding(H, "B", "Retried mutation duplication",
                            dash_014),
    "KAI-DASH-019": Finding(H, "B", "Dream mutation exposure",
                            route_auth(("post", "/api/dream"))),
    "KAI-DASH-026": Finding(H, "B", "Goal mutation exposure",
                            route_auth(("post", "/api/goals"), ("post", "/api/goals/update"))),
    "KAI-DASH-027": Finding(H, "B", "Feedback poisoning gateway",
                            route_auth(("post", "/api/feedback"))),
    "KAI-DASH-028": Finding(H, "B", "Narrative-state poisoning gateway",
                            prefix_auth(r"^/api/(autobiography|legacy|imagine)/", MUTATING)),
    "KAI-DASH-029": Finding(H, "B", "Escalation poisoning gateway",
                            route_auth(("post", "/api/nudge/escalate"))),
    "KAI-DASH-030": Finding(H, "B", "Operator-model manipulation/disclosure",
                            prefix_auth(r"^/api/(echo|cross-mode|oracle|shadow)/")),
    "KAI-DASH-032": Finding(H, "B", "Financial record mutation",
                            route_auth(("post", "/api/finance/cis/record"))),
    "KAI-DASH-035": Finding(H, "B", "Clipboard disclosure/control",
                            prefix_auth(r"^/api/clipboard/")),
    "KAI-DASH-036": Finding(H, "B", "Notification control",
                            prefix_auth(r"^/api/notify/")),
    "KAI-DASH-037": Finding(H, "B", "Screen-watcher control",
                            prefix_auth(r"^/api/screen-watcher/watch/", MUTATING)),
    "KAI-DASH-038": Finding(H, "B", "News refresh control",
                            route_auth(("post", "/api/news/refresh"))),
    "KAI-DASH-039": Finding(H, "B", "Sensitive-text forwarding",
                            route_auth(("post", "/api/pii/scan"))),

    # ── Track C — sensitive read authorisation ──
    "KAI-DASH-020": Finding(H, "C", "Security exploit disclosure",
                            route_auth(("get", "/api/security-audit"))),
    "KAI-DASH-021": Finding(H, "C", "Episode disclosure",
                            route_auth(("get", "/api/thinking"))),
    "KAI-DASH-022": Finding(H, "C", "Memory browsing disclosure",
                            prefix_auth(r"^/api/(memories|memory)")),
    "KAI-DASH-023": Finding(H, "C", "Hard-coded keeper scope", dash_023),
    "KAI-DASH-024": Finding(H, "C", "Graph view leaks security/ranking metadata",
                            route_auth(("get", "/api/memory/graph-data"))),
    "KAI-DASH-025": Finding(H, "C", "Personal-model disclosure",
                            prefix_auth(r"^/api/(emotion|identity|relationship|operator-model|eq)")),
    "KAI-DASH-031": Finding(H, "C", "Financial disclosure",
                            prefix_auth(r"^/api/finance/", {"get"})),
    "KAI-DASH-033": Finding(H, "C", "Broker disclosure",
                            prefix_auth(r"^/api/broker/(balance|positions|orders|pnl)")),
    "KAI-DASH-034": Finding(H, "C", "Email disclosure/control",
                            prefix_auth(r"^/api/email/")),
    "KAI-DASH-040": Finding(H, "C", "Log aggregation disclosure",
                            route_auth(("get", "/api/logs"))),
    "KAI-DASH-041": Finding(H, "C", "Public internal event bus",
                            route_auth(("get", "/api/events"))),
    "KAI-DASH-044": Finding(H, "C", "No event-level isolation",
                            dash_044),

    # ── Track D — failure semantics ──
    "KAI-DASH-013": Finding(H, "D", "Mode failure appears as normal success", dash_013),
    "KAI-DASH-015": Finding(H, "D", "4xx treated as dependency success",
                            dash_015),
    "KAI-DASH-016": Finding(H, "D", "Success-shaped fallbacks", dash_016),
    "KAI-DASH-054": Finding(H, "D", "Chat status not validated", dash_054),
    "KAI-DASH-061": Finding(H, "D", "HTTP success equals node health", dash_061),
    "KAI-DASH-062": Finding(H, "D", "False core readiness",
                            dash_062),
    "KAI-DASH-063": Finding(H, "D", "Invalid proof metric", dash_063),
    "KAI-DASH-064": Finding(H, "D", "Wrong reliability metric", dash_064),
    "KAI-DASH-065": Finding(H, "D", "False backup status", dash_065),
    "KAI-DASH-066": Finding(H, "D", "Fabricated correction chronology", dash_066),
    "KAI-DASH-067": Finding(H, "D", "Evidence outage becomes absence", dash_067),
    "KAI-DASH-080": Finding(M, "D", "Advisory-only NO_GO", dash_080),
    "KAI-DASH-082": Finding(M, "D", "Silent empty-state fallbacks", dash_082),

    # ── Track E — request/response bounds ──
    "KAI-DASH-017": Finding(H, "E", "Unbounded JSON fan-in", dash_017),
    "KAI-DASH-042": Finding(H, "E", "SSE connection exhaustion",
                            source_marker("MAX_SSE_CLIENTS",
                                          "no SSE admission limit; each client holds a Redis pubsub",
                                          "SSE admission limit present",
                                          present_means_live=False)),
    "KAI-DASH-045": Finding(H, "E", "Post-read upload limit",
                            unbounded_upload("api_upload")),
    "KAI-DASH-046": Finding(H, "E", "Unlimited audio upload",
                            unbounded_upload("api_audio_transcribe")),
    "KAI-DASH-047": Finding(H, "E", "Unlimited vision upload",
                            unbounded_upload("api_vision_analyze",
                                             "api_vision_presence")),
    "KAI-DASH-048": Finding(H, "E", "Unlimited TTS work/response",
                            unbounded_binary_response("api_tts_synthesize", "TTS")),
    "KAI-DASH-049": Finding(H, "E", "Unlimited screenshot response",
                            unbounded_binary_response("api_browser_screenshot",
                                                      "screenshot")),
    "KAI-DASH-053": Finding(H, "E", "Unbounded chat body", dash_053),
    "KAI-DASH-056": Finding(H, "E", "No gateway workload controls", dash_056),
    "KAI-DASH-076": Finding(M, "E", "Unbounded backend responses",
                            dash_076),
    "KAI-DASH-092": Finding(M, "E", "Binary response buffering",
                            dash_092),
    "KAI-DASH-093": Finding(M, "E", "Weak query limits", dash_093),

    # ── Track F — media and filename trust ──
    "KAI-DASH-050": Finding(H, "F", "Extension/MIME trust", dash_050),
    "KAI-DASH-051": Finding(H, "F", "Filename propagation", dash_051),
    "KAI-DASH-089": Finding(M, "F", "Forced TTS media type",
                            forced_media_type("api_tts_synthesize", "TTS")),
    "KAI-DASH-090": Finding(M, "F", "Forced screenshot media type",
                            forced_media_type("api_browser_screenshot", "screenshot")),
    "KAI-DASH-091": Finding(M, "F", "Caller media metadata trusted", dash_091),

    # ── Track G — disclosure minimisation ──
    "KAI-DASH-052": Finding(H, "G", "Internal error disclosure", dash_052),
    "KAI-DASH-055": Finding(H, "G", "Chat diagnostics leak", dash_055),
    "KAI-DASH-068": Finding(H, "G", "Root operational disclosure",
                            dash_068),
    "KAI-DASH-069": Finding(M, "G", "Health topology disclosure", dash_069),
    "KAI-DASH-077": Finding(M, "G", "Excess health detail",
                            dash_077),

    # ── Track H — fan-out and lifecycle ──
    "KAI-DASH-043": Finding(H, "H", "Malformed event denial",
                            dash_043),
    "KAI-DASH-057": Finding(H, "H", "Sequential health fan-out", dash_057),
    "KAI-DASH-058": Finding(H, "H", "Duplicate root fan-out",
                            dash_058),
    "KAI-DASH-059": Finding(H, "H", "UI amplification loop",
                            dash_059),
    "KAI-DASH-060": Finding(H, "H", "Readiness amplification",
                            dash_060),
    "KAI-DASH-074": Finding(M, "H", "Direct-client churn", dash_074),
    "KAI-DASH-075": Finding(M, "H", "Retry-client churn",
                            dash_075),
    "KAI-DASH-085": Finding(M, "H", "Redis lifecycle churn",
                            dash_085),
    "KAI-DASH-087": Finding(M, "H", "Blocking app-shell read", dash_087),

    # ── Track I — configuration, validation and hygiene ──
    "KAI-DASH-070": Finding(M, "I", "Liveness mislabeled as health",
                            dash_070),
    "KAI-DASH-071": Finding(M, "I", "Inventory drift",
                            dash_071),
    "KAI-DASH-072": Finding(M, "I", "Unvalidated backend destinations",
                            dash_072),
    "KAI-DASH-073": Finding(M, "I", "No backend identity proof",
                            manual("backend identity verification is a transport-layer change")),
    "KAI-DASH-078": Finding(M, "I", "Unsafe go/no-go configuration",
                            dash_078),
    "KAI-DASH-079": Finding(M, "I", "Malformed numeric backend data",
                            dash_079),
    "KAI-DASH-081": Finding(M, "I", "Deliberate mode split",
                            dash_002),  # same condition: resolved with the token removal
    "KAI-DASH-083": Finding(M, "I", "Naive synthetic times", dash_083),
    "KAI-DASH-084": Finding(M, "I", "Naive SSE heartbeat time", dash_084),
    "KAI-DASH-086": Finding(M, "I", "Silent event loss",
                            dash_086),
    "KAI-DASH-088": Finding(M, "I", "Missing browser security headers", dash_088),
    "KAI-DASH-094": Finding(M, "I", "Path interpolation",
                            dash_094),
    "KAI-DASH-095": Finding(M, "I", "Weak broker-watch rule validation",
                            dash_095),
    "KAI-DASH-096": Finding(M, "I", "Weak optional audit", dash_096),
}

TRACK_NAMES = {
    "A": "Inbound identity (foundation)",
    "B": "Authority on mutating routes",
    "C": "Sensitive read authorisation",
    "D": "Failure semantics",
    "E": "Request/response bounds",
    "F": "Media and filename trust",
    "G": "Disclosure minimisation",
    "H": "Fan-out and lifecycle",
    "I": "Configuration, validation and hygiene",
}


def _rel(path: Path) -> str:
    """Repo-relative display path, tolerating paths outside the repo."""
    try:
        return str(path.relative_to(REPO))
    except ValueError:
        return str(path)


def dash_d01() -> Result:
    """The browser UI must send the credentials the gateway now demands.

    Track A authenticated 179 routes. The shipped UI makes 121 `fetch()`
    calls and opens an `EventSource`, none of which carried a credential,
    so closing the gateway also closed it to its only real client. Found
    while checking whether Track A had broken anything it did not own.
    """
    static = REPO / "dashboard" / "static"
    if not (static / "auth.js").exists():
        return LIVE, "no dashboard/static/auth.js; the UI sends no credentials"

    unwired = []
    for page in sorted(static.glob("*.html")):
        text = page.read_text(encoding="utf-8")
        if "fetch(" not in text and "EventSource" not in text:
            continue
        if "/static/auth.js" not in text:
            unwired.append(page.name)

    if unwired:
        return LIVE, f"pages calling the API without auth.js: {', '.join(unwired)}"

    # EventSource cannot send headers at all, so a page still using it is
    # unauthenticated however well the fetch path is wired.
    raw_sse = [p.name for p in sorted(static.glob("*.html"))
               if "new EventSource(" in p.read_text(encoding="utf-8")]
    if raw_sse:
        return PARTIAL, (f"fetch is wired but raw EventSource remains in "
                         f"{', '.join(raw_sse)}; it cannot send credentials")
    return REMEDIATED, "UI attaches credentials on fetch and streams SSE authenticated"


# Findings discovered while remediating, which the original audit never
# saw. Numbered separately so they can never stand in for one of the 96.
DISCOVERED_ID_RX = re.compile(r"KAI-DASH-D\d{2}")

DISCOVERED: Dict[str, "Finding"] = {
    "KAI-DASH-D01": Finding(
        H, "A", "UI sends no credentials (regression from Track A)", dash_d01),
    "KAI-DASH-D02": Finding(
        H, "C", "Memory search omitted a required parameter (422)", dash_d02),
    "KAI-DASH-D03": Finding(
        H, "A", "Escape hatch bypassed authorisation, not just authentication",
        dash_d03),
    "KAI-DASH-D04": Finding(
        H, "A", "Default credential carried keeper authority", dash_d04),
}


def operator_directive() -> Result:
    """Standing operator directive: broker secrets stay in broker-bridge.

    Not a numbered audit finding — an explicit standing instruction that
    outranks the finding list, so it is checked on every run.
    """
    offenders = []
    targets = [DASHBOARD] + sorted((REPO / "dashboard").rglob("*.html"))
    for path in targets:
        try:
            content = path.read_text(encoding="utf-8")
        except OSError:
            continue
        for i, line in enumerate(content.splitlines(), 1):
            for marker in ("BINANCE_API_KEY", "BINANCE_API_SECRET"):
                if marker not in line:
                    continue
                # Naming the variable in help text is fine; reading its
                # value into the dashboard process is not.
                if "os.getenv" in line or "environ" in line:
                    offenders.append(f"{_rel(path)}:{i} reads {marker}")
    if offenders:
        return LIVE, "; ".join(offenders)
    return REMEDIATED, "dashboard never reads broker credentials"


# ── Coverage self-audit ──────────────────────────────────────────────

def coverage_gaps() -> List[str]:
    """Every finding 001..096 must appear in the table.

    The architecture-rules gate once reported a clean pass while silently
    omitting 6 of its 15 rules. A coverage table that does not audit its
    own coverage is not evidence, so this runs on every invocation.

    ``DISCOVERED`` is audited separately and on purpose. Remediation turns
    up defects the original audit never saw, and they need somewhere to
    live that cannot quietly stand in for one of the 96 — a register that
    lets new work dilute the original count is worse than no register.
    """
    expected = {f"KAI-DASH-{n:03d}" for n in range(1, TOTAL_DASH_FINDINGS + 1)}
    missing = sorted(expected - set(FINDINGS))
    extra = sorted(set(FINDINGS) - expected)
    gaps = [f"unaccounted finding: {fid}" for fid in missing]
    gaps += [f"unknown finding in table: {fid}" for fid in extra]
    for fid, finding in sorted(FINDINGS.items()):
        if finding.track not in TRACK_NAMES:
            gaps.append(f"{fid}: unknown track {finding.track!r}")

    for fid, finding in sorted(DISCOVERED.items()):
        if not DISCOVERED_ID_RX.fullmatch(fid):
            gaps.append(f"discovered finding {fid} must match KAI-DASH-D##")
        if fid in FINDINGS:
            gaps.append(f"discovered finding {fid} collides with an audit finding")
        if finding.track not in TRACK_NAMES:
            gaps.append(f"{fid}: unknown track {finding.track!r}")
    return gaps


def evaluate(include_discovered: bool = False) -> List[Dict[str, str]]:
    """Evaluate the audit table, optionally with the discovered register.

    Discovered findings are excluded by default so that every existing
    caller keeps measuring the original 96 and nothing else. Counts for
    the two sets are reported separately for the same reason.
    """
    table = dict(FINDINGS)
    if include_discovered:
        table.update(DISCOVERED)
    results = []
    for fid, finding in sorted(table.items()):
        try:
            status, detail = finding.check()
        except Exception as exc:
            status, detail = MANUAL, f"check errored: {type(exc).__name__}: {exc}"
        results.append({
            "finding": fid,
            "severity": finding.severity,
            "track": finding.track,
            "title": finding.title,
            "status": status,
            "detail": detail,
        })
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description="Revalidate KAI-DASH findings")
    parser.add_argument("--json", action="store_true", help="machine-readable output")
    parser.add_argument("--track", help="only report one track (A-I)")
    parser.add_argument("--gate", action="store_true",
                        help="exit non-zero while any finding is LIVE")
    args = parser.parse_args()

    # Nothing is judged until the tree is proven real. Every verdict
    # below is a claim *about the dashboard*; rendering one against a
    # tree that is absent or unrecognisable is how this reported
    # REMEDIATED=52 for a directory that did not exist.
    anchor = anchor_scan()
    if anchor:
        print("REFUSED: cannot judge these 96 findings.\n")
        print(f"  {anchor.detail}\n")
        if anchor.kind == "absent":
            print("  A finding that reports REMEDIATED against a tree that is")
            print("  not there is not evidence of anything. Most of these")
            print("  checks look for the *absence* of a defect, and absence")
            print("  of everything satisfies them all.")
        else:
            print("  The file is present but is not what these findings were")
            print("  written against. Re-anchor the tracker before trusting")
            print("  any verdict it produces.")
        return anchor.exit_code

    gaps = coverage_gaps()
    results = evaluate()
    discovered = [r for r in evaluate(include_discovered=True)
                  if r["finding"] in DISCOVERED]
    directive_status, directive_detail = operator_directive()

    if args.track:
        track = args.track.upper()
        results = [r for r in results if r["track"] == track]
        discovered = [r for r in discovered if r["track"] == track]

    counts = {s: sum(1 for r in results if r["status"] == s)
              for s in (LIVE, PARTIAL, REMEDIATED, MANUAL)}
    routes = _routes()
    mutating = [r for r in routes if r.method in MUTATING]
    unauth_mutating = [r for r in mutating if not r.authed]

    if args.json:
        print(json.dumps({
            "coverage_gaps": gaps,
            "results": results,
            "counts": counts,
            "discovered": discovered,
            "discovered_counts": {
                s: sum(1 for r in discovered if r["status"] == s)
                for s in (LIVE, PARTIAL, REMEDIATED, MANUAL)
            },
            "operator_directive": {
                "status": directive_status, "detail": directive_detail,
            },
            "routes": {
                "total": len(routes),
                "mutating": len(mutating),
                "mutating_unauthenticated": len(unauth_mutating),
                "unauthenticated": sum(1 for r in routes if not r.authed),
            },
        }, indent=2))
        live_total = counts[LIVE] + sum(
            1 for r in discovered if r["status"] == LIVE)
        return 1 if gaps or (args.gate and live_total) else 0

    print("Dashboard finding revalidation — Wave 1")
    print(f"  anchored: {len(ANCHOR_SYMBOLS)} symbols present, "
          f"{len(routes)} routes parsed\n")
    for track in sorted({r["track"] for r in results}):
        print(f"── Track {track}: {TRACK_NAMES[track]} " + "─" * 24)
        for r in (x for x in results if x["track"] == track):
            print(f"  {r['status']:<11}{r['severity']:<9}{r['finding']}  {r['title']}")
            print(f"      └─ {r['detail']}")
        print()

    if discovered:
        print("── Discovered during remediation (not in the original 96) "
              + "─" * 12)
        for r in discovered:
            print(f"  {r['status']:<11}{r['severity']:<9}{r['finding']}  {r['title']}")
            print(f"      └─ {r['detail']}")
        print()

    print(f"OPERATOR DIRECTIVE — broker credentials never reach the dashboard")
    print(f"  {directive_status}: {directive_detail}\n")

    print(f"  LIVE={counts[LIVE]}  PARTIAL={counts[PARTIAL]}  "
          f"REMEDIATED={counts[REMEDIATED]}  MANUAL={counts[MANUAL]}"
          f"   (of {len(results)} audit findings reported)")
    if discovered:
        d_live = sum(1 for r in discovered if r["status"] == LIVE)
        print(f"  Discovered during remediation: {len(discovered)} "
              f"({d_live} LIVE). Counted separately — they never reduce "
              f"the {TOTAL_DASH_FINDINGS}.")
    print(f"  Routes: {len(routes)} total, {len(mutating)} mutating, "
          f"{len(unauth_mutating)} mutating without auth")

    if gaps:
        print("\n  COVERAGE GAP — the table does not account for every finding:")
        for gap in gaps:
            print(f"    - {gap}")
        return 1

    print(f"\n  Coverage: all {TOTAL_DASH_FINDINGS} findings accounted for "
          f"({counts[MANUAL]} need manual review).")
    print("  This tool reports state; it does not close findings (Rule 7).")

    live_total = counts[LIVE] + sum(1 for r in discovered if r["status"] == LIVE)
    if args.gate and live_total:
        print(f"\n  GATE FAILED: {live_total} findings still LIVE "
              f"({counts[LIVE]} audit, {live_total - counts[LIVE]} discovered).")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
