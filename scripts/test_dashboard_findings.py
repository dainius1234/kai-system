"""Tests for the Dashboard finding tracker — Wave 1.

The tracker is the tracking mechanism for 96 findings. If it reports LIVE
for everything regardless of the code, it is not tracking anything; it is
just restating the audit. So every check that can flip is made to flip
here, against a synthetic dashboard that has the remediation applied.

This discipline is not hypothetical. Twice in this programme a check that
looked green was checking nothing: a negative test whose injected
violation was never written to disk, and an architecture gate that
silently omitted 6 of its 15 rules. Both passed. Both were worthless.
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from scripts.security import check_dashboard_findings as dash

passed = 0
failed = 0


def check(name: str, condition: bool, detail: str = "") -> None:
    global passed, failed
    if condition:
        passed += 1
    else:
        failed += 1
        msg = f"  FAIL: {name}"
        if detail:
            msg += f" — {detail}"
        print(msg)


class _Dashboard:
    """Point the tracker at synthetic source, then restore it.

    Asserts the substitution actually took effect, so a test cannot
    quietly assert against the real tree while believing otherwise.
    """

    def __init__(self, source: str) -> None:
        self.source = source
        self._original = dash.DASHBOARD
        self._tmp: Path | None = None

    def __enter__(self):
        fd, name = tempfile.mkstemp(suffix=".py")
        os.close(fd)
        self._tmp = Path(name)
        self._tmp.write_text(self.source, encoding="utf-8")
        dash.DASHBOARD = self._tmp
        dash._CACHE.clear()
        # Prove the substitution is real, not assumed.
        assert dash._text() == self.source, "synthetic dashboard did not take effect"
        return self

    def __exit__(self, *exc):
        dash.DASHBOARD = self._original
        dash._CACHE.clear()
        if self._tmp is not None:
            self._tmp.unlink(missing_ok=True)
        return False


# A dashboard with the Track A/B/C remediation applied, used to prove the
# auth checks report REMEDIATED when the condition genuinely no longer holds.
REMEDIATED_SOURCE = '''
from fastapi import Depends
from common.dashboard_auth import require_dashboard_auth, DashboardPrincipal

@app.post("/api/soul", dependencies=[Depends(require_dashboard_auth)])
async def api_soul_post(request):
    return {}

@app.post("/api/chat")
async def api_chat_proxy(principal: DashboardPrincipal = Depends(require_dashboard_auth)):
    return {}

@app.post("/api/browser/navigate", dependencies=[Depends(require_dashboard_auth)])
async def api_browser_navigate(request):
    return {}

@app.get("/health")
async def health():
    return {"status": "running"}
'''

UNAUTH_SOURCE = '''
@app.post("/api/soul")
async def api_soul_post(request):
    return {}

@app.post("/api/browser/navigate")
async def api_browser_navigate(request):
    return {}
'''


# ── Coverage self-audit ──────────────────────────────────────────────

def test_coverage_clean_on_real_table():
    check("real finding table has no coverage gaps",
          dash.coverage_gaps() == [],
          str(dash.coverage_gaps()))


def test_coverage_detects_missing_finding():
    victim = "KAI-DASH-042"
    removed = dash.FINDINGS.pop(victim)
    try:
        gaps = dash.coverage_gaps()
        check("coverage audit catches a removed finding",
              any(victim in g for g in gaps), str(gaps))
    finally:
        dash.FINDINGS[victim] = removed
    check("finding table restored after removal test",
          dash.coverage_gaps() == [])


def test_coverage_detects_unknown_finding():
    dash.FINDINGS["KAI-DASH-999"] = dash.Finding(
        "HIGH", "A", "not real", dash.manual("synthetic"))
    try:
        gaps = dash.coverage_gaps()
        check("coverage audit catches an out-of-range finding",
              any("KAI-DASH-999" in g for g in gaps), str(gaps))
    finally:
        del dash.FINDINGS["KAI-DASH-999"]


def test_coverage_detects_unknown_track():
    victim = "KAI-DASH-001"
    original = dash.FINDINGS[victim]
    dash.FINDINGS[victim] = original._replace(track="Z")
    try:
        gaps = dash.coverage_gaps()
        check("coverage audit catches an unknown track",
              any("unknown track" in g for g in gaps), str(gaps))
    finally:
        dash.FINDINGS[victim] = original


def test_every_finding_has_exactly_one_track():
    tracks = {}
    for fid, finding in dash.FINDINGS.items():
        tracks.setdefault(finding.track, []).append(fid)
    total = sum(len(v) for v in tracks.values())
    check("all 96 findings partitioned across tracks",
          total == dash.TOTAL_DASH_FINDINGS, f"got {total}")
    check("every track used is named",
          set(tracks) <= set(dash.TRACK_NAMES), str(set(tracks)))


# ── Route authentication detection ───────────────────────────────────

def test_route_auth_flips_to_remediated():
    with _Dashboard(REMEDIATED_SOURCE):
        status, detail = dash.route_auth(("post", "/api/soul"))()
        check("route_auth reports REMEDIATED for a decorator-guarded route",
              status == dash.REMEDIATED, f"{status}: {detail}")


def test_route_auth_detects_signature_dependency():
    with _Dashboard(REMEDIATED_SOURCE):
        status, detail = dash.route_auth(("post", "/api/chat"))()
        check("route_auth detects Depends() in the handler signature",
              status == dash.REMEDIATED, f"{status}: {detail}")


def test_route_auth_reports_live_when_unguarded():
    with _Dashboard(UNAUTH_SOURCE):
        status, detail = dash.route_auth(("post", "/api/soul"))()
        check("route_auth reports LIVE for an unguarded route",
              status == dash.LIVE, f"{status}: {detail}")


def test_prefix_auth_flips_both_ways():
    with _Dashboard(UNAUTH_SOURCE):
        live, _ = dash.prefix_auth(r"^/api/browser/", dash.MUTATING)()
    with _Dashboard(REMEDIATED_SOURCE):
        fixed, detail = dash.prefix_auth(r"^/api/browser/", dash.MUTATING)()
    check("prefix_auth reports LIVE while unguarded", live == dash.LIVE, live)
    check("prefix_auth reports REMEDIATED once guarded",
          fixed == dash.REMEDIATED, f"{fixed}: {detail}")


def test_absent_route_is_not_reported_live():
    with _Dashboard(UNAUTH_SOURCE):
        status, _ = dash.route_auth(("post", "/api/does-not-exist"))()
    check("a removed route is REMEDIATED, not LIVE", status == dash.REMEDIATED, status)


# ── Individual finding checks flip ───────────────────────────────────

def test_dash_002_detects_reintroduced_gate_token():
    with _Dashboard('DASHBOARD_GATE_TOKEN = os.getenv("DASHBOARD_GATE_TOKEN")\n'):
        status, detail = dash.dash_002()
    check("DASH-002 goes LIVE if a gate token returns",
          status == dash.LIVE, f"{status}: {detail}")


def test_dash_002_remediated_without_token():
    src = '@app.post("/api/mode")\nasync def api_set_mode(body):\n    return {"status": "local_only"}\n'
    with _Dashboard(src):
        status, detail = dash.dash_002()
    check("DASH-002 REMEDIATED when /api/mode is display-only",
          status == dash.REMEDIATED, f"{status}: {detail}")


def test_dash_011_flips_with_principal_model():
    with _Dashboard(REMEDIATED_SOURCE):
        fixed, _ = dash.dash_011()
    with _Dashboard(UNAUTH_SOURCE):
        live, _ = dash.dash_011()
    check("DASH-011 REMEDIATED once a principal model exists", fixed == dash.REMEDIATED, fixed)
    check("DASH-011 LIVE while none exists", live == dash.LIVE, live)


def test_dash_016_flips_when_failures_are_raised():
    good = '''
@app.get("/api/thing")
async def api_thing():
    try:
        return await call()
    except Exception:
        raise HTTPException(status_code=503, detail="upstream unavailable")
'''
    bad = '''
@app.get("/api/thing")
async def api_thing():
    try:
        return await call()
    except Exception:
        return {"nudges": []}
'''
    with _Dashboard(good):
        fixed, _ = dash.dash_016()
    with _Dashboard(bad):
        live, detail = dash.dash_016()
    check("DASH-016 REMEDIATED when failures raise", fixed == dash.REMEDIATED, fixed)
    check("DASH-016 LIVE when failures become 200 bodies", live == dash.LIVE, detail)


def test_dash_083_counts_naive_timestamps():
    with _Dashboard("x = datetime.utcnow()\ny = datetime.utcnow()\n"):
        live, detail = dash.dash_083()
    with _Dashboard("x = datetime.now(timezone.utc)\n"):
        fixed, _ = dash.dash_083()
    check("DASH-083 LIVE and counts naive uses",
          live == dash.LIVE and "2" in detail, f"{live}: {detail}")
    check("DASH-083 REMEDIATED with aware timestamps", fixed == dash.REMEDIATED, fixed)


def test_dash_069_flips_on_health_payload():
    with _Dashboard('async def health():\n    return {"tool_gate_url": u, "policy_hash": h}\n'):
        live, detail = dash.dash_069()
    with _Dashboard('async def health():\n    return {"status": "ok"}\n'):
        fixed, _ = dash.dash_069()
    check("DASH-069 LIVE when /health leaks topology", live == dash.LIVE, detail)
    check("DASH-069 REMEDIATED when it does not", fixed == dash.REMEDIATED, fixed)


def test_dash_088_reports_partial_then_remediated():
    with _Dashboard('h = "Content-Security-Policy"\n'):
        partial, detail = dash.dash_088()
    with _Dashboard('a = "Content-Security-Policy"\nb = "X-Frame-Options"\nc = "Referrer-Policy"\n'):
        fixed, _ = dash.dash_088()
    check("DASH-088 PARTIAL with only some headers", partial == dash.PARTIAL, detail)
    check("DASH-088 REMEDIATED with all three", fixed == dash.REMEDIATED, fixed)


def test_dash_057_flips_on_concurrent_fanout():
    with _Dashboard('async def fetch_status():\n    return await asyncio.gather(*probes)\n'):
        fixed, _ = dash.dash_057()
    with _Dashboard('async def fetch_status():\n    for n, u in NODES.items():\n        await get(u)\n'):
        live, _ = dash.dash_057()
    check("DASH-057 REMEDIATED when probes gather", fixed == dash.REMEDIATED, fixed)
    check("DASH-057 LIVE while sequential", live == dash.LIVE, live)


def test_dash_001_is_live_while_a_mutating_route_is_open():
    with _Dashboard(UNAUTH_SOURCE):
        status, detail = dash.dash_001()
    check("DASH-001 LIVE while any mutating route is unauthenticated",
          status == dash.LIVE, f"{status}: {detail}")


def test_dash_001_flags_an_undeclared_public_route():
    """A new anonymous GET must not be absorbed into the public allowlist."""
    src = REMEDIATED_SOURCE + '''
@app.get("/api/secrets")
async def api_secrets():
    return {}
'''
    with _Dashboard(src):
        status, detail = dash.dash_001()
    check("DASH-001 refuses to pass an undeclared unauthenticated route",
          status in (dash.LIVE, dash.PARTIAL) and "/api/secrets" in detail,
          f"{status}: {detail}")


def test_dash_001_accepts_only_the_declared_public_list():
    src = '''
from fastapi import Depends
from common.dashboard_auth import require_dashboard_auth, Scope

@app.get("/health")
async def health():
    return {}

@app.get("/api/thing", dependencies=[Depends(require_dashboard_auth(Scope.READ_SENSITIVE))])
async def api_thing():
    return {}
'''
    with _Dashboard(src):
        status, detail = dash.dash_001()
    check("DASH-001 REMEDIATED when the only open route is declared public",
          status == dash.REMEDIATED, f"{status}: {detail}")
    check("the public list holds nothing mutating",
          all(m == "get" for m, _ in dash.PUBLIC_ROUTES), str(dash.PUBLIC_ROUTES))


def test_dash_018_rejects_a_uniform_scope():
    """One scope everywhere is a shared authority, not least privilege."""
    src = '''
from fastapi import Depends
from common.dashboard_auth import require_dashboard_auth, Scope

@app.post("/api/a", dependencies=[Depends(require_dashboard_auth(Scope.WRITE_EXTERNAL))])
async def a():
    return {}

@app.get("/api/b", dependencies=[Depends(require_dashboard_auth(Scope.WRITE_EXTERNAL))])
async def b():
    return {}
'''
    with _Dashboard(src):
        status, detail = dash.dash_018()
    check("DASH-018 LIVE when every route declares the same scope",
          status == dash.LIVE, f"{status}: {detail}")


def test_dash_018_reports_partial_when_a_guarded_route_has_no_scope():
    src = '''
from fastapi import Depends
from common.dashboard_auth import require_dashboard_auth, Scope, require_service_auth

@app.post("/api/a", dependencies=[Depends(require_dashboard_auth(Scope.WRITE_EXTERNAL))])
async def a():
    return {}

@app.get("/api/b", dependencies=[Depends(require_dashboard_auth(Scope.READ_SENSITIVE))])
async def b():
    return {}

@app.get("/api/c", dependencies=[Depends(require_service_auth("c"))])
async def c():
    return {}
'''
    with _Dashboard(src):
        status, detail = dash.dash_018()
    check("DASH-018 PARTIAL when an authenticated route declares no scope",
          status == dash.PARTIAL and "/api/c" in detail, f"{status}: {detail}")


def test_dash_018_remediated_with_a_real_distribution():
    src = '''
from fastapi import Depends
from common.dashboard_auth import require_dashboard_auth, Scope

@app.post("/api/a", dependencies=[Depends(require_dashboard_auth(Scope.WRITE_EXTERNAL))])
async def a():
    return {}

@app.get("/api/b", dependencies=[Depends(require_dashboard_auth(Scope.READ_SENSITIVE))])
async def b():
    return {}
'''
    with _Dashboard(src):
        status, detail = dash.dash_018()
    check("DASH-018 REMEDIATED with distinct scopes on every route",
          status == dash.REMEDIATED, f"{status}: {detail}")


# ── Standing operator directive ──────────────────────────────────────

def test_operator_directive_detects_credential_read():
    with _Dashboard('KEY = os.getenv("BINANCE_API_KEY")\n'):
        status, detail = dash.operator_directive()
    check("operator directive catches a dashboard credential read",
          status == dash.LIVE and "BINANCE_API_KEY" in detail,
          f"{status}: {detail}")


def test_operator_directive_allows_help_text():
    with _Dashboard('MSG = "Configure BINANCE_API_KEY to see balance"\n'):
        status, detail = dash.operator_directive()
    check("operator directive tolerates naming the variable in help text",
          status == dash.REMEDIATED, f"{status}: {detail}")


def test_operator_directive_clean_on_real_tree():
    status, detail = dash.operator_directive()
    check("real dashboard does not read broker credentials",
          status == dash.REMEDIATED, f"{status}: {detail}")


# ── Whole-run invariants ─────────────────────────────────────────────

def test_every_check_returns_a_valid_status():
    valid = {dash.LIVE, dash.REMEDIATED, dash.PARTIAL, dash.MANUAL}
    results = dash.evaluate()
    bad = [r for r in results if r["status"] not in valid]
    check("every check returns a known status", not bad, str(bad[:3]))
    check("evaluate() covers the whole table",
          len(results) == dash.TOTAL_DASH_FINDINGS, str(len(results)))


def test_every_check_gives_a_reason():
    empty = [r["finding"] for r in dash.evaluate() if not r["detail"].strip()]
    check("every result carries a detail line", not empty, str(empty))


def test_manual_findings_name_what_to_review():
    vague = [r["finding"] for r in dash.evaluate()
             if r["status"] == dash.MANUAL and len(r["detail"]) < 20]
    check("MANUAL findings say what needs reviewing", not vague, str(vague))


def test_no_finding_is_remediated_without_evidence():
    """REMEDIATED must come from a check, never from a MANUAL placeholder."""
    for r in dash.evaluate():
        if r["status"] in (dash.REMEDIATED, dash.PARTIAL):
            fn = dash.FINDINGS[r["finding"]].check
            check(f"{r['finding']} {r['status']} by a real check",
                  not getattr(fn, "is_manual", False),
                  "closed by a manual placeholder, not evidence")


def test_manual_placeholders_can_never_report_closed():
    """A MANUAL entry must be inert — it cannot drift into a false pass."""
    placeholder = dash.manual("anything at all")
    status, _ = placeholder()
    check("manual() always reports MANUAL", status == dash.MANUAL, status)
    check("manual() is tagged as a placeholder",
          getattr(placeholder, "is_manual", False) is True)
    tagged = sum(1 for f in dash.FINDINGS.values()
                 if getattr(f.check, "is_manual", False))
    reported = sum(1 for r in dash.evaluate() if r["status"] == dash.MANUAL)
    check("every tagged placeholder reports MANUAL and nothing else does",
          tagged == reported, f"tagged={tagged} reported={reported}")



# ── Discovered-findings register ─────────────────────────────────────

def test_discovered_register_is_separate_from_the_96():
    """New findings must never stand in for one of the original 96."""
    check("discovered register does not overlap the audit table",
          not (set(dash.DISCOVERED) & set(dash.FINDINGS)),
          str(set(dash.DISCOVERED) & set(dash.FINDINGS)))
    check("audit table is still exactly 96",
          len(dash.FINDINGS) == dash.TOTAL_DASH_FINDINGS, str(len(dash.FINDINGS)))
    check("evaluate() excludes discovered findings by default",
          len(dash.evaluate()) == dash.TOTAL_DASH_FINDINGS)
    check("evaluate(include_discovered=True) adds them",
          len(dash.evaluate(include_discovered=True))
          == dash.TOTAL_DASH_FINDINGS + len(dash.DISCOVERED))


def test_discovered_ids_must_be_well_formed():
    dash.DISCOVERED["KAI-DASH-BOGUS"] = dash.Finding(
        "HIGH", "A", "bad id", dash.manual("synthetic"))
    try:
        gaps = dash.coverage_gaps()
        check("register rejects a malformed discovered id",
              any("KAI-DASH-BOGUS" in g for g in gaps), str(gaps))
    finally:
        del dash.DISCOVERED["KAI-DASH-BOGUS"]
    check("register clean after malformed-id test", dash.coverage_gaps() == [])


def test_discovered_cannot_collide_with_an_audit_finding():
    original = dash.FINDINGS["KAI-DASH-001"]
    dash.DISCOVERED["KAI-DASH-001"] = original
    try:
        gaps = dash.coverage_gaps()
        check("register rejects an id that collides with the audit table",
              any("collides" in g for g in gaps), str(gaps))
    finally:
        del dash.DISCOVERED["KAI-DASH-001"]


def test_d01_flips_on_the_ui_shim():
    """D01 is LIVE until the UI actually carries credentials."""
    import tempfile, pathlib
    real = dash.REPO
    with tempfile.TemporaryDirectory() as tmp:
        static = pathlib.Path(tmp) / "dashboard" / "static"
        static.mkdir(parents=True)
        page = static / "app.html"
        page.write_text("<script>fetch('/api/x')</script>", encoding="utf-8")
        dash.REPO = pathlib.Path(tmp)
        try:
            no_shim, d1 = dash.dash_d01()
            (static / "auth.js").write_text("// shim", encoding="utf-8")
            unwired, d2 = dash.dash_d01()
            page.write_text(
                '<script src="/static/auth.js"></script>'
                "<script>fetch('/api/x'); new EventSource('/api/events')</script>",
                encoding="utf-8")
            raw_sse, d3 = dash.dash_d01()
            page.write_text(
                '<script src="/static/auth.js"></script>'
                "<script>fetch('/api/x'); KaiAuth.eventStream('/api/events')</script>",
                encoding="utf-8")
            fixed, d4 = dash.dash_d01()
        finally:
            dash.REPO = real
    check("D01 LIVE with no shim present", no_shim == dash.LIVE, d1)
    check("D01 LIVE when a page does not load the shim", unwired == dash.LIVE, d2)
    check("D01 PARTIAL while raw EventSource remains", raw_sse == dash.PARTIAL, d3)
    check("D01 REMEDIATED once fetch and SSE both authenticate",
          fixed == dash.REMEDIATED, d4)


def test_d01_is_remediated_on_the_real_tree():
    status, detail = dash.dash_d01()
    check("the shipped UI carries credentials",
          status == dash.REMEDIATED, f"{status}: {detail}")



def test_dash_023_flips_on_hard_coded_identity():
    hard = """
async def api_memories_recent(top_k):
    return await get(url, params={'query': 'x', 'user_id': 'keeper', 'top_k': top_k})
"""
    scoped = """
async def api_memories_recent(top_k, principal):
    return await get(url, params={'query': 'x', 'user_id': principal.identity})
"""
    with _Dashboard(hard):
        live, d1 = dash.dash_023()
    with _Dashboard(scoped):
        fixed, d2 = dash.dash_023()
    check("DASH-023 LIVE with a literal identity", live == dash.LIVE, d1)
    check("DASH-023 names the literal it found", "keeper" in d1, d1)
    check("DASH-023 REMEDIATED once the caller's identity is used",
          fixed == dash.REMEDIATED, d2)


def test_dash_023_catches_any_literal_not_just_keeper():
    """The defect is a hard-coded identity, whatever it is called."""
    other = """
async def api_thing(principal):
    return await get(url, params={'user_id': 'admin'})
"""
    with _Dashboard(other):
        status, detail = dash.dash_023()
    check("DASH-023 catches a hard-coded identity other than 'keeper'",
          status == dash.LIVE and "admin" in detail, f"{status}: {detail}")


def test_dash_d02_flips_on_the_missing_parameter():
    broken = """
async def api_memories(query, principal):
    return await get(f'{MEMU_URL}/memory/retrieve', params={'query': query})
"""
    fixed_src = """
async def api_memories(query, principal):
    return await get(f'{MEMU_URL}/memory/retrieve',
                     params={'query': query, 'user_id': principal.identity})
"""
    with _Dashboard(broken):
        live, d1 = dash.dash_d02()
    with _Dashboard(fixed_src):
        fixed, d2 = dash.dash_d02()
    check("D02 LIVE while the required user_id is missing", live == dash.LIVE, d1)
    check("D02 REMEDIATED once it is passed", fixed == dash.REMEDIATED, d2)


def test_track_c_has_no_live_findings():
    live = [r["finding"] for r in dash.evaluate()
            if r["track"] == "C" and r["status"] == dash.LIVE]
    check("Track C is clear of LIVE findings", not live, str(live))



# ── Track D — failure semantics ──────────────────────────────────────

def test_dash_016_ignores_helpers_and_catches_routes():
    """Only a route's failure path becomes an HTTP 200."""
    helper_only = """
def _helper():
    try:
        return call()
    except Exception:
        return {'status': 'down'}
"""
    routed = """
@app.get("/api/thing")
async def api_thing():
    try:
        return await call()
    except Exception:
        return {'status': 'unavailable'}
"""
    fixed_src = """
from common.degraded import degraded_response

@app.get("/api/thing")
async def api_thing():
    try:
        return await call()
    except Exception as exc:
        return degraded_response('memu', str(exc), {'status': 'unavailable'})
"""
    with _Dashboard(helper_only):
        helper, d0 = dash.dash_016()
    with _Dashboard(routed):
        live, d1 = dash.dash_016()
    with _Dashboard(fixed_src):
        fixed, d2 = dash.dash_016()
    check("DASH-016 ignores a non-route helper", helper == dash.REMEDIATED, d0)
    check("DASH-016 LIVE for a route returning a 200 fallback",
          live == dash.LIVE, d1)
    check("DASH-016 REMEDIATED once the route answers degraded",
          fixed == dash.REMEDIATED, d2)


def test_dash_061_requires_reading_the_backend_status():
    naive = """
async def fetch_status():
    resp.raise_for_status()
    results[name] = {'status': 'ok', 'details': resp.json()}
"""
    honest = """
async def fetch_status():
    status, note = _classify_node(payload)
    return await asyncio.gather(*probes)
"""
    with _Dashboard(naive):
        live, d1 = dash.dash_061()
    with _Dashboard(honest):
        fixed, d2 = dash.dash_061()
    check("DASH-061 LIVE while any 2xx counts as healthy", live == dash.LIVE, d1)
    check("DASH-061 REMEDIATED once the self-report is read",
          fixed == dash.REMEDIATED, d2)


def test_dash_063_accepts_declining_to_measure_but_not_substituting():
    substituted = """
async def build_go_no_go_report():
    if ledger_count < NO_GO_GRACE_REQUESTS:
        reasons.append('not enough proof')
    return {'checks': {'minimum_gate_decisions': NO_GO_GRACE_REQUESTS}}
"""
    declared = """
async def build_go_no_go_report():
    proof = unavailable_metric('recent_approved_decisions', 'no ledger credential')
    return {'checks': {'proof_of_safe_operation': proof}}
"""
    with _Dashboard(substituted):
        live, d1 = dash.dash_063()
    with _Dashboard(declared):
        fixed, d2 = dash.dash_063()
    check("DASH-063 LIVE while a total count stands in for proof",
          live == dash.LIVE, d1)
    check("DASH-063 REMEDIATED when the metric is declared unavailable",
          fixed == dash.REMEDIATED, d2)


def test_dash_064_requires_the_decision_to_turn_on_fleet_health():
    wrong = """
async def build_go_no_go_report():
    error_ratio = float(metrics.get('error_ratio', 0.0))
    if error_ratio > MAX_ERROR_RATIO:
        reasons.append('too many errors')
"""
    right = """
async def build_go_no_go_report():
    fleet_unhealthy_ratio = 1.0 - healthy_nodes / total_nodes
    if fleet_unhealthy_ratio > MAX_ERROR_RATIO:
        reasons.append('fleet unhealthy')
"""
    with _Dashboard(wrong):
        live, d1 = dash.dash_064()
    with _Dashboard(right):
        fixed, d2 = dash.dash_064()
    check("DASH-064 LIVE while caller error ratio gates the decision",
          live == dash.LIVE, d1)
    check("DASH-064 REMEDIATED once fleet health gates it",
          fixed == dash.REMEDIATED, d2)


def test_dash_065_rejects_a_liveness_probe_as_backup_proof():
    fake = """
async def api_backup_status():
    now = datetime.utcnow().strftime('%Y')
    return {'status': f'{now} (service healthy)'}
"""
    real = """
async def api_backup_status():
    resp = await client.get(f'{backup_url}/backup/list')
    return {'status': latest['modified'], 'verified': True}
"""
    with _Dashboard(fake):
        live, d1 = dash.dash_065()
    with _Dashboard(real):
        fixed, d2 = dash.dash_065()
    check("DASH-065 LIVE while a liveness probe stands in for a backup",
          live == dash.LIVE, d1)
    check("DASH-065 REMEDIATED once a real backup is read",
          fixed == dash.REMEDIATED, d2)


def test_dash_080_requires_an_enforcing_status():
    advisory = """
async def go_no_go():
    return await build_go_no_go_report()
"""
    enforcing = """
async def go_no_go():
    report = await build_go_no_go_report()
    if report.get('decision') == 'GO':
        return report
    return JSONResponse(status_code=503, content=report)
"""
    with _Dashboard(advisory):
        live, d1 = dash.dash_080()
    with _Dashboard(enforcing):
        fixed, d2 = dash.dash_080()
    check("DASH-080 LIVE while NO_GO answers 200", live == dash.LIVE, d1)
    check("DASH-080 REMEDIATED once it carries a status",
          fixed == dash.REMEDIATED, d2)


def test_dash_054_catches_a_guard_that_runs_too_late():
    """A status check after streaming has begun is not a status check."""
    too_late = """
async def api_chat_proxy(request):
    async for chunk in resp.aiter_bytes():
        yield chunk
    if resp.status_code >= 400:
        return
"""
    in_time = """
async def api_chat_proxy(request):
    if resp.status_code >= 400:
        return
    async for chunk in resp.aiter_bytes():
        yield chunk
"""
    absent = """
async def api_chat_proxy(request):
    async for chunk in resp.aiter_bytes():
        yield chunk
"""
    with _Dashboard(too_late):
        late, d1 = dash.dash_054()
    with _Dashboard(in_time):
        good, d2 = dash.dash_054()
    with _Dashboard(absent):
        none, d3 = dash.dash_054()
    check("DASH-054 LIVE when the guard runs after streaming", late == dash.LIVE, d1)
    check("DASH-054 REMEDIATED when the guard runs first", good == dash.REMEDIATED, d2)
    check("DASH-054 LIVE when there is no guard at all", none == dash.LIVE, d3)


def test_dash_055_catches_exception_text_reaching_the_client():
    leaky = """
async def api_chat_proxy(request):
    try:
        pass
    except Exception as exc:
        yield f'error: {exc}'
"""
    quiet = """
async def api_chat_proxy(request):
    try:
        pass
    except Exception as exc:
        logger.warning('failed: %s', exc)
        yield _sse_error('unavailable')
"""
    with _Dashboard(leaky):
        live, d1 = dash.dash_055()
    with _Dashboard(quiet):
        fixed, d2 = dash.dash_055()
    check("DASH-055 LIVE while exception text is yielded", live == dash.LIVE, d1)
    check("DASH-055 REMEDIATED when it is only logged", fixed == dash.REMEDIATED, d2)


def test_track_d_has_no_live_findings():
    live = [r["finding"] for r in dash.evaluate()
            if r["track"] == "D" and r["status"] == dash.LIVE]
    check("Track D is clear of LIVE findings", not live, str(live))



# ── Shared-resilience findings (014, 015, 075, 076) ──────────────────

class _Resilience:
    """Point the resilience-backed checks at synthetic source."""

    def __init__(self, source):
        self.source = source
        self._orig = dash._resilience_src

    def __enter__(self):
        dash._resilience_src = lambda: self.source
        assert dash._resilience_src() == self.source
        return self

    def __exit__(self, *exc):
        dash._resilience_src = self._orig
        return False


def test_dash_015_reads_the_success_guard_not_the_string():
    """The module's own docstring mentions the old test; that must not count."""
    old = """
async def resilient_call(method, url):
    if resp.status_code < 500:
        cb.record_success()
        return resp.json()
"""
    new = """
async def resilient_call(method, url):
    '''Success was status_code < 500, so a 404 recorded a success.'''
    if 200 <= resp.status_code < 300:
        cb.record_success()
        return resp.json()
    if 400 <= resp.status_code < 500:
        return fallback
"""
    with _Resilience(old):
        live, d1 = dash.dash_015()
    with _Resilience(new):
        fixed, d2 = dash.dash_015()
    check("DASH-015 LIVE while <500 records success", live == dash.LIVE, d1)
    check("DASH-015 REMEDIATED with a 2xx guard, despite the docstring "
          "still naming the old test", fixed == dash.REMEDIATED, d2)


def test_dash_014_requires_idempotence_to_gate_retries():
    naive = "async def resilient_call(m, u):\n    for attempt in range(retries):\n        pass\n"
    flagged = "async def resilient_call(m, u, idempotent=None):\n    pass\n"
    gated = ("async def resilient_call(m, u, idempotent=None):\n"
             "    attempts = retries if idempotent else 1\n")
    with _Resilience(naive):
        live, d1 = dash.dash_014()
    with _Resilience(flagged):
        part, d2 = dash.dash_014()
    with _Resilience(gated):
        fixed, d3 = dash.dash_014()
    check("DASH-014 LIVE with unconditional retries", live == dash.LIVE, d1)
    check("DASH-014 PARTIAL when a flag exists but gates nothing",
          part == dash.PARTIAL, d2)
    check("DASH-014 REMEDIATED once the flag gates retries",
          fixed == dash.REMEDIATED, d3)


def test_dash_075_catches_a_client_inside_the_loop():
    inside = ("async def resilient_call(m, u):\n"
              "    for attempt in range(2):\n"
              "        async with pooled_client(timeout=1) as c:\n            pass\n")
    outside = ("async def resilient_call(m, u):\n"
               "    async with pooled_client(timeout=1) as c:\n"
               "        for attempt in range(2):\n            pass\n")
    with _Resilience(inside):
        live, d1 = dash.dash_075()
    with _Resilience(outside):
        fixed, d2 = dash.dash_075()
    check("DASH-075 LIVE when the client is built inside the loop",
          live == dash.LIVE, d1)
    check("DASH-075 REMEDIATED when it is built outside",
          fixed == dash.REMEDIATED, d2)


# ── Dashboard-side conversions ───────────────────────────────────────

def test_dash_062_catches_the_always_true_readiness_test():
    always_true = """
@app.get("/")
async def index():
    ledger_size = 0
    return {"core_ready": ledger_size >= 0}
"""
    observed = """
@app.get("/")
async def index():
    ledger_size = None
    return {"core_ready": ledger_size is not None}
"""
    with _Dashboard(always_true):
        live, d1 = dash.dash_062()
    with _Dashboard(observed):
        fixed, d2 = dash.dash_062()
    check("DASH-062 LIVE while a fallback zero satisfies readiness",
          live == dash.LIVE, d1)
    check("DASH-062 REMEDIATED once observation is required",
          fixed == dash.REMEDIATED, d2)


def test_dash_072_flips_on_url_validation():
    raw = 'TOOL_GATE_URL = os.getenv("TOOL_GATE_URL", "http://tool-gate:8000")\n'
    validated = 'TOOL_GATE_URL = backend_url("TOOL_GATE_URL", "http://tool-gate:8000")\n'
    with _Dashboard(raw):
        live, d1 = dash.dash_072()
    with _Dashboard(validated):
        fixed, d2 = dash.dash_072()
    check("DASH-072 LIVE with a raw getenv URL", live == dash.LIVE, d1)
    check("DASH-072 REMEDIATED once validated", fixed == dash.REMEDIATED, d2)


def test_dash_094_only_flags_routes_that_interpolate():
    unguarded = """
@app.get("/api/broker/ticker/{symbol}")
async def api_broker_ticker(symbol: str):
    return await _proxy_get(f"{BROKER_URL}/ticker/{symbol}")
"""
    guarded = """
@app.get("/api/broker/ticker/{symbol}")
async def api_broker_ticker(symbol: str):
    symbol = safe_symbol(symbol)
    return await _proxy_get(f"{BROKER_URL}/ticker/{symbol}")
"""
    with _Dashboard(unguarded):
        live, d1 = dash.dash_094()
    with _Dashboard(guarded):
        fixed, d2 = dash.dash_094()
    check("DASH-094 LIVE with unvalidated path interpolation",
          live == dash.LIVE, d1)
    check("DASH-094 REMEDIATED once the parameter is validated",
          fixed == dash.REMEDIATED, d2)


def test_dash_044_needs_both_a_principal_and_a_filter():
    none = "async def sse_events(request):\n    pass\n"
    principal_only = ("async def sse_events(request, principal: DashboardPrincipal = 1):\n"
                      "    pass\n")
    filtered = ("async def sse_events(request, principal: DashboardPrincipal = 1):\n"
                "    if not _event_visible_to(data, principal):\n        pass\n")
    with _Dashboard(none):
        live, d1 = dash.dash_044()
    with _Dashboard(principal_only):
        half, d2 = dash.dash_044()
    with _Dashboard(filtered):
        fixed, d3 = dash.dash_044()
    check("DASH-044 LIVE with no subscriber identity", live == dash.LIVE, d1)
    check("DASH-044 still LIVE with an identity but no filter",
          half == dash.LIVE, d2)
    check("DASH-044 REMEDIATED once events are filtered",
          fixed == dash.REMEDIATED, d3)


# ── The anchor pre-scan (KAI-GATE-005 — category confusion) ──────────

def _anchor_probe(mutate):
    """Run main() against a mutated tree and return (exit_code, output)."""
    import importlib, io, contextlib
    for name in list(sys.modules):
        if "check_dashboard_findings" in name:
            del sys.modules[name]
    mod = importlib.import_module("scripts.security.check_dashboard_findings")
    mutate(mod)
    argv, sys.argv = sys.argv, ["check_dashboard_findings.py"]
    buffer = io.StringIO()
    try:
        with contextlib.redirect_stdout(buffer):
            code = mod.main()
    except SystemExit as exc:
        code = exc.code
    finally:
        sys.argv = argv
    return code, buffer.getvalue()


def test_an_absent_tree_is_refused_not_reported_remediated():
    """The finding itself: this reported REMEDIATED=52 for a missing tree.

    Not one of those 52 checks was wrong in isolation — "the dashboard
    never reads broker credentials" is correctly true when there is no
    dashboard. A check for the absence of something bad passes when
    everything is absent. The operator named it category confusion.
    """
    import tempfile
    from pathlib import Path as _P
    with tempfile.TemporaryDirectory() as tmp:
        code, out = _anchor_probe(
            lambda m: setattr(m, "DASHBOARD", _P(tmp) / "gone.py"))
    check("an absent tree is refused", code == 2, f"exit={code}")
    # Assert on verdict *lines*, not the word: the refusal message
    # itself explains what a REMEDIATED-against-nothing would mean, so
    # a substring test matches its own prose — the same self-matching
    # mistake `dash_015` made when its grep hit its own docstring.
    check("no finding verdict is rendered",
          "KAI-DASH-0" not in out, out[:200])
    check("the refusal explains why", "not there is not evidence" in out,
          out[:300])


def test_an_unrecognisable_tree_fails_differently_from_an_absent_one():
    """The operator's nuance: 'file not found' and 'symbol not in tree'
    are different problems, so they must not share an exit code."""
    import tempfile
    from pathlib import Path as _P
    with tempfile.TemporaryDirectory() as tmp:
        wrong = _P(tmp) / "wrong.py"
        wrong.write_text("app = 1\n")
        code, out = _anchor_probe(lambda m: setattr(m, "DASHBOARD", wrong))
    check("an unrecognisable tree is refused", code == 3, f"exit={code}")
    check("its exit code differs from absent", code != 2, f"exit={code}")
    check("it names the missing anchors", "missing" in out, out[:200])


def test_a_thin_tree_is_refused_even_with_every_symbol_present():
    """Symbols alone are not enough — the scan must actually see routes."""
    import tempfile
    from pathlib import Path as _P
    import scripts.security.check_dashboard_findings as live
    with tempfile.TemporaryDirectory() as tmp:
        thin = _P(tmp) / "thin.py"
        thin.write_text("\n".join(live.ANCHOR_SYMBOLS) + "\n")
        code, out = _anchor_probe(lambda m: setattr(m, "DASHBOARD", thin))
    check("a tree with no routes is refused", code == 3, f"exit={code}")
    check("the route count is named", "routes parsed" in out, out[:300])


def test_the_real_tree_anchors_and_reports_its_denominator():
    code, out = _anchor_probe(lambda m: None)
    check("the real tree is judged", code == 0, f"exit={code}")
    check("the anchor is stated in the output", "anchored:" in out, out[:200])
    check("the route count is stated", "routes parsed" in out, out[:200])


def test_every_anchor_symbol_is_actually_present():
    """A guessed anchor is worse than none — it fails against a healthy
    tree. My first draft named two symbols that do not exist."""
    import scripts.security.check_dashboard_findings as live
    source = live.DASHBOARD.read_text(encoding="utf-8")
    missing = [s for s in live.ANCHOR_SYMBOLS if s not in source]
    check("no anchor symbol is invented", not missing, str(missing))


def run() -> None:
    test_an_absent_tree_is_refused_not_reported_remediated()
    test_an_unrecognisable_tree_fails_differently_from_an_absent_one()
    test_a_thin_tree_is_refused_even_with_every_symbol_present()
    test_the_real_tree_anchors_and_reports_its_denominator()
    test_every_anchor_symbol_is_actually_present()
    test_coverage_clean_on_real_table()
    test_coverage_detects_missing_finding()
    test_coverage_detects_unknown_finding()
    test_coverage_detects_unknown_track()
    test_every_finding_has_exactly_one_track()
    test_route_auth_flips_to_remediated()
    test_route_auth_detects_signature_dependency()
    test_route_auth_reports_live_when_unguarded()
    test_prefix_auth_flips_both_ways()
    test_absent_route_is_not_reported_live()
    test_dash_002_detects_reintroduced_gate_token()
    test_dash_002_remediated_without_token()
    test_dash_011_flips_with_principal_model()
    test_dash_016_flips_when_failures_are_raised()
    test_dash_083_counts_naive_timestamps()
    test_dash_069_flips_on_health_payload()
    test_dash_088_reports_partial_then_remediated()
    test_dash_057_flips_on_concurrent_fanout()
    test_dash_001_is_live_while_a_mutating_route_is_open()
    test_dash_001_flags_an_undeclared_public_route()
    test_dash_001_accepts_only_the_declared_public_list()
    test_dash_018_rejects_a_uniform_scope()
    test_dash_018_reports_partial_when_a_guarded_route_has_no_scope()
    test_dash_018_remediated_with_a_real_distribution()
    test_operator_directive_detects_credential_read()
    test_operator_directive_allows_help_text()
    test_operator_directive_clean_on_real_tree()
    test_every_check_returns_a_valid_status()
    test_every_check_gives_a_reason()
    test_manual_findings_name_what_to_review()
    test_no_finding_is_remediated_without_evidence()
    test_manual_placeholders_can_never_report_closed()
    test_discovered_register_is_separate_from_the_96()
    test_discovered_ids_must_be_well_formed()
    test_discovered_cannot_collide_with_an_audit_finding()
    test_d01_flips_on_the_ui_shim()
    test_d01_is_remediated_on_the_real_tree()
    test_dash_023_flips_on_hard_coded_identity()
    test_dash_023_catches_any_literal_not_just_keeper()
    test_dash_d02_flips_on_the_missing_parameter()
    test_track_c_has_no_live_findings()
    test_dash_016_ignores_helpers_and_catches_routes()
    test_dash_061_requires_reading_the_backend_status()
    test_dash_063_accepts_declining_to_measure_but_not_substituting()
    test_dash_064_requires_the_decision_to_turn_on_fleet_health()
    test_dash_065_rejects_a_liveness_probe_as_backup_proof()
    test_dash_080_requires_an_enforcing_status()
    test_dash_054_catches_a_guard_that_runs_too_late()
    test_dash_055_catches_exception_text_reaching_the_client()
    test_track_d_has_no_live_findings()


if __name__ == "__main__":
    run()
    print(f"\n{'='*60}")
    print(f"Dashboard Finding Tracker Tests: {passed} passed, {failed} failed")
    if failed:
        print("EXIT GATE: FAIL")
        sys.exit(1)
    print("EXIT GATE: PASS")
