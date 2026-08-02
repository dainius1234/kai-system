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
async def api_thing():
    try:
        return await call()
    except Exception:
        raise HTTPException(status_code=503, detail="upstream unavailable")
'''
    bad = '''
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


def run() -> None:
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


if __name__ == "__main__":
    run()
    print(f"\n{'='*60}")
    print(f"Dashboard Finding Tracker Tests: {passed} passed, {failed} failed")
    if failed:
        print("EXIT GATE: FAIL")
        sys.exit(1)
    print("EXIT GATE: PASS")
