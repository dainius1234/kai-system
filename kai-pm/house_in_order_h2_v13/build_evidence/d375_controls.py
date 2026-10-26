#!/usr/bin/env python3
"""D375 CONTROLS — the bounded remediation tranche, proved can-fail.

Banked WITH the repairs. Every repair here is paired with a case that
must still FAIL, because a control demonstrated only where it passes has
not been demonstrated at all (I-8).

WHAT THIS COVERS, AND WHAT IT DOES NOT

  RC-1  the developer-checkout recurrence.  THE PRODUCTION REPAIR IS NOT
        IN THIS TRANCHE — it is HELD on a contract collision recorded in
        section 1. What IS proved here is the thing Kai asked for
        regardless: that the existing preventive control still fires on
        a reintroduced literal, and still names it.
  RC-4  the demonstrated isolation leg only. The four UNKNOWN A-05
        observations are NOT touched and NOT baselined.
  RC-5  the Item-8 calibration's trigger-variable dependence.
  RC-6  the UH bind-mount calibration's history requirement.

NOTHING HERE LOWERS A FLOOR, INFLATES A BASELINE OR SKIPS A GATE. Where a
control must still refuse, this file proves it still refuses.

    python3 d375_controls.py
"""
from __future__ import annotations
import hashlib
import json
import os
import pathlib
import re
import subprocess
import sys
import tempfile

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parents[2]
OUT, FAILED = [], []

# Built, never written out: the rule that flags this literal scans every
# .py in the repository, and the first version of it flagged its own
# explanation.
NEEDLE = "/" + "home/user/" + "kai-system"


def A(s=""):
    OUT.append(s)


def check(name, got, want, extra=""):
    ok = got == want
    if not ok:
        FAILED.append(name)
    A(f"   {'OK  ' if ok else '<<< '}{name:<62}{str(got)[:40]}")
    if not ok:
        A(f"        EXPECTED: {str(want)[:90]}")
    if extra:
        A(f"        {extra}")
    return ok


def run(cmd, cwd=None, env=None, timeout=600):
    r = subprocess.run(cmd, cwd=None if cwd is None else str(cwd),
                       capture_output=True, text=True, env=env,
                       timeout=timeout)
    return r.returncode, r.stdout + r.stderr


def offenders(tree: pathlib.Path):
    """The developer-path rule's own predicate, applied to a tree."""
    hits = []
    skip = {"_archive", ".venv", "__pycache__", "node_modules", ".git"}
    for p in sorted(tree.rglob("*.py")):
        if skip & set(p.parts):
            continue
        try:
            text = p.read_text(errors="ignore")
        except OSError:
            continue
        for i, line in enumerate(text.splitlines(), 1):
            if NEEDLE in line:
                hits.append(f"{p.relative_to(tree)}:{i}")
    return hits


# ── 1. RC-1 — the control, and why the repair is held ─────────────────
def section1():
    A("1. RC-1 / D375 — THE PREVENTIVE CONTROL STILL FIRES")
    A("   The production repair is HELD (section 1c). What is proved here")
    A("   is that the control which should have stopped the recurrence is")
    A("   intact and still detects it.")
    A("")
    A("   1a. KNOWN-POSITIVE — the live tree")
    live = offenders(REPO)
    check("the rule finds the recurrence on the current tree",
          len(live) >= 7, True, f"{len(live)} occurrence(s)")
    for h in live:
        A(f"        {h}")
    files = sorted({h.rsplit(':', 1)[0] for h in live})
    check("across the five instrument files named in the diagnosis",
          len(files), 5, "; ".join(files))
    rc, out = run([sys.executable, "-m", "pytest", "-q",
                   "scripts/test_p1_p4_enhancements.py"
                   "::TestNoDeprecatedCalls::test_no_developer_home_paths"],
                  cwd=REPO)
    check("and the guard itself FAILS on it", rc != 0, True,
          [l for l in out.splitlines() if "developer" in l.lower()][:1])
    A("")

    A("   1b. KNOWN-POSITIVE — a NEWLY reintroduced literal is caught too")
    A("   The list of offenders is DERIVED, not maintained beside the rule:")
    A("   a file it has never seen must trip it the moment it appears.")
    probe = REPO / "kai_d375_reintroduction_probe.py"
    try:
        probe.write_text(f'SUBJECT = "{NEEDLE}/data/SOUL.md"\n')
        after = offenders(REPO)
        check("the probe file is detected", probe.name in " ".join(after),
              True)
        check("  and the count rises by exactly one", len(after) - len(live),
              1)
        rc2, out2 = run([sys.executable, "-m", "pytest", "-q",
                         "scripts/test_p1_p4_enhancements.py"
                         "::TestNoDeprecatedCalls::"
                         "test_no_developer_home_paths"], cwd=REPO)
        check("  and the guard names it", probe.name in out2, True)
        check("  and still fails", rc2 != 0, True)
    finally:
        probe.unlink(missing_ok=True)
    check("the probe was removed", probe.exists(), False)
    A("")

    A("   1c. KNOWN-NEGATIVE — the rule is not vacuous")
    with tempfile.TemporaryDirectory() as d:
        clean = pathlib.Path(d)
        (clean / "a.py").write_text("import pathlib\n"
                                    "ROOT = pathlib.Path(__file__)"
                                    ".resolve().parents[2]\n")
        (clean / "b.py").write_text('OTHER = "/home/user/repo/docs/foo.md"\n')
        check("a tree using self-derived paths reports nothing",
              offenders(clean), [])
    A("")

    A("   1d. WHY THE PRODUCTION REPAIR IS HELD — MEASURED, NOT ASSERTED")
    A("   Every offending file sits inside a hash-frozen evidence package.")
    A("   Repairing any of them invalidates that package's MANIFEST, and")
    A("   regenerating the MANIFEST moves the package aggregate. Kai's")
    A("   authorisation requires the repair AND requires hashes and")
    A("   evidence identity preserved; for these files those two cannot")
    A("   both hold, so the tranche stops here rather than choosing one")
    A("   silently (R14).")
    A("")
    pkgs = {
        "kai-pm/house_in_order_h2": ["pass_a.py", "cal_env.py"],
        "kai-pm/census_v11_claim_sensitivity": ["run_mutations.py"],
        "kai-pm/house_in_order_h2_v11": ["passa.py"],
        "kai-pm/house_in_order_census_v11": ["cal_claims.py"],
    }
    decisions = (REPO / "kai-pm" / "DECISIONS.md").read_text(errors="ignore")
    A(f"   {'package':<40}{'aggregate':<20}{'cited in DECISIONS'}")
    for pkg, names in pkgs.items():
        man = REPO / pkg / "MANIFEST.sha256"
        if not man.exists():
            A(f"   {pkg:<40}NO MANIFEST")
            continue
        agg = hashlib.sha256(man.read_bytes()).hexdigest()
        cited = decisions.count(agg[:16])
        A(f"   {pkg:<40}{agg[:16]:<20}{cited}   ({', '.join(names)})")
        # Not every manifest is a bare sha256sum list — some carry a
        # header. A line that does not split into (hash, name) is not a
        # checked entry, and silently treating it as one would abort the
        # control on a formatting difference.
        bad = []
        for ln in man.read_text().splitlines():
            parts = ln.split("  ", 1)
            if len(parts) != 2 or not re.fullmatch(r"[0-9a-f]{64}", parts[0]):
                continue
            h, n = parts
            f = REPO / pkg / n
            if not f.exists() or hashlib.sha256(f.read_bytes()).hexdigest() != h:
                bad.append(n)
        check(f"  {pkg} MANIFEST intact (repair not applied)", bad, [])
    A("")


# ── 2. RC-4 — the demonstrated isolation leg ──────────────────────────
def section2():
    A("2. RC-4 — THE DEMONSTRATED LEG ONLY. NOTHING BASELINED.")
    A("   memu-core/app.py:1052-1053 sets HF_HUB_OFFLINE and")
    A("   TRANSFORMERS_OFFLINE in its module body; test_contradiction.py")
    A("   loads that module, so the two variables outlived the file and")
    A("   reached every test after it. They are now restored, not declared.")
    A("")
    with tempfile.TemporaryDirectory() as d:
        rep = pathlib.Path(d) / "iso.json"
        env = dict(os.environ, KAI_ISOLATION_REPORT=str(rep),
                   MEMU_ALLOW_FAKE_EMBEDDINGS="true")
        rc, out = run([sys.executable, "-m", "pytest", "-q", "-p",
                       "scripts.security.isolation_plugin",
                       "scripts/test_contradiction.py"], cwd=REPO, env=env)
        check("test_contradiction.py still passes", rc, 0,
              [l for l in out.splitlines() if "passed" in l][-1:])
        data = json.loads(rep.read_text()) if rep.exists() else {}
        entry = next((v for k, v in data.items()
                      if "test_contradiction" in k), None)
        check("it now records NO cross-file leakage", entry, None,
              "was env_set ['HF_HUB_OFFLINE', 'TRANSFORMERS_OFFLINE']")
    A("")
    A("   KNOWN-POSITIVE — the ratchet still refuses undeclared leakage")
    with tempfile.TemporaryDirectory() as d:
        rep = pathlib.Path(d) / "leak.json"
        rep.write_text(json.dumps({
            str(REPO / "scripts" / "test_d375_probe.py"): {
                "added": [], "env_changed": [], "path_added": [],
                "replaced": [], "env_set": ["D375_PROBE_VAR"]}}))
        rc, out = run([sys.executable,
                       "scripts/security/check_test_isolation.py",
                       "--from-report", str(rep)], cwd=REPO)
        check("a synthetic undeclared leak is refused",
              "not declared" in out, True,
              [l.strip() for l in out.splitlines() if "not declared" in l][:1])
    A("")
    A("   NOT TOUCHED, AND STILL OPEN — the four UNKNOWN observations:")
    for n in ("scripts/test_hse_rams.py (local ratchet growth)",
              "scripts/test_llm_contract.py (CI-only, undeclared)",
              "scripts/test_audio_transcribe.py (CI-only +1 added)",
              "scripts/test_tts_service.py (CI-only 2 -> 3)"):
        A(f"     {n}")
    base = REPO / "scripts" / "security" / "test_isolation_baseline.json"
    check("the isolation baseline is UNCHANGED",
          run(["git", "diff", "--quiet", "--",
               str(base.relative_to(REPO))], cwd=REPO)[0], 0,
          "no leak was declared away")
    A("")


# ── 3. RC-5 — the trigger-variable dependence ─────────────────────────
def section3():
    A("3. RC-5 — THE CALIBRATION NOW CONTROLS ITS OWN ENVIRONMENT")
    A("   The production authority logic is UNCHANGED; only the")
    A("   calibration's environment is made deterministic.")
    A("")
    results = {}
    for label, extra in (("event unset", None),
                         ("GITHUB_EVENT_NAME=push", {"GITHUB_EVENT_NAME": "push"}),
                         ("GITHUB_EVENT_NAME=pull_request",
                          {"GITHUB_EVENT_NAME": "pull_request"})):
        env = dict(os.environ)
        env.pop("GITHUB_EVENT_NAME", None)
        env.update(extra or {})
        rc, out = run([sys.executable, "scripts/test_item8_verdicts.py"],
                      cwd=REPO, env=env)
        m = re.search(r"(\d+) passed, (\d+) failed", out)
        results[label] = (rc, m.group(0) if m else "?")
        A(f"   {label:<34}exit={rc}  {results[label][1]}")
    A("")
    check("every trigger state passes", [r for r, _ in results.values()],
          [0, 0, 0], "pull_request was exit=1, 405 passed / 1 failed")
    counts = {c for _, c in results.values()}
    check("and the assertion count is IDENTICAL in all three",
          len(counts), 1, f"{counts} — no check was neutralised to get green")
    check("production authority refusal logic untouched",
          run(["git", "diff", "--quiet", "--",
               "scripts/security/check_item8_authority.py"], cwd=REPO)[0], 0)
    # Asserted by INVOKING the guard, not by grepping its source. The
    # message is built from f-string fragments, so no contiguous literal
    # exists to match — and a source substring would prove the text is
    # present, not that the refusal happens (R13).
    with tempfile.TemporaryDirectory() as d:
        s = pathlib.Path(d) / "ITEM8_GO"
        s.write_text("frozen_r2=deadbeef\napproved_commit=HEAD\n"
                     "approved_tree=x\nauthorises=experiment\n")
        env = dict(os.environ, GITHUB_EVENT_NAME="pull_request",
                   GITHUB_RUN_ATTEMPT="1")
        rc, out = run([sys.executable,
                       "scripts/security/check_item8_authority.py",
                       "--sentinel", str(s)], cwd=REPO, env=env)
        check("  and the guard STILL refuses a non-sentinel event", rc, 1)
        check("  naming the event it refused on",
              "pull_request" in out, True,
              [l.strip() for l in out.splitlines()
               if "GITHUB_EVENT_NAME" in l][:1])
    A("")


# ── 4. RC-6 — the history requirement ─────────────────────────────────
def section4():
    A("4. RC-6 — THE ENVIRONMENT NOW SUPPLIES THE HISTORY")
    A("   The calibration is NOT relaxed. It still refuses when it cannot")
    A("   read its known answer; the checkout now gives it one.")
    A("")
    suite = "scripts/test_bind_mount_portability.py"
    rc, out = run([sys.executable, suite], cwd=REPO)
    m = re.search(r"(\d+) passed, (\d+) failed", out)
    check("KNOWN-NEGATIVE full clone: the calibration runs", rc, 0,
          m.group(0) if m else out[-120:])
    check("  with all four calibration checks present",
          out.count("CALIBRATION:"), 4)

    with tempfile.TemporaryDirectory() as d:
        sh = pathlib.Path(d) / "shallow"
        rc0, _ = run(["git", "clone", "--depth", "2", "--branch",
                      run(["git", "rev-parse", "--abbrev-ref", "HEAD"],
                          cwd=REPO)[1].strip(),
                      f"file://{REPO}", str(sh)], timeout=900)
        check("a depth-2 clone was created", rc0, 0)
        check("  and it is shallow",
              run(["git", "rev-parse", "--is-shallow-repository"],
                  cwd=sh)[1].strip(), "true")
        rc1, out1 = run([sys.executable, suite], cwd=sh)
        m1 = re.search(r"(\d+) passed, (\d+) failed", out1)
        check("KNOWN-POSITIVE shallow clone: the calibration STILL FAILS",
              rc1 != 0, True, m1.group(0) if m1 else out1[-120:])
        check("  naming shallow clone as the reason",
              "shallow clone" in out1, True)
        check("  and inability to read history is NOT a PASS",
              "EXIT GATE: PASS" in out1, False)
    A("")
    wf = (REPO / ".github" / "workflows" / "unified-hunter.yml").read_text()
    check("the UH checkout now requests full history",
          bool(re.search(r"fetch-depth:\s*0\b", wf)), True)
    check("  and no fetch-depth: 2 remains in that workflow",
          bool(re.search(r"fetch-depth:\s*2\b", wf)), False)
    A("")


def main():
    A("D375 CONTROLS. PRODUCED BY ORION. ZERO ADJUDICATION WEIGHT.")
    A("Banked WITH the repairs, before any full CI run.")
    A("")
    A("0. FINGERPRINT")
    for k, v in (("repository HEAD", run(["git", "rev-parse", "HEAD"],
                                         cwd=REPO)[1].strip()),
                 ("branch", run(["git", "rev-parse", "--abbrev-ref", "HEAD"],
                                cwd=REPO)[1].strip()),
                 ("instrument", hashlib.sha256(
                     pathlib.Path(__file__).read_bytes()).hexdigest())):
        A(f"   {k:<20}{v}")
    A("")
    section1()
    section2()
    section3()
    section4()
    A("OUTCOME")
    n = len([l for l in OUT if l.startswith(("   OK  ", "   <<< "))])
    A(f"   checks {n}   failures {len(FAILED)}")
    if FAILED:
        A("   REPORTED, NOT REPAIRED:")
        for f in FAILED:
            A(f"     <<< {f}")
    else:
        A("   Every control held. No floor lowered, no baseline inflated,")
        A("   no gate skipped, and every case that must still refuse does.")
    text = "\n".join(OUT) + "\n"
    (HERE / "D375_CONTROLS.txt").write_text(text, encoding="utf-8")
    print(text)
    raise SystemExit(1 if FAILED else 0)


if __name__ == "__main__":
    main()
