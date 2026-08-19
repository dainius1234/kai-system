#!/usr/bin/env python3
"""Calibration for the layer that turns six builds into Item-8 verdicts.

WHY THIS FILE EXISTS
====================

`scripts/test_item8_instruments.py` calibrated the freeze guard, the
deriver and the explicit collector — 71 assertions, and it described
itself as calibration for *"three new instruments"*. The runner and the
summariser were not among them.

**That smaller denominator is exactly why six verdict defects shipped**,
found in adversarial review rather than here:

  1. a failed `.Image` binding rewrote the branch's single verdict, and
     the summariser printed that field as Axis 1 — an image-provenance
     fault laundered into a contingency measurement;
  2. B3 could PASS without the five attempts the frozen design requires,
     while its own note *asserted* "five attempts";
  3. B2's retry detector matched `attempt.*failed`, which its own
     injected line satisfied — the detector measured the treatment;
  4. `--iidfile` was written and never compared with the collected id;
  5. the summariser keyed rows into a dict, collapsing duplicates, then
     checked only `len(rows) == 6`;
  6. the runner promised a non-zero instrument-failure path and ended
     `exit 0` unconditionally.

Every one of those is a fixture below, because a defect that has been
found once and not made permanent is a defect waiting for the next
person. R4 step 4: calibrate against a known answer.

HOW THE RUNNER IS EXERCISED WITHOUT A DAEMON
============================================

Through its shipped entry point (rule 17), with `DOCKER` pointed at a
fake that produces scenario-specific build logs and inspect payloads.
The real Dockerfiles' genuine retry text — `retrying in` — is reproduced
faithfully, because the whole point of defect 3 is that the detector must
key on the SUBJECT's output and not on ours.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
RUNNER = REPO / "scripts" / "security" / "run_item8_experiment.sh"
SUMMARISE = REPO / "scripts" / "security" / "summarise_item8.py"
AUTHORITY = REPO / "scripts" / "security" / "check_item8_authority.py"

passed = 0
failed = 0
EXPECTED_SCENARIOS = 7
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


# A docker whose behaviour is driven entirely by FAKE_MODE, reproducing
# the SHIPPED Dockerfiles' genuine retry text.
FAKE_DOCKER = r'''#!/usr/bin/env python3
import json, os, sys
argv = sys.argv[1:]
mode = os.environ["FAKE_MODE"]
joined = " ".join(argv).upper()
branch = "B3" if "B3" in joined else ("B2" if "B2" in joined else "B1")

def retries(n):
    for _ in range(n):
        print("model download attempt /5 failed; retrying in 10s")

if argv and argv[0] == "build":
    if branch == "B3":
        n = 4 if mode == "b3_four_attempts" else 5
        retries(n)
        print("REFUSING TO BUILD: could not fetch the embedding model in 5 attempts.")
        sys.exit(1)
    if branch == "B2":
        print("ITEM8-B2: first attempt failed by injection")
        if mode != "b2_no_genuine_retry":
            retries(1)
        print("BAKED ok")
        # a successful build writes the iidfile
        if "--iidfile" in argv:
            p = argv[argv.index("--iidfile") + 1]
            open(p, "w").write("sha256:" + ("f" * 64 if mode != "iid_mismatch" else "0" * 64))
        sys.exit(0)
    print("BAKED ok")
    if "--iidfile" in argv:
        p = argv[argv.index("--iidfile") + 1]
        open(p, "w").write("sha256:" + ("f" * 64 if mode != "iid_mismatch" else "0" * 64))
    sys.exit(0)

if argv[:2] == ["image", "inspect"]:
    if branch == "B3":
        sys.exit(1)                      # no image, by design
    print(json.dumps({"Id": "sha256:" + "f" * 64, "RepoDigests": [],
                      "Os": "linux", "Architecture": "amd64"}))
    sys.exit(0)

if argv[:2] == ["container", "inspect"]:
    img = "sha256:" + ("f" * 64 if mode != "bind_mismatch" else "9" * 64)
    print(json.dumps({"Image": img}))
    sys.exit(0)

if argv and argv[0] == "run":
    if mode == "offline_fails":
        sys.stderr.write("offline load failed\n")
        sys.exit(1)
    print("OFFLINE LOAD OK")
    sys.exit(0)

if argv and argv[0] == "rm":
    sys.exit(0)
sys.exit(0)
'''


def run_runner(mode: str, td: Path) -> tuple[int, str, list[dict]]:
    """Drive the shipped runner with a fake docker and pre-derived files."""
    fake = td / "docker"
    fake.write_text(FAKE_DOCKER)
    fake.chmod(0o755)
    derived = td / "derived"
    ident = td / "ident"
    derived.mkdir(exist_ok=True)
    ident.mkdir(exist_ok=True)
    for image in ("memu-core", "memu-graph"):
        for branch in ("B1", "B2", "B3"):
            (derived / f"Dockerfile.{image}.{branch}").write_text(
                f"# derived {image} {branch}\nFROM scratch\n")
    results = td / "results.jsonl"
    env = {**os.environ, "FAKE_MODE": mode, "DOCKER": str(fake),
           "ITEM8_DERIVED": str(derived), "ITEM8_IDENT": str(ident),
           "ITEM8_RESULTS": str(results), "GITHUB_RUN_ID": "555"}
    p = subprocess.run(["bash", str(RUNNER)], capture_output=True, text=True,
                       cwd=str(REPO), env=env)
    rows = [json.loads(l) for l in results.read_text().splitlines()
            if l.strip()] if results.exists() else []
    return p.returncode, p.stdout + p.stderr, rows


def summarise(rows: list[dict], td: Path) -> tuple[int, str]:
    f = td / "s.jsonl"
    f.write_text("".join(json.dumps(r) + "\n" for r in rows))
    p = subprocess.run([sys.executable, str(SUMMARISE), "--results", str(f)],
                       capture_output=True, text=True, cwd=str(REPO))
    return p.returncode, p.stdout + p.stderr


def row(image="memu-core", branch="B1", a1="PASS", a2="BOUND", qual=True):
    return {"image": image, "branch": branch, "axis1_verdict": a1,
            "axis2_provenance": a2, "qualified_for_closure": qual,
            "genuine_retries_observed": 1, "elapsed_seconds": 1, "note": ""}


def six(**over):
    out = []
    for i in ("memu-core", "memu-graph"):
        for b in ("B1", "B2", "B3"):
            r = row(i, b, a2="IMAGE_NOT_PRODUCED_BY_DESIGN" if b == "B3"
                    else "BOUND")
            r.update(over)
            out.append(r)
    return out


# ── defect 1: axes laundered ────────────────────────────────────────────

def test_axis2_failure_leaves_axis1_standing() -> None:
    scenario("axes: a binding failure must not rewrite Axis 1")
    with tempfile.TemporaryDirectory() as d:
        td = Path(d)
        code, out, rows = run_runner("bind_mismatch", td)
        b1 = [r for r in rows if r["branch"] == "B1"]
        check("B1 rows exist", len(b1) == 2, str(rows))
        for r in b1:
            check(f"{r['image']} B1 Axis 1 still PASS",
                  r["axis1_verdict"] == "PASS", str(r))
            check(f"{r['image']} B1 Axis 2 records the fault",
                  r["axis2_provenance"] == "MISMATCH", str(r))
            check(f"{r['image']} B1 does NOT qualify for closure",
                  r["qualified_for_closure"] is False, str(r))
        _, sout = summarise(rows, td)
        check("the summary keeps Axis 1 complete",
              "AXIS 1 COMPLETE, PROVENANCE INCOMPLETE" in sout, sout)


# ── defect 2: B3 five attempts ──────────────────────────────────────────

def test_b3_requires_five_attempts() -> None:
    scenario("B3: four attempts must not PASS")
    with tempfile.TemporaryDirectory() as d:
        td = Path(d)
        code, out, rows = run_runner("b3_four_attempts", td)
        b3 = [r for r in rows if r["branch"] == "B3"]
        check("B3 rows exist", len(b3) == 2, str(rows))
        for r in b3:
            check(f"{r['image']} B3 with 4 retries is NOT PASS",
                  r["axis1_verdict"] != "PASS", str(r))
            check(f"{r['image']} B3 names the count it saw",
                  "4 genuine retry line(s)" in r.get("note", ""), str(r))
            check(f"{r['image']} B3 records the measured count",
                  r["genuine_retries_observed"] == 4, str(r))
    with tempfile.TemporaryDirectory() as d:
        td = Path(d)
        _, _, rows = run_runner("ok", td)
        b3 = [r for r in rows if r["branch"] == "B3"]
        for r in b3:
            check(f"{r['image']} B3 with 5 retries PASSES",
                  r["axis1_verdict"] == "PASS", str(r))
            check(f"{r['image']} B3 image state is by-design",
                  r["axis2_provenance"] == "IMAGE_NOT_PRODUCED_BY_DESIGN",
                  str(r))


# ── defect 3: B2's detector must be independent of the injection ────────

def test_b2_retry_detector_is_independent() -> None:
    scenario("B2: the injection marker alone is not a retry")
    with tempfile.TemporaryDirectory() as d:
        td = Path(d)
        code, out, rows = run_runner("b2_no_genuine_retry", td)
        b2 = [r for r in rows if r["branch"] == "B2"]
        check("B2 rows exist", len(b2) == 2, str(rows))
        for r in b2:
            check(f"{r['image']} B2 with only the injection is NOT PASS",
                  r["axis1_verdict"] != "PASS", str(r))
            check(f"{r['image']} B2 says the injection alone is not recovery",
                  "injection marker alone" in r.get("note", ""), str(r))
            check(f"{r['image']} B2 counted ZERO genuine retries",
                  r["genuine_retries_observed"] == 0, str(r))


# ── defect 4: iidfile corroboration ─────────────────────────────────────

def test_iidfile_is_actually_compared() -> None:
    scenario("iidfile: disagreement is an Axis-2 fault")
    with tempfile.TemporaryDirectory() as d:
        td = Path(d)
        _, _, rows = run_runner("ok", td)
        built = [r for r in rows if r["branch"] in ("B1", "B2")]
        for r in built:
            check(f"{r['image']} {r['branch']} iidfile CORROBORATED",
                  r["iidfile_corroboration"] == "CORROBORATED", str(r))
    with tempfile.TemporaryDirectory() as d:
        td = Path(d)
        _, _, rows = run_runner("iid_mismatch", td)
        built = [r for r in rows if r["branch"] in ("B1", "B2")]
        for r in built:
            check(f"{r['image']} {r['branch']} iidfile MISMATCH detected",
                  r["iidfile_corroboration"] == "MISMATCH", str(r))
            check(f"{r['image']} {r['branch']} it is an AXIS 2 fault",
                  r["axis2_provenance"] == "MISMATCH", str(r))
            check(f"{r['image']} {r['branch']} Axis 1 is untouched by it",
                  r["axis1_verdict"] == "PASS", str(r))


# ── defect 5: the denominator is a key set, not a row count ─────────────

def test_denominator_is_the_six_subjects() -> None:
    scenario("denominator: duplicates never substitute for a missing branch")
    with tempfile.TemporaryDirectory() as d:
        td = Path(d)
        code, out = summarise(six(), td)
        check("the true six QUALIFY", code == 0, out)
        check("and it says so", "ALL SIX QUALIFY" in out, out)

        rows = six()
        rows[5] = row("memu-core", "B1", a2="BOUND")   # dupe, drops graph/B3
        code, out = summarise(rows, td)
        check("six rows with a duplicate and a gap REFUSE", code == 4, out)
        check("the duplicate is named", "DUPLICATE: memu-core/B1" in out, out)
        check("the missing subject is named",
              "MISSING: memu-graph/B3" in out, out)
        check("and it says a count is not a denominator",
              "not a count of lines" in out, out)

        rows = six() + [row("memu-core", "B9")]
        code, out = summarise(rows, td)
        check("an unexpected subject REFUSES", code == 4, out)
        check("and is named", "UNEXPECTED: memu-core/B9" in out, out)


# ── defect 6: instrument failure must propagate ─────────────────────────

def test_instrument_failure_is_not_swallowed() -> None:
    scenario("exit status: a real instrument failure exits non-zero")
    with tempfile.TemporaryDirectory() as d:
        td = Path(d)
        code, out, rows = run_runner("ok", td)
        check("a clean six-branch run exits 0", code == 0, out)
        check("and writes six rows", len(rows) == 6, str(len(rows)))
        # an unwritable results path is a genuine inability to measure
        fake = td / "docker"; fake.write_text(FAKE_DOCKER); fake.chmod(0o755)
        p = subprocess.run(["bash", str(RUNNER)], capture_output=True,
                           text=True, cwd=str(REPO),
                           env={**os.environ, "FAKE_MODE": "ok",
                                "DOCKER": str(fake),
                                "ITEM8_RESULTS": str(td / "nodir" / "r.jsonl")})
        check("an unwritable results file exits NON-ZERO", p.returncode != 0,
              p.stdout + p.stderr)
        check("and says INSTRUMENT FAILURE",
              "INSTRUMENT FAILURE" in (p.stdout + p.stderr), p.stdout)
        # A missing instrument is likewise not a subject result. The
        # runner resolves its repo from its OWN location, so absence has
        # to be reproduced by placing a copy in a tree that lacks the
        # collectors -- changing cwd proves nothing, which is what this
        # fixture originally did.
        alt = td / "alt" / "scripts" / "security"
        alt.mkdir(parents=True, exist_ok=True)
        (alt / "run_item8_experiment.sh").write_text(RUNNER.read_text())
        p = subprocess.run(["bash", str(alt / "run_item8_experiment.sh")],
                           capture_output=True, text=True,
                           env={**os.environ, "FAKE_MODE": "ok",
                                "DOCKER": str(fake)})
        check("a tree without the collectors exits NON-ZERO",
              p.returncode != 0, p.stdout + p.stderr)
        check("and names the missing instrument",
              "INSTRUMENT FAILURE" in (p.stdout + p.stderr), p.stdout)


# ── the authority envelope ──────────────────────────────────────────────

def test_authority_envelope() -> None:
    scenario("authority: no envelope, no build")
    with tempfile.TemporaryDirectory() as d:
        td = Path(d)
        p = subprocess.run([sys.executable, str(AUTHORITY), "--sentinel",
                            str(td / "absent")], capture_output=True,
                           text=True, cwd=str(REPO))
        check("an absent envelope REFUSES", p.returncode == 1, p.stdout)
        check("and says however the job was triggered",
              "however the job was triggered" in p.stdout, p.stdout)

        s = td / "ITEM8_GO"
        s.write_text("frozen_r2=deadbeef\napproved_commit=HEAD\n"
                     "approved_tree=x\n")
        p = subprocess.run([sys.executable, str(AUTHORITY), "--sentinel",
                            str(s)], capture_output=True, text=True,
                           cwd=str(REPO))
        check("an envelope naming the wrong design REFUSES",
              p.returncode == 1, p.stdout)
        check("and says it cannot authorise a moved design",
              "has since moved" in p.stdout, p.stdout)

        s.write_text("approved_commit=HEAD\n")
        p = subprocess.run([sys.executable, str(AUTHORITY), "--sentinel",
                            str(s)], capture_output=True, text=True,
                           cwd=str(REPO))
        check("an incomplete envelope REFUSES", p.returncode == 1, p.stdout)
        check("and says it authorises nothing",
              "authorises nothing" in p.stdout, p.stdout)


def run_all() -> None:
    test_axis2_failure_leaves_axis1_standing()
    test_b3_requires_five_attempts()
    test_b2_retry_detector_is_independent()
    test_iidfile_is_actually_compared()
    test_denominator_is_the_six_subjects()
    test_instrument_failure_is_not_swallowed()
    test_authority_envelope()
    print(f"  inspected: {EXPECTED_SCENARIOS} verdict-layer scenario(s) "
          f"across 3 shipped entry points")
    check(f"all {EXPECTED_SCENARIOS} scenarios ran",
          len(executed) == EXPECTED_SCENARIOS, f"{len(executed)}: {executed}")


if __name__ == "__main__":
    print("=" * 60)
    run_all()
    print()
    print("=" * 60)
    print(f"Item-8 Verdict-Layer Calibration: {passed} passed, {failed} failed")
    print(f"EXIT GATE: {'PASS' if failed == 0 else 'FAIL'}")
    sys.exit(1 if failed else 0)
