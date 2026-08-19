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
PARSER = REPO / "scripts" / "security" / "parse_buildkit_events.py"

passed = 0
failed = 0
EXPECTED_SCENARIOS = 15
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
"""A docker emitting BuildKit rawjson: instruction text and runtime
output in DIFFERENT fields, exactly as the real one does.
"""
import base64, json, os, sys
argv = sys.argv[1:]
mode = os.environ["FAKE_MODE"]
# Branch from the -f BASENAME suffix ONLY. Deriving it from the whole
# argv meant a random temp-dir name containing "b3" silently switched
# the scenario -- the same "matched incidental text" defect this whole
# repair is about, reproduced inside the fixture that tests for it.
_df = argv[argv.index("-f") + 1] if "-f" in argv else ""
branch = _df.rsplit(".", 1)[-1].upper() if _df else "B1"
if branch not in ("B1", "B2", "B3"):
    branch = "B1"
DIG = "sha256:" + "1" * 64

def ev(o): print(json.dumps(o))
def log(s): ev({"logs": [{"vertex": DIG, "stream": 1,
                          "data": base64.b64encode(s.encode()).decode()}]})

if argv and argv[0] == "build":
    name = '[3/9] RUN for attempt in 1 2 3 4 5; do python fetch && exit 0; echo "retrying in 10s"; done; echo "REFUSING TO BUILD"'
    cached = (mode == "cached_target")
    ev({"vertexes": [{"digest": DIG, "name": name, "cached": cached,
                      "started": None if mode == "never_started" else "t0"}]})
    if mode == "no_target_vertex":
        ev({"vertexes": [{"digest": "sha256:" + "2" * 64,
                          "name": "[1/9] FROM python:3.11-slim",
                          "started": "t0", "completed": "t1"}]})
    if mode == "unparseable":
        print("{not json")
        sys.exit(1)
    n = 0
    if branch == "B3":
        n = 4 if mode == "b3_four_attempts" else 5
        for _ in range(n):
            log("model download attempt /5 failed; retrying in 10s\n")
        log("REFUSING TO BUILD: could not fetch the model in 5 attempts.\n")
        ev({"vertexes": [{"digest": DIG, "name": name, "completed": "t1",
                          "error": "process did not complete successfully"}]})
        if mode == "b3_leaves_iid":
            open(argv[argv.index("--iidfile") + 1], "w").write("sha256:" + "f" * 64)
        if mode == "b3_leaves_empty_iid":
            open(argv[argv.index("--iidfile") + 1], "w").close()
        sys.exit(1)
    if branch == "B2":
        log("ITEM8-B2-INJECTED-ATTEMPT=%s\n" % ("2" if mode == "b2_inject_late" else "1"))
        if mode == "b2_double_inject":
            log("ITEM8-B2-INJECTED-ATTEMPT=1\n")
        if mode != "b2_no_genuine_retry":
            log("model download attempt /5 failed; retrying in 10s\n")
        log("BAKED ok\n")
    else:
        log("BAKED ok\n")
    ev({"vertexes": [{"digest": DIG, "name": name, "completed": "t1"}]})
    if mode != "iid_absent" and "--iidfile" in argv:
        open(argv[argv.index("--iidfile") + 1], "w").write(
            "sha256:" + ("f" * 64 if mode != "iid_mismatch" else "0" * 64))
    sys.exit(0)

if argv[:2] == ["image", "inspect"]:
    # `image inspect` has no -f; the BRANCH comes from the tag it is
    # asked about, e.g. kai-item8:b3-memu-core.
    ref = argv[-1]
    tb = "B3" if ":b3-" in ref else ("B2" if ":b2-" in ref else "B1")
    if tb == "B3" and mode != "b3_leaves_image":
        sys.exit(1)
    print(json.dumps({"Id": "sha256:" + "f" * 64, "RepoDigests": [],
                      "Os": "linux", "Architecture": "amd64"}))
    sys.exit(0)
if argv[:2] == ["container", "inspect"]:
    print(json.dumps({"Image": "sha256:" + ("f" * 64 if mode != "bind_mismatch" else "9" * 64)}))
    sys.exit(0)
if argv and argv[0] == "run":
    if mode == "offline_fails":
        sys.exit(1)
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
    tool = td / "toolchain.txt"
    tool.write_text("docker_version=fake\nbase_image_digest=sha256:abc\n")
    env = {**os.environ, "FAKE_MODE": mode, "DOCKER": str(fake),
           "ITEM8_DERIVED": str(derived), "ITEM8_IDENT": str(ident),
           "ITEM8_RESULTS": str(results), "GITHUB_RUN_ID": "555",
           "ITEM8_TOOLCHAIN": str(tool)}
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


def row(image="memu-core", branch="B1", a1="PASS", a2="BOUND", **extra):
    r = {"image": image, "branch": branch, "axis1_verdict": a1,
         "axis2_provenance": a2, "genuine_retries_observed": 1,
         "elapsed_seconds": 1, "note": "",
         "iidfile_corroboration": "n/a" if branch == "B3" else "CORROBORATED",
         "toolchain_sha256": "t" * 64}
    r.update(extra)
    return r


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
            check(f"{r['image']} B1 emits NO composite claim",
                  "qualified_for_closure" not in r, str(r))
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
                  "4 runtime retry line(s)" in r.get("note", ""), str(r))
            check(f"{r['image']} B3 records the measured retries",
                  r["runtime_retries_observed"] == 4, str(r))
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
            check(f"{r['image']} B2 says recovery is not established",
                  "recovery is not established" in r.get("note", ""), str(r))
            check(f"{r['image']} B2 saw NO runtime retry line",
                  r["runtime_retries_observed"] == 0, str(r))


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
                            str(td / "absent"), "--allow-no-ci"],
                           capture_output=True, text=True, cwd=str(REPO))
        check("an absent envelope REFUSES", p.returncode == 1, p.stdout)
        check("and says however the job was triggered",
              "however the job was triggered" in p.stdout, p.stdout)

        s = td / "ITEM8_GO"
        s.write_text("frozen_r2=deadbeef\napproved_commit=HEAD\n"
                     "approved_tree=x\n")
        p = subprocess.run([sys.executable, str(AUTHORITY), "--sentinel",
                            str(s), "--allow-no-ci"], capture_output=True,
                           text=True, cwd=str(REPO))
        check("an envelope naming the wrong design REFUSES",
              p.returncode == 1, p.stdout)
        check("and says it cannot authorise a moved design",
              "has since moved" in p.stdout, p.stdout)

        s.write_text("approved_commit=HEAD\n")
        p = subprocess.run([sys.executable, str(AUTHORITY), "--sentinel",
                            str(s), "--allow-no-ci"], capture_output=True,
                           text=True, cwd=str(REPO))
        check("an incomplete envelope REFUSES", p.returncode == 1, p.stdout)
        check("and says it authorises nothing",
              "authorises nothing" in p.stdout, p.stdout)


# ── new blocker 1: a MISSING iidfile must block qualification ───────────

def test_absent_iidfile_blocks_qualification() -> None:
    scenario("iidfile: ABSENT is not 'no objection'")
    with tempfile.TemporaryDirectory() as d:
        td = Path(d)
        _, _, rows = run_runner("iid_absent", td)
        built = [r for r in rows if r["branch"] in ("B1", "B2")]
        check("built branches exist", len(built) == 4, str(len(built)))
        for r in built:
            check(f"{r['image']} {r['branch']} iidfile is ABSENT",
                  r["iidfile_corroboration"] == "ABSENT", str(r))
            check(f"{r['image']} {r['branch']} Axis 1 still PASS",
                  r["axis1_verdict"] == "PASS", str(r))
        code, out = summarise(rows, td)
        check("an absent iidfile blocks closure", code != 0, out)
        check("and the reason is named",
              "iidfile corroboration is ABSENT" in out, out)


# ── new blocker: presentation must not satisfy the attempt detector ────

def test_presentation_cannot_satisfy_the_detector() -> None:
    scenario("markers: instruction text is not execution")
    with tempfile.TemporaryDirectory() as d:
        td = Path(d)
        _, _, rows = run_runner("b3_presentation_only", td)
        b3 = [r for r in rows if r["branch"] == "B3"]
        for r in b3:
            check(f"{r['image']} B3 unaffected by instruction text",
                  r["runtime_retries_observed"] == 5, str(r))
            check(f"{r['image']} B3 still PASSES on runtime evidence",
                  r["axis1_verdict"] == "PASS", str(r))


# ── new blocker: B3 absence uses EXISTENCE, not size ──────────────────

def test_b3_absence_is_existence_not_size() -> None:
    scenario("B3: a leftover iidfile blocks IMAGE_NOT_PRODUCED_BY_DESIGN")
    with tempfile.TemporaryDirectory() as d:
        td = Path(d)
        _, _, rows = run_runner("b3_leaves_iid", td)
        for r in [r for r in rows if r["branch"] == "B3"]:
            check(f"{r['image']} B3 with a leftover iidfile is NOT PASS",
                  r["axis1_verdict"] != "PASS", str(r))
            check(f"{r['image']} B3 provenance is not by-design",
                  r["axis2_provenance"] != "IMAGE_NOT_PRODUCED_BY_DESIGN",
                  str(r))
    # A ZERO-BYTE iidfile is the case that distinguishes -e from -s. The
    # first version of this fixture wrote a NON-empty file, so both
    # operators behaved identically and the reinjection could not fire.
    with tempfile.TemporaryDirectory() as d:
        td = Path(d)
        _, _, rows = run_runner("b3_leaves_empty_iid", td)
        for r in [r for r in rows if r["branch"] == "B3"]:
            check(f"{r['image']} B3 with a ZERO-BYTE iidfile is NOT PASS",
                  r["axis1_verdict"] != "PASS", str(r))
    with tempfile.TemporaryDirectory() as d:
        td = Path(d)
        _, _, rows = run_runner("b3_leaves_image", td)
        for r in [r for r in rows if r["branch"] == "B3"]:
            check(f"{r['image']} B3 with a surviving image is NOT PASS",
                  r["axis1_verdict"] != "PASS", str(r))


def test_a_self_certified_row_is_contradicted() -> None:
    """Rule 26: no consequential mechanism self-approves."""
    scenario("qualification: a row's own claim is never trusted")
    with tempfile.TemporaryDirectory() as d:
        td = Path(d)
        rows = six()
        # Axis 1 failed, yet the row asserts it qualifies.
        rows[0] = row("memu-core", "B1", a1="UNMEASURED", a2="BOUND",
                      qualified_for_closure=True)
        code, out = summarise(rows, td)
        check("a self-certified row does NOT qualify", code != 0, out)
        check("and the disagreement is printed",
              "DISAGREEMENT: memu-core/B1" in out, out)
        check("and the derived reason is given",
              "Axis 1 is UNMEASURED" in out, out)


# ── the parser itself ──────────────────────────────────────────────────

def parse_events(events: str, td: Path, *, needle="for attempt in 1 2 3 4 5",
                 counts=("retrying in",)) -> tuple[int, str]:
    f = td / "ev.jsonl"
    f.write_text(events)
    argv = [sys.executable, str(PARSER), "--events", str(f),
            "--target-substring", needle, "--json"]
    for c in counts:
        argv += ["--count", c]
    r = subprocess.run(argv, capture_output=True, text=True, cwd=str(REPO))
    return r.returncode, r.stdout + r.stderr


def test_parser_separates_instruction_from_runtime() -> None:
    """The whole reason for rawjson: a name is not an execution."""
    scenario("parser: instruction text is never counted as output")
    import base64 as b64
    D = "sha256:" + "1" * 64

    def evline(**kw):
        return json.dumps(kw)

    def logline(s):
        return json.dumps({"logs": [{"vertex": D, "stream": 1,
                                     "data": b64.b64encode(s.encode()).decode()}]})

    with tempfile.TemporaryDirectory() as d:
        td = Path(d)
        # The INSTRUCTION contains the phrase three times. RUNTIME contains
        # it twice. A parser that reads the name would say three.
        name = 'RUN for attempt in 1 2 3 4 5; do echo "retrying in"; echo "retrying in"; echo "retrying in"; done'
        ev = "\n".join([
            evline(vertexes=[{"digest": D, "name": name, "started": "t0"}]),
            logline("retrying in 10s\n"),
            logline("retrying in 20s\n"),
            evline(vertexes=[{"digest": D, "name": name, "completed": "t1"}]),
        ])
        code, out = parse_events(ev, td)
        check("a well-formed stream parses", code == 0, out)
        facts = json.loads(out)
        check("counts come from RUNTIME, not the instruction",
              facts["counts"]["retrying in"] == 2,
              f"got {facts['counts']} from a name containing it 3x")
        check("execution is observed from the vertex", facts["executed"] is True)
        check("cached is False here", facts["cached"] is False)

        # known-negatives
        code, out = parse_events("{not json\n", td)
        check("an unparseable stream REFUSES", code == 1, out)
        check("and says a partial stream is not an empty one",
              "cannot be distinguished" in out, out)
        code, out = parse_events(evline(vertexes=[{"digest": D, "name": "FROM x"}]), td)
        check("a missing target vertex REFUSES", code == 1, out)
        check("and says nothing about it can be measured",
              "can be measured" in out, out)
        dupe = "\n".join([
            evline(vertexes=[{"digest": D, "name": name}]),
            evline(vertexes=[{"digest": "sha256:" + "2" * 64, "name": name}]),
        ])
        code, out = parse_events(dupe, td)
        check("an ambiguous target REFUSES", code == 1, out)
        code, out = parse_events("", td)
        check("an empty stream REFUSES", code == 1, out)
        check("and says it licenses no conclusion", "licenses" in out, out)


# ── the authority guard's new one-shot controls ────────────────────────

def authority(envelope: str, td: Path, env_extra: dict | None = None,
              allow_no_ci=True) -> tuple[int, str]:
    s = td / "ITEM8_GO"
    s.write_text(envelope)
    argv = [sys.executable, str(AUTHORITY), "--sentinel", str(s)]
    if allow_no_ci:
        argv.append("--allow-no-ci")
    env = {k: v for k, v in os.environ.items()
           if k not in ("GITHUB_RUN_ATTEMPT", "GITHUB_EVENT_NAME")}
    env.update(env_extra or {})
    r = subprocess.run(argv, capture_output=True, text=True, cwd=str(REPO),
                       env=env)
    return r.returncode, r.stdout + r.stderr


def test_authority_one_shot_controls() -> None:
    """The newest controls were also the least calibrated. Not any more."""
    scenario("authority: one-shot controls fail closed")
    frozen = subprocess.run(
        [sys.executable, str(REPO / "scripts" / "security" /
                             "check_item8_design.py"), "--quiet"],
        capture_output=True, text=True, cwd=str(REPO)).stdout.strip()
    head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                          text=True, cwd=str(REPO)).stdout.strip()
    tree = subprocess.run(["git", "rev-parse", "HEAD^{tree}"],
                          capture_output=True, text=True,
                          cwd=str(REPO)).stdout.strip()
    good = f"frozen_r2={frozen}\napproved_commit={head}\napproved_tree={tree}\n"

    with tempfile.TemporaryDirectory() as d:
        td = Path(d)
        # ABSENCE must refuse, not pass. This is the defect: the first
        # version accepted a missing variable as consent.
        code, out = authority(good, td, allow_no_ci=False)
        check("absent GITHUB_RUN_ATTEMPT REFUSES", code == 1, out)
        check("and says unestablished is not satisfied",
              "cannot be established" in out, out)
        code, out = authority(good, td, {"GITHUB_RUN_ATTEMPT": "1"},
                              allow_no_ci=False)
        check("absent GITHUB_EVENT_NAME REFUSES", code == 1, out)
        # wrong values
        code, out = authority(good, td, {"GITHUB_RUN_ATTEMPT": "2",
                                         "GITHUB_EVENT_NAME": "push"},
                              allow_no_ci=False)
        check("a RE-RUN REFUSES", code == 1, out)
        check("and names replacement execution", "replacement execution" in out, out)
        code, out = authority(good, td, {"GITHUB_RUN_ATTEMPT": "1",
                                         "GITHUB_EVENT_NAME": "workflow_dispatch"},
                              allow_no_ci=False)
        check("a manual dispatch REFUSES", code == 1, out)
        check("whatever the platform permits", "whatever the platform" in out, out)
        # git-side bindings: HEAD is its own approved_commit, so the diff
        # is empty and the ADD requirement must refuse.
        code, out = authority(good, td)
        check("no ADD of the envelope REFUSES", code == 1, out)
        check("and demands diff status A", "not 'A'" in out, out)
        # a parent that is not the approved commit
        parent = subprocess.run(["git", "rev-parse", "HEAD~1"],
                                capture_output=True, text=True,
                                cwd=str(REPO)).stdout.strip()
        ptree = subprocess.run(["git", "rev-parse", "HEAD~1^{tree}"],
                               capture_output=True, text=True,
                               cwd=str(REPO)).stdout.strip()
        code, out = authority(
            f"frozen_r2={frozen}\napproved_commit={parent}\n"
            f"approved_tree={ptree}\n", td)
        check("a tree differing beyond the envelope REFUSES", code == 1, out)
        check("and says review approved one artefact",
              "would run another" in out, out)


# ── toolchain binding ──────────────────────────────────────────────────

def test_toolchain_binding_is_required() -> None:
    scenario("toolchain: an unbound row cannot qualify")
    with tempfile.TemporaryDirectory() as d:
        td = Path(d)
        rows = six()
        for r in rows:
            r.pop("toolchain_sha256", None)
        code, out = summarise(rows, td)
        check("rows with no toolchain binding do NOT qualify", code != 0, out)
        check("and the reason is named", "toolchain binding is" in out, out)
        rows = six()
        for r in rows:
            r["toolchain_sha256"] = "ABSENT"
        code, out = summarise(rows, td)
        check("an ABSENT toolchain does NOT qualify", code != 0, out)


def test_b1_must_prove_uncached_execution() -> None:
    """R2's B1 needs the fetch to RUN, not to have been requested."""
    scenario("B1: --no-cache is a request; execution is the observation")
    with tempfile.TemporaryDirectory() as d:
        td = Path(d)
        _, _, rows = run_runner("cached_target", td)
        built = [r for r in rows if r["branch"] in ("B1", "B2")]
        check("cached-target rows exist", len(built) == 4, str(len(built)))
        for r in built:
            check(f"{r['image']} {r['branch']} a CACHED target is NOT PASS",
                  r["axis1_verdict"] != "PASS", str(r))
            check(f"{r['image']} {r['branch']} says it came from cache",
                  "FROM CACHE" in r.get("note", ""), str(r))
            check(f"{r['image']} {r['branch']} records cached=True",
                  r["target_vertex_cached"] == "True", str(r))
    with tempfile.TemporaryDirectory() as d:
        td = Path(d)
        _, _, rows = run_runner("never_started", td)
        for r in [r for r in rows if r["branch"] == "B1"]:
            check(f"{r['image']} B1 an unexecuted target is NOT PASS",
                  r["axis1_verdict"] != "PASS", str(r))
            check(f"{r['image']} B1 says it did not execute",
                  "did not execute" in r.get("note", ""), str(r))


def run_all() -> None:
    test_axis2_failure_leaves_axis1_standing()
    test_b3_requires_five_attempts()
    test_b2_retry_detector_is_independent()
    test_iidfile_is_actually_compared()
    test_denominator_is_the_six_subjects()
    test_instrument_failure_is_not_swallowed()
    test_authority_envelope()
    test_absent_iidfile_blocks_qualification()
    test_presentation_cannot_satisfy_the_detector()
    test_b3_absence_is_existence_not_size()
    test_a_self_certified_row_is_contradicted()
    test_parser_separates_instruction_from_runtime()
    test_authority_one_shot_controls()
    test_toolchain_binding_is_required()
    test_b1_must_prove_uncached_execution()
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
