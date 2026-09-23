#!/usr/bin/env python3
"""CAI v1.0 hostile calibration — the ruleset DESIGN (order §19, §26).

The order's §18 matrix has no ruleset cases; these B-cases are
SUPPLEMENTARY and calibrate check_cai_bootstrap.py, whose PASS would
otherwise be evidence only that it runs. Each case writes a design
directory, runs the SHIPPED checker as a subprocess, and asserts the exit
code AND the reason.

Known-positive: B1 (the committed design, as design), B10 (filled with a
non-executor authority and a real sha), B11 (live == design).
"""
from __future__ import annotations

import copy
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cai_testkit import Suite  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DESIGN = ROOT / "kai-pm" / "change-control" / "rulesets"
CHECKER = ROOT / "scripts" / "security" / "check_cai_bootstrap.py"
EXECUTOR = 216391246            # measured: the identity this executor pushes as
OTHER = 999000111               # a fixture authority identity, NOT the executor
SHA = "1" * 40
DECLARED = [f"B{i}" for i in range(1, 14)]


def load():
    return {f.name: json.loads(f.read_text()) for f in sorted(DESIGN.glob("*.json"))}


def fill(d, actor=OTHER, sha=SHA):
    d = copy.deepcopy(d)
    d["10-authority-tags-create.json"]["bypass_actors"][0]["actor_id"] = actor
    wf = d["30-main-integration.json"]["rules"][3]["parameters"]["workflows"][0]
    wf["sha"] = sha
    return d


def run(d, *extra, live=None):
    tmp = Path(tempfile.mkdtemp(prefix="cai-rs-"))
    try:
        for n, body in d.items():
            (tmp / n).write_text(json.dumps(body))
        args = [sys.executable, str(CHECKER), "--design", str(tmp)] + list(extra)
        if live is not None:
            (tmp.parent / (tmp.name + "-live.json")).write_text(json.dumps(live))
            args += ["--live", str(tmp.parent / (tmp.name + "-live.json"))]
        pr = subprocess.run(args, capture_output=True, text=True, timeout=120)
        return pr.returncode, pr.stdout + pr.stderr
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main() -> int:
    s = Suite("test_cai_bootstrap")
    base = load()
    READY = ["--apply-ready", "--executor-id", str(EXECUTOR)]

    rc, out = run(base)
    s.expect("B1", "committed design satisfies the design invariants", rc,
             out, want_rc=0, must_contain=["design rulesets read: 3", "PASS"])

    rc, out = run(base, *READY)
    s.expect("B2", "committed design is NOT apply-ready (2 placeholders)",
             rc, out, want_rc=1, must_contain=["unresolved placeholder",
                                               "placeholders in design: 2"])

    d = fill(base, actor=EXECUTOR)
    rc, out = run(d, *READY)
    s.expect("B3", "authority bypass == executor identity -> IDENTITY "
                   "COLLAPSE REFUSE", rc, out, want_rc=1,
             must_contain=["IDENTITY COLLAPSE"])

    d = fill(base)
    d["20-authority-tags-immutable.json"]["bypass_actors"] = [
        {"actor_type": "User", "actor_id": OTHER, "bypass_mode": "always"}]
    rc, out = run(d)
    s.expect("B4", "a bypass on the immutability ruleset -> REFUSE", rc, out,
             want_rc=1, must_contain=["immutability must have NO bypass"])

    d = fill(base)
    d["20-authority-tags-immutable.json"]["rules"] = [
        r for r in d["20-authority-tags-immutable.json"]["rules"]
        if r["type"] != "deletion"]
    rc, out = run(d)
    s.expect("B5", "deletion not restricted -> REFUSE", rc, out, want_rc=1,
             must_contain=["exactly 'update' + 'deletion'"])

    d = fill(base)
    for n in ("10-authority-tags-create.json", "20-authority-tags-immutable.json"):
        d[n]["conditions"]["ref_name"]["include"] = ["refs/tags/kai-admitted/**"]
    rc, out = run(d)
    s.expect("B6", "a namespace left uncovered (kai-wa/ dropped) -> REFUSE",
             rc, out, want_rc=1, must_contain=["must cover exactly"])

    d = fill(base)
    d["20-authority-tags-immutable.json"]["conditions"]["ref_name"]["exclude"] = [
        "refs/tags/kai-admitted/tmp-*"]
    rc, out = run(d)
    s.expect("B7", "an exclude pattern carving a hole -> REFUSE", rc, out,
             want_rc=1, must_contain=["exclude patterns carve holes"])

    d = fill(base)
    wf = d["30-main-integration.json"]["rules"][3]["parameters"]["workflows"][0]
    del wf["sha"]
    wf["ref"] = "refs/heads/main"
    rc, out = run(d)
    s.expect("B8", "verifier pinned by movable 'ref' instead of exact 'sha' "
                   "-> REFUSE", rc, out, want_rc=1,
             must_contain=["pinned by 'ref'", "carries no exact 'sha'"])

    d = fill(base)
    d["30-main-integration.json"]["enforcement"] = "evaluate"
    rc, out = run(d)
    s.expect("B9", "a ruleset not 'active' -> REFUSE", rc, out, want_rc=1,
             must_contain=["enforcement is 'evaluate'"])

    d = fill(base)
    rc, out = run(d, *READY)
    s.expect("B10", "filled with a NON-executor authority and a real sha -> "
                    "apply-ready PASS", rc, out, want_rc=0,
             must_contain=["placeholders in design: 0", "PASS"])

    d = fill(base)
    rc, out = run(d, "--executor-id", str(EXECUTOR),
                  live=list(d.values()))
    s.expect("B11", "live rulesets equal the design -> PASS", rc, out,
             want_rc=0, must_contain=["live rulesets: 3", "PASS"])

    d = fill(base)
    live = copy.deepcopy(list(d.values()))
    live[1]["bypass_actors"] = [{"actor_type": "User", "actor_id": OTHER,
                                 "bypass_mode": "always"}]
    rc, out = run(d, "--executor-id", str(EXECUTOR), live=live)
    s.expect("B12", "live server grants a bypass the design does not -> "
                    "REFUSE", rc, out, want_rc=1,
             must_contain=["live rulesets do not equal the design"])

    d = fill(base)
    rc, out = run(d, "--executor-id", str(EXECUTOR), live=[])
    s.expect("B13", "live server has NO rulesets (today's measured state) -> "
                    "REFUSE", rc, out, want_rc=1,
             must_contain=["live rulesets: 0",
                           "3 designed ruleset(s) absent"])
    return s.finish(DECLARED)


if __name__ == "__main__":
    sys.exit(main())
