#!/usr/bin/env python3
"""Calibration for Item 8's three new instruments.

The frozen design (D285, `0055ead8…8796`) is implemented by three pieces
that did not exist before:

  * `check_item8_design.py`          — refuses to build if the freeze moved
  * `derive_item8_dockerfile.py`     — derives experimental Dockerfiles
  * `collect_explicit_image_identity.py` — identity for a named image

Each is exercised through its SHIPPED entry point as a subprocess (rule
17), with a fake `docker` injected where a daemon would otherwise be
needed — this host has none, and a calibration that only runs where the
subject runs never runs.

Every instrument gets a known-POSITIVE and a known-NEGATIVE, because a
check that has only ever been shown to pass is comfort, not control
(rule 15). The negatives here are the ones that would actually hurt:

  * a frozen design that has MOVED must stop the build, and the
    superseded R1 digest must be diagnosed rather than merely mismatched;
  * a derivation whose anchor no longer matches must REFUSE, not silently
    produce the original file — that is rule 18's earning failure, and it
    is the one most likely to recur when somebody rewords a Dockerfile;
  * a B3 derivation must deny network to EXACTLY the HF instruction, and
    the calibration asserts the pip layers are untouched, because a
    build-level flag would fail for the wrong reason and read as a pass;
  * an explicit-image record must stay contract-compatible with the
    unchanged sibling module, asserted by feeding one into it.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SEC = REPO / "scripts" / "security"
GUARD = SEC / "check_item8_design.py"
DERIVE = SEC / "derive_item8_dockerfile.py"
EXPLICIT = SEC / "collect_explicit_image_identity.py"
SIBLING = SEC / "collect_image_identity.py"

FROZEN = "0055ead8f51d8758bcd6f05b9b1fff84dd9509e91e79c79b6a2500ab78488796"
R1_DEAD = "b8ba2ae363d827b33e8d10c54a44789f35c22f0ad14f04b306897fa416e8ff98"

passed = 0
failed = 0
EXPECTED_SCENARIOS = 8
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


FAKE_DOCKER = '''#!/usr/bin/env python3
import json, os, sys
argv = sys.argv[1:]
mode = os.environ["FAKE_MODE"]
if argv[:2] == ["image", "inspect"]:
    if mode == "missing":
        sys.stderr.write("Error: No such image\\n")
        sys.exit(1)
    if mode == "no_id":
        print(json.dumps({"Os": "linux", "Architecture": "amd64"}))
        sys.exit(0)
    print(json.dumps({"Id": "sha256:" + "e" * 64, "RepoDigests": [],
                      "Os": "linux", "Architecture": "amd64"}))
    sys.exit(0)
if argv[:2] == ["container", "inspect"]:
    print(json.dumps({"Image": "sha256:" + "e" * 64}))
    sys.exit(0)
sys.stderr.write("fake docker: unexpected argv %r\\n" % (argv,))
sys.exit(99)
'''


def fake_docker(td: Path) -> Path:
    p = td / "docker"
    p.write_text(FAKE_DOCKER)
    p.chmod(0o755)
    return p


# ── the freeze guard ────────────────────────────────────────────────────

def test_guard_known_positive() -> None:
    scenario("guard: the real frozen design passes")
    r = subprocess.run([sys.executable, str(GUARD)], capture_output=True,
                       text=True, cwd=str(REPO))
    check("the live tree's frozen design PASSES", r.returncode == 0, r.stdout)
    check("and reports the frozen byte count", "7776" in r.stdout, r.stdout)
    check("and states its denominator",
          "inspected: 1 canonical region" in r.stdout, r.stdout)


def test_guard_known_negative() -> None:
    """A moved design must stop the build. This is the whole point."""
    scenario("guard: a moved design refuses")
    with tempfile.TemporaryDirectory() as d:
        td = Path(d)
        real = (REPO / "kai-pm" / "DECISIONS.md").read_text()
        # one character inside the region is enough to move the digest
        stem = "### CANONICAL ITEM-8 DESIGN R2 "
        i = real.index(stem + "— BEGIN")
        j = real.index(stem + "— END")
        moved = real[:i] + real[i:j].replace("N = 6 builds", "N = 5 builds", 1) \
            + real[j:]
        p = td / "moved.md"
        p.write_text(moved)
        r = subprocess.run([sys.executable, str(GUARD), "--file", str(p)],
                           capture_output=True, text=True, cwd=str(REPO))
        check("a moved frozen design FAILS", r.returncode == 1, r.stdout)
        check("and says no build may proceed",
              "NO BUILD MAY PROCEED" in r.stdout, r.stdout)
        # the superseded R1 digest must be DIAGNOSED, not merely mismatched
        r = subprocess.run([sys.executable, str(GUARD), "--expect", R1_DEAD],
                           capture_output=True, text=True, cwd=str(REPO))
        check("expecting the dead R1 digest FAILS", r.returncode == 1, r.stdout)
        # THE WHOLE digest must be compared, not a prefix. Found by
        # reinjection: weakening the guard to `got[:8] == expect[:8]` passed
        # every other assertion here, because each fixture moves the digest
        # entirely. A comparison whose STRENGTH is untested is a comparison
        # nobody has actually checked.
        for near in (FROZEN[:-1] + ("0" if FROZEN[-1] != "0" else "1"),
                     ("0" if FROZEN[0] != "0" else "1") + FROZEN[1:]):
            r = subprocess.run([sys.executable, str(GUARD), "--expect", near],
                               capture_output=True, text=True, cwd=str(REPO))
            check("a digest differing by ONE character FAILS",
                  r.returncode == 1, f"{near}\n{r.stdout}")
        # missing subject
        r = subprocess.run([sys.executable, str(GUARD), "--file",
                            str(td / "nope.md")], capture_output=True,
                           text=True, cwd=str(REPO))
        check("an unreadable decisions file REFUSES", r.returncode == 1, r.stdout)
        check("and refuses to call it probably fine",
              "probably fine" in r.stdout, r.stdout)
        # ambiguous region
        p2 = td / "dupe.md"
        p2.write_text(real + "\n" + stem + "— BEGIN\nx\n" + stem + "— END\n")
        r = subprocess.run([sys.executable, str(GUARD), "--file", str(p2)],
                           capture_output=True, text=True, cwd=str(REPO))
        check("duplicated region markers REFUSE", r.returncode == 1, r.stdout)
        check("and say an ambiguous region pins nothing",
              "pins nothing" in r.stdout, r.stdout)


# ── the derivation ──────────────────────────────────────────────────────

def derive(image: str, branch: str, td: Path, source: str | None = None
           ) -> tuple[int, str, str]:
    out = td / f"df.{image}.{branch}"
    argv = [sys.executable, str(DERIVE), "--image", image,
            "--branch", branch, "--out", str(out)]
    if source:
        argv += ["--source", source]
    r = subprocess.run(argv, capture_output=True, text=True, cwd=str(REPO))
    return r.returncode, r.stdout + r.stderr, (out.read_text()
                                               if out.exists() else "")


def test_derivation_all_six() -> None:
    scenario("derivation: all six derive, shipped files untouched")
    before = {i: (REPO / i / "Dockerfile").read_bytes()
              for i in ("memu-core", "memu-graph")}
    with tempfile.TemporaryDirectory() as d:
        td = Path(d)
        for image in ("memu-core", "memu-graph"):
            for branch in ("B1", "B2", "B3"):
                code, out, text = derive(image, branch, td)
                check(f"{image}/{branch} derives", code == 0, out)
                check(f"{image}/{branch} carries the PINNED frontend",
                      text.startswith("# syntax=docker/dockerfile:1.9.0@sha256:"),
                      text[:120])
                check(f"{image}/{branch} does NOT float the frontend",
                      "dockerfile:1\n" not in text[:200], text[:120])
    after = {i: (REPO / i / "Dockerfile").read_bytes()
             for i in ("memu-core", "memu-graph")}
    check("the SHIPPED Dockerfiles are byte-identical afterwards",
          before == after)


def test_b3_denies_exactly_one_instruction() -> None:
    """Build-level denial would kill pip and fail for the wrong reason."""
    scenario("derivation: B3 denies exactly the HF instruction")
    with tempfile.TemporaryDirectory() as d:
        td = Path(d)
        for image in ("memu-core", "memu-graph"):
            _, _, text = derive(image, "B3", td)
            check(f"{image} B3 has exactly ONE --network=none",
                  text.count("--network=none") == 1, str(text.count("--network=none")))
            check(f"{image} B3 denies the retry RUN",
                  "RUN --network=none for attempt in 1 2 3 4 5" in text)
            for line in text.splitlines():
                if line.startswith("RUN") and "pip install" in line:
                    check(f"{image} B3 leaves pip install with network",
                          "--network=none" not in line, line)


def test_b2_injects_one_first_attempt_failure() -> None:
    scenario("derivation: B2 wraps the real command, once")
    with tempfile.TemporaryDirectory() as d:
        td = Path(d)
        for image in ("memu-core", "memu-graph"):
            _, _, text = derive(image, "B2", td)
            check(f"{image} B2 carries the scaffolding attempt marker",
                  'ITEM8-MARK ATTEMPT=\\$attempt' in text, text[:200])
            check(f"{image} B2 injects the sentinel once",
                  text.count("item8-b2-first-attempt-consumed") == 2,
                  str(text.count("item8-b2-first-attempt-consumed")))
            check(f"{image} B2 announces the injection with a RUNTIME value",
                  'ITEM8-MARK B2INJECT=\\$attempt' in text, text[:200])
            # THE STRUCTURAL PROPERTY: the measured form must not exist in
            # the source, or BuildKit echoing the instruction could satisfy
            # the detector. Source carries `$attempt`; only an execution
            # ever prints a number.
            check(f"{image} B2 source contains NO expanded marker",
                  "B2INJECT=1" not in text and "ATTEMPT=1" not in text, text[:200])
            check(f"{image} B2 keeps the genuine command reachable",
                  "else" in text and "fi && exit 0" in text)
            check(f"{image} B2 does NOT deny the network",
                  "--network=none" not in text)


def test_derivation_refuses_rather_than_silently_doing_nothing() -> None:
    """Rule 18's earning failure, made permanent as a fixture."""
    scenario("derivation: a moved anchor REFUSES")
    with tempfile.TemporaryDirectory() as d:
        td = Path(d)
        src = td / "Dockerfile"
        src.write_text("FROM python:3.11-slim\nRUN pip install foo\n")
        for branch in ("B1", "B2", "B3"):
            code, out, _ = derive("memu-core", branch, td, source=str(src))
            check(f"{branch} with no retry loop REFUSES", code == 1, out)
            check(f"{branch} says it must not guess which RUN was meant",
                  "must not guess" in out, out)
        # refusing to overwrite the shipped file
        r = subprocess.run([sys.executable, str(DERIVE), "--image",
                            "memu-core", "--branch", "B1", "--out",
                            str(REPO / "memu-core" / "Dockerfile")],
                           capture_output=True, text=True, cwd=str(REPO))
        check("deriving ONTO the shipped Dockerfile REFUSES",
              r.returncode == 1, r.stdout)
        check("and says it would destroy the thing under test",
              "destroys the thing under test" in r.stdout, r.stdout)


# ── the explicit-image collector ────────────────────────────────────────

def explicit(mode: str, td: Path, ref: str = "kai-item8:b1-memu-core",
             label: str = "item8-b1-memu-core") -> tuple[int, str, list[dict]]:
    out = td / "explicit.jsonl"
    r = subprocess.run(
        [sys.executable, str(EXPLICIT), "--image-ref", ref, "--label", label,
         "--docker", str(fake_docker(td)), "--run-id", "777", "--out", str(out)],
        capture_output=True, text=True, cwd=str(REPO),
        env={**os.environ, "FAKE_MODE": mode})
    rows = [json.loads(l) for l in out.read_text().splitlines()] \
        if out.exists() else []
    return r.returncode, r.stdout + r.stderr, rows


def test_explicit_collector() -> None:
    scenario("explicit collector: positive and negatives")
    with tempfile.TemporaryDirectory() as d:
        td = Path(d)
        code, out, rows = explicit("ok", td)
        check("a resolvable explicit ref EXITS 0", code == 0, out)
        check("state is RECORDED", rows[0]["identity_state"] == "RECORDED", str(rows[0]))
        check("the id is carried",
              rows[0]["docker_image_id"] == "sha256:" + "e" * 64, str(rows[0]))
        check("the label lands in `service` for the sibling's --service",
              rows[0]["service"] == "item8-b1-memu-core", str(rows[0]))
        check("never called a digest", "image_digest" not in rows[0], str(rows[0]))
        check("empty RepoDigests is NULL, not ABSENT",
              rows[0]["repo_digest_state"] == "NULL", str(rows[0]))
        check("the denominator is stated",
              "inspected: 1 explicit image reference" in out, out)

        code, out, rows = explicit("missing", td)
        check("a missing image EXITS 3", code == 3, out)
        check("state is UNRECORDED", rows[0]["identity_state"] == "UNRECORDED", str(rows[0]))
        check("the id is null, NOT an empty string",
              rows[0]["docker_image_id"] is None, str(rows[0]))

        code, out, rows = explicit("no_id", td)
        check("a payload with no Id EXITS 3", code == 3, out)
        check("and says there is no identity to record",
              "no identity to record" in rows[0].get("unmet_prerequisite", ""),
              str(rows[0]))

        r = subprocess.run(
            [sys.executable, str(EXPLICIT), "--image-ref", "  ",
             "--label", "x", "--docker", str(fake_docker(td))],
            capture_output=True, text=True, cwd=str(REPO),
            env={**os.environ, "FAKE_MODE": "ok"})
        check("an empty --image-ref REFUSES with exit 1", r.returncode == 1, r.stdout)


def test_the_record_feeds_the_unchanged_sibling() -> None:
    """R2's contract clause, asserted rather than hoped for."""
    scenario("contract: the sibling reads this module's record")
    with tempfile.TemporaryDirectory() as d:
        td = Path(d)
        code, _, rows = explicit("ok", td)
        check("precondition: a record exists", code == 0 and len(rows) == 1)
        r = subprocess.run(
            [sys.executable, str(SIBLING), "--verify-executed",
             "item8-disposable", "--against", str(td / "explicit.jsonl"),
             "--service", "item8-b1-memu-core", "--docker", str(fake_docker(td))],
            capture_output=True, text=True, cwd=str(REPO),
            env={**os.environ, "FAKE_MODE": "ok"})
        check("the UNCHANGED sibling accepts this module's record",
              r.returncode == 0, r.stdout)
        check("and binds it as MATCH", "binding  : MATCH" in r.stdout, r.stdout)


def run_all() -> None:
    test_guard_known_positive()
    test_guard_known_negative()
    test_derivation_all_six()
    test_b3_denies_exactly_one_instruction()
    test_b2_injects_one_first_attempt_failure()
    test_derivation_refuses_rather_than_silently_doing_nothing()
    test_explicit_collector()
    test_the_record_feeds_the_unchanged_sibling()
    print(f"  inspected: {EXPECTED_SCENARIOS} Item-8 instrument scenario(s) "
          f"across 3 instruments and 6 derivations")
    check(f"all {EXPECTED_SCENARIOS} scenarios ran",
          len(executed) == EXPECTED_SCENARIOS, f"{len(executed)}: {executed}")


if __name__ == "__main__":
    print("=" * 60)
    run_all()
    print()
    print("=" * 60)
    print(f"Item-8 Instrument Calibration: {passed} passed, {failed} failed")
    print(f"EXIT GATE: {'PASS' if failed == 0 else 'FAIL'}")
    sys.exit(1 if failed else 0)
