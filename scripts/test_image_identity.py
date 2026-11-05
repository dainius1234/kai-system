#!/usr/bin/env python3
"""Calibration for the image-identity collector.

The collector exists because D247 §6 item 10 demanded *"tree, image and
run id"* and the tree carried two of the three for the whole of Stage 1
(D277). So the properties under test are the ways this collector could
recreate that gap in a new form — by writing an identity that is not one.

Every scenario drives the SHIPPED entry point as a subprocess with a
fake `docker` injected through `--docker`. That is deliberate on three
counts:

  * rule 17 — the shipped CLI must be exercised, not its internals;
  * there is no Docker daemon on this host, and a calibration that can
    only run where the subject runs is a calibration that never runs;
  * the failure modes worth testing are exactly the ones a real daemon
    will not produce on demand — inspect exiting 0 with no payload,
    an object with no Id, a service resolving to two images.

The known-negative half is the point. A collector that records an
identity when one exists is uninteresting; the question is whether it
refuses when one does not, and whether the refusal is legible.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
COLLECTOR = REPO / "scripts" / "security" / "collect_image_identity.py"

passed = 0
failed = 0
EXPECTED_SCENARIOS = 13
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
"""A docker that does exactly what the scenario says, and nothing else."""
import json, os, sys

argv = sys.argv[1:]
mode = os.environ["FAKE_MODE"]

if argv[:1] == ["compose"]:
    # ... config --images <service>
    if mode == "compose_fails":
        sys.stderr.write("no configuration file provided\\n")
        sys.exit(14)
    if mode == "compose_silent":
        sys.exit(0)
    if mode == "compose_two":
        print("proj-a")
        print("proj-b")
        sys.exit(0)
    print("kai-system-memu-graph")
    sys.exit(0)

if argv[:2] == ["container", "inspect"]:
    if mode == "exec_container_gone":
        sys.stderr.write("Error: No such container: stage1-replay-run\\n")
        sys.exit(1)
    if mode == "exec_silent":
        sys.exit(0)
    if mode == "exec_no_image":
        print(json.dumps({"Name": "/stage1-replay-run"}))
        sys.exit(0)
    if mode == "exec_mismatch":
        print(json.dumps({"Image": "sha256:" + "c" * 64}))
        sys.exit(0)
    print(json.dumps({"Image": "sha256:" + "a" * 64}))
    sys.exit(0)

if argv[:2] == ["image", "inspect"]:
    if mode == "inspect_fails":
        sys.stderr.write("Error: No such image: kai-system-memu-graph\\n")
        sys.exit(1)
    if mode == "inspect_silent":
        sys.exit(0)
    if mode == "inspect_no_id":
        print(json.dumps({"Os": "linux", "Architecture": "amd64"}))
        sys.exit(0)
    payload = {"Id": "sha256:" + "a" * 64,
               "Os": "linux", "Architecture": "amd64"}
    if mode == "repo_digest_value":
        payload["RepoDigests"] = ["ghcr.io/x/y@sha256:" + "b" * 64]
    elif mode == "repo_digest_absent":
        pass                      # the key is simply not there
    else:
        payload["RepoDigests"] = []   # present and empty: never pushed
    print(json.dumps(payload))
    sys.exit(0)

sys.stderr.write("fake docker: unexpected argv %r\\n" % (argv,))
sys.exit(99)
'''


def run(mode: str, *, services=("memu-graph",), compose="docker-compose.full.yml",
        out: str | None = None) -> tuple[int, str, list[dict]]:
    with tempfile.TemporaryDirectory() as td:
        fake = Path(td) / "docker"
        fake.write_text(FAKE_DOCKER)
        fake.chmod(0o755)
        argv = [sys.executable, str(COLLECTOR), "--compose-file", compose,
                "--docker", str(fake), "--run-id", "999"]
        for s in services:
            argv += ["--service", s]
        outfile = Path(td) / "rows.jsonl" if out is None else Path(out)
        argv += ["--out", str(outfile)]
        env = {**os.environ, "FAKE_MODE": mode}
        p = subprocess.run(argv, capture_output=True, text=True,
                           cwd=str(REPO), env=env)
        rows = []
        if outfile.exists():
            rows = [json.loads(l) for l in
                    outfile.read_text().splitlines() if l.strip()]
        return p.returncode, p.stdout + p.stderr, rows


def test_known_positive() -> None:
    """An image that exists, built in-job and never pushed."""
    scenario("known-positive: identity recorded")
    code, out, rows = run("ok")
    check("a resolvable, inspectable image EXITS 0", code == 0, out)
    check("one row is written", len(rows) == 1, str(rows))
    r = rows[0]
    check("state is RECORDED", r["identity_state"] == "RECORDED", str(r))
    check("the id is carried", r["docker_image_id"] == "sha256:" + "a" * 64, str(r))
    check("labelled DOCKER_LOCAL_IMAGE_ID",
          r["identity_type"] == "DOCKER_LOCAL_IMAGE_ID", str(r))
    check("the ref came from compose, not from a guess",
          r["image_ref"] == "kai-system-memu-graph", str(r))
    check("platform is recorded", r["platform"] == "linux/amd64", str(r))
    check("the run id is bound", r["run_id"] == "999", str(r))
    check("the tree sha is bound", bool(r["tree_sha"]), str(r))
    check("the denominator is stated",
          "inspected: 1 service(s), 1 recorded, 0 UNRECORDED" in out, out)


def test_the_field_is_never_called_a_digest() -> None:
    """The naming IS the finding. image_digest would be read as OCI."""
    scenario("naming: id, never digest")
    code, out, rows = run("ok")
    r = rows[0]
    check("there is no `image_digest` field", "image_digest" not in r, str(r))
    check("the field is `docker_image_id`", "docker_image_id" in r, str(r))
    check("the record states the retrievability bound",
          "does NOT make that" in out and "retrievable" in out, out)


def test_absent_null_value_stay_distinct() -> None:
    """Rule 20. Never-pushed and pushed-reporting-nothing are different."""
    scenario("ABSENT / NULL / VALUE kept apart")
    _, _, rows = run("repo_digest_absent")
    check("a MISSING RepoDigests key is ABSENT",
          rows[0]["repo_digest_state"] == "ABSENT", str(rows[0]))
    _, _, rows = run("ok")
    check("an EMPTY RepoDigests list is NULL",
          rows[0]["repo_digest_state"] == "NULL", str(rows[0]))
    check("and carries no digest", rows[0]["repo_digest"] is None, str(rows[0]))
    _, _, rows = run("repo_digest_value")
    check("a populated RepoDigests is VALUE",
          rows[0]["repo_digest_state"] == "VALUE", str(rows[0]))
    check("and carries the digest",
          rows[0]["repo_digest"].startswith("ghcr.io/x/y@sha256:"), str(rows[0]))


def test_inspect_failure_is_unrecorded() -> None:
    """Known-negative 1: the daemon says no."""
    scenario("known-negative: inspect fails")
    code, out, rows = run("inspect_fails")
    check("a failed inspect EXITS 3", code == 3, out)
    check("state is UNRECORDED", rows[0]["identity_state"] == "UNRECORDED", str(rows[0]))
    check("the id is null, NOT an empty string",
          rows[0]["docker_image_id"] is None, str(rows[0]))
    check("the unmet prerequisite names inspect",
          "image inspect" in rows[0].get("unmet_prerequisite", "<ABSENT>"), str(rows[0]))
    check("and the report says item 10 is not met",
          "item 10 is not met" in out, out)


def test_exit_zero_with_no_payload_is_unrecorded() -> None:
    """Known-negative 2: the /dev/null-with-manners case (R10)."""
    scenario("known-negative: exit 0, no payload")
    code, out, rows = run("inspect_silent")
    check("exit 0 printing nothing EXITS 3", code == 3, out)
    check("an exit status is not a payload",
          "not a payload" in rows[0].get("unmet_prerequisite", "<ABSENT>"), str(rows[0]))
    check("nothing is invented for the id",
          rows[0]["docker_image_id"] is None, str(rows[0]))


def test_a_payload_with_no_id_is_unrecorded() -> None:
    """Known-negative 3: success-shaped, identity-free."""
    scenario("known-negative: payload without Id")
    code, out, rows = run("inspect_no_id")
    check("an object with no Id EXITS 3", code == 3, out)
    check("state is UNRECORDED", rows[0]["identity_state"] == "UNRECORDED", str(rows[0]))
    check("and says there is no identity to record",
          "no identity to record" in rows[0].get("unmet_prerequisite", "<ABSENT>"), str(rows[0]))


def test_compose_resolution_failures() -> None:
    """R11: no subject, no observation — at the resolution boundary too."""
    scenario("known-negative: compose cannot resolve")
    code, _, rows = run("compose_fails")
    check("a failing compose resolution EXITS 3", code == 3)
    check("and names compose, not inspect",
          "compose config --images" in rows[0].get("unmet_prerequisite", "<ABSENT>"), str(rows[0]))
    code, _, rows = run("compose_silent")
    check("compose exiting 0 naming nothing EXITS 3", code == 3)
    check("and refuses rather than inspecting a guessed ref",
          "no image" in rows[0].get("unmet_prerequisite", "<ABSENT>"), str(rows[0]))
    code, _, rows = run("compose_two")
    check("a service resolving to TWO images EXITS 3", code == 3)
    check("and does not silently take the first",
          rows[0]["docker_image_id"] is None
          and "not decidable" in rows[0].get("unmet_prerequisite", "<ABSENT>"), str(rows[0]))


def test_the_gap_survives_in_the_artifact() -> None:
    """An UNRECORDED row must be WRITTEN, not merely reported and lost."""
    scenario("the gap is written, not just printed")
    code, out, rows = run("inspect_fails")
    check("the record file still exists on failure", len(rows) == 1, str(rows))
    check("and the row itself says UNRECORDED",
          rows[0]["identity_state"] == "UNRECORDED", str(rows[0]))
    check("the denominator counts it",
          "1 service(s), 0 recorded, 1 UNRECORDED" in out, out)


def test_refusals_before_collecting() -> None:
    """An empty population reports success over nothing."""
    scenario("refusals: empty population, absent compose file")
    code, out, _ = run("ok", services=())
    check("no --service REFUSES with exit 1", code == 1, out)
    check("and says why an empty population is the failure",
          "success over nothing" in out, out)
    code, out, _ = run("ok", compose="does-not-exist.yml")
    check("an absent compose file REFUSES with exit 1", code == 1, out)
    check("and says a guessed reference is not an identity",
          "not an identity" in out, out)


def run_verify(mode: str, collected_rows: list[dict] | None,
               *, container="stage1-replay-run", service="memu-graph",
               against=True) -> tuple[int, str, list[dict]]:
    """Drive --verify-executed against a collected record we control."""
    with tempfile.TemporaryDirectory() as td:
        fake = Path(td) / "docker"
        fake.write_text(FAKE_DOCKER)
        fake.chmod(0o755)
        rec = Path(td) / "collected.jsonl"
        if collected_rows is not None:
            rec.write_text("".join(json.dumps(r) + "\n" for r in collected_rows))
        outfile = Path(td) / "executed.jsonl"
        argv = [sys.executable, str(COLLECTOR), "--docker", str(fake),
                "--verify-executed", container, "--out", str(outfile)]
        if against:
            argv += ["--against", str(rec)]
        if service:
            argv += ["--service", service]
        env = {**os.environ, "FAKE_MODE": mode}
        p = subprocess.run(argv, capture_output=True, text=True,
                           cwd=str(REPO), env=env)
        rows = []
        if outfile.exists():
            rows = [json.loads(l) for l in
                    outfile.read_text().splitlines() if l.strip()]
        return p.returncode, p.stdout + p.stderr, rows


GOOD = [{"identity_type": "DOCKER_LOCAL_IMAGE_ID", "service": "memu-graph",
         "identity_state": "RECORDED",
         "docker_image_id": "sha256:" + "a" * 64,
         "run_id": "999", "tree_sha": "deadbeef"}]


def test_executed_match() -> None:
    """Known-positive for the SECOND measurement."""
    scenario("executed: MATCH")
    code, out, rows = run_verify("ok", GOOD)
    check("a container running the recorded image EXITS 0", code == 0, out)
    check("binding is MATCH", rows[0]["execution_binding"] == "MATCH", str(rows[0]))
    check("both ids are carried",
          rows[0]["collected_image_id"] == rows[0]["executed_image_id"], str(rows[0]))
    check("and it says this is now a measurement, not an inference",
          "not an" in out and "inference" in out, out)


def test_executed_mismatch_is_never_absorbed() -> None:
    """The case the whole repair exists for: a DIFFERENT image ran."""
    scenario("executed: MISMATCH")
    code, out, rows = run_verify("exec_mismatch", GOOD)
    check("a different executed image EXITS 3", code == 3, out)
    check("binding is MISMATCH",
          rows[0]["execution_binding"] == "MISMATCH", str(rows[0]))
    check("the two ids differ in the record",
          rows[0]["collected_image_id"] != rows[0]["executed_image_id"], str(rows[0]))
    check("and it says the claims are bound to the wrong image",
          "wrong image" in out, out)


def test_executed_unrecorded_paths() -> None:
    """R11 at every boundary of the second measurement."""
    scenario("executed: UNRECORDED boundaries")
    for mode, needle in [("exec_container_gone", "container inspect"),
                         ("exec_silent", "not a payload"),
                         ("exec_no_image", "no Image")]:
        code, out, rows = run_verify(mode, GOOD)
        check(f"{mode} EXITS 3", code == 3, out)
        check(f"{mode} is UNRECORDED",
              rows[0]["execution_binding"] == "UNRECORDED", str(rows[0]))
        check(f"{mode} names its unmet prerequisite",
              needle in rows[0].get("unmet_prerequisite", "<ABSENT>"), str(rows[0]))
    check("UNKNOWN is not a match", "UNKNOWN is not a match" in out, out)
    # a collected row that was itself UNRECORDED cannot be bound afterwards
    bad = [{**GOOD[0], "identity_state": "UNRECORDED", "docker_image_id": None}]
    code, out, rows = run_verify("ok", bad)
    check("an UNRECORDED collected row cannot become MATCH", code == 3, out)
    check("and says an unbound run cannot be bound retroactively",
          "retroactively" in rows[0].get("unmet_prerequisite", ""), str(rows[0]))
    # no collected file at all
    code, out, rows = run_verify("ok", None)
    check("a missing collected record EXITS 3", code == 3, out)
    check("and refuses rather than trusting the container alone",
          "no recorded" in rows[0].get("unmet_prerequisite", ""), str(rows[0]))


def test_verify_refuses_without_a_comparator() -> None:
    """Reading the container and calling it agreement is self-verification."""
    scenario("executed: refuses without --against")
    code, out, _ = run_verify("ok", GOOD, against=False)
    check("--verify-executed without --against REFUSES (exit 1)", code == 1, out)
    check("and says why comparing against nothing is not agreement",
          "call it agreement" in out, out)


def run_all() -> None:
    test_known_positive()
    test_the_field_is_never_called_a_digest()
    test_absent_null_value_stay_distinct()
    test_inspect_failure_is_unrecorded()
    test_exit_zero_with_no_payload_is_unrecorded()
    test_a_payload_with_no_id_is_unrecorded()
    test_compose_resolution_failures()
    test_the_gap_survives_in_the_artifact()
    test_refusals_before_collecting()
    test_executed_match()
    test_executed_mismatch_is_never_absorbed()
    test_executed_unrecorded_paths()
    test_verify_refuses_without_a_comparator()
    print(f"  inspected: {EXPECTED_SCENARIOS} image-identity scenario(s) "
          f"across 1 collector, both measurements")
    check(f"all {EXPECTED_SCENARIOS} scenarios ran",
          len(executed) == EXPECTED_SCENARIOS, f"{len(executed)}: {executed}")


if __name__ == "__main__":
    print("=" * 60)
    run_all()
    print()
    print("=" * 60)
    print(f"Image Identity Calibration: {passed} passed, {failed} failed")
    print(f"EXIT GATE: {'PASS' if failed == 0 else 'FAIL'}")
    sys.exit(1 if failed else 0)
