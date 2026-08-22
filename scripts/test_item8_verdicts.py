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

import base64
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
RUNNER = REPO / "scripts" / "security" / "run_item8_experiment.sh"
SUMMARISE = REPO / "scripts" / "security" / "summarise_item8.py"
AUTHORITY = REPO / "scripts" / "security" / "check_item8_authority.py"
PARSER = REPO / "scripts" / "security" / "parse_buildkit_events.py"


def _mod(name):
    import importlib.util
    s = importlib.util.spec_from_file_location(
        name, REPO / "scripts" / "security" / f"{name}.py")
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


PARSER_MOD = _mod("parse_buildkit_events")
DERIVER = _mod("derive_item8_dockerfile")

passed = 0
failed = 0
EXPECTED_SCENARIOS = 52
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

AND ON THE SAME FILE DESCRIPTOR AS THE REAL ONE. buildx writes its
progress printer -- rawjson included -- to STDERR. This fake wrote it to
stdout, so the calibration modelled a transport the shipped command path
does not have: the runner captured stdout, every fixture passed, and the
real six builds would have produced an empty stream and reported six
UNMEASURED after the denominator was spent. Measured against
`docker buildx build --progress=rawjson >/dev/null`, not reasoned about.
(D294)
"""
import base64, json, os, sys
argv = sys.argv[1:]
mode = os.environ["FAKE_MODE"]
# STDERR, like buildx. `events_on_stdout` is the known-negative for the
# opposite mistake: the runner captures BOTH descriptors, so it must
# measure correctly whichever one carries the events.
OUT = sys.stdout if mode == "events_on_stdout" else sys.stderr
# Branch from the -f BASENAME suffix ONLY. Deriving it from the whole
# argv meant a random temp-dir name containing "b3" silently switched
# the scenario -- the same "matched incidental text" defect this whole
# repair is about, reproduced inside the fixture that tests for it.
_df = argv[argv.index("-f") + 1] if "-f" in argv else ""
branch = _df.rsplit(".", 1)[-1].upper() if _df else "B1"
if branch not in ("B1", "B2", "B3"):
    branch = "B1"
# A DISTINCT VERTEX DIGEST PER SUBJECT, as a real daemon gives -- the
# LLB op differs per subject, and one constant digest across six builds
# made the fake weaker than the thing it models. Derived from the
# Dockerfile this build was actually handed, so a swapped -f swaps the
# digest too, which is the property under test.
import hashlib as _h
DIG = "sha256:" + _h.sha256(
    (open(_df).read() if _df else "none").encode()).hexdigest()

def ev(o, stream=None): print(json.dumps(o), file=stream or OUT)
def log(s, stream=None):
    ev({"logs": [{"vertex": DIG, "stream": 1,
                  "data": base64.b64encode(s.encode()).decode()}]}, stream)
def diag(s): print(s, file=sys.stderr)

if argv and argv[0] == "build":
    # THE VERTEX NAME COMES FROM THE SUBJECT'S OWN DERIVED DOCKERFILE,
    # exactly as BuildKit's would. A generic name here would let every
    # fixture pass while the shipped claim engine binds on a specific
    # one -- the fake being easier than the real thing, which is D294's
    # defect in a new place. (D298)
    import re as _re
    _txt = open(_df).read() if _df else ""
    _m = _re.search(r"^RUN (?:--\S+ )*for attempt in 1 2 3 4 5; do \\\s*$",
                    _txt, _re.M)
    if _m:
        _s = _m.start(); _i = _s
        for _l in _txt[_s:].splitlines(keepends=True):
            _i += len(_l)
            if not _l.rstrip("\n").endswith("\\"):
                break
        name = "[3/9] " + " ".join(
            _txt[_s:_i].replace("\\\n", " ").split())
    else:
        name = "[3/9] RUN for attempt in 1 2 3 4 5; do :; done"
    cached = (mode == "cached_target")
    ev({"vertexes": [{"digest": DIG, "name": name, "cached": cached,
                      "started": None if mode == "never_started" else "t0"}]})
    if mode == "no_target_vertex":
        ev({"vertexes": [{"digest": "sha256:" + "2" * 64,
                          "name": "[1/9] FROM python:3.11-slim",
                          "started": "t0", "completed": "t1"}]})
    if mode == "unparseable":
        print("{not json", file=OUT)
        sys.exit(1)
    n = 0
    if branch == "B3":
        n = 4 if mode == "b3_four_attempts" else 5
        for _ in range(n):
            log("model download attempt /5 failed; retrying in 10s\n")
        log("REFUSING TO BUILD: could not fetch the model in 5 attempts.\n")
        # The vertex's OWN error. `b3_no_vertex_error` is the case R2
        # cares about: the build failed and our text appeared, but the
        # target step carries no error, so the failure is somewhere else.
        verr = ("" if mode == "b3_no_vertex_error"
                else "process did not complete successfully")
        ev({"vertexes": [{"digest": DIG, "name": name, "completed": "t1",
                          "error": verr}]})
        # A real failed build also prints a CLI diagnostic on stderr,
        # which is NOT a BuildKit event. The parser must keep it and
        # report it, never treat it as a truncated event (R10).
        diag('ERROR: failed to solve: process "/bin/sh -c for attempt in '
             '1 2 3 4 5" did not complete successfully: exit code: 1')
        if mode == "b3_leaves_iid":
            open(argv[argv.index("--iidfile") + 1], "w").write("sha256:" + "f" * 64)
        if mode == "b3_leaves_empty_iid":
            open(argv[argv.index("--iidfile") + 1], "w").close()
        sys.exit(1)
    if branch == "B2":
        # A CONSTANT marker. The derivation cannot interpolate the attempt
        # number -- `\$attempt` inside a double-quoted RUN string is the
        # literal text -- so a fake that manufactured `=1` was proving a
        # behaviour the shipped Dockerfile did not have. (D294)
        MARK = "ITEM8-B2-INJECTED-FIRST-ATTEMPT\n"
        if mode == "split_fd":
            # The ordered sequence, SPLIT ACROSS BOTH DESCRIPTORS. File
            # order is not chronology: concatenating them could
            # manufacture the order R2 requires, or destroy a real one.
            log(MARK, sys.stdout)
            log("model download attempt /5 failed; retrying in 10s\n",
                sys.stderr)
            log("BAKED ok\n", sys.stderr)
            ev({"vertexes": [{"digest": DIG, "name": name,
                              "completed": "t1"}]}, sys.stderr)
            if "--iidfile" in argv:
                open(argv[argv.index("--iidfile") + 1], "w").write(
                    "sha256:" + "f" * 64)
            sys.exit(0)
        if mode == "b2_out_of_order":
            # retry BEFORE the injection: the sequence R2 requires is
            # injection -> retry -> success, and order is the criterion.
            log("model download attempt /5 failed; retrying in 10s\n")
            log(MARK)
            log("BAKED ok\n")
        else:
            log(MARK)
            if mode == "b2_double_inject":
                log(MARK)
            if mode != "b2_no_genuine_retry":
                log("model download attempt /5 failed; retrying in 10s\n")
            if mode != "b2_never_bakes":
                log("BAKED ok\n")
    else:
        log("BAKED ok\n")
    ev({"vertexes": [{"digest": DIG, "name": name, "completed": "t1"}]})
    if mode != "iid_absent" and "--iidfile" in argv:
        open(argv[argv.index("--iidfile") + 1], "w").write(
            "sha256:" + ("f" * 64 if mode != "iid_mismatch" else "0" * 64))
    sys.exit(0)

if argv[:2] == ["buildx", "imagetools"]:
    # The mutable base tag, resolved once per branch. There is no branch
    # in this argv -- it inspects python:3.11-slim -- so the counter is
    # kept beside the fake, and `base_digest_moves` lets the tag move
    # PART WAY THROUGH the experiment, which is the real failure shape:
    # the first arms build on one base image and the rest on another.
    if mode == "base_digest_unresolved":
        sys.exit(1)
    st = os.path.join(os.path.dirname(os.path.abspath(sys.argv[0])),
                      ".base-calls")
    n = (int(open(st).read()) if os.path.exists(st) else 0) + 1
    open(st, "w").write(str(n))
    print("sha256:" + ("b" * 64 if mode == "base_digest_moves" and n > 3
                       else "a" * 64))
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


LAST_DIRS: tuple = ()

PINNED_FRONTEND = ("docker/dockerfile:1.9.0@sha256:"
                   "fe40cf4e92cd0c467be2cfc30657a680ae2398318afd50b0c80585784c604f28")


def git(*a: str) -> str:
    return subprocess.run(["git", *a], capture_output=True, text=True,
                          cwd=str(REPO)).stdout.strip()


def toolchain_text(**over: str) -> str:
    """A COMPLETE toolchain record, of the shape R2 requires.

    The previous fixture had two fields, and six branches qualified
    against it. That is the defect this file now calibrates: the record
    is validated before build 1, so the fixture has to be a real one and
    the reinjections below break it one field at a time.
    """
    rec = {"frontend": PINNED_FRONTEND,
           "docker_version": "Docker version 27.0.3, build fake",
           "buildx_version": "github.com/docker/buildx v0.16.0 fake",
           "base_image_digest": "sha256:" + "a" * 64,
           "runner_os": "Linux 6.8.0 x86_64",
           "commit_sha": git("rev-parse", "HEAD"),
           "tree_sha": git("rev-parse", "HEAD^{tree}"),
           "run_id": "555"}
    rec.update(over)
    return "".join(f"{k}={v}\n" for k, v in rec.items())


# THE CANONICAL ARTEFACT the default fixtures reconcile against. The
# summariser now REQUIRES it, and requires each row's tree, run and base
# image to agree with it -- so a row helper that invents those fields
# independently would be testing a world the runner cannot produce.
TC_TEXT = toolchain_text()
TC_SHA = hashlib.sha256(TC_TEXT.encode()).hexdigest()
TC_TREE = git("rev-parse", "HEAD^{tree}")
TC_COMMIT = git("rev-parse", "HEAD")
TC_RUN = "555"
TC_BASE = "sha256:" + "a" * 64


FLAGLESS_DOCKER = r'''#!/usr/bin/env python3
"""Well-formed rawjson, but RUN flags never reach the vertex name."""
import base64, json, sys
argv = sys.argv[1:]
D = "sha256:" + "1" * 64
def ev(o): print(json.dumps(o), file=sys.stderr)
if argv and argv[0] == "build":
    df = argv[argv.index("-f") + 1]
    body = open(df).read().split("\nRUN ", 1)[1].replace("\\\n", " ")
    # the flag is STRIPPED, which is the property under test
    import re
    body = re.sub(r"^(?:--\S+ )*", "", " ".join(body.split()))
    name = "[2/2] RUN " + body
    ev({"vertexes": [{"digest": D, "name": name, "started": "t0",
                      "cached": "--no-cache" not in argv}]})
    for _ in range(3):
        ev({"logs": [{"vertex": D, "stream": 1,
                      "data": base64.b64encode(
                          b"PREFLIGHT-RUNTIME-LINE x\n").decode()}]})
    fail = "exit 7" in open(df).read()
    ev({"vertexes": [{"digest": D, "name": name, "completed": "t1",
                      "error": "process did not complete successfully"
                               if fail else ""}]})
    sys.exit(7 if fail else 0)
sys.exit(0)
'''


def run_runner(mode: str, td: Path,
               toolchain: str | None = None) -> tuple[int, str, list[dict]]:
    """Drive the shipped runner with a fake docker and pre-derived files."""
    fake = td / "docker"
    fake.write_text(FAKE_DOCKER)
    fake.chmod(0o755)
    derived = td / "derived"
    ident = td / "ident"
    derived.mkdir(exist_ok=True)
    ident.mkdir(exist_ok=True)
    # REAL derived Dockerfiles: the claim engine re-derives them from the
    # shipped sources and requires byte equality, so a stub would fail
    # for a reason unrelated to the scenario under test.
    for image in ("memu-core", "memu-graph"):
        for branch in ("B1", "B2", "B3"):
            text, _n, err = DERIVER.derive(
                (REPO / image / "Dockerfile").read_text(), branch)
            assert not err, err
            (derived / f"Dockerfile.{image}.{branch}").write_text(text)
    (derived / "binding-rule.json").write_text(json.dumps(
        {"flags_in_vertex_name": True,
         "full_instruction_in_vertex_name": True,
         "digest_stable_across_invocations": True,
         "netmode_changes_vertex_digest": True,
         "run_id": "555", "tree_sha": git("rev-parse", "HEAD^{tree}")}) + "\n")
    results = td / "results.jsonl"
    tool = td / "toolchain.txt"
    tool.write_text(toolchain_text() if toolchain is None else toolchain)
    env = {**os.environ, "FAKE_MODE": mode, "DOCKER": str(fake),
           "ITEM8_DERIVED": str(derived), "ITEM8_IDENT": str(ident),
           "ITEM8_RESULTS": str(results), "GITHUB_RUN_ID": "555",
           "ITEM8_TOOLCHAIN": str(tool)}
    p = subprocess.run(["bash", str(RUNNER)], capture_output=True, text=True,
                       cwd=str(REPO), env=env)
    rows = [json.loads(l) for l in results.read_text().splitlines()
            if l.strip()] if results.exists() else []
    # The runner's OWN artefact package. Scenarios that drive the runner
    # summarise against what it actually left behind, not against a
    # hand-written package -- otherwise the derivation would be checked
    # against evidence nothing produced.
    global LAST_DIRS
    LAST_DIRS = (derived, ident)
    return p.returncode, p.stdout + p.stderr, rows


DIGEST = "sha256:" + "1" * 64
IMAGE_ID = "sha256:" + "f" * 64


def _ev(o: dict) -> str:
    return json.dumps(o) + "\n"


def _lg(s: str, dg: str = DIGEST) -> str:
    return _ev({"logs": [{"vertex": dg, "stream": 1,
                          "data": base64.b64encode(s.encode()).decode()}]})


def write_evidence(td: Path, **over) -> tuple[Path, Path]:
    """A COMPLETE artefact package, as the runner would leave one.

    The summariser derives both axes from these files. Synthetic rows
    alone can no longer reach a closure claim — which is the point of
    D296, and the reason this helper had to exist at all: the previous
    six() fixture qualified while the package it described was empty.

    `over` names one artefact to corrupt, so every reinjection below is a
    single contradiction between a row and the evidence for it.
    """
    derived = td / "ev-derived"
    ident = td / "ev-identity"
    derived.mkdir(exist_ok=True)
    ident.mkdir(exist_ok=True)
    # THE VERTEX NAME IS NOW SUBJECT-SPECIFIC, because the claim engine
    # binds a capture to its subject by the derived target instruction.
    # A generic name would make every fixture pass for the wrong reason
    # -- which is the defect this repair is about. (D298)
    (derived / "binding-rule.json").write_text(json.dumps(
        over.get("binding_rule",
                 {"flags_in_vertex_name": True,
                  "full_instruction_in_vertex_name": True,
                  "digest_stable_across_invocations": True,
                  "netmode_changes_vertex_digest": True,
                  "run_id": TC_RUN, "tree_sha": TC_TREE})) + "\n")
    names = {}
    digests: dict = {}
    captures: dict = {}
    for _im in ("memu-core", "memu-graph"):
        for _br in ("B1", "B2", "B3"):
            text, _n, err = DERIVER.derive(
                (REPO / _im / "Dockerfile").read_text(), _br)
            assert not err, err
            (derived / f"Dockerfile.{_im}.{_br}").write_text(text)
            run = PARSER_MOD.find_target_run(text)
            names[(_im, _br)] = "[3/9] " + PARSER_MOD.normalise_command(run)
            _dsrc = over.get(f"samedigest:{_im}.{_br}", (_im, _br))
            digests[(_im, _br)] = "sha256:" + hashlib.sha256(
                f"{_dsrc[0]}/{_dsrc[1]}".encode()).hexdigest()
    for image in ("memu-core", "memu-graph"):
        for branch in ("B1", "B2", "B3"):
            label = f"item8-{branch.lower()}-{image}"
            key = f"{image}.{branch}"
            # `swap:<key>` files ANOTHER subject's capture under this
            # subject's filename -- the exact corruption D298 closes.
            name = names[(image, branch)]
            dg = digests[(image, branch)]
            out = [_ev({"vertexes": [{"digest": dg, "name": name,
                                      "cached": over.get(f"cached:{key}",
                                                         False),
                                      "started": None if over.get(
                                          f"unstarted:{key}") else "t0"}]})]
            if branch == "B3":
                n = over.get(f"retries:{key}", 5)
                out += [_lg("model download attempt /5 failed; "
                            "retrying in 10s\n", dg)] * n
                out.append(_lg("REFUSING TO BUILD: could not fetch the "
                               "model in 5 attempts.\n", dg))
                out.append(_ev({"vertexes": [{
                    "digest": dg, "name": name, "completed": "t1",
                    "error": "" if over.get(f"noerr:{key}")
                             else "process did not complete successfully"}]}))
                abs_rec = {"service": over.get(f"svc:{key}", label),
                           "image_ref": over.get(
                               f"ref:{key}",
                               f"kai-item8:{branch.lower()}-{image}"),
                           "image": image, "branch": branch,
                           "run_id": over.get(f"run:{key}", TC_RUN),
                           "tree_sha": over.get(f"tree:{key}", TC_TREE),
                           "commit_sha": over.get(f"commit:{key}", TC_COMMIT),
                           "pre_build_state": "clean",
                           "post_build_tag": "absent",
                           "post_build_iidfile": "absent"}
                abs_rec.update(over.get(f"absence:{key}", {}))
                (ident / f"{label}.absence.json").write_text(
                    json.dumps(abs_rec) + "\n")
                if over.get(f"b3iid:{key}"):
                    (derived / f"{key}.iid").write_text(IMAGE_ID)
            elif over.get(f"outage:{key}"):
                # THE B1 GENUINE-OUTAGE SHAPE. memu-core/Dockerfile:92-107
                # -- the UNMUTATED control -- retries five times, prints
                # REFUSING TO BUILD and exits 1 when upstream is
                # unreachable. That is byte for byte B3's required
                # evidence, which is why there is no degraded binding.
                out += [_lg("model download attempt /5 failed; "
                            "retrying in 10s\n", dg)] * 5
                out.append(_lg("REFUSING TO BUILD: could not fetch the "
                               "model in 5 attempts.\n", dg))
                out.append(_ev({"vertexes": [{
                    "digest": dg, "name": name, "completed": "t1",
                    "error": "process did not complete successfully"}]}))
            else:
                if branch == "B2" and not over.get(f"noinject:{key}"):
                    if over.get(f"disorder:{key}"):
                        out.append(_lg("retrying in 10s\n", dg))
                        out.append(_lg("ITEM8-B2-INJECTED-FIRST-ATTEMPT\n", dg))
                        out.append(_lg("BAKED ok\n", dg))
                    else:
                        out.append(_lg("ITEM8-B2-INJECTED-FIRST-ATTEMPT\n", dg))
                        out.append(_lg("model download attempt /5 failed; "
                                       "retrying in 10s\n", dg))
                        out.append(_lg("BAKED ok\n", dg))
                else:
                    out.append(_lg("BAKED ok\n", dg))
                out.append(_ev({"vertexes": [{"digest": dg, "name": name,
                                              "completed": "t1"}]}))
                iid = over.get(f"iid:{key}", IMAGE_ID)
                if iid is not None:
                    (derived / f"{key}.iid").write_text(iid)
                ref = over.get(f"ref:{key}",
                               f"kai-item8:{branch.lower()}-{image}")
                stamp = {"service": over.get(f"svc:{key}", label),
                         "image_ref": ref,
                         "run_id": over.get(f"run:{key}", TC_RUN),
                         "tree_sha": over.get(f"tree:{key}", TC_TREE),
                         "commit_sha": over.get(f"commit:{key}", TC_COMMIT)}
                (ident / f"{label}.offline.json").write_text(json.dumps({
                    **stamp, "image": image, "branch": branch,
                    "exit_status": over.get(f"offline:{key}", 0)}) + "\n")
                idrec = json.dumps({
                    **stamp,
                    "identity_state": over.get(f"idstate:{key}", "RECORDED"),
                    "docker_image_id": IMAGE_ID}) + "\n"
                if over.get(f"dupe:{key}"):
                    idrec += idrec.replace("RECORDED", "UNRECORDED")
                (ident / f"{label}.jsonl").write_text(idrec)
                (ident / f"{label}.executed.jsonl").write_text(json.dumps({
                    **stamp,
                    "execution_binding": over.get(f"binding:{key}", "MATCH"),
                    "collected_image_id": IMAGE_ID,
                    "executed_image_id": over.get(f"execid:{key}",
                                                  IMAGE_ID)}) + "\n")
            captures[(image, branch)] = "".join(out)
            (derived / f"{key}.events-stdout.jsonl").write_text("")

    # WRITTEN LAST, so `swap:` can file ANOTHER subject's WHOLE capture
    # under this subject's name -- vertex name and runtime output
    # together, which is what an accidental copy actually looks like.
    # `swap` is also usable RECIPROCALLY (B1<->B3), which distinctness
    # cannot see and the invocation binding must.
    for image in ("memu-core", "memu-graph"):
        for branch in ("B1", "B2", "B3"):
            key = f"{image}.{branch}"
            src = over.get(f"swap:{key}", (image, branch))
            (derived / f"{key}.events-stderr.jsonl").write_text(captures[src])

    # THE INVOCATION RECORDS, written from what is actually on disk --
    # so an unrepaired swap is visible as a hash mismatch, exactly as it
    # would be in a real package. `invswap:` moves the RECORD too, which
    # is the harder case: then only the re-derived Dockerfile SHA can
    # tell that the pair is reversed.
    for image in ("memu-core", "memu-graph"):
        for branch in ("B1", "B2", "B3"):
            key = f"{image}.{branch}"
            label = f"item8-{branch.lower()}-{image}"
            src = over.get(f"invswap:{key}", (image, branch))
            skey = f"{src[0]}.{src[1]}"
            df = derived / f"Dockerfile.{skey}"
            outf = derived / f"{skey}.events-stdout.jsonl"
            sh = lambda f: (hashlib.sha256(f.read_bytes()).hexdigest()
                            if f.is_file() else "ABSENT")
            # THE RECORD HASHES THE CAPTURE AS PRODUCED, not as it sits
            # on disk afterwards. The runner writes it immediately after
            # its own build; a swap performed later is exactly what the
            # byte binding is for, and generating the record from
            # post-swap disk state would model a threat nobody has.
            err_sha = hashlib.sha256(captures[src].encode()).hexdigest()
            rec = {"service": f"item8-{src[1].lower()}-{src[0]}",
                   "image": src[0], "branch": src[1],
                   "image_ref": f"kai-item8:{src[1].lower()}-{src[0]}",
                   "run_id": over.get(f"run:{key}", TC_RUN),
                   "tree_sha": over.get(f"tree:{key}", TC_TREE),
                   **({} if over.get(f"nocommit:{key}") else
                      {"commit_sha": over.get(f"commit:{key}", TC_COMMIT)}),
                   "derived_dockerfile_path": str(df),
                   "derived_dockerfile_sha256": over.get(
                       f"dfsha:{key}", sh(df)),
                   "invocation": {"subcommand": over.get(
                                      f"sub:{key}", "build"),
                                  "no_cache": over.get(
                                      f"nocache:{key}", src[1] == "B1"),
                                  "progress": over.get(
                                      f"progress:{key}", "rawjson"),
                                  "file": str(df),
                                  "tag": f"kai-item8:{src[1].lower()}-{src[0]}",
                                  "iidfile": over.get(
                                      f"iidpath:{key}",
                                      str(derived / f"{skey}.iid")),
                                  "context": over.get(f"ctx:{key}", ".")},
                   "events_stderr_sha256": over.get(f"errsha:{key}", err_sha),
                   "events_stdout_sha256": sh(outf),
                   "build_exit": 1 if src[1] == "B3" else 0}
            (ident / f"{label}.invocation.json").write_text(
                json.dumps(rec) + "\n")
    return derived, ident


def summarise(rows: list[dict], td: Path, toolchain: str | None = None,
              omit_toolchain: bool = False,
              dirs: tuple[Path, Path] | None = None,
              **over) -> tuple[int, str]:
    f = td / "s.jsonl"
    f.write_text("".join(json.dumps(r) + "\n" for r in rows))
    argv = [sys.executable, str(SUMMARISE), "--results", str(f)]
    if not omit_toolchain:
        if toolchain is None:
            art = td / "canonical-toolchain.txt"
            art.write_text(TC_TEXT)
            toolchain = str(art)
        argv += ["--toolchain", toolchain]
    if dirs is None:
        dirs = write_evidence(td, **over)
    argv += ["--derived-dir", str(dirs[0]), "--identity-dir", str(dirs[1])]
    p = subprocess.run(argv, capture_output=True, text=True, cwd=str(REPO))
    return p.returncode, p.stdout + p.stderr


def row(image="memu-core", branch="B1", a1="PASS", a2="BOUND", **extra):
    r = {"image": image, "branch": branch, "axis1_verdict": a1,
         "axis2_provenance": a2, "runtime_retries_observed": 1,
         "elapsed_seconds": 1, "note": "",
         "iidfile_corroboration": "n/a" if branch == "B3" else "CORROBORATED",
         "toolchain_sha256": TC_SHA, "tree_sha": TC_TREE,
         "run_id": TC_RUN, "base_image_digest": TC_BASE}
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
        _, sout = summarise(rows, td, dirs=LAST_DIRS)
        check("the summary keeps Axis 1 complete",
              "AXIS 1 COMPLETE, PROVENANCE INCOMPLETE" in sout, sout)
        check("and Axis 2 was DERIVED as the fault, not read from the row",
              "MISMATCH" in sout, sout)


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
                     "approved_tree=x\nauthorises=experiment\n")
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
        code, out = summarise(rows, td, dirs=LAST_DIRS)
        check("an absent iidfile blocks closure", code != 0, out)
        check("and the reason is named — from the PACKAGE, not the row",
              "no iidfile in the package" in out, out)


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
        check("and the artefacts' answer is given",
              "the artefacts give PASS" in out, out)


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
    good = (f"frozen_r2={frozen}\napproved_commit={head}\n"
            f"approved_tree={tree}\nauthorises=experiment\n")

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
            f"approved_tree={ptree}\nauthorises=experiment\n", td)
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


# ── D294 1: the transport. rawjson is on STDERR, and both are captured ──

def test_events_are_read_from_either_descriptor() -> None:
    """The fake used to write events to stdout. buildx does not."""
    scenario("transport: rawjson arrives on stderr, and both FDs are read")
    with tempfile.TemporaryDirectory() as d:
        td = Path(d)
        # DEFAULT mode now emits on stderr, like real buildx. If the
        # runner captured only stdout this would produce six UNMEASURED.
        _, out, rows = run_runner("ok", td)
        check("stderr-borne events are measured", len(rows) == 6, out)
        for r in rows:
            check(f"{r['image']} {r['branch']} was actually measured",
                  r["axis1_verdict"] == "PASS", str(r))
    with tempfile.TemporaryDirectory() as d:
        td = Path(d)
        # And the opposite mistake must not break it either: the runner
        # passes BOTH descriptors, so events on stdout measure the same.
        _, out, rows = run_runner("events_on_stdout", td)
        check("stdout-borne events are measured too", len(rows) == 6, out)
        for r in rows:
            check(f"{r['image']} {r['branch']} measured from stdout",
                  r["axis1_verdict"] == "PASS", str(r))


def test_parser_names_the_wrong_descriptor_possibility() -> None:
    """A capture that watched the wrong FD looks exactly like no build."""
    scenario("transport: an eventless capture refuses and names the FD")
    with tempfile.TemporaryDirectory() as d:
        td = Path(d)
        empty = td / "stdout.jsonl"
        empty.write_text("")
        p = subprocess.run(
            [sys.executable, str(PARSER), "--events", str(empty),
             "--target-substring", "for attempt in 1 2 3 4 5", "--json"],
            capture_output=True, text=True, cwd=str(REPO))
        check("a single empty descriptor REFUSES", p.returncode == 1, p.stdout)
        check("and names the wrong-descriptor possibility",
              "DIFFERENT FILE DESCRIPTOR" in p.stdout
              or "descriptor" in p.stdout, p.stdout)
        # CLI diagnostics are kept, not mistaken for truncated events.
        both = td / "stderr.jsonl"
        both.write_text(
            'ERROR: failed to solve: process "/bin/sh -c x" did not '
            'complete successfully\n'
            + json.dumps({"vertexes": [{"digest": "sha256:" + "1" * 64,
                                        "name": "RUN for attempt in 1 2 3 4 5",
                                        "started": "t0"}]}) + "\n")
        p = subprocess.run(
            [sys.executable, str(PARSER), "--events", str(both),
             "--events", str(empty), "--target-substring",
             "for attempt in 1 2 3 4 5", "--json"],
            capture_output=True, text=True, cwd=str(REPO))
        check("a mixed stream parses", p.returncode == 0, p.stdout + p.stderr)
        facts = json.loads(p.stdout)
        check("and the CLI diagnostic is COUNTED, not discarded",
              facts["cli_diagnostic_lines"] == 1, p.stdout)


# ── D294 2: B2's marker is a constant, and order is the criterion ───────

def test_b2_marker_is_a_constant_and_order_matters() -> None:
    scenario("B2: one marker, and injection then retry then success")
    with tempfile.TemporaryDirectory() as d:
        td = Path(d)
        _, _, rows = run_runner("b2_double_inject", td)
        for r in [r for r in rows if r["branch"] == "B2"]:
            check(f"{r['image']} B2 two markers is NOT PASS",
                  r["axis1_verdict"] != "PASS", str(r))
            check(f"{r['image']} B2 counts them", r["injection_markers"] == 2,
                  str(r))
            check(f"{r['image']} B2 says exactly one is required",
                  "exactly one is required" in r.get("note", ""), str(r))
    with tempfile.TemporaryDirectory() as d:
        td = Path(d)
        _, _, rows = run_runner("b2_out_of_order", td)
        for r in [r for r in rows if r["branch"] == "B2"]:
            check(f"{r['image']} B2 retry BEFORE injection is NOT PASS",
                  r["axis1_verdict"] != "PASS", str(r))
            check(f"{r['image']} B2 names the ordering",
                  "in that order" in r.get("note", ""), str(r))
    with tempfile.TemporaryDirectory() as d:
        td = Path(d)
        _, _, rows = run_runner("b2_never_bakes", td)
        for r in [r for r in rows if r["branch"] == "B2"]:
            check(f"{r['image']} B2 with no later success is NOT PASS",
                  r["axis1_verdict"] != "PASS", str(r))


def test_b2_shim_prints_a_constant_in_a_real_shell() -> None:
    """The fake must not be better than the shipped command path.

    The derived Dockerfile is fed to /bin/sh here, not reasoned about.
    An earlier shim wrote `\\$attempt` intending expansion; the shell
    prints the literal, and only running it says so.
    """
    scenario("B2: the derived shim's marker, measured against /bin/sh")
    with tempfile.TemporaryDirectory() as d:
        td = Path(d)
        out = td / "Dockerfile.memu-graph.B2"
        p = subprocess.run(
            [sys.executable, str(REPO / "scripts" / "security" /
                                 "derive_item8_dockerfile.py"),
             "--image", "memu-graph", "--branch", "B2", "--out", str(out)],
            capture_output=True, text=True, cwd=str(REPO))
        check("the B2 derivation succeeds", p.returncode == 0,
              p.stdout + p.stderr)
        text = out.read_text()
        check("the marker is a CONSTANT",
              "ITEM8-B2-INJECTED-FIRST-ATTEMPT" in text, text[:400])
        check("and carries no interpolation at all",
              "$attempt" not in text.split("ITEM8-B2-INJECTED-FIRST-ATTEMPT")[0]
              .rsplit("\n", 1)[-1], text[:400])


# ── D294 3: B3's failure must arise from the target vertex ─────────────

def test_b3_requires_the_target_vertexs_own_error() -> None:
    scenario("B3: our text in a build that failed elsewhere is WRONG_FAILURE")
    with tempfile.TemporaryDirectory() as d:
        td = Path(d)
        _, _, rows = run_runner("b3_no_vertex_error", td)
        for r in [r for r in rows if r["branch"] == "B3"]:
            check(f"{r['image']} B3 with no vertex error is WRONG_FAILURE",
                  r["axis1_verdict"] == "WRONG_FAILURE", str(r))
            check(f"{r['image']} B3 says it is not attributable",
                  "not attributable" in r.get("note", ""), str(r))
            check(f"{r['image']} B3 records the empty error",
                  r["target_vertex_error"] == "", str(r))
    with tempfile.TemporaryDirectory() as d:
        td = Path(d)
        _, _, rows = run_runner("ok", td)
        for r in [r for r in rows if r["branch"] == "B3"]:
            check(f"{r['image']} B3 WITH a vertex error PASSES",
                  r["axis1_verdict"] == "PASS", str(r))
            check(f"{r['image']} B3 records that error",
                  r["target_vertex_error"] != "", str(r))


# ── D294 4: the toolchain record, validated before build 1 ─────────────

def toolchain_check(text: str | None, td: Path,
                    extra: list[str] | None = None) -> tuple[int, str]:
    f = td / "toolchain.txt"
    if text is not None:
        f.write_text(text)
    p = subprocess.run(
        [sys.executable, str(REPO / "scripts" / "security" /
                             "check_item8_toolchain.py"),
         "--toolchain", str(f), *(extra or [])],
        capture_output=True, text=True, cwd=str(REPO))
    return p.returncode, p.stdout + p.stderr


def test_toolchain_record_is_validated_not_merely_hashed() -> None:
    """A SHA-256 of an incomplete record is a perfect hash of bad evidence."""
    scenario("toolchain: completeness is checked before any build")
    with tempfile.TemporaryDirectory() as d:
        td = Path(d)
        # KNOWN-POSITIVE first, so the refusals below mean something.
        code, out = toolchain_check(toolchain_text(), td)
        check("a complete record PASSES", code == 0, out)
        check("and reports its own denominator",
              "8 required identity(s)" in out, out)
        # the original two-field fixture -- the defect, exactly as it was
        code, out = toolchain_check(
            "docker_version=fake\nbase_image_digest=sha256:abc\n", td)
        check("the old two-field fixture REFUSES", code == 1, out)
        check("and names every missing identity",
              "frontend" in out and "runner_os" in out, out)
        # `key=` -- present, empty. `set -uo pipefail` produces exactly this.
        code, out = toolchain_check(toolchain_text(buildx_version=""), td)
        check("an EMPTY value REFUSES", code == 1, out)
        check("and says absence wearing a key's clothes",
              "absence wearing a key" in out, out)
        code, out = toolchain_check(
            toolchain_text(base_image_digest="UNRESOLVED"), td)
        check("UNRESOLVED REFUSES", code == 1, out)
        code, out = toolchain_check(
            toolchain_text(frontend="docker/dockerfile:1"), td)
        check("a floating frontend REFUSES", code == 1, out)
        check("and says it is not the pinned value",
              "not the pinned value R2 froze" in out, out)
        code, out = toolchain_check(toolchain_text(tree_sha="0" * 40), td)
        check("a stale tree_sha REFUSES", code == 1, out)
        code, out = toolchain_check(toolchain_text(commit_sha="0" * 40), td)
        check("a stale commit_sha REFUSES", code == 1, out)
        code, out = toolchain_check(toolchain_text(), td,
                                    ["--expect-run-id", "999"])
        check("a record describing another run REFUSES", code == 1, out)
        (td / "toolchain.txt").unlink()
        code, out = toolchain_check(None, td)
        check("an absent record REFUSES", code == 1, out)
        check("and says no build may start", "NO BUILD MAY START" in out, out)


def test_runner_refuses_before_build_1_on_a_bad_toolchain() -> None:
    """Zero builds spent, not six. That is the whole point of the ordering."""
    scenario("toolchain: a bad record costs zero experimental builds")
    for label, tc in (("two fields", "docker_version=fake\n"),
                      ("an empty value", toolchain_text(runner_os="")),
                      ("UNRESOLVED", toolchain_text(
                          base_image_digest="UNRESOLVED"))):
        with tempfile.TemporaryDirectory() as d:
            td = Path(d)
            code, out, rows = run_runner("ok", td, toolchain=tc)
            check(f"{label}: the runner exits NON-ZERO", code != 0, out)
            check(f"{label}: it is an INSTRUMENT FAILURE",
                  "INSTRUMENT FAILURE" in out, out)
            check(f"{label}: ZERO result rows were written",
                  len(rows) == 0, str(rows))
            check(f"{label}: no build was attempted",
                  "::group::" not in out, out[:400])


def test_summariser_rehashes_the_toolchain_artefact() -> None:
    """Six rows agreeing with each other are six statements from one source."""
    scenario("toolchain: rows are matched against the artefact, not each other")
    with tempfile.TemporaryDirectory() as d:
        td = Path(d)
        art = td / "toolchain.txt"
        art.write_text(toolchain_text())
        import hashlib
        real = hashlib.sha256(art.read_bytes()).hexdigest()

        code, out = summarise(six(toolchain_sha256=real), td, str(art))
        check("rows bound to the artefact QUALIFY", code == 0, out)
        check("and the recomputation is reported", "TOOLCHAIN  recomputed" in out, out)

        # All six agree with each other and none with the artefact. This
        # is the case a row-to-row comparison cannot see.
        code, out = summarise(six(toolchain_sha256="d" * 64), td, str(art))
        check("six self-consistent rows still REFUSE", code == 4, out)
        check("and each is named", out.count("TOOLCHAIN: ") == 6, out)

        rows = six(toolchain_sha256=real)
        rows[3]["toolchain_sha256"] = "e" * 64
        code, out = summarise(rows, td, str(art))
        check("one divergent row REFUSES", code == 4, out)
        check("and only that one is named",
              out.count("TOOLCHAIN: ") == 1, out)
        check("naming the subject", "memu-graph/B1" in out, out)

        code, out = summarise(six(toolchain_sha256=real), td,
                              str(td / "not-there.txt"))
        check("an absent artefact REFUSES", code == 4, out)


# ── D295 1: the independent evidence may not be optional ───────────────

def test_toolchain_artefact_is_mandatory() -> None:
    """Optional independent evidence is not independent evidence."""
    scenario("toolchain: six perfect rows cannot qualify without the artefact")
    with tempfile.TemporaryDirectory() as d:
        td = Path(d)
        code, out = summarise(six(), td)
        check("with the artefact, the true six QUALIFY", code == 0, out)
        # The same six rows, through the same shipped entry point, with
        # the flag omitted. This USED to reach ALL SIX QUALIFY on the
        # producer's word alone.
        code, out = summarise(six(), td, omit_toolchain=True)
        check("without it the summariser REFUSES to run at all", code != 0,
              out)
        check("and argparse names the required flag",
              "--toolchain" in out, out)
        check("and it never reaches a qualification verdict",
              "ALL SIX QUALIFY" not in out, out)


# ── D295 2: each branch has ONE admissible provenance state ────────────

def test_branch_contract_is_enforced_per_branch() -> None:
    scenario("provenance: a BOUND B3 is a contradiction, not sound evidence")
    with tempfile.TemporaryDirectory() as d:
        td = Path(d)
        # B3's whole contract is that no image is produced. A B3 row
        # claiming a successful binding is describing something else --
        # and since D296 the claim never reaches the qualification test,
        # because the ARTEFACTS are what answer the question. The row is
        # contradicted instead, which is a stronger outcome: the
        # impossible state is now unreachable rather than merely refused.
        rows = six()
        rows[2]["axis2_provenance"] = "BOUND"
        rows[2]["iidfile_corroboration"] = "CORROBORATED"
        code, out = summarise(rows, td)
        check("a BOUND B3 does NOT qualify", code != 0, out)
        check("and the row is contradicted by the artefacts",
              "DISAGREEMENT: memu-core/B3 Axis 2" in out, out)
        check("which give the state the branch must carry",
              "artefacts give IMAGE_NOT_PRODUCED_BY_DESIGN" in out, out)
        # The mirror image, likewise contradicted rather than believed.
        rows = six()
        rows[0]["axis2_provenance"] = "IMAGE_NOT_PRODUCED_BY_DESIGN"
        code, out = summarise(rows, td)
        check("an IMAGE_NOT_PRODUCED_BY_DESIGN B1 does NOT qualify",
              code != 0, out)
        check("and it too is contradicted",
              "DISAGREEMENT: memu-core/B1 Axis 2" in out, out)
        check("the artefacts giving BOUND for a built branch",
              "artefacts give BOUND" in out, out)


# ── D295 3: tree and run reconciled against the artefact ───────────────

def test_row_identity_is_reconciled_with_the_artefact() -> None:
    """Hash equality binds a row to a FILE, not to a run."""
    scenario("toolchain: tree and run are compared, not just the digest")
    with tempfile.TemporaryDirectory() as d:
        td = Path(d)
        rows = six()
        rows[1]["tree_sha"] = "0" * 40
        code, out = summarise(rows, td)
        check("one wrong tree REFUSES", code == 4, out)
        check("and names that subject once",
              out.count("names tree") == 1, out)

        rows = six()
        rows[4]["run_id"] = "999"
        code, out = summarise(rows, td)
        check("one wrong run REFUSES", code == 4, out)
        check("and names the run it expected", "names run 999" in out, out)

        # ALL SIX wrong but mutually consistent -- the case row-to-row
        # comparison can never see.
        code, out = summarise(six(tree_sha="0" * 40, run_id="999"), td)
        check("six self-consistent wrong identities REFUSE", code == 4, out)
        check("and every one is named", out.count("names tree") == 6, out)

        rows = six()
        rows[0]["tree_sha"] = "0" * 40
        rows[3]["run_id"] = "888"
        code, out = summarise(rows, td)
        check("mixed identities REFUSE", code == 4, out)
        check("and both faults are named",
              out.count("names tree") == 1 and out.count("names run") == 1,
              out)

        # An artefact that cannot supply the identities is itself a
        # refusal: there is nothing to reconcile against.
        art = td / "thin.txt"
        art.write_text("docker_version=fake\n")
        code, out = summarise(six(), td, str(art))
        check("an artefact naming no tree/run REFUSES", code == 4, out)
        check("and says the rows cannot be reconciled",
              "cannot be reconciled" in out, out)


# ── D295 4: two event-bearing descriptors have no shared chronology ────

def test_split_descriptors_refuse_rather_than_invent_order() -> None:
    scenario("transport: a split capture must not manufacture an order")
    with tempfile.TemporaryDirectory() as d:
        td = Path(d)
        a = td / "a.jsonl"
        b = td / "b.jsonl"
        D = "sha256:" + "1" * 64
        import base64 as b64

        def lg(s):
            return json.dumps({"logs": [{"vertex": D, "stream": 1,
                                         "data": b64.b64encode(
                                             s.encode()).decode()}]})
        a.write_text(json.dumps({"vertexes": [
            {"digest": D, "name": "RUN for attempt in 1 2 3 4 5",
             "started": "t0"}]}) + "\n" + lg("ITEM8-B2-INJECTED-FIRST-ATTEMPT\n") + "\n")
        b.write_text(lg("retrying in 10s\n") + "\n" + lg("BAKED ok\n") + "\n")
        p = subprocess.run(
            [sys.executable, str(PARSER), "--events", str(a),
             "--events", str(b), "--target-substring",
             "for attempt in 1 2 3 4 5", "--json"],
            capture_output=True, text=True, cwd=str(REPO))
        check("two event-bearing descriptors REFUSE", p.returncode == 1,
              p.stdout)
        check("and say the chronology is unestablished",
              "chronology" in p.stdout, p.stdout)
        check("and no facts are emitted", "counts" not in p.stdout, p.stdout)

    # Through the shipped runner: B2's evidence split across the two
    # captures must not become a PASS.
    with tempfile.TemporaryDirectory() as d:
        td = Path(d)
        _, out, rows = run_runner("split_fd", td)
        b2 = [r for r in rows if r["branch"] == "B2"]
        check("split-FD B2 rows exist", len(b2) == 2, str(rows))
        for r in b2:
            check(f"{r['image']} B2 split across descriptors is NOT PASS",
                  r["axis1_verdict"] != "PASS", str(r))
            check(f"{r['image']} B2 says nothing was observed",
                  "could not be parsed" in r.get("note", ""), str(r))


# ── D295 proactive: the base tag is mutable ────────────────────────────

def test_base_image_digest_must_hold_across_the_six() -> None:
    """Six arms on two base images are not six arms of one experiment."""
    scenario("base image: a tag that moves mid-experiment blocks closure")
    with tempfile.TemporaryDirectory() as d:
        td = Path(d)
        _, out, rows = run_runner("ok", td)
        check("every branch records what the tag resolved to",
              all(r.get("base_image_digest") == TC_BASE for r in rows),
              str([r.get("base_image_digest") for r in rows]))
        code, sout = summarise(rows, td)
        check("a stable base image qualifies", code == 0, sout)
    with tempfile.TemporaryDirectory() as d:
        td = Path(d)
        _, out, rows = run_runner("base_digest_moves", td)
        digests = {r.get("base_image_digest") for r in rows}
        check("the movement is visible in the rows", len(digests) == 2,
              str(digests))
        code, sout = summarise(rows, td)
        check("and it blocks interpretation", code == 4, sout)
        check("naming it as a cross-arm confound",
              "not six arms of one experiment" in sout, sout)
        for r in rows:
            check(f"{r['image']} {r['branch']} Axis 1 is untouched by it",
                  r["axis1_verdict"] in ("PASS", "WRONG_FAILURE",
                                         "UNMEASURED"), str(r))
    with tempfile.TemporaryDirectory() as d:
        td = Path(d)
        _, out, rows = run_runner("base_digest_unresolved", td)
        check("an unresolvable tag is recorded as UNRESOLVED",
              all(r.get("base_image_digest") == "UNRESOLVED" for r in rows),
              str([r.get("base_image_digest") for r in rows]))
        code, sout = summarise(rows, td)
        check("and blocks closure rather than passing silently",
              code == 4, sout)


# ── D296: the claim engine derives both axes from the artefacts ────────

def test_axis1_is_derived_not_read() -> None:
    """The smoking gun: rows saying PASS over evidence that says otherwise."""
    scenario("claim engine: Axis 1 is derived from BuildKit's evidence")
    with tempfile.TemporaryDirectory() as d:
        td = Path(d)
        # THE EXACT FIXTURE GPT NAMED. Six rows saying PASS, and a B3
        # package holding ONE retry where the frozen design requires
        # five. Before D296 this was a valid ALL SIX QUALIFY case.
        code, out = summarise(six(), td, **{"retries:memu-core.B3": 1})
        check("B3 with one retry does NOT qualify", code != 0, out)
        check("and the count is derived, not read",
              "1 runtime retry line(s)" in out, out)
        check("naming the five the design requires", "not the 5" in out, out)
        check("and the row is contradicted",
              "DISAGREEMENT: memu-core/B3 Axis 1" in out, out)

        code, out = summarise(six(), td, **{"cached:memu-core.B1": True})
        check("a CACHED target does not qualify", code != 0, out)
        check("and says the genuine fetch did not run",
              "FROM CACHE" in out, out)

        code, out = summarise(six(), td, **{"unstarted:memu-graph.B1": True})
        check("a target that never executed does not qualify", code != 0, out)
        check("and says so", "did not execute" in out, out)

        code, out = summarise(six(), td, **{"noinject:memu-core.B2": True})
        check("B2 without its injection does not qualify", code != 0, out)
        check("and names the missing marker",
              "0 injection marker(s)" in out, out)

        code, out = summarise(six(), td, **{"disorder:memu-graph.B2": True})
        check("B2 out of order does not qualify", code != 0, out)
        check("and names the ordering", "in that order" in out, out)

        code, out = summarise(six(), td, **{"noerr:memu-core.B3": True})
        check("B3 without its own vertex error does not qualify",
              code != 0, out)
        check("and says it is not attributable",
              "not attributable" in out, out)

        code, out = summarise(six(), td, **{"offline:memu-core.B1": 1})
        check("a failed offline load does not qualify", code != 0, out)
        check("and reads the container's own exit status",
              "offline asset load exited 1" in out, out)


def test_axis2_is_derived_from_the_identity_artefacts() -> None:
    scenario("claim engine: Axis 2 is derived from the identity records")
    with tempfile.TemporaryDirectory() as d:
        td = Path(d)
        code, out = summarise(six(), td, **{"binding:memu-core.B1": "MISMATCH"})
        check("a MISMATCH binding does not qualify", code != 0, out)
        check("and a record whose word contradicts its own ids is named",
              "contradicts itself" in out, out)

        code, out = summarise(six(), td,
                              **{"iid:memu-graph.B2": "sha256:" + "0" * 64})
        check("an iidfile disagreeing with the collector does not qualify",
              code != 0, out)
        check("and both values are named", "the collector says" in out, out)

        code, out = summarise(six(), td, **{"idstate:memu-core.B1": "UNRECORDED"})
        check("an UNRECORDED identity does not qualify", code != 0, out)

        code, out = summarise(six(), td, **{"b3iid:memu-core.B3": True})
        check("a B3 with an iidfile does not qualify", code != 0, out)
        check("and says the no-image contract is unestablished",
              "no-image contract is not established" in out, out)

        code, out = summarise(six(), td, **{
            "absence:memu-graph.B3": {"pre_build_state": "clean",
                                      "post_build_tag": "present",
                                      "post_build_iidfile": "absent"}})
        check("a B3 whose absence record shows a surviving tag fails",
              code != 0, out)


def test_the_package_must_actually_be_present() -> None:
    """R11 at the claim boundary: no artefacts, no claim."""
    scenario("claim engine: an empty evidence package qualifies nothing")
    with tempfile.TemporaryDirectory() as d:
        td = Path(d)
        empty_d = td / "no-derived"
        empty_i = td / "no-identity"
        empty_d.mkdir()
        empty_i.mkdir()
        code, out = summarise(six(), td, dirs=(empty_d, empty_i))
        check("six perfect rows over an EMPTY package REFUSE", code != 0, out)
        check("and it says the raw evidence is not in the package",
              "not in the package" in out, out)
        check("every subject is named",
              out.count("the raw evidence for this branch") >= 6, out)


def test_closure_toolchain_contract_matches_pre_build() -> None:
    """One contract, both boundaries — not two that can drift."""
    scenario("toolchain: closure applies the same eight-field contract")
    with tempfile.TemporaryDirectory() as d:
        td = Path(d)
        # Exactly the artefact GPT described: enough to reconcile
        # hash/tree/run/base, missing the other five identities.
        thin = td / "thin.txt"
        thin.write_text(f"tree_sha={TC_TREE}\nrun_id={TC_RUN}\n"
                        f"base_image_digest={TC_BASE}\n")
        thin_sha = hashlib.sha256(thin.read_bytes()).hexdigest()
        rows = six(toolchain_sha256=thin_sha)
        code, out = summarise(rows, td, str(thin))
        check("rows correctly bound to a THIN artefact still REFUSE",
              code == 4, out)
        for k in ("frontend", "docker_version", "buildx_version",
                  "runner_os", "commit_sha"):
            check(f"and {k} is named as missing", k in out, out)
        # and the shared contract is reported by size, from the module
        code, out = summarise(six(), td)
        check("the complete artefact passes the same contract",
              code == 0, out)
        check("and the summary says which contract it applied",
              "8-field contract" in out, out)


def test_runner_binds_the_run_id_before_build_1() -> None:
    scenario("toolchain: a stale run id costs the runner zero builds")
    with tempfile.TemporaryDirectory() as d:
        td = Path(d)
        code, out, rows = run_runner(
            "ok", td, toolchain=toolchain_text(run_id="999"))
        check("a record describing another run exits NON-ZERO",
              code != 0, out)
        check("it is an INSTRUMENT FAILURE", "INSTRUMENT FAILURE" in out, out)
        check("ZERO rows were written", len(rows) == 0, str(rows))
        check("and no build was attempted", "::group::" not in out, out[:400])
        check("the runner names the run it expected",
              "is not this run" in out, out)


# ── D297: the evidence must be provably about THIS subject and run ─────

def test_binding_match_is_rederived_from_the_raw_ids() -> None:
    """MATCH is somebody's reading of two ids. The ids are the evidence."""
    scenario("binding: MATCH is re-derived, not accepted as a word")
    with tempfile.TemporaryDirectory() as d:
        td = Path(d)
        # THE CASE GPT NAMED: the record says MATCH while carrying two
        # different ids. Reading the verdict and ignoring the ids left
        # one classification still trusted inside a reparsed artefact.
        code, out = summarise(six(), td,
                              **{"execid:memu-core.B1": "sha256:" + "9" * 64})
        check("MATCH over disagreeing ids does NOT qualify", code != 0, out)
        check("and the executed id is named",
              "the executed container ran" in out, out)
        check("Axis 2 is the fault, not Axis 1",
              "DISAGREEMENT: memu-core/B1 Axis 2" in out, out)
        # A binding missing an id cannot have its comparison redone.
        code, out = summarise(six(), td, **{"execid:memu-graph.B2": ""})
        check("a binding without both ids does NOT qualify", code != 0, out)
        check("and says the comparison cannot be redone",
              "cannot be redone here" in out, out)


def test_artefact_subject_identity_is_reconciled() -> None:
    """A correctly named file is not evidence about the right subject."""
    scenario("evidence: filename is not identity")
    with tempfile.TemporaryDirectory() as d:
        td = Path(d)
        for label, over, why in (
            ("a swapped RUN id", {"run:memu-core.B1": "999"}, "names run"),
            ("a swapped TREE", {"tree:memu-core.B1": "0" * 40}, "names tree"),
            ("a swapped SERVICE", {"svc:memu-core.B1": "item8-b1-memu-graph"},
             "names service"),
            ("a swapped IMAGE REF", {"ref:memu-core.B1": "kai-item8:b3-memu-graph"},
             "names image_ref"),
        ):
            code, out = summarise(six(), td, **over)
            check(f"{label} does NOT qualify", code != 0, out)
            check(f"{label} is named as the reason", why in out, out)
        # and the same for B3's absence record
        code, out = summarise(six(), td, **{"run:memu-graph.B3": "999"})
        check("a B3 absence record from another run does NOT qualify",
              code != 0, out)


def test_one_subject_one_record() -> None:
    scenario("evidence: two contradictory records is not 'take the first'")
    with tempfile.TemporaryDirectory() as d:
        td = Path(d)
        code, out = summarise(six(), td, **{"dupe:memu-core.B1": True})
        check("a duplicated identity record does NOT qualify", code != 0, out)
        check("and says one subject, one record",
              "one subject and one record" in out, out)
        check("rather than silently using the first",
              "holds 2 records" in out, out)
        # a malformed record is refused, not skipped
        derived, ident = write_evidence(td)
        (ident / "item8-b1-memu-core.jsonl").write_text("{not json\n")
        code, out = summarise(six(), td, dirs=(derived, ident))
        check("a malformed identity record does NOT qualify", code != 0, out)
        check("and names the line", "is not valid JSON" in out, out)
        # zero records is its own state
        (ident / "item8-b1-memu-core.jsonl").write_text("")
        code, out = summarise(six(), td, dirs=(derived, ident))
        check("an empty identity record does NOT qualify", code != 0, out)
        check("and says it holds no records", "holds no records" in out, out)


def test_offline_observation_is_a_stamped_record() -> None:
    scenario("evidence: the offline load carries who, which run, which tree")
    with tempfile.TemporaryDirectory() as d:
        td = Path(d)
        derived, ident = write_evidence(td)
        rec = json.loads(
            (ident / "item8-b1-memu-core.offline.json").read_text())
        for k in ("service", "image_ref", "run_id", "tree_sha",
                  "exit_status"):
            check(f"the offline record carries {k}", k in rec, str(rec))
        # a bare exit code in a well-named file is no longer enough
        (ident / "item8-b1-memu-core.offline.json").write_text(
            json.dumps({"exit_status": 0}) + "\n")
        code, out = summarise(six(), td, dirs=(derived, ident))
        check("an unstamped offline record does NOT qualify", code != 0, out)
        check("and the missing subject is named",
              "the offline-load record names service" in out, out)
        check("and Axis 1 is where it lands, not Axis 2",
              "DISAGREEMENT: memu-core/B1 Axis 1" in out, out)
        # an offline record from ANOTHER run, correctly named
        code, out = summarise(six(), td, **{"run:memu-graph.B1": "999"})
        check("an offline record from another run does NOT qualify",
              code != 0, out)
        # present, stamped, and carrying no exit status at all: ABSENT is
        # not zero, and a record that observes nothing is not evidence.
        derived, ident = write_evidence(td)
        rec = json.loads(
            (ident / "item8-b1-memu-core.offline.json").read_text())
        rec.pop("exit_status")
        (ident / "item8-b1-memu-core.offline.json").write_text(
            json.dumps(rec) + "\n")
        code, out = summarise(six(), td, dirs=(derived, ident))
        check("an offline record with no exit_status does NOT qualify",
              code != 0, out)
        check("and says so", "no integer exit_status" in out, out)


def test_preflight_refuses_without_a_daemon() -> None:
    """R11 for the preflight itself: it cannot qualify what it can't invoke."""
    scenario("preflight: a toolchain it cannot invoke is a refusal")
    with tempfile.TemporaryDirectory() as d:
        td = Path(d)
        p = subprocess.run(
            [sys.executable, str(REPO / "scripts" / "security" /
                                 "preflight_buildkit_rawjson.py"),
             "--docker", str(td / "no-such-docker")],
            capture_output=True, text=True, cwd=str(REPO))
        check("an absent docker REFUSES", p.returncode == 2, p.stdout)
        check("and says it cannot qualify what it cannot invoke",
              "cannot qualify a toolchain it cannot invoke" in p.stdout,
              p.stdout)
        # A daemon that emits NO events must refuse, not report zero.
        fake = td / "silent-docker"
        fake.write_text("#!/bin/sh\nexit 0\n")
        fake.chmod(0o755)
        p = subprocess.run(
            [sys.executable, str(REPO / "scripts" / "security" /
                                 "preflight_buildkit_rawjson.py"),
             "--docker", str(fake)],
            capture_output=True, text=True, cwd=str(REPO))
        check("a daemon emitting no rawjson REFUSES", p.returncode == 1,
              p.stdout)
        check("and says which builds were spent: none",
              "ZERO Item-8 builds have been spent" in p.stdout, p.stdout)
        check("and names the rawjson possibility",
              "--progress=rawjson" in p.stdout, p.stdout)
        # THE SHAPE, NOT THE VALUE. This asserted "6 required propert"
        # -- a second hand-written copy of a denominator the preflight
        # also typed in, so adding the chronology property broke a test
        # that has nothing to do with chronology. That is R5 twice over:
        # the count lived beside the thing in two places, and neither
        # derived it.
        #
        # What this check is NAMED for is that the preflight REPORTS a
        # denominator. The value has exactly one maintained expectation,
        # in test_item8_preflight.py, declared there as a drift detector.
        # Pinning it here as well would mean a property could not be
        # added without editing two unrelated suites.
        check("and reports its own denominator",
              re.search(r"\d+ required propert", p.stdout) is not None,
              p.stdout)


# ── D298: the raw BuildKit capture must be THIS subject's ──────────────

def test_capture_is_bound_to_the_derived_subject() -> None:
    """A filename is not an identity — one layer below D297."""
    scenario("capture: a substituted BuildKit capture is refused")
    with tempfile.TemporaryDirectory() as d:
        td = Path(d)
        # THE CASE GPT NAMED: memu-core/B3's capture filed as
        # memu-graph/B3. Both have a five-attempt target, five retries,
        # a refusal and a vertex error; only the derived instruction
        # tells them apart.
        # LAYER 1 — the bytes. The invocation recorded what its own
        # build returned; this capture is not it.
        code, out = summarise(six(), td,
                              **{"swap:memu-graph.B3": ("memu-core", "B3")})
        check("a cross-IMAGE substitution REFUSES", code != 0, out)
        check("on the bytes first",
              "not the bytes that build returned" in out, out)

        # LAYER 2 — move the invocation record with the capture, so the
        # hashes agree, and the INSTRUCTION binding has to catch it.
        code, out = summarise(six(), td,
                              **{"swap:memu-graph.B3": ("memu-core", "B3"),
                                 "invswap:memu-graph.B3": ("memu-core", "B3")})
        check("moving the record too still REFUSES", code != 0, out)
        check("and names the subject mismatch",
              "names service" in out or "names -f" in out
              or "is not evidence about" in out, out)

        code, out = summarise(six(), td,
                              **{"swap:memu-core.B2": ("memu-graph", "B2"),
                                 "invswap:memu-core.B2": ("memu-graph", "B2")})
        check("the reverse cross-image substitution REFUSES", code != 0, out)
        check("naming the subject it was filed as",
              "memu-core/B2" in out, out)

        # B1's capture filed as B3, same image. On a daemon that carries
        # RUN flags in vertex names this is caught by the instruction;
        # the fixture's binding rule says it does.
        code, out = summarise(six(), td,
                              **{"swap:memu-core.B3": ("memu-core", "B1")})
        check("a B1-for-B3 substitution REFUSES", code != 0, out)

        # ...and it must STILL refuse with every digest corroborator
        # switched off. That is the point of the primary chain: the
        # refusal comes from the bytes the invocation recorded, not from
        # a structural signal BuildKit does not license across separate
        # invocations. If this ever passes, the chain is not carrying
        # the weight the design says it carries. (D301)
        code, out = summarise(
            six(), td,
            **{"swap:memu-core.B3": ("memu-core", "B1"),
               "binding_rule": {"flags_in_vertex_name": False,
                                "full_instruction_in_vertex_name": True,
                                "digest_stable_across_invocations": False,
                                "netmode_changes_vertex_digest": False,
                                "run_id": TC_RUN, "tree_sha": TC_TREE}})
        check("the swap REFUSES with NO digest corroboration at all",
              code == 4, out)
        check("and the refusal comes from the invocation chain",
              "not the bytes that build returned" in out
              or "was not given this subject" in out, out)
        # A clean six with the corroborators unavailable must NOT be
        # blocked by their absence -- corroboration is not authority.
        code, out = summarise(
            six(), td,
            **{"binding_rule": {"flags_in_vertex_name": False,
                                "full_instruction_in_vertex_name": True,
                                "digest_stable_across_invocations": False,
                                "netmode_changes_vertex_digest": False,
                                "run_id": TC_RUN, "tree_sha": TC_TREE}})
        check("an unstable digest does NOT block a sound chain",
              code == 0, out)
        check("and the corroboration is reported as unavailable",
              "DIGEST CORROBORATION" in out, out)


def test_derived_dockerfile_must_be_the_re_derivation() -> None:
    scenario("capture: the subject is re-derived from the shipped source")
    with tempfile.TemporaryDirectory() as d:
        td = Path(d)
        derived, ident = write_evidence(td)
        (derived / "Dockerfile.memu-core.B1").unlink()
        code, out = summarise(six(), td, dirs=(derived, ident))
        check("an ABSENT derived Dockerfile REFUSES", code != 0, out)
        check("and says the subject cannot be established",
              "cannot be established" in out, out)

        derived, ident = write_evidence(td)
        f = derived / "Dockerfile.memu-graph.B2"
        before = f.read_text()
        after = before.replace("for attempt in 1 2 3 4 5",
                               "for attempt in 1 2 3 4 5 6", 1)
        assert after != before, "the tamper must actually change the file"
        f.write_text(after)
        code, out = summarise(six(), td, dirs=(derived, ident))
        check("a TAMPERED derived Dockerfile REFUSES", code != 0, out)
        check("and says it is not what the shipped source produces",
              "NOT what deriving the shipped" in out, out)


def test_binding_rule_must_have_been_measured() -> None:
    """R11: an unknown-strength binding is not a binding."""
    scenario("capture: no measured binding rule, no claim")
    with tempfile.TemporaryDirectory() as d:
        td = Path(d)
        derived, ident = write_evidence(td)
        (derived / "binding-rule.json").unlink()
        code, out = summarise(six(), td, dirs=(derived, ident))
        check("a missing binding rule REFUSES", code != 0, out)
        check("and says the preflight did not run",
              "did not run" in out or "holds no records" in out, out)
        check("naming the rule as inadmissible, not assuming a strength",
              "BINDING RULE" in out, out)


def test_generic_loop_alone_does_not_bind() -> None:
    """The old anchor matched all six. It must no longer be sufficient."""
    scenario("capture: the generic retry loop is not a subject")
    with tempfile.TemporaryDirectory() as d:
        td = Path(d)
        derived, ident = write_evidence(td)
        # A capture whose target carries the GENERIC anchor and nothing
        # else. This satisfied the old selector for every branch.
        import base64 as b64
        generic = json.dumps({"vertexes": [{
            "digest": DIGEST,
            "name": "[3/9] RUN for attempt in 1 2 3 4 5; do :; done",
            "started": "t0"}]}) + "\n"
        generic += json.dumps({"logs": [{"vertex": DIGEST, "stream": 1,
                                         "data": b64.b64encode(
                                             b"BAKED ok\n").decode()}]}) + "\n"
        cap = derived / "memu-core.B1.events-stderr.jsonl"
        cap.write_text(generic)
        # LAYER 1: the invocation that produced this branch recorded the
        # bytes it got back, and these are not those bytes.
        code, out = summarise(six(), td, dirs=(derived, ident))
        check("a substituted capture fails the BYTE binding first",
              code != 0, out)
        check("and says these are not the bytes that build returned",
              "not the bytes that build returned" in out, out)
        # LAYER 2: now repair the record so the bytes agree, and the
        # INSTRUCTION binding is what has to catch it. The old anchor
        # matched all six; this proves it no longer suffices.
        inv = ident / "item8-b1-memu-core.invocation.json"
        rec = json.loads(inv.read_text())
        rec["events_stderr_sha256"] = hashlib.sha256(
            cap.read_bytes()).hexdigest()
        inv.write_text(json.dumps(rec) + "\n")
        code, out = summarise(six(), td, dirs=(derived, ident))
        check("the generic loop alone still does NOT bind", code != 0, out)
        check("and says no vertex carries this subject's instruction",
              "carries this subject's target instruction" in out, out)


def test_commit_is_compared_as_D297_said_it_was() -> None:
    scenario("evidence: commit_sha is compared, matching the statement")
    with tempfile.TemporaryDirectory() as d:
        td = Path(d)
        derived, ident = write_evidence(td)
        f = ident / "item8-b1-memu-core.jsonl"
        rec = json.loads(f.read_text())
        rec["commit_sha"] = "0" * 40
        f.write_text(json.dumps(rec) + "\n")
        code, out = summarise(six(), td, dirs=(derived, ident))
        check("a record from another COMMIT does NOT qualify", code != 0, out)
        check("and the commit is named", "names commit" in out, out)


# ── D299: there is no degraded binding, because B1 and B3 are not ──────
#         separated by their outcomes either

def test_b1_outage_looks_exactly_like_b3() -> None:
    """The premise D298 got wrong, made permanent as a fixture."""
    scenario("binding: a B1 outage capture is B3's evidence shape")
    with tempfile.TemporaryDirectory() as d:
        td = Path(d)
        # A REAL B1 outage: five retries, REFUSING TO BUILD, vertex
        # error, no image -- exactly what B3 requires. Filed as B3.
        code, out = summarise(six(), td, **{
            "outage:memu-core.B1": True,
            "swap:memu-core.B3": ("memu-core", "B1")})
        check("a B1-outage capture filed as B3 does NOT qualify",
              code != 0, out)
        check("and it is the PROVENANCE that catches it, not the outcome",
              "not the bytes that build returned" in out, out)
        # Even with the record moved to match the bytes -- so provenance
        # agrees and the outcome is indistinguishable from a real B3 --
        # the re-derived subject still refuses it.
        code, out = summarise(six(), td, **{
            "outage:memu-core.B1": True,
            "swap:memu-core.B3": ("memu-core", "B1"),
            "invswap:memu-core.B3": ("memu-core", "B1")})
        check("with the record moved too it STILL does not qualify",
              code != 0, out)
        # THE HARDEST FORM: a genuine B1 OUTAGE capture -- five retries,
        # refusal, vertex error, no image, indistinguishable from B3 by
        # outcome -- substituted as B3, with EVERY digest corroborator
        # switched off. The refusal must come from the invocation chain
        # alone, because that is the only thing left. (D301)
        code, out = summarise(six(), td, **{
            "outage:memu-core.B1": True,
            "swap:memu-core.B3": ("memu-core", "B1"),
            "binding_rule": {"flags_in_vertex_name": False,
                             "full_instruction_in_vertex_name": True,
                             "digest_stable_across_invocations": False,
                             "netmode_changes_vertex_digest": False,
                             "run_id": TC_RUN, "tree_sha": TC_TREE}})
        check("a B1 OUTAGE capture as B3 REFUSES with no corroboration",
              code == 4, out)
        check("and the refusal is the invocation chain, not a corroborator",
              "not the bytes that build returned" in out
              or "was not given this subject" in out, out)
        check("and the unavailable corroboration is stated",
              "DIGEST CORROBORATION  UNAVAILABLE" in out, out)


def test_preflight_flat_digest_is_not_a_verdict_layer_concern() -> None:
    """The overruled hard-fail, and where its replacement now lives.

    D300 made the preflight FAIL when netmode changed no vertex digest.
    That was overruled: BuildKit licenses vertex-digest comparison
    within a running solver, not across the six separate invocations
    this experiment makes, so a corroborator that is unavailable may not
    stop the measurement. The behaviour is now calibrated in the
    PREFLIGHT suite, which is the entry point that owns it. (D301)
    """
    scenario("preflight: an unavailable corroborator is not a verdict fault")
    with tempfile.TemporaryDirectory() as d:
        td = Path(d)
        # The verdict layer's concern is only that it is TOLD, and that
        # the primary chain still carries the weight.
        code, out = summarise(
            six(), td,
            **{"binding_rule": {"flags_in_vertex_name": False,
                                "full_instruction_in_vertex_name": True,
                                "digest_stable_across_invocations": False,
                                "netmode_changes_vertex_digest": False,
                                "run_id": TC_RUN, "tree_sha": TC_TREE}})
        check("a sound chain qualifies without any corroboration",
              code == 0, out)
        check("and the unavailability is stated, not silent",
              "DIGEST CORROBORATION  UNAVAILABLE" in out, out)

def test_binding_rule_is_itself_admissible_evidence() -> None:
    scenario("binding rule: from this run, this tree, both capabilities")
    with tempfile.TemporaryDirectory() as d:
        td = Path(d)
        for label, rule, why in (
            ("a rule from another run",
             {"flags_in_vertex_name": True,
              "full_instruction_in_vertex_name": True,
              "run_id": "999", "tree_sha": TC_TREE}, "measured in run"),
            ("a rule from another tree",
             {"flags_in_vertex_name": True,
              "full_instruction_in_vertex_name": True,
              "run_id": TC_RUN, "tree_sha": "0" * 40},
             "measured against tree"),
            ("a rule with truncated instructions",
             {"flags_in_vertex_name": True,
              "full_instruction_in_vertex_name": False,
              "run_id": TC_RUN, "tree_sha": TC_TREE},
             "full_instruction_in_vertex_name is False"),
        ):
            code, out = summarise(six(), td, **{"binding_rule": rule})
            check(f"{label} REFUSES", code == 4, out)
            check(f"{label} is named", why in out, out)
        # two records is not "take the first"
        derived, ident = write_evidence(td)
        f = derived / "binding-rule.json"
        f.write_text(f.read_text() + f.read_text())
        code, out = summarise(six(), td, dirs=(derived, ident))
        check("a duplicated binding rule REFUSES", code == 4, out)


def test_commit_is_required_not_merely_compared() -> None:
    scenario("evidence: absence of a commit is not agreement")
    with tempfile.TemporaryDirectory() as d:
        td = Path(d)
        derived, ident = write_evidence(td)
        for name in ("item8-b1-memu-core.offline.json",
                     "item8-b3-memu-graph.absence.json"):
            derived, ident = write_evidence(td)
            f = ident / name
            rec = json.loads(f.read_text())
            check(f"{name} carries a commit", "commit_sha" in rec, str(rec))
            rec.pop("commit_sha")
            f.write_text(json.dumps(rec) + "\n")
            code, out = summarise(six(), td, dirs=(derived, ident))
            check(f"{name} without a commit does NOT qualify",
                  code != 0, out)
            check(f"{name} says absence is not agreement",
                  "absence is not agreement" in out, out)


# ── D300: the invocation chain, and what distinctness cannot see ───────

def test_reciprocal_swap_is_caught() -> None:
    """Six distinct digests, subjects reversed. The case that killed the
    distinctness proposal."""
    scenario("provenance: a reciprocal B1<->B3 swap is refused")
    with tempfile.TemporaryDirectory() as d:
        td = Path(d)
        recip = {"swap:memu-core.B1": ("memu-core", "B3"),
                 "swap:memu-core.B3": ("memu-core", "B1")}
        # First, the thing that makes this case interesting: after a
        # RECIPROCAL swap the six digests are still all different, so a
        # distinctness rule sees nothing wrong.
        derived, ident = write_evidence(td, **recip)
        seen = set()
        for im in ("memu-core", "memu-graph"):
            for br in ("B1", "B2", "B3"):
                cap = (derived / f"{im}.{br}.events-stderr.jsonl").read_text()
                for line in cap.splitlines():
                    o = json.loads(line)
                    for v in o.get("vertexes") or []:
                        seen.add(v["digest"])
        check("a reciprocal swap leaves six DISTINCT digests",
              len(seen) == 6, str(len(seen)))
        # ...and the invocation chain refuses it anyway. Tested with
        # flag corroboration OFF, so the refusal cannot be coming from
        # the one signal that may not exist on the real daemon.
        noflag = dict(recip)
        noflag["binding_rule"] = {"flags_in_vertex_name": False,
                                  "full_instruction_in_vertex_name": True,
                                  "digest_stable_across_invocations": True,
                                  "netmode_changes_vertex_digest": True,
                                  "run_id": TC_RUN, "tree_sha": TC_TREE}
        code, out = summarise(six(), td, **noflag)
        check("the reciprocal swap REFUSES without flag corroboration",
              code != 0, out)
        check("on the bytes the invocation recorded",
              "not the bytes that build returned" in out, out)

        # HARDER: swap the invocation RECORDS with the captures, so the
        # byte hashes agree again. Only re-deriving the expected
        # Dockerfile for the slot can tell the pair is reversed.
        both = dict(recip)
        both.update({"invswap:memu-core.B1": ("memu-core", "B3"),
                     "invswap:memu-core.B3": ("memu-core", "B1")})
        code, out = summarise(six(), td, **both)
        check("swapping the records TOO still REFUSES", code != 0, out)
        check("because the re-derived subject disagrees",
              "was not given this subject" in out
              or "names -f" in out or "names service" in out, out)


def test_invocation_chain_links_are_each_load_bearing() -> None:
    scenario("provenance: every link of subject->invocation->capture")
    with tempfile.TemporaryDirectory() as d:
        td = Path(d)
        # a record naming another branch's Dockerfile
        code, out = summarise(six(), td, **{
            "dfsha:memu-graph.B2": "0" * 64})
        check("a wrong derived-Dockerfile sha REFUSES", code != 0, out)
        check("and says the build was not given this subject",
              "was not given this subject" in out, out)
        # a record whose stderr hash does not match the archived capture
        code, out = summarise(six(), td, **{
            "errsha:memu-core.B3": "0" * 64})
        check("a wrong capture hash REFUSES", code != 0, out)
        check("naming the file", "events-stderr.jsonl hashes to" in out, out)
        # a missing invocation record is not "no objection"
        derived, ident = write_evidence(td)
        (ident / "item8-b1-memu-core.invocation.json").unlink()
        code, out = summarise(six(), td, dirs=(derived, ident))
        check("an ABSENT invocation record REFUSES", code != 0, out)
        check("and says so", "invocation record unusable" in out, out)


def test_two_subjects_may_not_be_one_step() -> None:
    """Distinctness is corroboration, and corroboration still has to fire."""
    scenario("structural: a shared vertex digest is reported, not fatal")
    with tempfile.TemporaryDirectory() as d:
        td = Path(d)
        code, out = summarise(six(), td)
        check("six structurally distinct subjects qualify", code == 0, out)
        check("and no anomaly is invented", "DIGEST ANOMALY" not in out, out)
        # Two of the six are the SAME step. That IS an anomaly and it is
        # reported -- but it may not block closure, because comparing
        # vertex digests across six separate invocations is a comparison
        # BuildKit does not license. The bytes are what refuse a copied
        # capture, and they already do. (D301)
        code, out = summarise(six(), td, **{
            "samedigest:memu-graph.B2": ("memu-core", "B2")})
        check("a shared digest is REPORTED", "DIGEST ANOMALY" in out, out)
        check("and both subjects are named",
              "memu-core/B2, memu-graph/B2" in out, out)
        check("and it is labelled corroboration only",
              "CORROBORATION ONLY" in out, out)
        check("and it does NOT block an otherwise sound chain",
              code == 0, out)


def test_every_frozen_invocation_parameter_is_reconciled() -> None:
    """Two repairs shipped with no fixture. Removing them broke nothing."""
    scenario("invocation: each frozen build parameter is checked")
    with tempfile.TemporaryDirectory() as d:
        td = Path(d)
        # --no-cache IS THE FROZEN DESIGN. R2 gives it to B1 and to no
        # other branch, and it is what makes B1's genuine-fetch
        # criterion a fetch rather than a cache read. Left as producer
        # metadata, the design's own distinguishing flag was taken on
        # the runner's word.
        code, out = summarise(six(), td, **{"nocache:memu-core.B1": False})
        check("B1 invoked WITHOUT --no-cache REFUSES", code != 0, out)
        check("and names the parameter", "no_cache=False" in out, out)
        code, out = summarise(six(), td, **{"nocache:memu-graph.B3": True})
        check("B3 invoked WITH --no-cache REFUSES", code != 0, out)
        check("and says the branch requires otherwise",
              "the frozen design for this branch requires" in out, out)
        # and the rest of the invocation, each on its own
        for label, over, token in (
            ("a different subcommand", {"sub:memu-core.B2": "buildx"},
             "subcommand="),
            ("progress not rawjson", {"progress:memu-core.B1": "plain"},
             "progress="),
            ("an iidfile somewhere else", {"iidpath:memu-graph.B1": "/tmp/x"},
             "iidfile="),
            ("a different build context", {"ctx:memu-core.B3": "subdir"},
             "context="),
        ):
            code, out = summarise(six(), td, **over)
            check(f"{label} REFUSES", code != 0, out)
            check(f"{label} is named", token in out, out)


def test_invocation_record_must_carry_its_commit() -> None:
    scenario("invocation: absence of a commit is not agreement")
    with tempfile.TemporaryDirectory() as d:
        td = Path(d)
        code, out = summarise(six(), td, **{"nocommit:memu-core.B1": True})
        check("an invocation record with NO commit REFUSES", code != 0, out)
        check("and says absence is not agreement",
              "absence is not agreement" in out, out)
        code, out = summarise(six(), td,
                              **{"commit:memu-graph.B2": "0" * 40})
        check("an invocation record from another COMMIT REFUSES",
              code != 0, out)
        check("and names the commit", "names commit" in out, out)


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
    test_events_are_read_from_either_descriptor()
    test_parser_names_the_wrong_descriptor_possibility()
    test_b2_marker_is_a_constant_and_order_matters()
    test_b2_shim_prints_a_constant_in_a_real_shell()
    test_b3_requires_the_target_vertexs_own_error()
    test_toolchain_record_is_validated_not_merely_hashed()
    test_runner_refuses_before_build_1_on_a_bad_toolchain()
    test_summariser_rehashes_the_toolchain_artefact()
    test_toolchain_artefact_is_mandatory()
    test_branch_contract_is_enforced_per_branch()
    test_row_identity_is_reconciled_with_the_artefact()
    test_split_descriptors_refuse_rather_than_invent_order()
    test_base_image_digest_must_hold_across_the_six()
    test_axis1_is_derived_not_read()
    test_axis2_is_derived_from_the_identity_artefacts()
    test_the_package_must_actually_be_present()
    test_closure_toolchain_contract_matches_pre_build()
    test_runner_binds_the_run_id_before_build_1()
    test_binding_match_is_rederived_from_the_raw_ids()
    test_artefact_subject_identity_is_reconciled()
    test_one_subject_one_record()
    test_offline_observation_is_a_stamped_record()
    test_preflight_refuses_without_a_daemon()
    test_capture_is_bound_to_the_derived_subject()
    test_derived_dockerfile_must_be_the_re_derivation()
    test_binding_rule_must_have_been_measured()
    test_generic_loop_alone_does_not_bind()
    test_commit_is_compared_as_D297_said_it_was()
    test_b1_outage_looks_exactly_like_b3()
    test_preflight_flat_digest_is_not_a_verdict_layer_concern()
    test_binding_rule_is_itself_admissible_evidence()
    test_reciprocal_swap_is_caught()
    test_invocation_chain_links_are_each_load_bearing()
    test_two_subjects_may_not_be_one_step()
    test_every_frozen_invocation_parameter_is_reconciled()
    test_invocation_record_must_carry_its_commit()
    test_commit_is_required_not_merely_compared()
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
