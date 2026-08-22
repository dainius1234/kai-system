#!/usr/bin/env python3
"""Qualify rawjson against THIS daemon, before the denominator is spent.

WHAT THIS EXISTS TO STOP
========================

Item 8's entire verdict layer rests on one claim about the toolchain:

> BuildKit's `--progress=rawjson` emits `SolveStatus` objects in which a
> step's INSTRUCTION TEXT and its RUNTIME OUTPUT are different fields of
> different objects, the vertex reports `started` and `cached`, and a
> failing step carries its own `error`.

That claim has been checked against Docker's documentation and against
BuildKit's source. **It has never been checked against a running
daemon**, because no environment the implementation was written in had
one. Every calibration to date models the events rather than observing
them, and a model is exactly the thing that was wrong in D294: the fake
wrote to the wrong file descriptor and every fixture passed.

So this runs the real command path — same flags, same parser, same
capture — on whatever Docker and buildx the authorised runner actually
has. Properties 1-6 are REQUIRED -- the verdict layer cannot
read a capture without them. Property 7 is MEASURED AND
RECORDED, corroboration only:

    1. the target vertex is identifiable by its instruction text
    2. runtime output is attributed to it, and the INSTRUCTION's own
       occurrences of the marker are NOT counted
    3. `started` is readable and true for a step that ran
    4. `cached` is readable and MOVES — false on a forced build, true on
       a repeat (a field that is always false measures nothing)
    5. a deliberately failing target carries its OWN `error`
    6. the WHOLE INSTRUCTION survives into the vertex name — REQUIRED,
       because the claim engine locates a subject's step by containment
       of exactly that text, and a daemon that truncates would fail
       every branch after the denominator was spent
    7. whether RUN FLAGS survive into the name, and whether the vertex
       DIGEST is stable across invocations and moves with the network
       mode — MEASURED AND RECORDED, never required

~~D299 made property 7 a hard failure, on the grounds that B1 and B3
differ only by `--network=none` and do not separate on outcome either.~~
That reasoning about the subjects is correct and stands. The conclusion
was overruled in D301, and rightly: BuildKit licenses vertex-digest
comparison WITHIN a running solver, and Item 8 is six separate
invocations, so neither the printed flag nor the digest may be the thing
a closure claim rests on. The subject binding is the invocation chain --
SUBJECT -> INVOCATION -> HASH-BOUND CAPTURE -> VERTEX -- and properties
6-7 corroborate it. A corroborator promoted to an authority is the same
error as a cosmetic dependency, one layer in. (D300, D301)

IT IS NOT AN EXPERIMENTAL ARM
=============================

Nothing here touches memu-core or memu-graph, no derived Dockerfile is
used, no result row is written, and no outcome of this can become
evidence about the contingency. It is instrument qualification under
rule 15 — prove it can fail — and I-8: the expectation lives here, the
subject is the daemon.

If it fails, **zero Item-8 builds have been spent** and the frozen
denominator is untouched. That is the whole point of its position in the
workflow.

A PASS IS EVIDENCE, NOT EXECUTION AUTHORITY. It says this daemon can
be read; it does not say anything may be built. The sentinel is a
separate act by a separate party, and a measurement that ends by
announcing what may now proceed is a measurement quietly issuing
permission. (D302)

Exit 0 = the toolchain on this runner behaves as the verdict layer
         assumes. Nothing is thereby authorised.
Exit 1 = it does not. STOP. Nothing has been consumed.
Exit 2 = the preflight itself could not run (no docker, no workspace).
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import pathlib
import re
import shutil
import subprocess
import sys
import tempfile

_HERE = pathlib.Path(__file__).resolve().parent

# The pinned frontend R2 froze. The preflight runs the SAME toolchain the
# experiment will, or it is qualifying something else.
SYNTAX = ("# syntax=docker/dockerfile:1.9.0@sha256:"
          "fe40cf4e92cd0c467be2cfc30657a680ae2398318afd50b0c80585784c604f28")
BASE = "python:3.11-slim"

# Deliberately NOT the experiment's anchor. A preflight that shared the
# frozen target substring could be mistaken for a subject build in the
# artefacts, and the whole point is that this is not one.
TARGET = "PREFLIGHT-TARGET-INSTRUCTION"
MARK = "PREFLIGHT-RUNTIME-LINE"
RUNTIME_EMISSIONS = 3

# THE REQUIRED PROPERTIES, DECLARED ONCE.
#
# The summary line used to read "7 non-subject build(s) across 6 required
# propert(ies)" with BOTH numbers typed in by hand. That is R5's defect
# sitting inside the instrument: a denominator kept beside the thing
# instead of derived from it.
#
# It was found while adding the chronology property below, and the tell
# was noticing that the fix would be to change a "6" to a "7" -- writing
# a name I recognised rather than one I looked up. So the count is now
# len() of this tuple, and the build count is counted as builds run.
#
# The tuple is still a list of names, which R5 also distrusts. What
# stops it drifting is that the count printed to a human now moves only
# when this tuple moves; what it cannot do is prove every declared
# property was actually EVALUATED on a given run. That gap is real and
# is reported, not papered over. (see the note emitted at the end)
REQUIRED_PROPERTIES = (
    "rawjson is emitted and can be parsed",
    "runtime output is attributed to its vertex, apart from the instruction",
    "execution is observable (`started`)",
    "cache state is reported, and moves",
    "a failing step's error is attributed to that step",
    "the whole instruction survives into the vertex name",
    "runtime CHRONOLOGY is preserved, in emission order",
)

# Counted, not asserted. A run that aborts early has inspected FEWER
# builds than a full pass, and saying "7" regardless would be the same
# defect as a fifty-row table built on a failed prerequisite. (R11)
BUILDS_RUN = 0

def _longest_real_target() -> int:
    """How long a command must be before this preflight means anything.

    If BuildKit truncates long vertex names, the claim engine's
    containment check fails on every subject and all six builds are
    spent discovering it. So the preflight's own command is made AT
    LEAST as long as the longest instruction the experiment will
    actually present.

    Measured from the SHIPPED Dockerfiles plus the treatment that makes
    the longest branch longest -- and WITHOUT importing the deriver,
    which would put the module that produces the six subjects on the
    measurement's dependency path. The locator lives in the parser,
    which this already depends on. (D301)
    """
    longest = 0
    for image in ("memu-core", "memu-graph"):
        src = _HERE.parent.parent / image / "Dockerfile"
        if not src.is_file():
            continue
        run = EV.find_target_run(src.read_text())
        if run:
            longest = max(longest, len(EV.normalise_command(run)))
    # The B2 shim wraps the shipped command, so the longest DERIVED
    # instruction exceeds the longest shipped one. Doubling is a
    # deliberate over-allowance rather than a measured figure, and it is
    # over-allowance in the safe direction: a longer probe can only make
    # the truncation test harder to pass.
    return longest * 2


def _ok_dockerfile(pad_to: int) -> str:
    """The instruction MENTIONS the marker once (in the echo it will run)
    and the loop PRINTS it three times. A parser reading the vertex name
    would say 1, or 4; only runtime attribution says 3. That is property
    2, and it is the same trap three earlier reviews found in the
    verdict layer, aimed at the toolchain instead of at the subject.
    """
    # Pad with echoes until the normalised command is at least as long
    # as the longest real one. The padding is inert output, not a probe.
    body = (f'echo {TARGET} && for probe in 1 2 3; do \\\n'
            f'      echo "{MARK} $probe"; \\\n'
            f'    done')
    n = 0
    while len(EV.normalise_command("RUN " + body)) < pad_to:
        n += 1
        body += f' \\\n    && echo "PREFLIGHT-PADDING-{n:03d}-xxxxxxxxxxxxxxxx"'
    return f"{SYNTAX}\nFROM {BASE}\nRUN {body}\n"


FAIL_DOCKERFILE = f"""{SYNTAX}
FROM {BASE}
RUN echo {TARGET} && echo "{MARK} failing" && exit 7
"""

# THE A/B/A DIGEST DIFFERENTIAL.
#
# Not "does the flag appear in the rendered name" -- that is cosmetic,
# and depending on it would very likely have blocked Item 8 for a reason
# that has nothing to do with evidence quality.
#
# BuildKit's ExecOp carries `network` as a real field of the operation,
# and a vertex digest is a checksum of the definition graph through that
# vertex. So changing ONLY the network mode SHOULD change the structural
# identity of the step. That is an inference, not a measurement, which is
# exactly what this probe turns into a measurement:
#
#   A1  the command, default network
#   B   the same command, --network=none
#   A2  the command again, default network
#
#   digest(A1) == digest(A2)   the digest is stable across invocations
#                              on THIS daemon
#   digest(A1) != digest(B)    network mode changes structural identity
#
# Stated narrowly as RUNNER-LOCAL structural behaviour, because BuildKit
# documents vertex-digest comparison as valid within a running solver and
# does not license it as a universal cross-invocation identity. We may
# qualify a behaviour experimentally; we may not redefine somebody else's
# contract. (D300)
ABA_BODY = f'echo {TARGET}-ABA && echo "{MARK} aba"'
ABA_DEFAULT = f"""{SYNTAX}
FROM {BASE}
RUN {ABA_BODY}
"""
ABA_DENIED = f"""{SYNTAX}
FROM {BASE}
RUN --network=none {ABA_BODY}
"""


def _parser():
    spec = importlib.util.spec_from_file_location(
        "parse_buildkit_events", _HERE / "parse_buildkit_events.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


EV = _parser()


def build(docker: str, ctx: pathlib.Path, dockerfile: str, tag: str,
          no_cache: bool) -> tuple[int, pathlib.Path, pathlib.Path]:
    """The PRODUCTION command path, byte for byte in its shape."""
    global BUILDS_RUN
    BUILDS_RUN += 1
    df = ctx / f"Dockerfile.{tag}"
    df.write_text(dockerfile)
    err = ctx / f"{tag}.events-stderr.jsonl"
    out = ctx / f"{tag}.events-stdout.jsonl"
    cmd = [docker, "build"]
    if no_cache:
        cmd.append("--no-cache")
    cmd += ["--progress=rawjson", "-f", str(df), "-t", f"kai-item8-preflight:{tag}",
            str(ctx)]
    with open(out, "wb") as o, open(err, "wb") as e:
        rc = subprocess.run(cmd, stdout=o, stderr=e,
                            env={**__import__("os").environ,
                                 "DOCKER_BUILDKIT": "1"}).returncode
    return rc, err, out


def target_of(err: pathlib.Path, out: pathlib.Path) -> tuple[object, str]:
    vertices: dict = {}
    bearing = []
    for p in (err, out):
        if not p.is_file():
            continue
        vx, _diag, e = EV.parse(p.read_text(), p.name)
        if e and "held no BuildKit events" not in e:
            return None, f"{p.name}: {e}"
        if vx:
            bearing.append(p.name)
            vertices = vx
    if not bearing:
        return None, ("neither captured descriptor held BuildKit events. "
                      "Either this buildx does not implement "
                      "--progress=rawjson, or it writes them somewhere "
                      "neither descriptor sees")
    if len(bearing) > 1:
        return None, f"both descriptors carried events ({bearing})"
    t, e = EV.find_target(vertices, TARGET)
    return (t, "") if t else (None, e)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--docker", default="docker")
    ap.add_argument("--keep", action="store_true",
                    help="leave the build context and every capture in "
                         "place. REQUIRED whenever the evidence is to be "
                         "archived: without it this deletes the captures "
                         "it just produced, and a measurement whose raw "
                         "evidence no longer exists is a claim")
    ap.add_argument("--workdir",
                    help="create the context INSIDE this directory, so the "
                         "captures land in the workspace an artefact "
                         "uploader can see rather than in the system "
                         "temporary directory")
    ap.add_argument("--emit-binding-rule", metavar="PATH",
                    help="archive how this daemon represents a RUN, for the "
                         "claim engine to apply")
    ap.add_argument("--run-id", default="")
    ap.add_argument("--tree-sha", default="")
    args = ap.parse_args()

    if shutil.which(args.docker) is None and not pathlib.Path(args.docker).exists():
        print(f"REFUSED: {args.docker} is not executable. The preflight "
              f"cannot qualify a toolchain it cannot invoke.")
        return 2

    ctx = pathlib.Path(tempfile.mkdtemp(prefix="item8-preflight-",
                                        dir=str(args.workdir) if args.workdir
                                        else None))
    failures: list[str] = []
    notes: list[str] = []
    observed: dict = {}
    try:
        # ── 1: a forced build that SUCCEEDS ──────────────────────────
        pad = _longest_real_target()
        ok_df = _ok_dockerfile(pad)
        ok_cmd = EV.normalise_command(
            "RUN " + ok_df.split("\nRUN ", 1)[1])
        observed["longest_real_target_chars"] = pad
        observed["preflight_command_chars"] = len(ok_cmd)
        rc, err, out = build(args.docker, ctx, ok_df, "ok", no_cache=True)
        t, e = target_of(err, out)
        if e:
            failures.append(f"forced build: {e}")
        elif rc != 0:
            failures.append(f"the trivial preflight build exited {rc}; this "
                            f"daemon cannot build {BASE} at all, which the "
                            f"experiment also requires")
        else:
            runtime = "".join(t.log)
            n = runtime.count(MARK)
            observed["runtime_marker_count"] = n
            observed["instruction_mentions"] = t.name.count(MARK)
            observed["started"] = bool(t.started)
            observed["cached_forced"] = bool(t.cached)
            if n != RUNTIME_EMISSIONS:
                failures.append(
                    f"runtime attribution: the target printed {MARK} "
                    f"{RUNTIME_EMISSIONS} times and its instruction mentions "
                    f"it {t.name.count(MARK)} time(s); the parser counted "
                    f"{n}. Property 2 does not hold on this daemon")

            # CHRONOLOGY, NOT JUST CARDINALITY.
            #
            # Counting the marker three times says the runtime lines were
            # attributed. It says NOTHING about the order they arrived
            # in, and 3,1,2 counts exactly the same as 1,2,3.
            #
            # B2's frozen verdict is injection -> genuine retry -> BAKED
            # success IN THAT ORDER. The parser preserves log entries in
            # arrival order and the verdict layer then reads that
            # ordering, so a daemon that attributed correctly but
            # reordered would not produce a false PASS -- it would make
            # B2 UNMEASURABLE. That is not a wrong answer, it is a WASTED
            # IRREVERSIBLE BUILD, and the denominator cannot be redrawn.
            #
            # The emission is already distinguishable (`echo "$MARK
            # $probe"` for probe in 1 2 3); only the CHECK was blind to
            # it. Scope smaller than the name implied. (R5)
            order = re.findall(rf"{re.escape(MARK)} (\S+)", runtime)
            expected_order = [str(i) for i in range(1, RUNTIME_EMISSIONS + 1)]
            observed["runtime_marker_order"] = ",".join(order) or "(none)"
            if order != expected_order:
                failures.append(
                    f"runtime chronology: the target printed {MARK} "
                    f"{'->'.join(expected_order)} in that order; the parser "
                    f"recovered {'->'.join(order) if order else '(none)'}. "
                    f"B2's verdict is injection -> retry -> success IN THAT "
                    f"ORDER, so on this daemon B2 would be UNMEASURABLE "
                    f"after the build had already been spent")
            if not t.started:
                failures.append("the vertex reports no `started`, so "
                                "execution cannot be observed (property 3)")
            if t.cached:
                failures.append("the vertex reports `cached` on a "
                                "--no-cache build (property 4)")
            # PROPERTY 6a: does the WHOLE instruction survive into the
            # vertex name? The claim engine binds a capture to its
            # subject by containment of exactly this text, and a daemon
            # that truncates long names would fail every subject after
            # the denominator was spent.
            seen = EV.normalise_command(t.name)
            body, _flags = EV.strip_run_flags(ok_cmd)
            observed["full_instruction_in_vertex_name"] = body in seen
            if body not in seen:
                failures.append(
                    f"the vertex name does not carry the whole instruction "
                    f"({len(body)} chars); the claim engine binds a capture "
                    f"to its subject by exactly this containment, and on "
                    f"this daemon it would fail for every branch. Name "
                    f"seen: {seen[:160]!r}")

        # ── 2: the same build again, WITHOUT --no-cache ──────────────
        rc2, err2, out2 = build(args.docker, ctx, ok_df, "cached",
                                no_cache=False)
        t2, e2 = target_of(err2, out2)
        if e2:
            failures.append(f"repeat build: {e2}")
        else:
            observed["cached_repeat"] = bool(t2.cached)
            if not t2.cached:
                # A `cached` that never becomes true is not a measurement;
                # B1's whole "the fetch really ran" criterion rests on it.
                failures.append(
                    "the repeated build reports cached=False, so `cached` "
                    "never moves on this daemon and B1's uncached-execution "
                    "criterion cannot be established (property 4)")

        # ── 3: a build whose TARGET deliberately fails ───────────────
        rc3, err3, out3 = build(args.docker, ctx, FAIL_DOCKERFILE, "fail",
                                no_cache=True)
        t3, e3 = target_of(err3, out3)
        if e3:
            failures.append(f"failing build: {e3}")
        else:
            observed["failing_build_exit"] = rc3
            observed["target_error"] = (t3.error or "")[:120]
            if rc3 == 0:
                failures.append("a build whose RUN exits 7 returned 0")
            if not t3.error:
                failures.append(
                    "the deliberately failing target carries NO error of "
                    "its own, so B3's attribution criterion cannot be "
                    "established on this daemon (property 5)")
            if MARK not in "".join(t3.log):
                failures.append("no runtime output was attributed to the "
                                "failing target")
        # ── 4: PROPERTY 7 — THE A/B/A STRUCTURAL DIFFERENTIAL ───────
        #
        # CORROBORATION, NOT PASS/FAIL. It replaces the cosmetic
        # "does the flag appear in the vertex name" dependency that
        # D299 made load-bearing. B1 and B3 differ only by
        # `--network=none`; the question is whether that difference is
        # STRUCTURAL on this daemon, which is a different and far more
        # likely thing than whether it is PRINTED.
        #
        # The stale comment that used to sit here said "not a pass/fail"
        # while the code below it treated absence as fatal. GPT caught
        # the contradiction. It is gone rather than reworded, because a
        # comment that argues with its code is what the next reader
        # believes. (D300)
        digests = {}
        aba_err = ""
        for tag, df in (("aba1", ABA_DEFAULT), ("abab", ABA_DENIED),
                        ("aba2", ABA_DEFAULT)):
            _rc, _e, _o = build(args.docker, ctx, df, tag, no_cache=True)
            tv, terr = target_of(_e, _o)
            if terr:
                aba_err = f"A/B/A probe {tag}: {terr}"
                break
            digests[tag] = tv.digest
        if aba_err:
            # NOT A FAILURE. "The corroboration could not be obtained" and
            # "a required property is broken" are different facts, and
            # collapsing them lets a corroborator stop the run through
            # the back door -- the same promotion D301 removed from the
            # front door.
            #
            # If the A/B/A probe failed for a reason that IS a required
            # property, properties 1-6 above have already recorded it
            # against that property, on their own builds. What is left
            # here is only "we could not measure the corroborator", and
            # the answer to that is to say so. (D302)
            observed["digest_corroboration"] = "UNRESOLVED"
            observed["digest_corroboration_why"] = aba_err
        observed["aba_digests"] = {k: v[:19] for k, v in digests.items()}
        digest_stable = (len(digests) == 3
                         and digests["aba1"] == digests["aba2"])
        netmode_changes_digest = (len(digests) == 3
                                  and digests["aba1"] != digests["abab"])
        observed["digest_stable_across_invocations"] = digest_stable
        observed["netmode_changes_vertex_digest"] = netmode_changes_digest
        # RECORDED, NOT FATAL. Both of these were failures, and both
        # were wrong to be.
        #
        # BuildKit documents vertex-digest comparison as valid within a
        # RUNNING SOLVER, and this experiment is six separate
        # invocations. Making a corroborator BuildKit does not license
        # across invocations into a gate would stop the measurement for
        # a reason that is not about evidence quality -- the same
        # mistake the cosmetic flag-name dependency made, one layer in.
        #
        # The subject binding is the invocation chain. This probe says
        # how much CORROBORATION that chain will have, and either answer
        # is a result. (D301)
        if len(digests) == 3 and not digest_stable:
            notes.append(
                "the same command built twice produced DIFFERENT target "
                "vertex digests: on this daemon a vertex digest is not "
                "comparable across invocations, so it corroborates "
                "nothing. Recorded, not fatal — the invocation chain does "
                "not depend on it")
        if len(digests) == 3 and not netmode_changes_digest:
            notes.append(
                "--network=none did NOT change the target vertex digest, "
                "so B1 and B3 of one image are structurally alike to this "
                "daemon. Recorded, not fatal — they are separated by the "
                "invocation chain, which binds each capture to the exact "
                "derived Dockerfile and flags that produced it")

        # The rendered name is RECORDED, and no longer REQUIRED. It was a
        # cosmetic dependency; the structural one above replaces it.
        rc4, err4, out4 = build(args.docker, ctx, ABA_DENIED, "flag",
                                no_cache=False)
        t4, e4 = target_of(err4, out4)
        flags_in_name = False
        if not e4:
            flags_in_name = "--network=none" in EV.normalise_command(t4.name)
        observed["flags_in_vertex_name"] = flags_in_name
        # Emitted on PASS of the REQUIRED properties. The digest fields
        # carry whatever was measured, including False.
        if not failures and args.emit_binding_rule:
            pathlib.Path(args.emit_binding_rule).write_text(json.dumps({
                "flags_in_vertex_name": flags_in_name,
                "digest_stable_across_invocations": digest_stable,
                "netmode_changes_vertex_digest": netmode_changes_digest,
                "full_instruction_in_vertex_name": True,
                "measured_against": "a real daemon, by "
                                    "preflight_buildkit_rawjson.py",
                "run_id": args.run_id, "tree_sha": args.tree_sha,
                "longest_real_target_chars": pad}) + "\n")
    finally:
        for tag in ("ok", "cached", "fail", "flag", "aba1", "abab", "aba2"):
            subprocess.run([args.docker, "image", "rm", "-f",
                            f"kai-item8-preflight:{tag}"],
                           capture_output=True)
        if args.keep:
            print(f"  raw captures retained: {ctx}")
        else:
            # Said out loud. The first version deleted these silently
            # while the report above described them as archived, and
            # "all three captures archived" was mechanically false at
            # that tree. (D301)
            print("  raw captures DELETED (no --keep): this run produced "
                  "no archivable evidence")
            shutil.rmtree(ctx, ignore_errors=True)

    print("ITEM-8 BUILDKIT PREFLIGHT — instrument qualification")
    print("=" * 68)
    for k, v in observed.items():
        print(f"  {k:<24} {v}")
    print()
    print(f"  inspected: {BUILDS_RUN} non-subject build(s) across "
          f"{len(REQUIRED_PROPERTIES)} required propert(ies) of "
          f"--progress=rawjson")
    print(f"  the build count is COUNTED and the property count is len() "
          f"of the declared tuple; neither is typed in. What this does "
          f"NOT prove is that every declared property was reached on "
          f"this run -- an early abort inspects fewer.")

    print()
    for n in notes:
        print(f"  NOTE: {n}")
    if notes:
        print()
    if failures:
        for f in failures:
            print(f"FAIL: {f}")
        print()
        print("REFUSED. The toolchain on this runner does not behave the way "
              "Item 8's verdict layer assumes, and every conclusion the six "
              "builds would produce rests on that assumption.")
        print("ZERO Item-8 builds have been spent. The frozen denominator is "
              "untouched, and this is not a re-draw of anything. (D297)")
        return 1
    print("PASS: rawjson on THIS daemon separates instruction from runtime, "
          "reports execution and cache state, and attributes a failing "
          "step's error to that step.")
    print("PREFLIGHT PASS IS EVIDENCE, NOT EXECUTION AUTHORITY. It says "
          "this daemon can be read; it authorises nothing.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
