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
has, and proves the five properties the verdicts depend on:

    1. the target vertex is identifiable by its instruction text
    2. runtime output is attributed to it, and the INSTRUCTION's own
       occurrences of the marker are NOT counted
    3. `started` is readable and true for a step that ran
    4. `cached` is readable and MOVES — false on a forced build, true on
       a repeat (a field that is always false measures nothing)
    5. a deliberately failing target carries its OWN `error`

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

Exit 0 = the toolchain on this runner behaves as the verdict layer
         assumes. The six builds may proceed.
Exit 1 = it does not. STOP. Nothing has been consumed.
Exit 2 = the preflight itself could not run (no docker, no workspace).
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import pathlib
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

# The instruction MENTIONS the marker once (in the echo it will run) and
# the loop PRINTS it three times. A parser reading the vertex name would
# say 1, or 4; only runtime attribution says 3. That is property 2, and
# it is the same trap three earlier reviews found in the verdict layer.
OK_DOCKERFILE = f"""{SYNTAX}
FROM {BASE}
RUN echo {TARGET} && for probe in 1 2 3; do \\
      echo "{MARK} $probe"; \\
    done
"""

FAIL_DOCKERFILE = f"""{SYNTAX}
FROM {BASE}
RUN echo {TARGET} && echo "{MARK} failing" && exit 7
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
                    help="leave the temporary context for inspection")
    args = ap.parse_args()

    if shutil.which(args.docker) is None and not pathlib.Path(args.docker).exists():
        print(f"REFUSED: {args.docker} is not executable. The preflight "
              f"cannot qualify a toolchain it cannot invoke.")
        return 2

    ctx = pathlib.Path(tempfile.mkdtemp(prefix="item8-preflight-"))
    failures: list[str] = []
    observed: dict = {}
    try:
        # ── 1: a forced build that SUCCEEDS ──────────────────────────
        rc, err, out = build(args.docker, ctx, OK_DOCKERFILE, "ok",
                             no_cache=True)
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
            if not t.started:
                failures.append("the vertex reports no `started`, so "
                                "execution cannot be observed (property 3)")
            if t.cached:
                failures.append("the vertex reports `cached` on a "
                                "--no-cache build (property 4)")

        # ── 2: the same build again, WITHOUT --no-cache ──────────────
        rc2, err2, out2 = build(args.docker, ctx, OK_DOCKERFILE, "cached",
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
    finally:
        for tag in ("ok", "cached", "fail"):
            subprocess.run([args.docker, "image", "rm", "-f",
                            f"kai-item8-preflight:{tag}"],
                           capture_output=True)
        if not args.keep:
            shutil.rmtree(ctx, ignore_errors=True)

    print("ITEM-8 BUILDKIT PREFLIGHT — instrument qualification")
    print("=" * 68)
    for k, v in observed.items():
        print(f"  {k:<24} {v}")
    print()
    print(f"  inspected: 3 non-subject build(s) across 5 required "
          f"propert(ies) of --progress=rawjson")
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
    print("The six frozen builds may proceed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
