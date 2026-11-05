#!/usr/bin/env python3
"""The post-mortem must print what is wrong and skip what is silent.

Calibrated against the shape run 712 actually produced on 2026-08-07:
the image build failed and every downstream step was skipped, so exactly
one log had content and twelve were empty.
"""
from __future__ import annotations

import io
import sys
from contextlib import redirect_stdout
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from scripts.ci.post_mortem import SECTIONS, main, section_body  # noqa: E402

PASSED = 0
FAILED = 0


def check(label: str, condition: bool) -> None:
    global PASSED, FAILED
    if condition:
        PASSED += 1
        print(f"  ok    {label}")
    else:
        FAILED += 1
        print(f"  FAIL  {label}")


def run(sections) -> str:
    buf = io.StringIO()
    with redirect_stdout(buf):
        main(sections)
    return buf.getvalue()


def main_test(tmp: Path) -> None:
    build = tmp / "build-minimal.log"
    build.write_text("ERROR: failed to solve: process did not complete\n"
                     "REFUSING TO BUILD: could not fetch the embedding model\n")
    empty = tmp / "empty.log"
    empty.write_text("")
    whitespace = tmp / "ws.log"
    whitespace.write_text("   \n\n  \n")
    absent = tmp / "never-created.log"

    # ── section_body decides absence, emptiness and whitespace alike ──
    check("absent file yields None", section_body(str(absent), 10) is None)
    check("empty file yields None", section_body(str(empty), 10) is None)
    check("whitespace-only yields None",
          section_body(str(whitespace), 10) is None)
    check("file with content yields text",
          "REFUSING TO BUILD" in (section_body(str(build), 10) or ""))
    check("tail honours the line limit",
          len((section_body(str(build), 1) or "").splitlines()) == 1)

    # ── run 712's shape: one section with content, twelve without ──
    sections = [("live smoke", str(absent), 10),
                ("kill-isolation", str(empty), 10),
                ("full profile bring-up", str(absent), 10),
                ("sovereign boot", str(whitespace), 10),
                ("the minimal image build", str(build), 40)]
    out = run(sections)

    check("denominator is reported", "inspected: 5 captured step log(s)" in out)
    check("counts are right", "1 had output, 4 were empty or absent" in out)
    check("the failing build IS printed", "REFUSING TO BUILD" in out)
    check("silent steps are named once, not sectioned",
          out.count("── ") == 1)
    check("silent steps are still accounted for",
          "live smoke" in out and "sovereign boot" in out)

    # The whole point: a build-only failure must be SHORT, so it fits in
    # the log window that evicted it on run 712.
    check("a build-only post-mortem stays under 20 lines",
          len(out.splitlines()) < 20)

    # ── build logs land last, so they survive a tail-truncated window ──
    body_order = [ln for ln in out.splitlines() if ln.startswith("── ")]
    check("build section is the last section printed",
          body_order and "image build" in body_order[-1])

    sections_multi = [("the minimal image build", str(build), 40),
                      ("live smoke", str(build), 40)]
    out_multi = run(sections_multi)
    check("declared order is preserved when several have output",
          out_multi.index("live smoke (its real output)")
          > out_multi.index("image build (its real output)"))

    # ── I-1: nothing captured is not a clean bill of health ──
    out_none = run([("live smoke", str(absent), 10),
                    ("kill-isolation", str(empty), 10)])
    check("all-empty says so loudly", "NOTHING WAS CAPTURED" in out_none)
    check("all-empty does not read as success",
          "no problem" not in out_none.lower())

    # ── the real SECTIONS list is coherent ──
    check("real sections all use absolute /tmp paths",
          all(p.startswith("/tmp/") for _, p, _ in SECTIONS))
    check("real sections have no duplicate paths",
          len({p for _, p, _ in SECTIONS}) == len(SECTIONS))
    check("real sections all request a positive line count",
          all(n > 0 for _, _, n in SECTIONS))
    check("build logs are last in the real list",
          "build" in SECTIONS[-1][1] and "build" in SECTIONS[-2][1])
    check("every log the workflow tees is covered",
          {"/tmp/bringup.log", "/tmp/build-minimal.log", "/tmp/live-smoke.log",
           "/tmp/sovereign-boot.log"} <= {p for _, p, _ in SECTIONS})

    # ── it is a diagnostic: it must never add a failure of its own ──
    check("exit code is 0 even with nothing to show",
          main([("x", str(absent), 10)]) == 0)
    check("exit code is 0 with content",
          main([("x", str(build), 10)]) == 0)

    # ── unreadable path must not raise ──
    check("a directory in place of a file is handled",
          section_body(str(tmp), 10) is None)


if __name__ == "__main__":
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        print("Post-mortem readability tests")
        print("=" * 60)
        buf = io.StringIO()
        with redirect_stdout(buf):
            pass
        main_test(Path(d))
    print("=" * 60)
    print(f"Post-mortem tests: {PASSED} passed, {FAILED} failed")
    print(f"EXIT GATE: {'PASS' if FAILED == 0 else 'FAIL'}")
    sys.exit(1 if FAILED else 0)
