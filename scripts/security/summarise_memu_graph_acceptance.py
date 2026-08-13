#!/usr/bin/env python3
"""KAI-GATE-048 Phase 1 verdict, from the acceptance stage logs.

Separated from the collector so the deciding half can be calibrated
without a Docker daemon — the same split as
`summarise_memu_graph_startup` and `summarise_asset_contract`.

THE CLOSURE CONDITION IS NOT "THE BUILD WENT GREEN"
===================================================

D191: KAI-GATE-048 closes only when runtime proves memu-graph remains
correctly lazy AND its first model-dependent request succeeds without
external model-registry egress. So five checks, and every one must be
demonstrated rather than assumed:

    A  the shipped image loads the tokenizer with NO network at all
    B  readiness is still reached without the tokenizer being loaded
    C  under the INTENDED topology: no external egress, internal
       delegate reachable, real capability works, no HF retry storm
    D  the same check FAILS when the asset is withheld
    E  the evidence is bound to a committed tree and an exact image

C uses the intended topology, NOT `--network none`. memu-graph delegates
embedding work to `ollama`; requiring it to work with no network at all
would test a stricter and different system than production, and would
have made a correct design look broken.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import List, Optional, Tuple

PASS, FAIL, UNKNOWN = "PASS", "FAIL", "UNKNOWN"


def read(path: Path) -> Optional[str]:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None


def _hf_retry(text: Optional[str]) -> Optional[bool]:
    """A Hugging Face retry/backoff sequence, matched on transport
    evidence rather than on the word "model"."""
    if text is None:
        return None
    for pattern in (r"Retrying in \d+s", r"\[Retry \d+/\d+\]",
                    r"huggingface\.co", r"couldn't connect to 'https://huggingface\.co'"):
        if re.search(pattern, text, re.I):
            return True
    return False


def _maps_loaded(text: Optional[str]) -> Optional[bool]:
    if text is None:
        return None
    seen = False
    for marker in ("tokenizers", "torch", "safetensors"):
        m = re.search(rf"^{marker}:\s*(\d+)", text, re.M)
        if not m:
            continue
        seen = True
        if int(m.group(1)) > 0:
            return True
    return False if seen else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage-logs", required=True)
    for flag in ("--tree-sha", "--commit", "--image"):
        ap.add_argument(flag, default="")
    ap.add_argument("--dirty", default="")
    ap.add_argument("--build-inputs", default="")
    for flag in ("--probe-rc", "--a-rc", "--d-rc", "--c3-rc"):
        ap.add_argument(flag, type=int, default=None)
    args = ap.parse_args()
    d = Path(args.stage_logs)

    # The collector writes the exit codes it observed. Reading them from
    # a file rather than from flags is what lets the verdict be its own
    # workflow step: the step needs no knowledge of what the collector
    # saw. A missing rc.env leaves every code None, which propagates to
    # UNKNOWN rather than to a pass.
    rc_env = read(d / "rc.env") or ""
    for line in rc_env.splitlines():
        key, _, value = line.strip().partition("=")
        value = value.strip()
        if key == "A_RC" and args.a_rc is None and value.isdigit():
            args.a_rc = int(value)
        elif key == "D_RC" and args.d_rc is None and value.isdigit():
            args.d_rc = int(value)
        elif key == "C3_RC" and args.c3_rc is None and value.isdigit():
            args.c3_rc = int(value)
        elif key == "PROBE_RC" and args.probe_rc is None and value.isdigit():
            args.probe_rc = int(value)
        elif key == "TREE_SHA" and not args.tree_sha:
            args.tree_sha = value
        elif key == "COMMIT" and not args.commit:
            args.commit = value
        elif key == "IMAGE" and not args.image:
            args.image = value
        elif key == "DIRTY" and not args.dirty:
            args.dirty = value
        elif key == "BUILD_INPUTS" and not args.build_inputs:
            args.build_inputs = value

    expected = ("A-final-image-offline", "D-canfail-no-asset", "B-chronology",
                "B-maps-ready", "C1-external-egress", "C2-internal-reachability",
                "C3-live-cycle", "C4-service-logs")
    texts = {n: read(d / f"{n}.log") for n in expected}
    present = [n for n in expected if texts[n] is not None]
    print(f"  inspected: {len(present)} of {len(expected)} expected stage log(s)")
    missing = [n for n in expected if n not in present]
    if missing:
        print(f"  NOT COLLECTED: {', '.join(missing)} — the checks these "
              f"answer stay UNKNOWN")
    print()

    results: List[Tuple[str, str, str]] = []

    # A — final image, no network at all.
    a = texts["A-final-image-offline"]
    if a is None or args.a_rc is None:
        results.append(("A", UNKNOWN, "not collected"))
    elif args.a_rc == 0 and "OFFLINE-VERIFIED" in a:
        m = re.search(r"OFFLINE-VERIFIED (\S+) in ([\d.]+)s", a)
        results.append(("A", PASS, f"{m.group(1)} loaded in {m.group(2)}s from "
                                  f"the shipped image with --network none"))
    else:
        results.append(("A", FAIL, f"exit {args.a_rc}; the shipped image "
                                   f"cannot load its own asset offline"))

    # B — readiness reached, tokenizer NOT loaded before any request.
    chron = texts["B-chronology"]
    ready = None
    if chron:
        if "FIRST PASSING HEALTH PROBE" in chron:
            ready = True
        elif "NO PASSING HEALTH PROBE" in chron:
            ready = False
    loaded_early = _maps_loaded(texts["B-maps-ready"])
    if ready is None or loaded_early is None:
        results.append(("B", UNKNOWN, "readiness or map state not collected"))
    elif ready and not loaded_early:
        m = re.search(r"FIRST PASSING HEALTH PROBE at \+([\d.]+)s", chron or "")
        when = f" at +{m.group(1)}s" if m else ""
        results.append(("B", PASS, f"ready{when} with no model extension "
                                   f"mapped into the serving process"))
    elif ready and loaded_early:
        results.append(("B", FAIL, "the tokenizer is now loaded BEFORE "
                                   "readiness — the lazy design regressed"))
    else:
        results.append(("B", FAIL, "readiness was not reached"))

    # C — intended topology.
    ext = texts["C1-external-egress"]
    ext_blocked = None
    if ext is not None:
        ext_blocked = "EXTERNAL EGRESS AVAILABLE" not in ext
    internal = texts["C2-internal-reachability"]
    delegate_ok = None
    if internal is not None:
        delegate_ok = "internal delegate REACHABLE" in internal
    retried = _hf_retry(texts["C4-service-logs"])
    cap_ok = None if args.c3_rc is None else args.c3_rc == 0

    parts = []
    verdict = PASS
    for label, value, good, bad in (
        ("external egress blocked", ext_blocked, "blocked", "STILL AVAILABLE"),
        ("internal delegate reachable", delegate_ok, "reachable", "UNREACHABLE"),
        ("real /graph/ingest succeeded", cap_ok, "succeeded", "FAILED"),
        ("no HF retry storm", None if retried is None else not retried,
         "none seen", "RETRY SEQUENCE PRESENT"),
    ):
        if value is None:
            parts.append(f"{label}: NOT MEASURED")
            verdict = UNKNOWN if verdict != FAIL else FAIL
        elif value:
            parts.append(f"{label}: {good}")
        else:
            parts.append(f"{label}: {bad}")
            verdict = FAIL
    results.append(("C", verdict, "; ".join(parts)))

    # D — can-fail, demonstrated. A NON-ZERO exit is the pass.
    if args.d_rc is None or texts["D-canfail-no-asset"] is None:
        results.append(("D", UNKNOWN, "not collected"))
    elif args.d_rc != 0:
        results.append(("D", PASS, f"withholding the asset made the SAME "
                                   f"check fail (exit {args.d_rc}) — stage A "
                                   f"is therefore about the asset"))
    else:
        results.append(("D", FAIL, "the check passed with the asset "
                                   "withheld, so stage A proves nothing"))

    # E — artefact identity.
    if not args.tree_sha or args.tree_sha == "UNKNOWN":
        results.append(("E", UNKNOWN, "tree sha not recorded"))
    elif args.dirty not in ("0", ""):
        results.append(("E", FAIL, f"{args.dirty} uncommitted modification(s): "
                                   f"the tree under test is not a committed tree"))
    elif not args.image:
        results.append(("E", UNKNOWN, "image identity not recorded"))
    elif not args.build_inputs:
        # An image id paired with a tree sha implies a rebuild that the
        # pairing does not establish. Without the build-inputs digest
        # there is nothing the image id actually resolves TO, so this is
        # UNKNOWN rather than a pass.
        results.append(("E", UNKNOWN, "build-inputs digest not recorded, so "
                                      "the image id resolves to nothing"))
    else:
        results.append(("E", PASS, f"tree {args.tree_sha[:12]} commit "
                                   f"{args.commit[:12]} image {args.image[:19]} "
                                   f"build-inputs {args.build_inputs[:12]}"))

    for key, state, why in results:
        print(f"  {key}  {state:<8} {why}")
    print()

    states = {k: s for k, s, _ in results}
    if all(s == PASS for s in states.values()):
        print("  KAI-GATE-048 Phase 1: ACCEPTANCE MET.")
        print("  memu-graph stays lazy and readiness-independent, AND its "
              "first\n  model-dependent request succeeds without external "
              "model-registry\n  egress because the tokenizer is locally "
              "satisfiable.")
        return 0
    if any(s == FAIL for s in states.values()):
        failed = [k for k, s in states.items() if s == FAIL]
        print(f"  KAI-GATE-048 Phase 1: NOT MET — {', '.join(failed)} failed.")
        print("  The finding stays OPEN. A green build is not the contract.")
        return 1
    unknown = [k for k, s in states.items() if s == UNKNOWN]
    print(f"  KAI-GATE-048 Phase 1: INCOMPLETE — {', '.join(unknown)} UNKNOWN.")
    print("  Not a failure and not a pass. The finding stays OPEN because "
          "the\n  evidence does not exist, which is a different thing from "
          "the\n  evidence being bad.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
