#!/usr/bin/env python3
"""Turn KAI-GATE-048's stage logs into Observations, then classify.

Separated from the collector on purpose. The bash script gathers; this
reads what it gathered and decides. That split is what lets the deciding
half be calibrated on synthetic inputs
(`scripts/test_model_startup_classifier.py`) without a Docker daemon, and
it keeps the verdict from being written by the same code that chose what
to look at.

Every field it cannot establish stays `None`, which the classifier turns
into UNKNOWN. A missing stage log is NOT a False.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Optional

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO))

from scripts.security.classify_model_startup import (  # noqa: E402
    IMAGE, Observations, RUNTIME, classify, summarise,
)

#: Compiled extensions that only appear in a process's mapped files once
#: the tokenizer machinery has actually been loaded there.
LOADED_MARKERS = ("tokenizers", "torch", "safetensors")


def read(path: Path) -> Optional[str]:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None


def file_count(text: Optional[str]) -> Optional[int]:
    """The trailing `wc -l` from a cache snapshot stage."""
    if text is None:
        return None
    m = re.search(r"--- file count ---\s*\n\s*(\d+)", text)
    return int(m.group(1)) if m else None


def maps_loaded(text: Optional[str]) -> Optional[bool]:
    """True if any compiled model extension is mapped into pid 1."""
    if text is None:
        return None
    seen = False
    for marker in LOADED_MARKERS:
        m = re.search(rf"^{re.escape(marker)}:\s*(\d+)", text, re.M)
        if not m:
            continue
        seen = True
        if int(m.group(1)) > 0:
            return True
    return False if seen else None


def egress(text: Optional[str]) -> Optional[bool]:
    if text is None:
        return None
    if "egress AVAILABLE" in text:
        return True
    if "no egress on this path" in text:
        return False
    return None


def reached_ready(chronology: Optional[str], probe_rc: Optional[int]) -> Optional[bool]:
    """Prefer the daemon's own health record over the probe's exit code."""
    if chronology:
        if "FIRST PASSING HEALTH PROBE" in chronology:
            return True
        if "NO PASSING HEALTH PROBE" in chronology:
            return False
    if probe_rc is not None:
        return probe_rc == 0
    return None


def external_attempt(*texts: Optional[str]) -> Optional[bool]:
    """Did anything reach for a model registry?

    Matched on transport evidence — a named registry host, or the hub
    client's own retry line — not on the word "model".
    """
    seen_any = False
    for text in texts:
        if text is None:
            continue
        seen_any = True
        for pattern in (r"huggingface\.co", r"hf-mirror", r"Retrying in \d+s",
                        r"\[Retry \d+/\d+\]", r"OSError:.*not a local folder",
                        r"couldn't connect to 'https://huggingface\.co'",
                        r"We couldn't connect", r"LocalEntryNotFoundError",
                        r"extension\.kuzudb\.com"):
            if re.search(pattern, text, re.I):
                return True
    return False if seen_any else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage-logs", required=True)
    ap.add_argument("--probe-rc", type=int, default=None)
    ap.add_argument("--live-rc", type=int, default=None)
    args = ap.parse_args()
    d = Path(args.stage_logs)

    # I-2. How much evidence this verdict actually rests on. A summary
    # that does not say reads identically whether the collector produced
    # nine stage logs or none — and "none" is the case where the verdict
    # is worth least and looks the same.
    expected = ("image-cache", "cache-ready", "cache-after", "maps-ready",
                "maps-after", "chronology", "egress-probe", "live-cycle",
                "service-logs")
    present = [n for n in expected if (d / f"{n}.log").exists()]
    print(f"  inspected: {len(present)} of {len(expected)} expected stage "
          f"log(s) in {d}")
    missing = [n for n in expected if n not in present]
    if missing:
        print(f"  NOT COLLECTED: {', '.join(missing)} — the fields these "
              f"would have established stay NOT MEASURED")
    print()

    image_cache = read(d / "image-cache.log")
    cache_ready = read(d / "cache-ready.log")
    cache_after = read(d / "cache-after.log")
    maps_ready = read(d / "maps-ready.log")
    maps_after = read(d / "maps-after.log")
    chronology = read(d / "chronology.log")
    live = read(d / "live-cycle.log")
    svc_logs = read(d / "service-logs.log")

    n_image = file_count(image_cache)
    n_ready = file_count(cache_ready)
    n_after = file_count(cache_after)
    ready_loaded = maps_loaded(maps_ready)
    after_loaded = maps_loaded(maps_after)

    # Loaded BEFORE readiness: the cache grew past what the image
    # shipped, or the serving process already had the extensions mapped,
    # at a point where no request had been made.
    before: Optional[bool] = None
    if ready_loaded is True:
        before = True
    elif n_image is not None and n_ready is not None:
        before = n_ready > n_image
        if ready_loaded is False and before is False:
            before = False
    elif ready_loaded is False:
        before = False

    # Loaded AT ALL: either signal, at either snapshot.
    at_all: Optional[bool] = None
    signals = [s for s in (ready_loaded, after_loaded) if s is not None]
    counts = [c for c in (n_ready, n_after) if c is not None]
    if any(signals):
        at_all = True
    elif n_image is not None and counts and max(counts) > n_image:
        at_all = True
    elif signals or counts:
        at_all = False

    obs = Observations(
        reached_ready=reached_ready(chronology, args.probe_rc),
        loaded_before_ready=before,
        loaded_at_all=at_all,
        external_resolution_attempted=external_attempt(live, svc_logs),
        # The asset is local iff the IMAGE already carried cache files.
        asset_present_locally=(None if n_image is None else n_image > 0),
        egress_available=egress(read(d / "egress-probe.log")),
        evidence_level=RUNTIME if chronology else IMAGE,
    )

    obs.notes.append(
        f"cache files: image={_n(n_image)} at-readiness={_n(n_ready)} "
        f"after-request={_n(n_after)}")
    obs.notes.append(
        f"pid-1 mapped model extensions: at-readiness={_b(ready_loaded)} "
        f"after-request={_b(after_loaded)}")
    obs.notes.append(f"live-cycle exit: {args.live_rc}")
    if n_ready is not None and n_after is not None and n_after > n_ready:
        obs.notes.append(
            f"the cache GREW by {n_after - n_ready} file(s) across the first "
            f"request — acquisition is request-time, measured, not inferred")

    print(summarise("memu-graph", obs))
    verdict, _ = classify(obs)
    print(f"\n  KAI-GATE-048 VERDICT: {verdict}")
    # Never a non-zero exit for a verdict. This is a REPORT: a collector
    # that fails the build on what it found would make people stop
    # running it, and the decision here belongs to the operator.
    return 0


def _n(v: Optional[int]) -> str:
    return "NOT MEASURED" if v is None else str(v)


def _b(v: Optional[bool]) -> str:
    return "NOT MEASURED" if v is None else ("YES" if v else "NO")


if __name__ == "__main__":
    sys.exit(main())
