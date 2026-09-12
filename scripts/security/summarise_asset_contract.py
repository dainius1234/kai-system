#!/usr/bin/env python3
"""Answer KAI-GATE-048's five definition questions from the stage logs.

Separated from the collector for the same reason as
`summarise_memu_graph_startup`: the half that decides must be
calibratable without a Docker daemon, and must not be written by the
same code that chose what to look at.

Every question has exactly one place it can be answered from, and any
question whose stage did not run is reported NOT MEASURED rather than
inferred from a neighbouring stage.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import List, Optional, Tuple

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO))

STAGES = ("A-fetch", "B-offline-with-asset", "C-offline-no-asset",
          "D-noflag-no-asset")


def read(path: Path) -> Optional[str]:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None


def loaded(text: Optional[str]) -> Optional[bool]:
    if text is None:
        return None
    if "RESULT: LOADED" in text:
        return True
    if "RESULT: FAILED" in text:
        return False
    return None


def seconds(text: Optional[str]) -> Optional[float]:
    if text is None:
        return None
    m = re.search(r"RESULT: (?:LOADED in|FAILED after) ([\d.]+)s", text)
    return float(m.group(1)) if m else None


def field(text: Optional[str], label: str) -> Optional[str]:
    if text is None:
        return None
    m = re.search(rf"^{re.escape(label)}\s*(.*)$", text, re.M)
    return m.group(1).strip() if m else None


def cache_files(text: Optional[str]) -> Tuple[Optional[int], Optional[int],
                                              List[Tuple[int, str]]]:
    """(file count, total bytes, [(size, relpath)])."""
    if text is None:
        return None, None, []
    count = re.search(r"--- file count ---\s*\n\s*(\d+)", text)
    total = re.search(r"--- total bytes ---\s*\n\s*(\d+)", text)
    files: List[Tuple[int, str]] = []
    # Leading whitespace is tolerated on the markers. A first version
    # anchored `\n--- file count ---` at column 0, so any log written
    # with indentation parsed the COUNT (whose regex searches anywhere)
    # and silently lost the FILE LIST — a summary that knew how many
    # files there were and could not name one of them.
    block = re.search(
        r"---\s*cache tree under .*? ---\n(.*?)\n\s*---\s*file count\s*---",
        text, re.S)
    if block:
        for line in block.group(1).splitlines():
            m = re.match(r"\s*(\d+)\s+(.*)$", line)
            if m:
                files.append((int(m.group(1)), m.group(2)))
    return (int(count.group(1)) if count else None,
            int(total.group(1)) if total else None,
            files)


def revision(files: List[Tuple[int, str]]) -> Optional[str]:
    """The snapshot sha, read off the cache layout the stack itself built.

    Derived from the paths rather than from a `revision=` argument,
    because cognee never passes one — which is precisely the finding.
    """
    for _size, rel in files:
        m = re.search(r"snapshots/([0-9a-f]{7,40})/", rel)
        if m:
            return m.group(1)
    return None


def _b(v: Optional[bool]) -> str:
    return "NOT MEASURED" if v is None else ("YES" if v else "NO")


def _s(v) -> str:
    return "NOT MEASURED" if v is None else str(v)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage-logs", required=True)
    args = ap.parse_args()
    d = Path(args.stage_logs)

    texts = {name: read(d / f"{name}.log") for name in STAGES}
    present = [n for n in STAGES if texts[n] is not None]
    print(f"  inspected: {len(present)} of {len(STAGES)} expected stage log(s)")
    missing = [n for n in STAGES if n not in present]
    if missing:
        print(f"  NOT COLLECTED: {', '.join(missing)} — the questions these "
              f"answer stay NOT MEASURED")
    print()

    a, b, c, dd = (texts[n] for n in STAGES)
    n_files, n_bytes, files = cache_files(a)
    rev = revision(files)

    print("  Q1  what model does the failing path request?")
    print(f"        {_s(field(a, 'model requested'))}")
    print(f"        via AutoTokenizer.from_pretrained(model) — cognee "
          f"adapter.py:32")
    print()
    print("  Q2  where does the stack expect it locally?")
    print(f"        HF_HOME       {_s(field(a, 'HF_HOME'))}")
    print(f"        HF_HUB_CACHE  {_s(field(a, 'HF_HUB_CACHE'))}")
    print(f"        (the image sets HF_HOME=/data/hf_cache; the probe "
          f"redirects it so the")
    print(f"         measurement cannot write into the image's own path)")
    print()
    print("  Q3  can the revision be pinned?")
    print(f"        resolved snapshot: {_s(rev)}")
    print("        NOT through cognee's API — it calls from_pretrained(model)")
    print("        with no `revision=` and no `local_files_only=`. Pinning is")
    print("        only available at the ASSET level, by baking a specific")
    print("        snapshot into the cache.")
    print()
    print("  Q4  what offline switch does this stack honour?")
    print(f"        with asset + HF_HUB_OFFLINE=1/TRANSFORMERS_OFFLINE=1, "
          f"no network:")
    print(f"          loaded={_b(loaded(b))}  in {_s(seconds(b))}s")
    print(f"        without asset + same flags, no network:")
    print(f"          loaded={_b(loaded(c))}  in {_s(seconds(c))}s")
    print(f"        without asset, NO flags, no network (today's behaviour):")
    print(f"          loaded={_b(loaded(dd))}  in {_s(seconds(dd))}s")
    print()
    print("  Q5  one asset, or multiple transitive assets?")
    print(f"        files materialised: {_s(n_files)}   total bytes: {_s(n_bytes)}")
    for size, rel in sorted(files, key=lambda t: t[1])[:40]:
        print(f"          {size:>10}  {rel}")
    if len(files) > 40:
        print(f"          ... and {len(files) - 40} more (full list in "
              f"{d}/A-fetch.log)")
    print()

    # The contract is PROVEN only if the withheld-network stage succeeded.
    if loaded(b) is True and loaded(c) is False:
        print("  CONTRACT PROVEN: the asset set from stage A is SUFFICIENT "
              "with the network\n  removed, and its absence fails closed "
              "under the same flags.")
    elif loaded(b) is None:
        print("  CONTRACT NOT PROVEN: stage B did not run. Stage A is a list "
              "of files that\n  nobody has shown to be complete.")
    elif loaded(b) is False:
        print("  CONTRACT DISPROVEN: the asset set from stage A was NOT "
              "sufficient offline.\n  A bake built from it would ship a "
              "still-broken image. Re-measure before\n  planning any "
              "remediation.")
    else:
        print("  CONTRACT AMBIGUOUS: offline-with-asset succeeded but "
              "offline-without-asset\n  did not fail. The flags are not "
              "doing what the plan would assume.")

    fast, slow = seconds(c), seconds(dd)
    if fast is not None and slow is not None:
        print(f"\n  FAIL-CLOSED COST: {slow:.1f}s of retry today vs "
              f"{fast:.1f}s with the offline\n  switch — the same outcome, "
              f"reached {slow - fast:.1f}s sooner. This is what\n  "
              f"obligation 2 buys ON ITS OWN, and it is not a fix.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
