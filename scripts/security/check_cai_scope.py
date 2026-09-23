#!/usr/bin/env python3
"""CAI v1.0 — does baseline->candidate stay inside a WA's path scope?

PURE CHECKER (contract §6, order §12). Evaluates the COMPLETE Git change
population with rename detection disabled, so a rename is checked at both
ends. Symlinks, gitlinks, ambiguous and colliding paths are refused.

  python3 scripts/security/check_cai_scope.py --repo . \
      --baseline <commit> --candidate <commit> --scope <wa-payload.json>

The scope file is a WA payload (or any object carrying the four path
lists). The signed WA itself is verified by check_cai_authority.py; this
checker answers only the scope question and prints its denominator.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cai_lib as C  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--candidate", required=True)
    ap.add_argument("--scope", required=True)
    a = ap.parse_args(argv)
    try:
        scope = json.loads(Path(a.scope).read_text(encoding="utf-8"))
        state, reasons, n = C.evaluate_scope(a.repo, a.baseline, a.candidate,
                                             scope)
    except (C.CaiError, OSError, ValueError) as e:
        print(f"REFUSE\n  - UNKNOWN: {e}")
        return 1
    print(f"  changed entries evaluated: {n}")
    print(state)
    for r in reasons:
        print(f"  - {r}")
    return 0 if state == C.PASS else 1


if __name__ == "__main__":
    sys.exit(main())
