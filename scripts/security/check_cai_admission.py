#!/usr/bin/env python3
"""CAI v1.0 — derive the programme baseline from the COMPLETE admission
chain (contract §8, order §5 and §15).

PURE CHECKER. Never uses max(sequence) or "latest tag". Prints the
programme baseline ONLY when every record verifies; otherwise REFUSE or
UNRESOLVED with reasons, and no baseline.

  python3 scripts/security/check_cai_admission.py --repo . \
      --authority-config kai-pm/change-control/authority_keys.json
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cai_lib as C  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--authority-config", required=True)
    a = ap.parse_args(argv)
    try:
        key = C.load_authority_config(a.authority_config)
        with C.PinnedKeyring(key) as kr:
            ns = C.load_namespace(a.repo, kr)
            print(f"  records: {len(ns.was)} WA, {len(ns.admissions)} "
                  f"admissions, {len(ns.anomalies)} anomalies")
            d = C.derive(a.repo, ns)
            if kr.error:
                d = C.Derivation(C.REFUSE, [f"UNKNOWN: {kr.error}"] + d.reasons,
                                 d.chain)
    except C.CaiError as e:
        print(f"REFUSE\n  - UNKNOWN: {e}")
        return 1
    print(f"  chain length: {len(d.chain)}")
    print(d.state)
    if d.state == C.PASS:
        print(f"  programme baseline: {d.baseline}")
        return 0
    for r in d.reasons:
        print(f"  - {r}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
