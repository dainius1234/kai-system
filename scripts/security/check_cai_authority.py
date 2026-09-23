#!/usr/bin/env python3
"""CAI v1.0 — is a Work Authority usable now? (contract §7, order §13-14)

PURE CHECKER. Reads Git objects; writes nothing; creates no authority.

  python3 scripts/security/check_cai_authority.py --repo . \
      --authority-config kai-pm/change-control/authority_keys.json \
      --now 2026-09-23T00:00:00Z [--wa WA_ID]

With --wa: PASS only if that WA verifies, is unexpired at --now, has no
valid HOLD/REVOKE, has not already backed an admission (replay, P-6), and
— when an admission chain exists — is based on the current programme
baseline (staleness, P-7). An EMPTY chain imposes no baseline constraint;
any other non-PASS chain state refuses, because the baseline cannot be
established.

Without --wa: PASS only if every record in refs/tags/kai-wa/ verifies.

--now is an explicit input so the answer is deterministic. The verifier
never reads the wall clock.
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
    ap.add_argument("--now", required=True, help="UTC, YYYY-MM-DDTHH:MM:SSZ")
    ap.add_argument("--wa")
    a = ap.parse_args(argv)
    reasons = []
    try:
        now = C.parse_utc(a.now)
        key = C.load_authority_config(a.authority_config)
        with C.PinnedKeyring(key) as kr:
            # A verifier that cannot verify must not pass anything -- not
            # even an empty namespace, where "every record verifies" would
            # be vacuously true over zero records.
            if kr.error:
                reasons.append(f"UNKNOWN: {kr.error}")
            ns = C.load_namespace(a.repo, kr)
            wa_anoms = [x for x in ns.anomalies if x.startswith(C.NS_WA)]
            print(f"  kai-wa records: {len(ns.was)} WA, "
                  f"{sum(len(v) for v in ns.events.values())} terminal events, "
                  f"{len(wa_anoms)} anomalies")
            reasons += [f"namespace (P-3): {x}" for x in wa_anoms]
            if a.wa:
                wa = ns.was.get(a.wa)
                if wa is None:
                    reasons.append(f"no verifying record for WA {a.wa}")
                else:
                    reasons += C.wa_standing(a.repo, wa, now)
                    for ev in C.terminal_events(ns, wa):
                        reasons.append(f"{ev.payload['event']} "
                                       f"{ev.payload['event_id']} is in force")
                    d = C.derive(a.repo, ns)
                    used = {r.payload["wa_id"] for r in d.chain}
                    if a.wa in used:
                        reasons.append("already backed an admission (P-6)")
                    empty = d.state == C.REFUSE and not ns.admissions \
                        and not ns.anomalies
                    if d.state == C.PASS:
                        if wa.payload["baseline_commit"] != d.baseline:
                            reasons.append(
                                f"stale: baseline {wa.payload['baseline_commit'][:12]}"
                                f" != programme baseline {d.baseline[:12]} (P-7)")
                    elif not empty:
                        reasons.append(f"programme baseline not derivable "
                                       f"({d.state}); staleness cannot be "
                                       f"established")
    except C.CaiError as e:
        reasons.append(f"UNKNOWN: {e}")
    if reasons:
        print("REFUSE")
        for r in reasons:
            print(f"  - {r}")
        return 1
    print("PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
