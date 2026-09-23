#!/usr/bin/env python3
"""CAI v1.0 hostile calibration — IV&V binding, admission chain, admission
namespace (order §15, §5, §18: V1-V5, D1-D8, N4).

Every case: fresh real repository, real signed WA / admission tags, a real
committed IV&V evidence file resolved by digest, the SHIPPED
check_cai_admission.py run as a subprocess, exit code AND reason asserted.

Known-positive: V1, D1, D2 (the baseline must be the exact expected
commit, not merely "a PASS"). Everything else is a known-negative.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cai_testkit import Keys, Suite, World, run_checker  # noqa: E402

DECLARED = ["V1", "V2", "V3", "V4", "V5",
            "D1", "D2", "D3", "D4", "D5", "D6", "D7", "D8", "N4",
            # SUPPLEMENTARY, beyond the order's §18 minimum. They calibrate
            # the contract's PROPOSED choice points (P-n) and the remaining
            # admission predicates, which mutation calibration showed had
            # NO executing case. They carry the same standing as the P-n
            # they test: proposed, for Kai IV&V.
            "X1", "X2", "X3", "X4", "X5", "X6", "X7", "X8", "X9"]


def derive(w, cfg):
    return run_checker("check_cai_admission.py", "--repo", str(w.repo),
                       "--authority-config", str(cfg))


def main() -> int:
    keys = Keys()
    s = Suite("test_cai_admission")
    worlds = []

    def genesis_world():
        """B -WA1-> C1 admitted at sequence 1. Returns (w, cfg, c1, adm1)."""
        w = World(keys)
        worlds.append(w)
        w.issue_wa("WA1", w.B)
        c1 = w.change(w.B, {"allowed/a.txt": "a1\n"}, "candidate C1")
        return w, keys.config(w.dir), c1

    try:
        # ── IV&V binding ─────────────────────────────────────────────
        w, cfg, c1 = genesis_world()
        w.admit("ADM1", 1, "WA1", c1)
        rc, out = derive(w, cfg)
        s.expect("V1", "exact candidate + its evidence -> PASS, baseline is "
                       "that exact commit", rc, out, want_rc=0,
                 must_contain=["PASS", f"programme baseline: {c1}"])

        w, cfg, c1 = genesis_world()
        c2 = w.change(c1, {"allowed/b.txt": "b2\n"}, "a DIFFERENT candidate")
        ev = w.evidence("WA1", c1)                 # Kai reviewed c1
        w.admit("ADM1", 1, "WA1", c2, ev=ev)       # admission names c2
        rc, out = derive(w, cfg)
        s.expect("V2", "admission of a different candidate SHA than IV&V "
                       "reviewed -> REFUSE", rc, out, want_rc=1,
                 must_contain=["admission target is not the exact commit "
                               "IV&V reviewed"])

        w, cfg, c1 = genesis_world()
        twin = w.same_tree_new_commit(c1)
        ev = w.evidence("WA1", c1)
        w.admit("ADM1", 1, "WA1", twin, ev=ev)
        rc, out = derive(w, cfg)
        s.expect("V3", "same tree, different commit object -> REFUSE", rc, out,
                 want_rc=1, must_contain=["admission target is not the exact "
                                          "commit IV&V reviewed"])

        w, cfg, c1 = genesis_world()
        ref, digest = w.evidence("WA1", c1)
        missing = dict(ref, path="ivv/does-not-exist.json")
        w.admit("ADM1", 1, "WA1", c1, ev=(missing, digest))
        rc, out = derive(w, cfg)
        s.expect("V4", "IV&V evidence missing -> REFUSE", rc, out, want_rc=1,
                 must_contain=["IV&V evidence unavailable",
                               "ivv/does-not-exist.json not found"])

        w, cfg, c1 = genesis_world()
        ref, digest = w.evidence("WA1", c1)
        w.admit("ADM1", 1, "WA1", c1, ev=(ref, "0" * 64))
        rc, out = derive(w, cfg)
        s.expect("V5", "IV&V evidence digest mismatch -> REFUSE", rc, out,
                 want_rc=1, must_contain=["IV&V evidence digest mismatch"])

        # ── the chain ────────────────────────────────────────────────
        w, cfg, c1 = genesis_world()
        w.admit("ADM1", 1, "WA1", c1)
        rc, out = derive(w, cfg)
        s.expect("D1", "valid genesis -> PASS", rc, out, want_rc=0,
                 must_contain=["chain length: 1", f"programme baseline: {c1}"])

        def successor_world():
            w, cfg, c1 = genesis_world()
            a1 = w.admit("ADM1", 1, "WA1", c1)
            w.issue_wa("WA2", c1)
            c2 = w.change(c1, {"allowed/b.txt": "b2\n"}, "candidate C2")
            return w, cfg, c1, a1, c2

        w, cfg, c1, a1, c2 = successor_world()
        w.admit("ADM2", 2, "WA2", c2, prev=a1)
        rc, out = derive(w, cfg)
        s.expect("D2", "valid successor -> PASS, baseline advances to C2",
                 rc, out, want_rc=0,
                 must_contain=["chain length: 2", f"programme baseline: {c2}"])

        w, cfg, c1, a1, c2 = successor_world()
        w.admit("ADM3", 3, "WA2", c2, prev=a1)
        rc, out = derive(w, cfg)
        s.expect("D3", "skipped sequence (1 -> 3) -> REFUSE", rc, out,
                 want_rc=1, must_contain=["REFUSE", "sequence 3 is not on "
                                                    "the chain"])

        w, cfg, c1, a1, c2 = successor_world()
        w.admit("ADM2a", 2, "WA2", c2, prev=a1)
        w.issue_wa("WA3", c1)
        c3 = w.change(c1, {"allowed/b.txt": "b3\n"}, "sibling candidate")
        w.admit("ADM2b", 2, "WA3", c3, prev=a1)
        rc, out = derive(w, cfg)
        s.expect("D4", "two valid siblings at sequence 2 -> UNRESOLVED, no "
                       "baseline chosen", rc, out, want_rc=1,
                 must_contain=["UNRESOLVED", "fork at sequence 2",
                               "none is chosen"])

        w, cfg, c1, a1, c2 = successor_world()
        w.admit("ADM2", 2, "WA2", c2, prev="e" * 40)
        rc, out = derive(w, cfg)
        s.expect("D5", "wrong predecessor tag object -> REFUSE", rc, out,
                 want_rc=1, must_contain=["sequence 2 is not on the chain"])

        w, cfg, c1 = genesis_world()
        w.admit("ADM1", 1, "WA1", c1, repository_id=42)
        rc, out = derive(w, cfg)
        s.expect("D6", "admission for another repository -> REFUSE", rc, out,
                 want_rc=1, must_contain=["repository identity mismatch"])

        w, cfg, c1 = genesis_world()
        c2 = w.change(c1, {"allowed/b.txt": "b2\n"}, "other commit")
        ev = w.evidence("WA1", c1)
        # the payload names the reviewed c1, but the TAG points at c2
        w.admit("ADM1", 1, "WA1", c2, candidate=c1, ev=ev)
        rc, out = derive(w, cfg)
        s.expect("D7", "admission tag target != IV&V candidate -> REFUSE",
                 rc, out, want_rc=1,
                 must_contain=["admission tag target != candidate_commit",
                               "admission target is not the exact commit "
                               "IV&V reviewed"])

        w, cfg, c1 = genesis_world()
        w.admit("ADM1", 1, "WA1", c1, signer=None)
        rc1, out1 = derive(w, cfg)
        w2, cfg2, c1b = genesis_world()
        w2.admit("ADM1", 1, "WA1", c1b, signer="ATTACKER")
        rc2, out2 = derive(w2, cfg2)
        s.expect("D8", "unsigned admission / wrong-signer admission -> REFUSE",
                 rc1 if rc1 == rc2 else 99, out1 + out2, want_rc=1,
                 must_contain=["signature UNSIGNED", "signature WRONG_KEY"])

        # ── admission namespace ──────────────────────────────────────
        w, cfg, c1 = genesis_world()
        a1 = w.admit("ADM1", 1, "WA1", c1)
        # outside the namespace: never enumerated, cannot move the baseline
        w.issue_wa("WA2", c1)
        c2 = w.change(c1, {"allowed/b.txt": "b2\n"})
        ev = w.evidence("WA2", c2)
        w.tag("kai-admitted-x/ADM2", c2, None, raw_message="ignored\n")
        rc_ign, out_ign = derive(w, cfg)
        # inside the namespace but off-grammar: an anomaly, REFUSE (P-3)
        w.tag("kai-admitted/nested/ADM2", c2, None, raw_message="nested\n")
        rc, out = derive(w, cfg)
        ok_ignored = (rc_ign == 0 and f"programme baseline: {c1}" in out_ign)
        s.expect("N4", "kai-admitted-x/... ignored (baseline unchanged); "
                       "kai-admitted/nested/... -> REFUSE", rc if ok_ignored
                 else 99, out, want_rc=1,
                 must_contain=["kai-admitted/nested/ADM2: look-alike"])
        # ── SUPPLEMENTARY X-cases ────────────────────────────────────
        import hashlib

        # X1 · P-2 — a terminal event dominates admissions AFTER its issue
        #      position, and does not retroactively undo earlier ones.
        w, cfg, c1 = genesis_world()
        w.issue_event("WA1", "HOLD0", "HOLD", seq_at_issue=0)
        w.admit("ADM1", 1, "WA1", c1)
        rc_a, out_a = derive(w, cfg)
        w2, cfg2, c1b = genesis_world()
        w2.admit("ADM1", 1, "WA1", c1b)
        w2.issue_event("WA1", "HOLD1", "HOLD", seq_at_issue=1)
        rc_b, out_b = derive(w2, cfg2)
        s.expect("X1", "P-2: HOLD issued at position 0 dominates admission 1; "
                       "HOLD issued at position 1 leaves admission 1 intact",
                 rc_a if rc_b == 0 else 99, out_a + out_b, want_rc=1,
                 must_contain=["dominated by HOLD HOLD0",
                               f"programme baseline: {c1b}"])

        # X2 · P-5 — expiry is judged at the admission's SIGNED tagger time
        #      (fixed 2026-09-21), never the wall clock (2026-09-23 here).
        w = World(keys); worlds.append(w); cfg = keys.config(w.dir)
        w.issue_wa("WA1", w.B, expires_at="2026-09-20T00:00:00Z")
        c1 = w.change(w.B, {"allowed/a.txt": "a1\n"})
        w.admit("ADM1", 1, "WA1", c1)
        rc_a, out_a = derive(w, cfg)
        w2 = World(keys); worlds.append(w2); cfg2 = keys.config(w2.dir)
        w2.issue_wa("WA1", w2.B, expires_at="2026-09-22T00:00:00Z")
        c1b = w2.change(w2.B, {"allowed/a.txt": "a1\n"})
        w2.admit("ADM1", 1, "WA1", c1b)
        rc_b, out_b = derive(w2, cfg2)
        s.expect("X2", "P-5: WA expired before the admission tagger time -> "
                       "REFUSE; expiring after it (but before today) -> PASS",
                 rc_a if rc_b == 0 else 99, out_a + out_b, want_rc=1,
                 must_contain=["expired at 2026-09-20T00:00:00Z",
                               f"programme baseline: {c1b}"])

        # X3 · P-7 — admission-level staleness: WA2 is based on B, not on
        #      the admitted c1, while ancestry and scope both hold.
        w, cfg, c1 = genesis_world()
        a1 = w.admit("ADM1", 1, "WA1", c1)
        w.issue_wa("WA2", w.B)
        c2 = w.change(c1, {"allowed/b.txt": "b2\n"})
        w.admit("ADM2", 2, "WA2", c2, prev=a1)
        rc, out = derive(w, cfg)
        s.expect("X3", "P-7: successor WA not based on the previous admitted "
                       "candidate -> REFUSE", rc, out, want_rc=1,
                 must_contain=["stale work authority"])

        # X4 · P-8 — the candidate must DESCEND from the WA baseline. An
        #      in-scope tree on an unrelated root commit does not.
        w, cfg, c1 = genesis_world()
        stray = w.git("commit-tree", w.tree(c1), "-m", "no parent").strip()
        w.admit("ADM1", 1, "WA1", stray)
        rc, out = derive(w, cfg)
        s.expect("X4", "P-8: candidate not descended from the baseline -> "
                       "REFUSE", rc, out, want_rc=1,
                 must_contain=["does not descend from the WA baseline"])

        # X5 — scope is enforced AT ADMISSION, not only by the scope CLI
        w, cfg, c1 = genesis_world()
        cx = w.change(w.B, {"docs/readme.md": "outside\n"})
        w.admit("ADM1", 1, "WA1", cx)
        rc, out = derive(w, cfg)
        s.expect("X5", "admission of an out-of-scope candidate -> REFUSE", rc,
                 out, want_rc=1,
                 must_contain=["docs/readme.md (M): outside every allowed"])

        # X6 — governing_refs are RESOLVED and HASHED, not trusted as fields
        good = {"kind": "git-blob", "path": "docs/readme.md",
                "sha256": hashlib.sha256(b"r\n").hexdigest()}
        w = World(keys); worlds.append(w); cfg = keys.config(w.dir)
        w.issue_wa("WA1", w.B, governing_refs=[dict(good, commit=w.B,
                                                    sha256="0" * 64)])
        c1 = w.change(w.B, {"allowed/a.txt": "a1\n"})
        w.admit("ADM1", 1, "WA1", c1)
        rc_a, out_a = derive(w, cfg)
        w2 = World(keys); worlds.append(w2); cfg2 = keys.config(w2.dir)
        w2.issue_wa("WA1", w2.B, governing_refs=[dict(good, commit=w2.B)])
        c1b = w2.change(w2.B, {"allowed/a.txt": "a1\n"})
        w2.admit("ADM1", 1, "WA1", c1b)
        rc_b, out_b = derive(w2, cfg2)
        s.expect("X6", "governing ref digest wrong -> REFUSE; correct -> PASS",
                 rc_a if rc_b == 0 else 99, out_a + out_b, want_rc=1,
                 must_contain=["governing ref docs/readme.md digest mismatch",
                               f"programme baseline: {c1b}"])

        # X7 — IV&V must actually ACCEPT; a REJECT record cannot admit
        w, cfg, c1 = genesis_world()
        ev = w.evidence("WA1", c1, disposition="REJECT")
        w.admit("ADM1", 1, "WA1", c1, ev=ev)
        rc, out = derive(w, cfg)
        s.expect("X7", "IV&V disposition REJECT -> REFUSE", rc, out,
                 want_rc=1, must_contain=["IV&V disposition is REJECT"])

        # X8 — evidence for a different work authority cannot be borrowed
        w, cfg, c1 = genesis_world()
        ev = w.evidence("WA9", c1)
        w.admit("ADM1", 1, "WA1", c1, ev=ev)
        rc, out = derive(w, cfg)
        s.expect("X8", "IV&V evidence names another WA -> REFUSE", rc, out,
                 want_rc=1,
                 must_contain=["IV&V evidence is for another work authority"])

        # X9 — genesis rule: sequence 1 <=> null predecessor
        w, cfg, c1 = genesis_world()
        w.admit("ADM1", 1, "WA1", c1, prev="a" * 40)
        rc, out = derive(w, cfg)
        s.expect("X9", "sequence 1 with a predecessor -> REFUSE", rc, out,
                 want_rc=1, must_contain=["genesis rule violated"])
    finally:
        for w in worlds:
            w.cleanup()
        keys.cleanup()
    return s.finish(DECLARED)


if __name__ == "__main__":
    sys.exit(main())
