#!/usr/bin/env python3
"""CAI v1.0 hostile calibration — Work Authority, HOLD/REVOKE, namespace.

Order §18: A1-A9, H1-H4, N1-N3 (N4 is an admission-namespace case and is
in test_cai_admission.py). Every case builds a FRESH real repository,
issues real signed tags, runs the SHIPPED checker as a subprocess, and
asserts the exit code AND the reason. A refusal for a different reason
does not pass the case.

Known-positive (must PASS): A1, H1, H4, N1.  Every other case is a
known-negative that must REFUSE for its own stated reason.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cai_testkit import Keys, Suite, World, run_checker  # noqa: E402

NOW = "2026-09-23T00:00:00Z"
DECLARED = ["A1", "A2", "A3", "A4", "A5", "A6", "A7", "A8", "A9",
            "H1", "H2", "H3", "H4", "N1", "N2", "N3"]


def auth(w, cfg, wa=None):
    args = ["--repo", str(w.repo), "--authority-config", str(cfg),
            "--now", NOW]
    return run_checker("check_cai_authority.py", *(args + (["--wa", wa]
                                                          if wa else [])))


def main() -> int:
    keys = Keys()
    s = Suite("test_cai_authority")
    worlds = []

    def world():
        w = World(keys)
        worlds.append(w)
        return w, keys.config(w.dir)

    try:
        w, cfg = world()
        w.issue_wa("WA1", w.B)
        rc, out = auth(w, cfg, "WA1")
        s.expect("A1", "correct authority-signed WA -> PASS", rc, out,
                 want_rc=0, must_contain=["PASS", "1 WA"])

        w, cfg = world()
        w.issue_wa("WA1", w.B, signer=None)
        rc, out = auth(w, cfg, "WA1")
        s.expect("A2", "unsigned WA -> REFUSE", rc, out, want_rc=1,
                 must_contain=["REFUSE", "signature UNSIGNED"])

        w, cfg = world()
        w.issue_wa("WA1", w.B, signer="ATTACKER")
        rc, out = auth(w, cfg, "WA1")
        ok1 = (rc == 1 and "signature WRONG_KEY" in out)
        # and a substituted key FILE cannot stand in for the pinned key
        bad = w.dir / "attacker.asc"
        bad.write_bytes(keys._gpg("--armor", "--export",
                                  keys.fpr["ATTACKER"]))
        import json
        c2 = json.loads(Path(cfg).read_text())
        c2["public_key_armored_path"] = str(bad)
        cfg2 = w.dir / "swapped.json"
        cfg2.write_text(json.dumps(c2))
        w2_rc, w2_out = auth(w, cfg2, "WA1")
        s.expect("A3", "wrong signer -> REFUSE; swapped key file -> REFUSE",
                 rc if ok1 else 99, out + w2_out, want_rc=1,
                 must_contain=["signature WRONG_KEY",
                               "does not contain exactly the pinned primary key"])

        w, cfg = world()
        w.issue_wa("WA1", w.B)
        w.tampered_copy("refs/tags/kai-wa/wa/WA1", "refs/tags/kai-wa/wa/WA1",
                        b'"control_level":"C2"', b'"control_level":"C9"')
        rc, out = auth(w, cfg, "WA1")
        s.expect("A4", "invalid signature (signed bytes altered) -> REFUSE",
                 rc, out, want_rc=1, must_contain=["signature INVALID"])

        w, cfg = world()
        w.issue_wa("WA1", w.B)
        c_missing = keys.config(w.dir, gpg_program="/nonexistent/gpg")
        rc1, out1 = auth(w, c_missing, "WA1")
        wrap = w.dir / "gpg-no-verify"
        wrap.write_text("#!/bin/sh\nfor a in \"$@\"; do case \"$a\" in "
                        "--verify) exit 2;; esac; done\nexec gpg \"$@\"\n")
        wrap.chmod(0o755)
        c_wrap = keys.config(w.dir, gpg_program=str(wrap))
        rc2, out2 = auth(w, c_wrap, "WA1")
        # (c) NOTHING pinned, over an EMPTY namespace. This is the vacuous-
        # PASS defect found in the smoke run before commit 2: "every record
        # verifies" is true of zero records. Without this sub-case the fix
        # had no execution path in the suite, and mutation calibration
        # showed its guard surviving removal.
        w3 = World(keys)
        worlds.append(w3)
        c_unpinned = keys.config(w3.dir, fingerprint=None)
        rc3, out3 = auth(w3, c_unpinned)
        s.expect("A5", "verification unavailable (no gpg; gpg that cannot "
                       "verify; nothing pinned over an empty namespace) -> "
                       "REFUSE UNKNOWN",
                 rc1 if rc1 == rc2 == rc3 else 99, out1 + out2 + out3,
                 want_rc=1,
                 must_contain=["UNKNOWN", "gpg unavailable",
                               "no GnuPG status output",
                               "no authority key is pinned"])

        w, cfg = world()
        w.issue_wa("WA1", w.B, repository_id=1)
        rc, out = auth(w, cfg, "WA1")
        w.issue_wa("WA2", w.B, repository_full_name="someone/kai-system")
        rc2, out2 = auth(w, cfg, "WA2")
        s.expect("A6", "wrong repository id / wrong full name -> REFUSE",
                 rc if rc == rc2 else 99, out + out2, want_rc=1,
                 must_contain=["repository identity mismatch",
                               "must equal 1004463473",
                               "must equal 'dainius1234/kai-system'"])

        w, cfg = world()
        p = w.wa_payload("WA1", w.B)
        del p["note"]
        w.tag("kai-wa/wa/WA1", w.B, p)
        rc, out = auth(w, cfg, "WA1")
        from cai_testkit import canonical
        noncanon = canonical(w.wa_payload("WA2", w.B)).replace(",", ", ")
        w.tag("kai-wa/wa/WA2", w.B, None, raw_message=noncanon)
        rc2, out2 = auth(w, cfg, "WA2")
        s.expect("A7", "malformed schema; non-canonical payload -> REFUSE",
                 rc if rc == rc2 else 99, out + out2, want_rc=1,
                 must_contain=["missing required field 'note'",
                               "not in canonical form"])

        w, cfg = world()
        w.issue_wa("WA1", w.B, expires_at="2020-01-01T00:00:00Z")
        rc, out = auth(w, cfg, "WA1")
        s.expect("A8", "expired WA -> REFUSE", rc, out, want_rc=1,
                 must_contain=["expired at 2020-01-01T00:00:00Z"])

        # A9 — replay and staleness both need a real admission chain
        w, cfg = world()
        w.issue_wa("WA1", w.B)
        c1 = w.change(w.B, {"allowed/a.txt": "a1\n"})
        w.admit("ADM1", 1, "WA1", c1)
        rc1, out1 = auth(w, cfg, "WA1")            # replay
        w.issue_wa("WA2", w.B)                     # stale: baseline is B, not c1
        rc2, out2 = auth(w, cfg, "WA2")
        s.expect("A9", "replayed WA (already admitted) / stale WA (baseline "
                       "!= programme baseline) -> REFUSE",
                 rc1 if rc1 == rc2 else 99, out1 + out2, want_rc=1,
                 must_contain=["already backed an admission (P-6)",
                               "stale: baseline"])

        w, cfg = world()
        w.issue_wa("WA1", w.B)
        w.issue_wa("WA2", w.B)
        w.issue_event("WA2", "HOLD1", "HOLD")
        rc, out = auth(w, cfg, "WA1")
        s.expect("H1", "active WA (HOLD exists on ANOTHER WA) -> eligible",
                 rc, out, want_rc=0, must_contain=["PASS", "1 terminal events"])

        w, cfg = world()
        w.issue_wa("WA1", w.B)
        w.issue_event("WA1", "HOLD1", "HOLD")
        rc, out = auth(w, cfg, "WA1")
        s.expect("H2", "valid signed HOLD -> REFUSE", rc, out, want_rc=1,
                 must_contain=["HOLD HOLD1 is in force"])

        w, cfg = world()
        w.issue_wa("WA1", w.B)
        w.issue_event("WA1", "RV1", "REVOKE")
        rc, out = auth(w, cfg, "WA1")
        s.expect("H3", "valid signed REVOKE -> REFUSE", rc, out, want_rc=1,
                 must_contain=["REVOKE RV1 is in force"])

        # H4 — the verifier has no chat input. A HOLD that exists only as
        # text -- a committed note, a commit message -- must change NOTHING.
        # That is the intended behaviour, and the reason CAI exists.
        w, cfg = world()
        w.issue_wa("WA1", w.B)
        w.write("HOLD.md", "KAI -> ORION: HOLD. NO FURTHER MUTATION. WA1.\n")
        w.commit("KAI -> ORION: HOLD AT WA1. NO FURTHER MUTATION.")
        rc, out = auth(w, cfg, "WA1")
        s.expect("H4", "chat/text-only HOLD, no signed event -> no mechanical "
                       "state change (WA still eligible)", rc, out, want_rc=0,
                 must_contain=["PASS", "0 terminal events"])

        w, cfg = world()
        w.issue_wa("WA1", w.B)
        rc, out = auth(w, cfg)
        s.expect("N1", "record at refs/tags/kai-wa/wa/<id> is recognised",
                 rc, out, want_rc=0, must_contain=["1 WA", "0 anomalies",
                                                   "PASS"])

        w, cfg = world()
        w.tag("kai-wa-evil/wa/WA9", w.B, w.wa_payload("WA9", w.B))
        rc, out = auth(w, cfg, "WA9")
        s.expect("N2", "kai-wa-evil/... is never enumerated: no record, no "
                       "anomaly", rc, out, want_rc=1,
                 must_contain=["no verifying record for WA WA9",
                               "0 WA", "0 anomalies"])

        w, cfg = world()
        w.issue_wa("WA1", w.B)
        w.tag("kai-wa/wa/X/extra", w.B, w.wa_payload("X", w.B))
        w.tag("kai-wa/WA/WA3", w.B, w.wa_payload("WA3", w.B))
        obj = w.issue_wa("WA4", w.B)
        w.git("update-ref", "refs/tags/kai-wa/wa/WA5", obj)   # renamed copy
        rc, out = auth(w, cfg, "WA1")
        s.expect("N3", "nested / look-alike / re-named refs inside kai-wa/ "
                       "-> anomaly -> REFUSE (P-3)", rc, out, want_rc=1,
                 must_contain=["kai-wa/wa/X/extra: look-alike",
                               "kai-wa/WA/WA3: look-alike",
                               "names itself 'kai-wa/wa/WA4' but is "
                               "published as 'kai-wa/wa/WA5'"])
    finally:
        for w in worlds:
            w.cleanup()
        keys.cleanup()
    return s.finish(DECLARED)


if __name__ == "__main__":
    sys.exit(main())
