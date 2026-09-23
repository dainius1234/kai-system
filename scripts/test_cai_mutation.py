#!/usr/bin/env python3
"""CAI v1.0 — calibrate the CALIBRATION (doctrine rule 15).

A suite that has only ever been seen to pass proves nothing about whether
it would notice a broken verifier. Each mutant below disables exactly ONE
control in a THROWAWAY COPY of the tree, then all three hostile suites run
against that copy. A mutant is KILLED when at least one of its named
target cases fails. A SURVIVING mutant is an uncalibrated control, and is
reported as one — it is not assumed covered.

Every edit asserts it matched exactly once (doctrine rule 18). A mutant
whose target text is not found is an ERROR, never a silent skip.

The working tree is never modified.
"""
from __future__ import annotations

import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SUITES = ["test_cai_authority.py", "test_cai_scope.py", "test_cai_admission.py"]

# id: (control, [(old, new), ...], target cases that MUST fail)
MUTANTS = {
    "M01": ("signature: everything is VALID_PINNED",
            [('    if tag.signature_format is None:\n        return UNSIGNED',
              '    return VALID_PINNED, "mutant"\n    if tag.signature_format is None:\n        return UNSIGNED')],
            ["A2", "A3", "A4", "D8"]),
    # Defined at the level of the PROTECTION ("an unpinned verifier
    # refuses"), not of one line. Removing only the library guard SURVIVED
    # in the first run, because the checker-level guards overlap it on
    # every input a checker can receive. A survivor of a redundant line
    # is not a calibrated protection, and neither is a claim about one.
    "M02": ("every unpinned / unavailable keyring guard removed",
            [('    if keyring.error or not keyring.home:\n        return UNKNOWN',
              '    if False:\n        return UNKNOWN'),
             ('            if kr.error:\n                reasons.append(f"UNKNOWN: {kr.error}")',
              '            if False:\n                pass'),
             ('            if kr.error:\n                d = C.Derivation(',
              '            if False:\n                d = C.Derivation(')],
            ["A5"]),
    "M03": ("WA expiry not checked",
            [('    if at_time >= parse_utc(p["expires_at"]):',
              '    if False:')],
            ["A8"]),
    "M04": ("terminal events never found",
            [('        if ev.tag.target == wa.tag.sha:\n            out.append(ev)',
              '        pass')],
            ["H2", "H3"]),
    "M05": ("forbidden scope not enforced",
            [('    if path in fp:\n        return False, "forbidden exact path"\n'
              '    if any(path.startswith(pre) for pre in fpre):\n'
              '        return False, "under a forbidden prefix"\n',
              ''),
             ('    if path in ap or any(path.startswith(pre) for pre in apre):',
              '    if path in ap or any(path.startswith(pre) for pre in apre + fpre):')],
            ["S3", "S4"]),
    "M06": ("symlink / gitlink modes not refused",
            [('            if mode == "120000":', '            if False:'),
             ('            if mode == "160000":', '            if False:')],
            ["S5", "S6"]),
    "M07": ("path ambiguity not refused",
            [('def path_problems(raw: bytes) -> Tuple[Optional[str], List[str]]:\n',
              'def path_problems(raw: bytes) -> Tuple[Optional[str], List[str]]:\n'
              '    return raw.decode("utf-8", "replace"), []\n')],
            ["S7"]),
    "M08": ("prefix boundary: prefixes need not end in '/'",
            [('    if path in ap or any(path.startswith(pre) for pre in apre):',
              '    if path in ap or any(path.startswith(pre.rstrip("/")) for pre in apre):')],
            ["S2"]),
    "M09": ("fork taken as the first sibling (a max/latest proxy)",
            [('        if len(succ) > 1:', '        if False:')],
            ["D4"]),
    "M10": ("orphans (skipped / wrong predecessor) ignored",
            [('    orphans = [a for a in adm if a not in chain]',
              '    orphans = []')],
            ["D3", "D5"]),
    "M11": ("IV&V: evidence candidate not compared to admission target",
            [('                if ev["candidate_commit"] != a.tag.target:',
              '                if False:')],
            ["V2", "V3"]),
    "M12": ("IV&V: digest not compared",
            [('        if sha256_hex(blob) != p["ivv_evidence_sha256"]:\n'
              '            r.append("IV&V evidence digest mismatch")\n        else:',
              '        if True:')],
            ["V5"]),
    "M13": ("admission tag target not compared to candidate_commit",
            [('    if a.tag.target != p["candidate_commit"]:',
              '    if False:')],
            ["D7"]),
    "M14": ("canonical-form requirement dropped",
            [('    if canonical(obj) != text:', '    if False:')],
            ["A7"]),
    "M15": ("repository identity: explicit check AND schema const removed",
            [('    if payload.get("repository_id") != REPOSITORY_ID or \\\n'
              '            payload.get("repository_full_name") != REPOSITORY_FULL_NAME:',
              '    if False:'),
             ('    if "const" in schema and inst != schema["const"]:',
              '    if False:')],
            ["A6", "D6"]),
    "M16": ("look-alike refs silently skipped",
            [('                ns.anomalies.append(f"{ref}: look-alike — does not match the "\n'
              '                                    f"record grammar of its namespace")\n',
              '')],
            ["N3", "N4"]),
    "M17": ("tag object name not compared to its ref",
            [('    if tag.tag_name != short:', '    if False:')],
            ["N3"]),
    "M18": ("replay (single use) not enforced in the authority check",
            [('                    if a.wa in used:', '                    if False:')],
            ["A9"]),
    "M19": ("staleness not enforced in the authority check",
            [('                        if wa.payload["baseline_commit"] != d.baseline:',
              '                        if False:')],
            ["A9"]),
    # M20 kills on REASON TEXT, not on disposition: with the explicit check
    # removed the schema const still refuses the record, but A6/D6 assert
    # the words "repository identity mismatch", which only the explicit
    # check emits. It proves that check produces the stated reason; it does
    # NOT prove the check is necessary for protection. Stated, not implied.
    "M20": ("repository identity: explicit check only (schema const kept)",
            [('    if payload.get("repository_id") != REPOSITORY_ID or \\\n'
              '            payload.get("repository_full_name") != REPOSITORY_FULL_NAME:',
              '    if False:')],
            ["A6", "D6"]),
    # ── controls the order's MINIMUM matrix does not reach: the PROPOSED
    #    choice points P-2..P-10 and the remaining admission predicates.
    #    Targets are the supplementary X-cases in test_cai_admission.py.
    "M21": ("P-2: terminal event never dominates an admission",
            [('        if p["sequence"] > ev.payload["admission_sequence_at_issue"]:',
              '        if False:')],
            ["X1"]),
    "M22": ("P-5: WA expiry not evaluated at the admission's tagger time",
            [('    if at_time >= parse_utc(p["expires_at"]):',
              '    if at_time >= parse_utc(p["expires_at"]) and at_time < 0:')],
            ["X2", "A8"]),
    "M23": ("P-7: admission-level staleness not enforced",
            [('    if prev is not None and w["baseline_commit"] != \\\n'
              '            prev.payload["candidate_commit"]:',
              '    if False:')],
            ["X3"]),
    "M24": ("P-8: ancestry not enforced",
            [('        if not is_ancestor(repo, w["baseline_commit"], p["candidate_commit"]):',
              '        if False:')],
            ["X4"]),
    "M25": ("scope not enforced at admission",
            [('        r += [f"scope ({n} changed entries): {x}" for x in why]',
              '        pass')],
            ["X5"]),
    "M26": ("governing_refs digest not compared",
            [('            if got != g["sha256"]:', '            if False:')],
            ["X6"]),
    "M27": ("IV&V disposition not required to be ACCEPT_FOR_ADMISSION",
            [('                if ev["disposition"] != "ACCEPT_FOR_ADMISSION":',
              '                if False:')],
            ["X7"]),
    "M28": ("IV&V evidence may name another work authority",
            [('                if ev["wa_id"] != p["wa_id"]:', '                if False:')],
            ["X8"]),
    "M29": ("genesis rule (sequence 1 <=> null predecessor) not enforced",
            [('    if bad:\n        return Derivation(REFUSE, [f"genesis rule violated',
              '    if False:\n        return Derivation(REFUSE, [f"genesis rule violated')],
            ["X9"]),
    # M22 turns expiry OFF, so it is killed by X2's "expired must refuse"
    # half. P-5's actual claim is WHICH CLOCK: this mutant swaps the signed
    # tagger time for the wall clock, which only X2's positive twin (expires
    # after the tagger time, before today) can detect.
    "M30": ("P-5: admission expiry judged by the WALL CLOCK",
            [('              for x in wa_standing(repo, wa, a.tag.tagger_time)]',
              '              for x in wa_standing(repo, wa, int(time.time()))]')],
            ["X2"]),
}


def copy_tree(dst: Path):
    for rel in ("scripts", "kai-pm/change-control"):
        shutil.copytree(ROOT / rel, dst / rel,
                        ignore=shutil.ignore_patterns("__pycache__"))


def apply(dst: Path, edits):
    for old, new in edits:
        hits = []
        for fn in ("scripts/security/cai_lib.py",
                   "scripts/security/check_cai_authority.py",
                   "scripts/security/check_cai_admission.py"):
            p = dst / fn
            if p.read_text().count(old):
                hits.append((p, p.read_text().count(old)))
        if len(hits) != 1 or hits[0][1] != 1:
            raise SystemExit(f"MUTANT EDIT NOT UNIQUE: {old[:60]!r} -> {hits}")
        p = hits[0][0]
        p.write_text(p.read_text().replace(old, new, 1))


def run_suites(dst: Path):
    failed = set()
    for s in SUITES:
        pr = subprocess.run([sys.executable, str(dst / "scripts" / s)],
                            capture_output=True, text=True, timeout=1200)
        failed |= set(re.findall(r"^  (\S+)\s+FAIL", pr.stdout, re.M))
        if pr.returncode not in (0, 1):
            failed.add(f"{s}:CRASH")
    return failed


def main() -> int:
    rows, survivors = [], []
    base = Path(tempfile.mkdtemp(prefix="cai-mut-"))
    try:
        ctl = base / "control"
        copy_tree(ctl)
        control_failed = run_suites(ctl)
        print(f"  CONTROL (unmutated copy): failing cases = "
              f"{sorted(control_failed) or 'none'}")
        if control_failed:
            print("  ABORT: the unmutated copy does not pass; no mutant result "
                  "would be attributable (R11).")
            return 1
        only = set(sys.argv[1:])            # optional: run a named subset
        for mid, (what, edits, targets) in MUTANTS.items():
            if only and mid not in only:
                continue
            d = base / mid
            copy_tree(d)
            apply(d, edits)
            failed = run_suites(d)
            hit = sorted(set(targets) & failed)
            killed = bool(hit)
            rows.append((mid, killed))
            if not killed:
                survivors.append(mid)
            print(f"  {mid} {'KILLED  ' if killed else 'SURVIVED'} {what}")
            print(f"        targets {targets}  failed {sorted(failed) or '-'}")
    finally:
        shutil.rmtree(base, ignore_errors=True)
    print(f"\n  mutants declared {len(MUTANTS)} (selected "
          f"{sys.argv[1:] or 'all'}), executed {len(rows)}, "
          f"killed {len(rows) - len(survivors)}, survived {len(survivors)}")
    if survivors:
        print(f"  SURVIVORS (uncalibrated controls): {survivors}")
    expected = len(set(sys.argv[1:]) & set(MUTANTS)) if sys.argv[1:] \
        else len(MUTANTS)
    return 1 if survivors or len(rows) != expected else 0


if __name__ == "__main__":
    sys.exit(main())
