#!/usr/bin/env python3
"""CAI-BOOT-001 evidence capture. Re-runnable.

Runs every CAI suite, the mutation calibration and the relevant existing
gates as SEPARATE child processes and writes each one's COMPLETE output
to its own file. Every status below is subprocess.CompletedProcess.
returncode of the child that produced that file -- never a shell or
pipeline status (INC-35), never parsed from text. Output is kept whole
(R10); the summary states each file's byte count.

  python3 kai-pm/change-control/evidence/capture.py
"""
from __future__ import annotations

import datetime
import hashlib
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent / "runs"

STEPS = [
    ("test_cai_authority", [sys.executable, "scripts/test_cai_authority.py"]),
    ("test_cai_scope", [sys.executable, "scripts/test_cai_scope.py"]),
    ("test_cai_admission", [sys.executable, "scripts/test_cai_admission.py"]),
    ("test_cai_bootstrap", [sys.executable, "scripts/test_cai_bootstrap.py"]),
    ("test_cai_mutation", [sys.executable, "scripts/test_cai_mutation.py"]),
    ("bootstrap_design", [sys.executable, "scripts/security/check_cai_bootstrap.py",
                          "--design", "kai-pm/change-control/rulesets"]),
    ("authority_real_repo", [sys.executable, "scripts/security/check_cai_authority.py",
                             "--repo", ".", "--authority-config",
                             "kai-pm/change-control/authority_keys.json",
                             "--now", "2026-09-23T00:00:00Z"]),
    ("admission_real_repo", [sys.executable, "scripts/security/check_cai_admission.py",
                             "--repo", ".", "--authority-config",
                             "kai-pm/change-control/authority_keys.json"]),
    ("gate_registry", [sys.executable, "scripts/security/check_gate_registry.py", "--gate"]),
    ("test_wiring", [sys.executable, "scripts/security/check_test_wiring.py"]),
]


def main() -> int:
    OUT.mkdir(exist_ok=True)
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                          capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT,
                           capture_output=True, text=True).stdout
    lines = [f"CAI-BOOT-001 evidence capture",
             f"captured {datetime.datetime.now(datetime.timezone.utc).isoformat()}",
             f"HEAD {head}",
             f"working tree {'CLEAN' if not dirty else 'DIRTY: ' + ' | '.join(dirty.splitlines())}",
             "", f"{'step':<22} {'rc':>3} {'bytes':>8}  sha256 (first 16)", ""]
    for name, cmd in STEPS:
        pr = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True,
                            timeout=3600)
        body = (f"$ {' '.join(cmd)}\n\n{pr.stdout}{pr.stderr}\n"
                f"process exit status = {pr.returncode}\n")
        f = OUT / f"{name}.txt"
        f.write_text(body)
        b = f.read_bytes()
        lines.append(f"{name:<22} {pr.returncode:>3} {len(b):>8}  "
                     f"{hashlib.sha256(b).hexdigest()[:16]}")
        print(lines[-1], flush=True)
    (OUT / "SUMMARY.txt").write_text("\n".join(lines) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
