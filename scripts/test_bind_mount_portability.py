#!/usr/bin/env python3
"""The bind-mount gate must fire on a local path and only on a local path.

Calibration figure asserted here, not left in a comment: against the tree
before the 2026-08-07 fix the gate reports exactly **2** — `git-watcher`
and `heartbeat`. A detector that stops detecting reports a comfortable
zero, and a ratchet cannot tell that apart from success (I-7). This test
can.
"""
from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from scripts.security.check_bind_mount_portability import (  # noqa: E402
    audit, is_machine_specific)

#: The commit before the machine-specific mounts were fixed. The gate
#: must report exactly 2 against this tree — the known answer.
_BEFORE_FIX = "f50715b"

PASSED = 0
FAILED = 0


def check(label: str, condition: bool) -> None:
    global PASSED, FAILED
    if condition:
        PASSED += 1
        print(f"  ok    {label}")
    else:
        FAILED += 1
        print(f"  FAIL  {label}")


def write(tmp: Path, name: str, body: str) -> None:
    (tmp / name).write_text(textwrap.dedent(body), encoding="utf-8")


def main() -> int:
    import tempfile

    # ── the predicate, directly ──
    check("a developer home path is machine-specific",
          is_machine_specific("/home/user/kai-system"))
    check("an arbitrary absolute path is machine-specific",
          is_machine_specific("/var/log/sovereign"))
    check("/opt is machine-specific too", is_machine_specific("/opt/models"))
    check("the docker socket is NOT flagged",
          not is_machine_specific("/var/run/docker.sock"))
    check("/dev/net/tun is NOT flagged",
          not is_machine_specific("/dev/net/tun"))
    check("/proc is NOT flagged", not is_machine_specific("/proc/self"))
    check("/sys is NOT flagged", not is_machine_specific("/sys/class/thermal"))
    check("/run is NOT flagged", not is_machine_specific("/run/user/1000"))
    check("a named volume is NOT flagged",
          not is_machine_specific("postgres_data"))
    check("a relative path is NOT flagged", not is_machine_specific("."))
    check("a ./-prefixed relative path is NOT flagged",
          not is_machine_specific("./runtime-logs"))
    check("a ${VAR}-prefixed path is NOT flagged",
          not is_machine_specific("${SOVEREIGN_LOG_DIR:-./x}"))

    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)

        # ── I-3: inject a violation, assert it fires ──
        write(tmp, "docker-compose.bad.yml", """
            services:
              watcher:
                build: {context: ., dockerfile: w/Dockerfile}
                volumes:
                  - /home/somebody/repo:/workspace:ro
            """)
        findings, checked, files = audit(tmp)
        check("a local path produces a finding", len(findings) == 1)
        check("the finding names the service", "watcher" in findings[0])
        check("the finding names the path",
              "/home/somebody/repo" in findings[0])
        check("the finding explains the EMPTY DIRECTORY behaviour",
              "EMPTY DIRECTORY" in findings[0])
        check("the denominator counts the declaration", checked == 1)

        # ── the exemptions, through the real audit path ──
        write(tmp, "docker-compose.bad.yml", """
            services:
              a:
                build: {context: ., dockerfile: a/Dockerfile}
                volumes:
                  - /var/run/docker.sock:/var/run/docker.sock:ro
                  - /dev/net/tun:/dev/net/tun
                  - named_vol:/data
                  - ./logs:/logs:ro
            """)
        findings, checked, _ = audit(tmp)
        check("portable mounts produce no findings", findings == [])
        check("but they are still counted in the denominator", checked == 4)

        # ── I-1: no compose files is a finding, not a pass ──
        with tempfile.TemporaryDirectory() as empty:
            findings, checked, files = audit(Path(empty))
            check("an empty tree REFUSES rather than passing",
                  len(findings) == 1 and "inspected nothing" in findings[0])
            check("an empty tree reports zero inspected", checked == 0)

        # ── long-form volume syntax must not crash it ──
        write(tmp, "docker-compose.bad.yml", """
            services:
              a:
                build: {context: ., dockerfile: a/Dockerfile}
                volumes:
                  - type: bind
                    source: /home/somebody/x
                    target: /x
            """)
        findings, checked, _ = audit(tmp)
        check("long-form syntax is skipped, not crashed on", checked == 0)

    # ── CALIBRATION against a known answer, run rather than described ──
    # The tree at the commit before the fix must report exactly 2:
    # git-watcher and heartbeat. Read from git rather than mutating the
    # working tree, so this is safe to run at any time.
    calib = Path(tempfile.mkdtemp())
    known = 0
    for name in ("docker-compose.minimal.yml", "docker-compose.sovereign.yml",
                 "docker-compose.full.yml"):
        blob = subprocess.run(["git", "show", f"{_BEFORE_FIX}:{name}"],
                              cwd=REPO, capture_output=True, text=True)
        if blob.returncode == 0:
            (calib / name).write_text(blob.stdout, encoding="utf-8")
            known += 1
    if known == 3:
        pre, pre_checked, _ = audit(calib)
        check("CALIBRATION: the pre-fix tree reports exactly 2",
              len(pre) == 2)
        check("CALIBRATION: it names git-watcher",
              any("git-watcher" in f for f in pre))
        check("CALIBRATION: it names heartbeat",
              any("heartbeat" in f for f in pre))
        check("CALIBRATION: the denominator is unchanged by the fix",
              pre_checked == 35)
    else:
        check(f"CALIBRATION: could not read {_BEFORE_FIX} — "
              f"got {known}/3 files (rebase or shallow clone?)", False)

    findings, checked, files = audit(REPO)
    check("the live tree is clean", findings == [])
    check("the live tree still has mounts to inspect", checked >= 30)
    check("it traverses every compose file", files == 3)

    print("=" * 60)
    print(f"Bind-mount portability tests: {PASSED} passed, {FAILED} failed")
    print(f"EXIT GATE: {'PASS' if FAILED == 0 else 'FAIL'}")
    return 1 if FAILED else 0


if __name__ == "__main__":
    print("Bind-mount portability tests")
    print("=" * 60)
    sys.exit(main())
