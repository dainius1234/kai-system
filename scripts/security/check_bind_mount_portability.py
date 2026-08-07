#!/usr/bin/env python3
"""A bind mount must not name a path that exists on one machine.

The defect
----------

`docker-compose.minimal.yml` declared, for `git-watcher`:

    volumes:
      - /home/user/kai-system:/workspace:ro

and `docker-compose.sovereign.yml`, for `heartbeat`:

    volumes:
      - /var/log/sovereign:/var/log/sovereign:ro

Both are absolute host paths that exist on exactly one machine.

**Docker creates a missing bind-mount source as an empty directory
rather than failing.** So on any other host — a CI runner, a colleague's
laptop, a real deployment — the mount succeeds, the container starts, the
healthcheck passes, and the service reads nothing. It is the worst
available failure mode: a green service doing no work.

`heartbeat` proved it. Its `_scan_executor_log()` read
`/var/log/sovereign/executor.log`, returned `0` when the file was absent,
and `/status` published that zero as `intrusion_hits`. It scans for
`timeout`, `blocked` and `injection`. The sovereign profile has been
booting green in 11 seconds while its intrusion monitor was structurally
unable to see an intrusion.

Scope, and what is deliberately NOT flagged
-------------------------------------------

Only absolute host paths. A named volume is Docker's to place, and a
relative path resolves against the compose file's own directory on every
host, so both are portable by construction.

Absolute paths under a **kernel or daemon interface** are exempt, by
prefix rather than by a list of names:

    /proc  /sys  /dev  /run  /var/run

`/var/run/docker.sock` and `/dev/net/tun` are correct — they name the
same thing on every Linux host, which is the property this gate is
actually about. Exempting them by *prefix* means a new one needs no
edit here; exempting them by name would be the list-beside-the-thing
defect this repository has spent a fortnight removing.

Calibration
-----------

Against the tree at the commit before the fix, this reports exactly 2:
`git-watcher` and `heartbeat`. `scripts/test_bind_mount_portability.py`
asserts that figure, so a detector that stops detecting fails its own
test rather than reporting a comfortable zero.

Exit 0 = every bind mount names something that exists everywhere.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Tuple

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO))

from scripts.security.gate_inputs import compose_files, inspected  # noqa: E402

#: Kernel and daemon interfaces. Identical on every Linux host, so an
#: absolute path here is a portable reference rather than a local one.
_PORTABLE_PREFIXES = ("/proc", "/sys", "/dev", "/run", "/var/run")


def is_machine_specific(host: str) -> bool:
    """True when *host* names a path that exists on one machine.

    Written as a positive test on purpose. The first draft asked `if
    host.startswith('/home')`, which is a guess at where people keep
    things; this asks whether the path is one the kernel guarantees, and
    treats everything else as local. The question "is this portable?"
    has a small answerable set; "is this local?" does not.
    """
    if not host.startswith("/"):
        return False        # named volume or relative path — portable
    if host.startswith(_PORTABLE_PREFIXES):
        return False
    return True


def audit(root: Path = None) -> Tuple[List[str], int, int]:
    """Return (findings, volume declarations inspected, compose files)."""
    import yaml

    root = root or REPO
    files = compose_files(root)
    if not files:
        # I-1: no input is not a clean bill of health.
        return ([f"{root}: no compose files found — this gate inspected "
                 f"nothing and must not report success"], 0, 0)

    findings: List[str] = []
    checked = 0
    for path in files:
        try:
            doc = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        except Exception:
            continue        # `check_ci_tolerations` owns unparseable files
        for name, cfg in (doc.get("services") or {}).items():
            for vol in ((cfg or {}).get("volumes") or []):
                if not isinstance(vol, str):
                    continue    # long-form syntax; source is explicit there
                checked += 1
                host = vol.split(":")[0]
                if not is_machine_specific(host):
                    continue
                findings.append(
                    f"{path.name}: `{name}` bind-mounts `{host}`, an "
                    f"absolute host path that exists on one machine. Docker "
                    f"creates a missing bind-mount source as an EMPTY "
                    f"DIRECTORY rather than failing, so anywhere else this "
                    f"service starts healthy and reads nothing. Use a "
                    f"relative path (resolved against the compose file) or "
                    f"a named volume.")
    return findings, checked, len(files)


def main() -> int:
    findings, checked, files = audit()

    print(inspected(checked, "bind mount(s) and volume declaration(s)",
                    f"across {files} compose file(s)"))
    print()

    if findings:
        print(f"FAIL: {len(findings)} mount(s) name a path that exists on "
              f"one machine:\n")
        for line in findings:
            print(f"  - {line}")
        print("\n  heartbeat mounted /var/log/sovereign and scanned it for "
              "`timeout`,\n  `blocked` and `injection`. On every host but "
              "one the directory was\n  empty, the scan returned 0, and "
              "/status published that zero as\n  `intrusion_hits`. The "
              "sovereign profile was green throughout.")
        return 1

    print(f"PASS: every bind mount names something that exists on any host "
          f"({checked} inspected).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
