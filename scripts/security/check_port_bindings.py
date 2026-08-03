#!/usr/bin/env python3
"""P0-PR-02 CI gate: reject disallowed host-port bindings in Compose files.

Only the 'dashboard' service may publish a port, and only on 127.0.0.1.
All other services must communicate over Docker networks only.

Exit 0 = clean.  Exit 1 = violations found.
"""
from __future__ import annotations

import sys
from pathlib import Path

try:
    import yaml
except ImportError:
    print("ERROR: PyYAML required. Install with: pip install pyyaml", file=sys.stderr)
    sys.exit(2)

ALLOWED_SERVICE = "dashboard"
ALLOWED_PREFIX = "127.0.0.1:"

COMPOSE_FILES = [
    "docker-compose.full.yml",
    "docker-compose.minimal.yml",
    "docker-compose.sovereign.yml",
]


def _loopback_binding(port) -> tuple[bool, str]:
    """Return (binds_to_loopback_only, human_description).

    Compose has two port syntaxes and this gate understood only one. The
    long form is a mapping::

        ports:
          - target: 8080
            host_ip: 127.0.0.1
            published: 8080

    which is *correct*, and `str(port)` on it produced a dict repr that
    could not start with "127.0.0.1:" — so the gate reported a properly
    loopback-bound dashboard as a violation, and told the operator to
    "bind to 127.0.0.1 only" when they already had. A message that names
    the wrong problem sends whoever is debugging in the wrong direction.
    """
    if isinstance(port, dict):
        host_ip = str(port.get("host_ip", ""))
        published = port.get("published")
        if published is None:
            return True, f"target {port.get('target')} (not published)"
        return host_ip in ("127.0.0.1", "::1"), \
            f"host_ip={host_ip or '<all interfaces>'} published={published}"
    text = str(port)
    return text.startswith(ALLOWED_PREFIX), text


def check_file(path: Path) -> list[str]:
    violations: list[str] = []
    try:
        with open(path) as f:
            doc = yaml.safe_load(f)
    except Exception as exc:
        violations.append(f"{path}: failed to parse: {exc}")
        return violations

    services = doc.get("services", {}) or {}
    for svc_name, svc_def in services.items():
        if not isinstance(svc_def, dict):
            continue
        ports = svc_def.get("ports")
        if not ports:
            continue

        # A `ports:` value that is not a list is a malformed file, not a
        # set of bad ports. Iterating a string yields characters, and the
        # gate used to report violations for ports named '8', '0' and ':'
        # — sending the reader to check a port when the parser is what
        # needs checking.
        if not isinstance(ports, list):
            violations.append(
                f"{path}: {svc_name} has a malformed 'ports' value of type "
                f"{type(ports).__name__} — expected a list. The port "
                f"configuration could not be checked."
            )
            continue

        if svc_name == ALLOWED_SERVICE:
            for port in ports:
                ok, described = _loopback_binding(port)
                if not ok:
                    violations.append(
                        f"{path}: {svc_name} port {described} reaches beyond "
                        f"loopback — it must publish on 127.0.0.1 only"
                    )
        else:
            for port in ports:
                _, described = _loopback_binding(port)
                violations.append(
                    f"{path}: {svc_name} has disallowed port binding "
                    f"{described} — only '{ALLOWED_SERVICE}' may publish ports"
                )
    return violations


def main() -> int:
    repo_root = Path(__file__).resolve().parent.parent.parent
    all_violations: list[str] = []

    for name in COMPOSE_FILES:
        path = repo_root / name
        if not path.exists():
            continue
        all_violations.extend(check_file(path))

    if all_violations:
        print(f"FAIL: {len(all_violations)} port-binding violation(s):\n")
        for v in all_violations:
            print(f"  - {v}")
        return 1

    print("PASS: No disallowed port bindings found.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
