#!/usr/bin/env python3
"""P0-PR-09 CI gate: reject mutable privileged image tags.

Services using pre-built images (not local builds) must pin to a specific
digest or version tag.  The :latest tag is mutable and can silently change
the running code, breaking reproducibility and audit trails.

Exit 0 = clean.  Exit 1 = violations found.
"""
from __future__ import annotations

import sys
from pathlib import Path

import yaml

COMPOSE_FILES = [
    "docker-compose.full.yml",
    "docker-compose.minimal.yml",
    "docker-compose.sovereign.yml",
]

MUTABLE_TAGS = frozenset({"latest", "stable", "edge", "nightly"})


def check_file(path: Path) -> list[str]:
    violations: list[str] = []
    try:
        data = yaml.safe_load(path.read_text())
    except Exception as exc:
        violations.append(f"{path}: failed to parse: {exc}")
        return violations

    if data is None:
        return violations

    services = data.get("services", {})

    for svc_name, svc_cfg in services.items():
        if svc_cfg is None:
            continue

        image = svc_cfg.get("image")
        if not image:
            continue

        if ":" in image:
            tag = image.rsplit(":", 1)[1]
        else:
            tag = "latest"

        if tag in MUTABLE_TAGS:
            violations.append(
                f"{path}: service '{svc_name}' uses mutable image tag "
                f"'{image}' — pin to a specific version"
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
        print(f"FAIL: {len(all_violations)} mutable image tag(s) found:\n")
        for v in all_violations:
            print(f"  - {v}")
        return 1

    print("PASS: All pre-built images use pinned version tags.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
