#!/usr/bin/env python3
"""P0-PR-09 CI gate: reject mutable privileged image tags.

Services using pre-built images (not local builds) must pin to a specific
digest or version tag.  The :latest tag is mutable and can silently change
the running code, breaking reproducibility and audit trails.

Exit 0 = clean.  Exit 1 = violations found.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from scripts.security.gate_inputs import (count_services, inspected,  # noqa: E402
                                          require)

COMPOSE_FILES = [
    "docker-compose.full.yml",
    "docker-compose.minimal.yml",
    "docker-compose.sovereign.yml",
]

# Named mutable tags, kept only so the message can say *why* a tag is
# known-bad rather than merely unversioned.
MUTABLE_TAGS = frozenset({
    "latest", "stable", "edge", "nightly", "main", "master", "dev",
    "develop", "test", "prod", "production", "release", "current",
})

# The rule, rather than a list. A denylist of four words let `myimg:main`
# through, and every future mutable name would need adding by hand — the
# same shape as the secret gate's nine-word denylist.
#
# A pinned tag either is a digest, or contains a version number. Measured
# before adopting: every one of the 18 image tags in this repository
# contains a digit (`7-alpine`, `pg15`, `v1.78`, `3.11-slim`), so the
# rule costs nothing today and catches every unversioned name.
_VERSIONED = re.compile(r"\d")


def tag_is_pinned(image: str, tag: str) -> bool:
    """True when this tag identifies one immutable image."""
    if "@sha256:" in image:
        return True                     # a digest is the strongest pin
    if tag in MUTABLE_TAGS:
        return False
    return bool(_VERSIONED.search(tag))


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

        if not tag_is_pinned(image, tag):
            why = ("a known-mutable name" if tag in MUTABLE_TAGS
                   else "carries no version")
            violations.append(
                f"{path}: service '{svc_name}' uses image '{image}' — tag "
                f"'{tag}' is {why}; pin to a version or a @sha256 digest"
            )

    return violations


def main() -> int:
    repo_root = Path(__file__).resolve().parent.parent.parent
    all_violations: list[str] = []

    # A missing compose file is not a clean bill of health: this gate
    # would inspect nothing and print PASS, byte-identical to a real one.
    paths = require(COMPOSE_FILES)
    for path in paths:
        all_violations.extend(check_file(path))

    print(inspected(count_services(paths), "service definitions",
                    f"across {len(paths)} compose files"))

    if all_violations:
        print(f"FAIL: {len(all_violations)} mutable image tag(s) found:\n")
        for v in all_violations:
            print(f"  - {v}")
        return 1

    print("PASS: All pre-built images use pinned version tags.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
