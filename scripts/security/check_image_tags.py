#!/usr/bin/env python3
"""P0-PR-09 CI gate: reject mutable privileged image tags.

Services using pre-built images (not local builds) must pin to a specific
digest or version tag.  The :latest tag is mutable and can silently change
the running code, breaking reproducibility and audit trails.

**A pinned tag is not the same as a tag that exists**, and on 2026-08-05
that gap cost the whole live-verification half of `core-tests.yml`.

`docker-compose.minimal.yml` pinned `ollama/ollama:0.6`. This gate passed
it — it carries a version, it is not on the mutable list, it looked like
a pin. It had also been **withdrawn from Docker Hub**, so every
`docker compose up` failed in under a second with

    manifest for ollama/ollama:0.6 not found: manifest unknown

and steps 47 through 59 never ran. The image build ahead of it passed,
because `docker compose build` builds `build:` services and never touches
an `image:` one; nothing looked at the pull until `up`.

`0.6` was a **minor-series** tag: upstream retargets those and eventually
drops them. `0.6.8`, the patch release, is still there. So the tag was
reproducible in form and disposable in fact, and this gate was measuring
the form.

`--verify-exists` measures the fact. It asks the registry whether each
tag resolves. That needs the network, so it is a separate mode rather
than part of the default run — and it **fails closed**: a tag it could
not check is reported, never assumed present.

Exit 0 = clean.  Exit 1 = violations found.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import List, Tuple

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


# ── Does the tag actually resolve? ───────────────────────────────────
# The registry v2 API, which both Docker Hub and ghcr.io speak: ask the
# auth endpoint for an anonymous pull token, then HEAD the manifest.
# Anonymous is enough for public images and keeps this runnable without
# credentials.

_REGISTRIES = {
    "ghcr.io": ("https://ghcr.io/token?scope=repository:{repo}:pull",
                "https://ghcr.io/v2/{repo}/manifests/{tag}"),
    None:      ("https://auth.docker.io/token?service=registry.docker.io"
                "&scope=repository:{repo}:pull",
                "https://registry-1.docker.io/v2/{repo}/manifests/{tag}"),
}

_ACCEPT = ", ".join((
    "application/vnd.oci.image.index.v1+json",
    "application/vnd.oci.image.manifest.v1+json",
    "application/vnd.docker.distribution.manifest.list.v2+json",
    "application/vnd.docker.distribution.manifest.v2+json",
))


def split_reference(image: str) -> Tuple[str, str, str]:
    """(registry-host-or-empty, repository, tag) for one image reference."""
    ref, _, digest = image.partition("@")
    head, _, maybe_tag = ref.rpartition(":")
    if head and "/" not in maybe_tag:
        repo, tag = head, maybe_tag
    else:
        repo, tag = ref, "latest"
    if digest:
        tag = f"sha256:{digest.split(':')[-1]}"
    host = ""
    first = repo.split("/")[0]
    if "." in first or ":" in first:
        host, repo = first, repo.split("/", 1)[1]
    elif "/" not in repo:
        repo = f"library/{repo}"        # official images live under library/
    return host, repo, tag


def tag_resolves(image: str, timeout: float = 20.0) -> Tuple[bool, str]:
    """(resolves, detail). Never guesses: an error is reported, not swallowed."""
    host, repo, tag = split_reference(image)
    token_url, manifest_url = _REGISTRIES.get(host or None, _REGISTRIES[None])
    try:
        with urllib.request.urlopen(
                token_url.format(repo=repo), timeout=timeout) as resp:
            token = json.loads(resp.read()).get("token")
        if not token:
            return False, "registry returned no pull token"
        req = urllib.request.Request(
            manifest_url.format(repo=repo, tag=tag), method="HEAD")
        req.add_header("Authorization", f"Bearer {token}")
        req.add_header("Accept", _ACCEPT)
        urllib.request.urlopen(req, timeout=timeout)
        return True, "resolves"
    except urllib.error.HTTPError as exc:
        if exc.code in (401, 403, 404):
            return False, f"not found in the registry (HTTP {exc.code})"
        return False, f"registry error HTTP {exc.code}"
    except Exception as exc:
        # I-1. "Could not ask" is not "it is there".
        return False, f"could not verify ({type(exc).__name__}: {exc})"


def images_in(paths) -> List[Tuple[str, str, str]]:
    """(compose file, service, image) for every pre-built image, deduped."""
    out, seen = [], set()
    for path in paths:
        data = yaml.safe_load(path.read_text()) or {}
        for name, cfg in sorted((data.get("services") or {}).items()):
            image = (cfg or {}).get("image")
            if not image or image in seen:
                continue
            seen.add(image)
            out.append((path.name, name, image))
    return out


def verify_existence(paths) -> Tuple[List[str], int]:
    """Return (violations, images checked)."""
    violations = []
    entries = images_in(paths)
    for filename, service, image in entries:
        ok, detail = tag_resolves(image)
        if not ok:
            violations.append(
                f"{filename}: service '{service}' uses '{image}' — {detail}. "
                f"A tag that is pinned but absent still stops every "
                f"`docker compose up` before it starts.")
    return violations, len(entries)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Image tag gate")
    parser.add_argument(
        "--verify-exists", action="store_true",
        help="also ask the registry whether each tag resolves (needs network)")
    args = parser.parse_args(argv)

    all_violations: list[str] = []

    # A missing compose file is not a clean bill of health: this gate
    # would inspect nothing and print PASS, byte-identical to a real one.
    paths = require(COMPOSE_FILES)
    for path in paths:
        all_violations.extend(check_file(path))

    print(inspected(count_services(paths), "service definitions",
                    f"across {len(paths)} compose files"))

    checked = 0
    if args.verify_exists:
        existence, checked = verify_existence(paths)
        all_violations.extend(existence)
        print(f"  resolved: {checked - len(existence)} of {checked} distinct "
              f"image reference(s) against their registry")
        if checked == 0:
            # I-2. Nothing asked must not read the same as everything answered.
            all_violations.append(
                "no pre-built images found to verify — this gate inspected "
                "nothing, which is not a pass")

    if all_violations:
        print(f"FAIL: {len(all_violations)} image tag problem(s):\n")
        for v in all_violations:
            print(f"  - {v}")
        return 1

    if args.verify_exists:
        print("PASS: every pre-built image is pinned and resolves today.")
    else:
        print("PASS: All pre-built images use pinned version tags.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
