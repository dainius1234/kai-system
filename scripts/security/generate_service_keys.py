#!/usr/bin/env python3
"""Generate per-service Ed25519 keys and the trusted receiver key map.

The property the whole mechanism rests on
-----------------------------------------

    A PRIVATE KEY IS SEEN BY EXACTLY ONE SERVICE.

If two services share a key they are the same principal, and the
measurement that started this work — 26 of 32 endpoints unable to tell
one caller from another — comes straight back. So this writes one file
per service at mode 0600, and the map that receivers read contains only
public halves.

What goes where
---------------

    <out>/private/<service>.key     ed25519:<hex>   mode 0600, one owner
    <out>/keymap.json               kid -> identity, algorithm, public
                                    key, plus the grant table

The map is not a secret: public keys are public, and the file is mounted
read-only. What matters is that nothing can *rewrite* it, because
anything that can rewrite the map can mint an identity — which is why
`KeyMap.load` refuses a group- or other-writable file.

Grants
------

`--grant <operation>=<identity>[,<identity>]` writes the route grant
table into the same document. An operation with no entry denies, and a
grant naming an identity with no key is refused at load rather than
silently ignored: a typo and a removed-but-still-authorised key both
read as a working grant, and neither is one.

Refuses to overwrite
--------------------

Regenerating a key silently would lock out every receiver still holding
the old public half, and the symptom — valid callers refused — looks
exactly like an attack. `--force` is required, and says so.

    python3 scripts/security/generate_service_keys.py \
        --service agentic --service cortex \
        --grant cortex_observe_turn=agentic \
        --out secrets/service-identity
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO))

from common.service_identity import (ALG_ED25519, IdentityError,  # noqa: E402
                                     generate_keypair)
from scripts.security.gate_inputs import inspected  # noqa: E402

DEFAULT_OUT = REPO / "secrets" / "service-identity"


def _shown(path: Path) -> str:
    """Repo-relative when it is inside the repo, absolute otherwise.

    `--out` is a parameter, so it can legitimately point anywhere.
    `relative_to` raises on any path outside the repository, which made
    the generator crash for exactly the argument it invites — found by
    the first test that pointed it at a temporary directory.
    """
    try:
        return str(path.relative_to(REPO))
    except ValueError:
        return str(path)


def _parse_grants(raw: List[str]) -> Dict[str, List[str]]:
    grants: Dict[str, List[str]] = {}
    for item in raw or []:
        operation, _, identities = item.partition("=")
        if not operation or not identities:
            raise SystemExit(f"--grant must be <operation>=<identity>[,...], "
                             f"got {item!r}")
        grants.setdefault(operation, [])
        for identity in identities.split(","):
            identity = identity.strip()
            if identity and identity not in grants[operation]:
                grants[operation].append(identity)
    return grants


def generate(services: List[str], grants: Dict[str, List[str]],
             out: Path, force: bool = False, version: str = "v1") -> int:
    if not services:
        raise SystemExit("no services named — refusing to write an empty key "
                         "map, which would verify nothing")

    unknown = sorted({i for ids in grants.values() for i in ids}
                     - set(services))
    if unknown:
        raise SystemExit(
            f"grant(s) name identity(ies) with no key requested: "
            f"{', '.join(unknown)}. Add --service for each, or fix the typo. "
            f"A grant for an identity that cannot sign is not a grant.")

    private_dir = out / "private"
    private_dir.mkdir(parents=True, exist_ok=True)
    os.chmod(private_dir, 0o700)
    keymap_path = out / "keymap.json"

    existing = [s for s in services
                if (private_dir / f"{s}.key").exists()]
    if existing and not force:
        raise SystemExit(
            f"private key(s) already exist for: {', '.join(existing)}.\n"
            f"Regenerating invalidates every receiver still holding the old "
            f"public half, and the symptom — legitimate callers refused — "
            f"looks exactly like an attack. Pass --force if that is what you "
            f"intend, and redeploy the map at the same time.")

    keys: Dict[str, Dict[str, str]] = {}
    for service in services:
        private_material, public_material = generate_keypair()
        key_path = private_dir / f"{service}.key"
        # Written 0600 BEFORE content, so the material is never briefly
        # readable by anything that happens to be watching the directory.
        fd = os.open(key_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(f"{ALG_ED25519}:{private_material.hex()}")
        os.chmod(key_path, 0o600)
        keys[f"{service}-{version}"] = {
            "identity": service,
            "algorithm": ALG_ED25519,
            "public_key": public_material.hex(),
        }
        print(f"  {service:<18} key id {service}-{version}  "
              f"private {_shown(key_path)} (0600)")

    document = {"version": 1, "keys": keys}
    if grants:
        document["grants"] = grants
    keymap_path.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n",
                           encoding="utf-8")
    os.chmod(keymap_path, 0o644)

    # Prove the artefact we just wrote is one the runtime will accept,
    # rather than assuming it. A generator that emits a map the loader
    # refuses is worse than no generator.
    from common.service_identity import KeyMap
    try:
        loaded = KeyMap.load(str(keymap_path))
    except IdentityError as exc:
        raise SystemExit(f"the generated key map is not loadable: {exc}")

    print()
    print(inspected(len(keys), "service key(s) generated",
                    "one private key per named service, each seen by exactly "
                    "one service"))
    print(f"\n  key map {_shown(keymap_path)} "
          f"({len(loaded)} key(s), sha256 {loaded.digest[:16]})")
    for operation in sorted(grants):
        print(f"  grant  {operation} -> {', '.join(loaded.grants_for(operation))}")
    print("\n  Mount the private key ONLY into its owning service. Mount the "
          "key map\n  read-only into every service that verifies. Neither "
          "belongs in git.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--service", action="append", default=[],
                        help="service needing its own signing key (repeatable)")
    parser.add_argument("--grant", action="append", default=[],
                        help="<operation>=<identity>[,<identity>] (repeatable)")
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--version", default="v1",
                        help="key id suffix; bump to rotate with overlap")
    parser.add_argument("--force", action="store_true",
                        help="overwrite existing private keys")
    args = parser.parse_args()
    print("Generating per-service ed25519 identity material\n")
    return generate(args.service, _parse_grants(args.grant),
                    Path(args.out), args.force, args.version)


if __name__ == "__main__":
    sys.exit(main())
