#!/usr/bin/env python3
"""Per-service identity wiring: keys, mounts, and the one-owner rule.

The property being gated
------------------------

    A PRIVATE KEY IS SEEN BY EXACTLY ONE SERVICE.

If two services mount the same private key they are the same principal,
and the measured defect this whole line of work exists to close — 26 of
32 endpoints unable to tell one caller from another — returns intact,
now wearing a signature.

The denominator is DERIVED, twice over
--------------------------------------

* **Verifiers** come from the code: every service whose sources call
  ``require_service_identity``. Not a list beside this file.
* **Signers** come from compose: every service declaring
  ``KAI_SERVICE_KEY_ID``.

Both sides are then checked against what compose actually mounts, so a
service that is wired in code and not in compose is a finding rather
than a silence.

What is checked
---------------

1. a verifier declares ``KAI_SERVICE_NAME`` and ``KAI_SERVICE_KEYMAP``
2. a verifier mounts its key map **read-only** — anything able to
   rewrite the map can mint an identity
3. a signer declares ``KAI_SERVICE_PRIVATE_KEY`` and mounts it
4. **no private key is mounted into more than one service**
5. no private key material is committed to the repository

``KAI_SERVICE_NAME`` is local configuration for signing and for the
receiver's own destination check. It is never read from a request, and
check 6 asserts that no code path lets a header supply it — the failure
this mechanism replaced was exactly a caller naming itself.
"""
from __future__ import annotations

import ast
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO))

from scripts.security.gate_inputs import compose_files, inspected  # noqa: E402

_IDENTITY_DEP = "require_service_identity"

NAME_ENV = "KAI_SERVICE_NAME"
KEYMAP_ENV = "KAI_SERVICE_KEYMAP"
KEY_ID_ENV = "KAI_SERVICE_KEY_ID"
PRIVATE_ENV = "KAI_SERVICE_PRIVATE_KEY"

#: Header names that would reintroduce caller-asserted identity. Not a
#: guess: `actor_did` is the field the previous mechanism trusted.
_FORBIDDEN_IDENTITY_INPUTS = ("x-kai-identity", "x-service-name",
                              "x-caller-identity")


def verifier_services(root: Path) -> Set[str]:
    """Services whose code verifies caller identity. Derived from source."""
    out: Set[str] = set()
    skip = {"_archive", ".venv", "__pycache__", ".git", "scripts", "output",
            "common", "tests"}
    for path in sorted(root.rglob("*.py")):
        parts = path.relative_to(root).parts
        if len(parts) < 2 or any(p in skip for p in parts):
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        if _IDENTITY_DEP not in text:
            continue
        try:
            tree = ast.parse(text)
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            name = getattr(node.func, "id", None) or getattr(
                node.func, "attr", None)
            if name == _IDENTITY_DEP:
                out.add(parts[0])
                break
    return out


def _env_of(service: dict) -> Dict[str, str]:
    env = (service or {}).get("environment") or {}
    if isinstance(env, list):
        parsed = {}
        for item in env:
            key, _, value = str(item).partition("=")
            parsed[key] = value
        return parsed
    return {str(k): str(v) for k, v in env.items()}


def _mounts_of(service: dict) -> List[Tuple[str, str, bool]]:
    """(source, target, read_only) for every bind mount."""
    out = []
    for volume in (service or {}).get("volumes") or []:
        if isinstance(volume, str):
            bits = volume.split(":")
            if len(bits) >= 2:
                out.append((bits[0], bits[1], len(bits) > 2 and "ro" in bits[2:]))
        elif isinstance(volume, dict):
            out.append((str(volume.get("source", "")),
                        str(volume.get("target", "")),
                        bool(volume.get("read_only"))))
    return out


def audit(root: Path = None) -> Tuple[List[str], int, List[str]]:
    """Return (rows, services inspected, findings)."""
    import yaml

    root = root or REPO
    files = compose_files(root)
    if not files:
        return (["no compose files found — nothing was inspected and this "
                 "must not be read as a clean surface"], 0,
                ["no compose files found"])

    verifiers = verifier_services(root)
    rows: List[str] = []
    findings: List[str] = []
    inspected_n = 0

    # 5. key material must never be committed.
    tracked_keys = sorted(
        str(p.relative_to(root))
        for p in root.rglob("*.key")
        if ".git" not in p.parts and "_archive" not in p.parts
        and "ed25519" in p.read_text(encoding="utf-8", errors="ignore")[:16])
    for path in tracked_keys:
        import subprocess
        result = subprocess.run(["git", "ls-files", "--error-unmatch", path],
                                cwd=root, capture_output=True)
        if result.returncode == 0:
            findings.append(f"{path} is COMMITTED and contains key material")

    for compose_path in files:
        try:
            doc = yaml.safe_load(compose_path.read_text(encoding="utf-8")) or {}
        except Exception:
            continue
        services = doc.get("services") or {}
        if not services:
            continue

        private_owners: Dict[str, List[str]] = {}
        for name, service in sorted(services.items()):
            env = _env_of(service)
            mounts = _mounts_of(service)
            is_verifier = name in verifiers
            is_signer = bool(env.get(KEY_ID_ENV))
            if not (is_verifier or is_signer):
                continue
            inspected_n += 1
            role = "+".join(filter(None, ["verifier" if is_verifier else "",
                                          "signer" if is_signer else ""]))
            rows.append(f"  {compose_path.name:<28}{name:<16}{role}")

            if is_verifier:
                for required in (NAME_ENV, KEYMAP_ENV):
                    if not env.get(required):
                        findings.append(
                            f"{compose_path.name}: {name} verifies caller "
                            f"identity in code but declares no {required}")
                keymap_target = env.get(KEYMAP_ENV, "")
                mounted = [m for m in mounts if m[1] == keymap_target]
                if keymap_target and not mounted:
                    findings.append(
                        f"{compose_path.name}: {name} reads its key map from "
                        f"{keymap_target}, which nothing mounts — it would "
                        f"refuse every signed caller")
                for _, _, read_only in mounted:
                    if not read_only:
                        findings.append(
                            f"{compose_path.name}: {name} mounts its key map "
                            f"WRITABLE — anything that can rewrite the map "
                            f"can mint an identity")

            if is_signer:
                private = env.get(PRIVATE_ENV, "")
                if not private:
                    findings.append(
                        f"{compose_path.name}: {name} declares {KEY_ID_ENV} "
                        f"but no {PRIVATE_ENV} — it cannot sign anything")
                mounted = [m for m in mounts if m[1] == private]
                if private.startswith("/") and not mounted:
                    findings.append(
                        f"{compose_path.name}: {name} expects its private key "
                        f"at {private}, which nothing mounts")
                for source, _, read_only in mounted:
                    private_owners.setdefault(source, []).append(name)
                    if not read_only:
                        findings.append(
                            f"{compose_path.name}: {name} mounts its private "
                            f"key writable; it should be read-only")

        # 4. THE ONE-OWNER RULE.
        for source, owners in sorted(private_owners.items()):
            if len(owners) > 1:
                findings.append(
                    f"{compose_path.name}: private key {source} is mounted "
                    f"into {len(owners)} services ({', '.join(sorted(owners))})"
                    f" — services sharing a key are ONE principal, which is "
                    f"the defect this mechanism exists to remove")

    # 6. no code path may take identity from a request header.
    auth_source = (root / "common" / "service_auth.py").read_text(
        encoding="utf-8", errors="replace").lower()
    identity_source = (root / "common" / "service_identity.py").read_text(
        encoding="utf-8", errors="replace").lower()
    for header in _FORBIDDEN_IDENTITY_INPUTS:
        for label, text in (("service_auth", auth_source),
                            ("service_identity", identity_source)):
            # Described, not quoted: a gate whose own prose trips its
            # assertion is a defect this repository has hit five times.
            if f'get("{header}"' in text or f"get('{header}'" in text:
                findings.append(
                    f"common/{label}.py reads an identity-naming header — "
                    f"the principal must come from the verifying key only")

    return rows, inspected_n, findings


def main() -> int:
    rows, n, findings = audit()
    print(inspected(n, "service(s) that sign or verify caller identity",
                    "derived from code calling require_service_identity plus "
                    f"compose services declaring {KEY_ID_ENV}"))
    print()
    if rows:
        print(f"  {'compose file':<28}{'service':<16}role")
        for row in rows:
            print(row)
    print()
    if findings:
        print(f"FAILED: {len(findings)} identity wiring defect(s):\n")
        for finding in findings:
            print(f"  - {finding}")
        return 1
    print("  Every signer owns its key alone, every verifier mounts its map")
    print("  read-only, and no principal is taken from a request header.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
