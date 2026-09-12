#!/usr/bin/env python3
"""The wiring gate must fire on each defect it claims to catch.

A gate that has only ever passed is a hypothesis. Every finding below is
produced from a synthetic tree, so the assertion is that the check
*fires*, not merely that today's compose happens to be clean.

The load-bearing one is the one-owner rule: two services mounting the
same private key are ONE principal, and that is the measured defect this
whole mechanism exists to remove.
"""
from __future__ import annotations

import sys
import tempfile
import textwrap
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from scripts.security.check_service_identity_wiring import (  # noqa: E402
    audit, verifier_services)

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


def _tree(compose: str, service_code: str = "") -> Path:
    root = Path(tempfile.mkdtemp())
    (root / "docker-compose.test.yml").write_text(
        textwrap.dedent(compose), encoding="utf-8")
    (root / "common").mkdir()
    for name in ("service_auth.py", "service_identity.py"):
        (root / "common" / name).write_text("# nothing to see\n",
                                            encoding="utf-8")
    if service_code:
        (root / "receiver").mkdir()
        (root / "receiver" / "app.py").write_text(
            textwrap.dedent(service_code), encoding="utf-8")
    return root


RECEIVER = """
    from common.service_auth import require_service_identity
    @app.post("/thing")
    async def thing(p=Depends(require_service_identity("op",
                                                      require_grant=True))):
        return {}
"""


def findings_for(compose: str, code: str = RECEIVER) -> list:
    return audit(_tree(compose, code))[2]


def main() -> int:
    # ── the denominator is derived from CODE, not a list ──
    root = _tree("services: {}\n", RECEIVER)
    check("a verifier is discovered by its use of the dependency",
          verifier_services(root) == {"receiver"})
    check("a service that does not use it is not a verifier",
          verifier_services(_tree("services: {}\n", "print('hi')\n")) == set())

    # ── THE ONE-OWNER RULE ──
    shared = findings_for("""
        services:
          alpha:
            environment:
              KAI_SERVICE_KEY_ID: alpha-v1
              KAI_SERVICE_PRIVATE_KEY: /run/secrets/id
            volumes:
              - ./keys/shared.key:/run/secrets/id:ro
          beta:
            environment:
              KAI_SERVICE_KEY_ID: beta-v1
              KAI_SERVICE_PRIVATE_KEY: /run/secrets/id
            volumes:
              - ./keys/shared.key:/run/secrets/id:ro
    """)
    check("TWO SERVICES SHARING A PRIVATE KEY IS A FINDING",
          any("ONE principal" in f for f in shared))

    # And the same wiring with distinct keys is clean.
    distinct = findings_for("""
        services:
          alpha:
            environment:
              KAI_SERVICE_KEY_ID: alpha-v1
              KAI_SERVICE_PRIVATE_KEY: /run/secrets/id
            volumes:
              - ./keys/alpha.key:/run/secrets/id:ro
          beta:
            environment:
              KAI_SERVICE_KEY_ID: beta-v1
              KAI_SERVICE_PRIVATE_KEY: /run/secrets/id
            volumes:
              - ./keys/beta.key:/run/secrets/id:ro
    """)
    check("one key each is NOT a finding — the check is not just noisy",
          not any("ONE principal" in f for f in distinct))

    # ── a writable key map can mint an identity ──
    writable = findings_for("""
        services:
          receiver:
            environment:
              KAI_SERVICE_NAME: receiver
              KAI_SERVICE_KEYMAP: /etc/kai/keymap.json
            volumes:
              - ./keys/keymap.json:/etc/kai/keymap.json
    """)
    check("a WRITABLE key map is a finding",
          any("mint an identity" in f for f in writable))

    read_only = findings_for("""
        services:
          receiver:
            environment:
              KAI_SERVICE_NAME: receiver
              KAI_SERVICE_KEYMAP: /etc/kai/keymap.json
            volumes:
              - ./keys/keymap.json:/etc/kai/keymap.json:ro
    """)
    check("a read-only key map is clean", read_only == [])

    # ── wired in code, missing from compose ──
    unwired = findings_for("""
        services:
          receiver:
            environment:
              KAI_SERVICE_TOKEN: tok
            volumes: []
    """)
    check("a verifier with no KAI_SERVICE_NAME is a finding",
          any("KAI_SERVICE_NAME" in f for f in unwired))
    check("a verifier with no KAI_SERVICE_KEYMAP is a finding",
          any("KAI_SERVICE_KEYMAP" in f for f in unwired))

    unmounted = findings_for("""
        services:
          receiver:
            environment:
              KAI_SERVICE_NAME: receiver
              KAI_SERVICE_KEYMAP: /etc/kai/keymap.json
            volumes: []
    """)
    check("a key map path nothing mounts is a finding — it would refuse "
          "every signed caller",
          any("nothing mounts" in f for f in unmounted))

    # ── a signer that cannot sign ──
    keyless = findings_for("""
        services:
          alpha:
            environment:
              KAI_SERVICE_KEY_ID: alpha-v1
    """, code="")
    check("a signer with a key id and no private key is a finding",
          any("cannot sign anything" in f for f in keyless))

    unmounted_key = findings_for("""
        services:
          alpha:
            environment:
              KAI_SERVICE_KEY_ID: alpha-v1
              KAI_SERVICE_PRIVATE_KEY: /run/secrets/id
            volumes: []
    """, code="")
    check("a private key path nothing mounts is a finding",
          any("nothing mounts" in f for f in unmounted_key))

    # ── I-1: an empty inspection is a broken scan, not a clean system ──
    empty = Path(tempfile.mkdtemp())
    rows, n, findings = audit(empty)
    check("a tree with no compose files refuses rather than passing",
          n == 0 and findings != [])

    # ── the live tree ──
    rows, n, findings = audit(REPO)
    check("CALIBRATION: the live tree has 2 identity-wired services", n == 2)
    check("CALIBRATION: and no wiring defects", findings == [])
    text = "\n".join(rows)
    check("agentic is the signer", "agentic" in text and "signer" in text)
    check("cortex is the verifier", "cortex" in text and "verifier" in text)

    # ── the key GENERATOR ──
    #
    # The registry declares probe=False for it, because probing a key
    # generator would write private key material as a side effect of
    # measuring. That skip is only honest if the behaviour is exercised
    # here instead, against a temporary directory.
    import io
    import contextlib
    import json as _json
    from scripts.security.generate_service_keys import generate
    from common.service_identity import KeyMap, IdentityError

    out = Path(tempfile.mkdtemp()) / "keys"
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        code = generate(["alpha", "beta"], {"some_op": ["alpha"]}, out)
    output = buffer.getvalue()
    check("the generator succeeds", code == 0)
    check("and reports its denominator, which is why probe=False is honest",
          "inspected: 2 service key(s) generated" in output)

    alpha = out / "private" / "alpha.key"
    beta = out / "private" / "beta.key"
    check("one private key file per service", alpha.exists() and beta.exists())
    check("PRIVATE KEYS ARE MODE 0600",
          (alpha.stat().st_mode & 0o777) == 0o600
          and (beta.stat().st_mode & 0o777) == 0o600)
    check("the two services do NOT share key material",
          alpha.read_text() != beta.read_text())
    check("private key material never reaches the key map",
          alpha.read_text().split(":")[1] not in (out / "keymap.json").read_text())

    keymap = KeyMap.load(str(out / "keymap.json"))
    check("the generated map is loadable by the runtime that must read it",
          len(keymap) == 2)
    check("the grant is carried through", keymap.granted("some_op", "alpha"))
    check("and grants nobody it was not asked to",
          not keymap.granted("some_op", "beta"))

    # Regenerating silently would lock out every receiver holding the old
    # public half, and the symptom looks exactly like an attack.
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            generate(["alpha"], {}, out)
        ok = False
    except SystemExit as exc:
        ok = "already exist" in str(exc)
    check("REGENERATING OVER AN EXISTING KEY IS REFUSED without --force", ok)

    # A grant for an identity with no key is a typo or a removed key left
    # authorised. Caught at generation, not only at load.
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            generate(["alpha"], {"op": ["ghost"]},
                     Path(tempfile.mkdtemp()) / "k")
        ok = False
    except SystemExit as exc:
        ok = "no key requested" in str(exc)
    check("a grant naming a service with no key is refused at generation", ok)

    try:
        with contextlib.redirect_stdout(io.StringIO()):
            generate([], {}, Path(tempfile.mkdtemp()) / "k")
        ok = False
    except SystemExit as exc:
        ok = "verify nothing" in str(exc)
    check("generating an EMPTY key map is refused — it verifies nothing", ok)

    print("=" * 66)
    print(f"Identity wiring gate tests: {PASSED} passed, {FAILED} failed")
    print(f"EXIT GATE: {'PASS' if FAILED == 0 else 'FAIL'}")
    return 1 if FAILED else 0


if __name__ == "__main__":
    print("Service identity wiring — proving the gate fires")
    print("=" * 66)
    sys.exit(main())
