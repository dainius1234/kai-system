#!/usr/bin/env python3
"""Per-service Ed25519 identity — the refusals, which are the product.

A signing layer is judged by what it REJECTS. Almost every assertion here
is negative, and the two that matter most are:

  * a service holding its own key cannot sign as another service, and
  * a caller-supplied identity header changes nothing.

The second is the whole reason this module exists. The mechanism it
replaces authenticated an `actor_did` the caller filled in, using a
secret three services shared — caller-asserted identity, signed. If a
header could still steer the principal, we would have rebuilt that.

There is no `skipUnless` here. Two existing tests in this repository skip
when `cryptography` cannot be imported, and a skipped security test reads
exactly like a passing one. If the backend is missing, this FAILS.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from common import service_identity as si  # noqa: E402

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


def _keymap(entries) -> si.KeyMap:
    return si.KeyMap.from_text(json.dumps({"keys": entries}))


def main() -> int:  # noqa: C901 - a list of refusals, not a branchy algorithm
    # ── the dependency is a requirement, not a nice-to-have ──
    available, detail = si.import_status()
    check(f"ed25519 backend is importable ({detail})", available)
    if not available:
        print("\n  Every remaining assertion needs the backend. Refusing to "
              "skip: a skipped security test reads like a passing one.")
        print(f"EXIT GATE: FAIL")
        return 1

    agentic_priv, agentic_pub = si.generate_keypair()
    cortex_priv, cortex_pub = si.generate_keypair()
    check("generated keys are the ed25519 sizes",
          len(agentic_priv) == 32 and len(agentic_pub) == 32)
    check("the public half is derivable from the private half",
          si.public_from_private(agentic_priv) == agentic_pub)
    check("two services do not get the same key", agentic_priv != cortex_priv)

    keymap = _keymap({
        "agentic-v1": {"identity": "agentic", "algorithm": "ed25519",
                       "public_key": agentic_pub.hex()},
        "cortex-v1": {"identity": "cortex", "algorithm": "ed25519",
                      "public_key": cortex_pub.hex()},
    })
    check("the key map carries both identities",
          keymap.identities() == ("agentic", "cortex"))

    body = b'{"turn": "hello"}'
    call = dict(destination="cortex", method="POST", path="/observe_turn",
                body=body)

    def sign_as(priv, kid, **over):
        args = dict(call)
        args.update(over)
        return si.signed_headers(key_id=kid, algorithm="ed25519",
                                 private_material=priv, **args)

    def verify(headers, cache=None, **over):
        args = dict(call)
        args.update(over)
        return si.verify_request(headers, keymap=keymap, cache=cache, **args)

    # ── the happy path, so the refusals below mean something ──
    good = sign_as(agentic_priv, "agentic-v1")
    principal, status, _ = verify(good)
    check("a correctly signed request is accepted",
          principal is not None and status == 200)
    check("THE PRINCIPAL IS DERIVED FROM THE KEY, and names the signer",
          principal.identity == "agentic" and principal.verified)
    check("the principal is usable for provenance",
          principal.usable_for_provenance)
    check("no identity header was sent, yet identity was established",
          not any("identity" in k.lower() for k in good))

    # ── 1. missing signature ──
    _, status, detail = verify({})
    check("missing signature -> 401", status == 401)
    check("and it says what is missing", "signature" in detail)

    # ── 2. unknown key id ──
    stranger_priv, stranger_pub = si.generate_keypair()
    forged = sign_as(stranger_priv, "unknown-v1")
    principal, status, detail = verify(forged)
    check("unknown key id -> 401", principal is None and status == 401)
    check("unknown key and bad signature give the SAME detail, so the "
          "response leaks no oracle",
          detail == "signature could not be verified")

    # ── 3. THE ONE THAT MATTERS: A cannot sign as B ──
    #
    # cortex's own key, but claiming agentic's key id. Under the shared
    # secret this succeeded, because the key did not identify anyone.
    impersonation = sign_as(cortex_priv, "agentic-v1")
    principal, status, _ = verify(impersonation)
    check("A SERVICE CANNOT SIGN AS ANOTHER SERVICE",
          principal is None and status == 401)

    # And with its own key id it is cortex, never agentic.
    honest = sign_as(cortex_priv, "cortex-v1")
    principal, _, _ = verify(honest)
    check("signing with its own key makes it itself, not the peer",
          principal is not None and principal.identity == "cortex")

    # ── 4. caller-supplied identity is ignored ──
    lying = dict(sign_as(cortex_priv, "cortex-v1"))
    lying.update({"X-Kai-Identity": "agentic", "X-Actor-Did": "agentic",
                  "actor_did": "agentic", "X-Service-Name": "agentic"})
    principal, _, _ = verify(lying)
    check("CALLER-SUPPLIED IDENTITY HEADERS CHANGE NOTHING",
          principal is not None and principal.identity == "cortex")

    # ── 5. tampered body ──
    principal, status, _ = verify(sign_as(agentic_priv, "agentic-v1"),
                                  body=b'{"turn": "goodbye"}')
    check("a tampered body -> 401", principal is None and status == 401)

    # ── 6. replay onto a different route / method / service ──
    signed = sign_as(agentic_priv, "agentic-v1")
    principal, _, _ = verify(signed, path="/erase")
    check("a signature cannot be replayed onto another PATH",
          principal is None)
    principal, _, _ = verify(signed, method="DELETE")
    check("a signature cannot be replayed onto another METHOD",
          principal is None)
    principal, _, _ = verify(signed, destination="executor")
    check("a signature cannot be replayed onto another SERVICE",
          principal is None)

    # ── 7. expired / future timestamp ──
    stale = sign_as(agentic_priv, "agentic-v1",
                    )  # signed at now
    old = si.signed_headers(key_id="agentic-v1", algorithm="ed25519",
                            private_material=agentic_priv,
                            timestamp=int(time.time()) - 10_000, **call)
    principal, status, detail = verify(old)
    check("an expired timestamp -> 401", principal is None and status == 401)
    check("and it names the window", "window" in detail)
    future = si.signed_headers(key_id="agentic-v1", algorithm="ed25519",
                               private_material=agentic_priv,
                               timestamp=int(time.time()) + 10_000, **call)
    check("a future timestamp is refused too", verify(future)[0] is None)
    check("a fresh signature is still fine", verify(stale)[0] is not None)

    # ── 8. replay of the identical request ──
    with tempfile.TemporaryDirectory() as tmp:
        cache = si.NonceCache(path=str(Path(tmp) / "nonces.json"), ttl=900)
        once = sign_as(agentic_priv, "agentic-v1")
        principal, _, _ = verify(once, cache=cache)
        check("first use of a nonce is accepted", principal is not None)
        principal, status, detail = verify(once, cache=cache)
        check("REPLAYING THE IDENTICAL REQUEST -> 401",
              principal is None and status == 401)
        check("and it says so plainly", "already been seen" in detail)

        # A failed verification must not burn the nonce, or an attacker
        # could pre-poison a legitimate caller's retry.
        nonce = si.new_nonce()
        bad = si.signed_headers(key_id="agentic-v1", algorithm="ed25519",
                                private_material=agentic_priv, nonce=nonce,
                                **call)
        verify(bad, cache=cache, body=b"tampered")     # fails verification
        principal, _, _ = verify(bad, cache=cache)     # honest retry
        check("a REFUSED request does not burn its nonce",
              principal is not None)

    # ── 9. the restart gap ──
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "nonces.json"
        path.write_text("{not json at all")
        started = time.time()
        cache = si.NonceCache(path=str(path), ttl=900, now=started)
        check("a corrupt cache is reported as NOT restored", not cache.restored)
        check("and the floor is start-up, so pre-restart stamps are refused",
              cache.floor(300, started + 1) == started)
        stale_sig = si.signed_headers(
            key_id="agentic-v1", algorithm="ed25519",
            private_material=agentic_priv,
            timestamp=int(started) - 60, **call)
        principal, status, detail = si.verify_request(
            stale_sig, keymap=keymap, cache=cache, now=started + 1, **call)
        check("a pre-restart timestamp is refused while the cache is blind",
              principal is None and "predates this instance" in detail)
        check("and the floor lifts once the skew window has passed",
              cache.floor(300, started + 301) == 0.0)

        # A cache that simply does not exist yet is NOT a failed restore:
        # a first start has no window behind it needing protection.
        fresh = si.NonceCache(path=str(Path(tmp) / "absent.json"))
        check("an absent cache is a clean first start, not a failure",
              fresh.restored and fresh.floor(300) == 0.0)

    # ── 10. revocation ──
    os.environ[si.REVOKED_ENV] = "agentic-v1"
    try:
        principal, status, _ = verify(sign_as(agentic_priv, "agentic-v1"))
        check("a REVOKED key id is refused even with a valid signature",
              principal is None and status == 401)
    finally:
        os.environ.pop(si.REVOKED_ENV, None)
    check("and accepted again once un-revoked",
          verify(sign_as(agentic_priv, "agentic-v1"))[0] is not None)

    # ── 11. algorithm downgrade ──
    downgrade = dict(sign_as(agentic_priv, "agentic-v1"))
    downgrade[si.SIGNATURE_HEADER] = downgrade[si.SIGNATURE_HEADER].replace(
        "ed25519:", "hmac-sha256:", 1)
    check("an algorithm downgrade is refused", verify(downgrade)[0] is None)
    check("the algorithm is INSIDE the signed string, so a downgrade is "
          "also a signature mismatch",
          b"7:ed25519" in si.canonical_request(
              algorithm="ed25519", key_id="k", destination="d", method="GET",
              path="/p", body=b"", timestamp=1, nonce="n"))

    # ── 12. the canonical string cannot be forged from inside a field ──
    #
    # Length prefixes, not delimiters. Two different requests whose fields
    # concatenate to the same text must still sign differently.
    a = si.canonical_request(algorithm="ed25519", key_id="ab",
                             destination="c", method="GET", path="/p",
                             body=b"", timestamp=1, nonce="n")
    b = si.canonical_request(algorithm="ed25519", key_id="a",
                             destination="bc", method="GET", path="/p",
                             body=b"", timestamp=1, nonce="n")
    check("field boundaries cannot be shifted by field CONTENT", a != b)
    tricky = si.canonical_request(
        algorithm="ed25519", key_id="k", destination="d", method="GET",
        path="/p:1:x", body=b"", timestamp=1, nonce="n")
    check("a path containing the separator does not break the encoding",
          tricky.count(b"5:/p:1:x") == 0 and b"6:/p:1:x" in tricky)

    # ── 13. malformed signature headers ──
    for label, header in (
            ("no colons", "abcdef"),
            ("too few parts", "ed25519:agentic-v1"),
            ("too many parts", "ed25519:agentic-v1:aa:bb"),
            ("empty part", "ed25519::aa"),
            ("non-hex signature", "ed25519:agentic-v1:zzzz")):
        headers = {si.SIGNATURE_HEADER: header,
                   si.TIMESTAMP_HEADER: str(int(time.time())),
                   si.NONCE_HEADER: si.new_nonce()}
        check(f"a malformed signature ({label}) is refused",
              verify(headers)[0] is None)

    missing_stamp = dict(sign_as(agentic_priv, "agentic-v1"))
    missing_stamp.pop(si.TIMESTAMP_HEADER)
    check("a signature without a timestamp is refused",
          verify(missing_stamp)[0] is None)
    bad_stamp = dict(sign_as(agentic_priv, "agentic-v1"))
    bad_stamp[si.TIMESTAMP_HEADER] = "not-a-number"
    check("a non-integer timestamp is refused",
          verify(bad_stamp)[0] is None)

    # ── 14. the key map refuses what makes it meaningless ──
    for label, text in (
        ("not JSON", "{nope"),
        ("no keys section", '{"version": 1}'),
        ("an EMPTY key set", '{"keys": {}}'),
        ("a key with no identity",
         '{"keys": {"k": {"algorithm": "ed25519", "public_key": "aa"}}}'),
        ("an unsupported algorithm",
         '{"keys": {"k": {"identity": "x", "algorithm": "rot13", '
         '"public_key": "aa"}}}'),
        ("a public key that is not hex",
         '{"keys": {"k": {"identity": "x", "algorithm": "ed25519", '
         '"public_key": "zz"}}}'),
        ("a public key of the wrong length",
         '{"keys": {"k": {"identity": "x", "algorithm": "ed25519", '
         '"public_key": "aabb"}}}'),
    ):
        try:
            si.KeyMap.from_text(text)
            ok = False
        except si.IdentityError:
            ok = True
        check(f"the key map refuses {label}", ok)

    with tempfile.TemporaryDirectory() as tmp:
        good_map = Path(tmp) / "keys.json"
        good_map.write_text(json.dumps({"keys": {
            "agentic-v1": {"identity": "agentic", "algorithm": "ed25519",
                           "public_key": agentic_pub.hex()}}}))
        good_map.chmod(0o644)
        check("a world-readable key map is fine — public keys are public",
              len(si.KeyMap.load(str(good_map))) == 1)
        good_map.chmod(0o666)
        try:
            si.KeyMap.load(str(good_map))
            ok = False
        except si.IdentityError as exc:
            ok = "writable" in str(exc)
        check("a world-WRITABLE key map is refused — it could mint identities",
              ok)
        try:
            si.KeyMap.load(str(Path(tmp) / "absent.json"))
            ok = False
        except si.IdentityError:
            ok = True
        check("an absent key map is an error, not an empty one", ok)

    # ── 15. private key handling ──
    with tempfile.TemporaryDirectory() as tmp:
        secret = Path(tmp) / "private.key"
        secret.write_text(f"ed25519:{agentic_priv.hex()}")
        secret.chmod(0o600)
        os.environ[si.KEY_ID_ENV] = "agentic-v1"
        os.environ[si.PRIVATE_KEY_ENV] = str(secret)
        try:
            kid, alg, material = si.load_private_key()
            check("a private key loads from a Docker-secret style path",
                  kid == "agentic-v1" and alg == "ed25519"
                  and material == agentic_priv)
            secret.chmod(0o644)
            try:
                si.load_private_key()
                ok = False
            except si.IdentityError as exc:
                ok = "readable by group or other" in str(exc)
            check("a group/world-readable PRIVATE key is refused", ok)
            secret.chmod(0o600)

            os.environ[si.PRIVATE_KEY_ENV] = f"ed25519:{agentic_priv.hex()}"
            check("an inline private key also works",
                  si.load_private_key()[2] == agentic_priv)
            os.environ[si.PRIVATE_KEY_ENV] = "ed25519:not-hex"
            try:
                si.load_private_key()
                ok = False
            except si.IdentityError:
                ok = True
            check("a non-hex private key is refused", ok)
            os.environ.pop(si.KEY_ID_ENV)
            os.environ[si.PRIVATE_KEY_ENV] = f"ed25519:{agentic_priv.hex()}"
            try:
                si.load_private_key()
                ok = False
            except si.IdentityError:
                ok = True
            check("a missing key id is refused", ok)
        finally:
            os.environ.pop(si.KEY_ID_ENV, None)
            os.environ.pop(si.PRIVATE_KEY_ENV, None)

    # ── 16. the transition principal is honestly worthless ──
    anon = si.unverified_principal()
    check("a shared-token caller is NOT usable for provenance",
          not anon.usable_for_provenance)
    check("and its identity is not a service name",
          anon.identity == si.UNVERIFIED_IDENTITY and not anon.verified)
    check("a verified principal cannot be forged by construction alone",
          not si.ServicePrincipal(identity="agentic",
                                  verified=False).usable_for_provenance)

    # ── 17. END TO END across the seam ──
    #
    # The caller hashes the body it is about to send; the receiver hashes
    # the body that arrived. If those two serialisations differ by one
    # space every signature fails, and it fails looking like a key
    # problem rather than an encoding one. So this signs through the real
    # caller-side function and verifies through the real receiver-side
    # one, rather than trusting that both call the same encoder.
    import common.actuator_registry.mutating_handlers as mh
    import common.service_auth as sa

    with tempfile.TemporaryDirectory() as tmp:
        map_path = Path(tmp) / "keys.json"
        map_path.write_text(json.dumps({"keys": {
            "agentic-v1": {"identity": "agentic", "algorithm": "ed25519",
                           "public_key": agentic_pub.hex()}}}))
        map_path.chmod(0o644)
        env = {
            si.KEY_ID_ENV: "agentic-v1",
            si.PRIVATE_KEY_ENV: f"ed25519:{agentic_priv.hex()}",
            si.KEYMAP_ENV: str(map_path),
            si.NONCE_CACHE_ENV: str(Path(tmp) / "nonces.json"),
        }
        saved = {k: os.environ.get(k) for k in env}
        os.environ.update(env)
        sa.reset_identity_context()
        try:
            payload = {"target": "postgres", "confirm": True}
            raw = mh._encode_body(payload)
            headers = mh._auth_headers("shell-sandbox", "POST", "/run", raw)
            check("the caller produced a signature", si.SIGNATURE_HEADER in
                  {k.lower() for k in headers})

            principal, status, detail = sa.check_identity(
                headers, "tool_execute",
                destination=mh._destination("shell-sandbox"),
                method="POST", path="/run", body=raw)
            check("END TO END: the receiver identifies the caller by key",
                  principal is not None and principal.identity == "agentic"
                  and principal.verified)
            check("and that principal may be used for provenance",
                  principal.usable_for_provenance)

            # The encoder is shared and key-order independent, which is
            # the trap being guarded: a dict built in another order must
            # produce the same bytes and therefore the same hash.
            check("key order in the source dict does not change the bytes",
                  mh._encode_body(dict(reversed(list(payload.items()))))
                  == raw)

            # Re-verifying that same request is correctly a REPLAY. This
            # assertion originally read "still verifies", and the cache
            # refused it — the test was wrong, not the code.
            principal, status, detail = sa.check_identity(
                headers, "tool_execute",
                destination=mh._destination("shell-sandbox"),
                method="POST", path="/run", body=raw)
            check("END TO END: the replay cache refuses the second use",
                  principal is None and "already been seen" in detail)

            # A bad signature must NOT fall back to the shared token,
            # or an attacker could strip identity and stay anonymous.
            os.environ["KAI_SERVICE_TOKEN"] = "a-valid-shared-token"
            tampered = dict(headers)
            tampered["Authorization"] = "Bearer a-valid-shared-token"
            principal, status, _ = sa.check_identity(
                tampered, "tool_execute",
                destination=mh._destination("shell-sandbox"),
                method="POST", path="/run", body=b"different body")
            check("A BAD SIGNATURE DOES NOT DOWNGRADE to shared-token auth",
                  principal is None and status == 401)

            # Unsigned + valid token is accepted during the window, but
            # the principal is honestly anonymous.
            principal, status, _ = sa.check_identity(
                {"Authorization": "Bearer a-valid-shared-token"},
                "tool_execute", destination="shell-sandbox", method="POST",
                path="/run", body=raw)
            check("an unsigned caller with a valid token is accepted during "
                  "the transition window",
                  principal is not None and status == 200)
            check("but is NOT usable for provenance",
                  not principal.usable_for_provenance)

            # A signed request that cannot be EVALUATED -- key map present
            # but the backend missing -- must refuse with a clear 503, not
            # a 500 from somewhere deep, and must never fall back. This
            # path had never run until it was tested; it was found by
            # running the contingency rather than by reading the code.
            real_verify = si.verify_request
            si.verify_request = lambda *a, **k: (_ for _ in ()).throw(
                si.IdentityError("backend unavailable"))
            try:
                principal, status, detail = sa.check_identity(
                    headers, "tool_execute",
                    destination=mh._destination("shell-sandbox"),
                    method="POST", path="/run", body=raw)
                check("an unevaluable signature gives 503, not a crash",
                      principal is None and status == 503)
                check("and says the backend is the problem",
                      "backend is unavailable" in detail)
            finally:
                si.verify_request = real_verify

            os.environ[sa.REQUIRE_IDENTITY_ENV] = "true"
            principal, status, detail = sa.check_identity(
                {"Authorization": "Bearer a-valid-shared-token"},
                "tool_execute", destination="shell-sandbox", method="POST",
                path="/run", body=raw)
            check("and refused outright once identity is required",
                  principal is None and status == 401)
            check("with a reason naming why", "which service called" in detail)
        finally:
            os.environ.pop(sa.REQUIRE_IDENTITY_ENV, None)
            os.environ.pop("KAI_SERVICE_TOKEN", None)
            for key, value in saved.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value
            sa.reset_identity_context()

    print("=" * 66)
    print(f"Service identity (ed25519) tests: {PASSED} passed, {FAILED} failed")
    print(f"EXIT GATE: {'PASS' if FAILED == 0 else 'FAIL'}")
    return 1 if FAILED else 0


if __name__ == "__main__":
    print("Per-service ed25519 identity — the refusals")
    print("=" * 66)
    sys.exit(main())
