#!/usr/bin/env python3
"""`/observe_turn` — the first class-B endpoint to require verified identity.

This is the vertical slice that unblocks Cortex promotion, and it is
tested through the real FastAPI app rather than by calling the auth
helper directly. Calling the helper would prove the helper works; only a
request through the route proves the *endpoint* is governed.

The chain under test, end to end:

    per-service key -> caller signs the exact bytes -> receiver verifies
    -> principal derived from the verifying key -> route grant enforced
    -> provenance taken from the verified principal, never a header

And the property that makes it a security boundary rather than a
formality: **no accumulator moves unless the request was accepted.**
Every refusal below re-checks the topic history, the tacit message
lengths and the hourly counts, because an endpoint that rejects you
*after* learning from you has not rejected you.
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


def main() -> int:  # noqa: C901 - a list of refusals
    from common import service_identity as si

    available, detail = si.import_status()
    check(f"ed25519 backend importable ({detail})", available)
    if not available:
        print("EXIT GATE: FAIL")
        return 1

    agentic_priv, agentic_pub = si.generate_keypair()
    executor_priv, executor_pub = si.generate_keypair()

    tmp = tempfile.mkdtemp()
    keymap_path = Path(tmp) / "keys.json"
    keymap_path.write_text(json.dumps({
        "keys": {
            "agentic-v1": {"identity": "agentic", "algorithm": "ed25519",
                           "public_key": agentic_pub.hex()},
            # executor has a valid key and NO grant for this operation.
            "executor-v1": {"identity": "executor", "algorithm": "ed25519",
                            "public_key": executor_pub.hex()},
        },
        "grants": {"cortex_observe_turn": ["agentic"]},
    }))
    keymap_path.chmod(0o644)

    os.environ.update({
        si.KEYMAP_ENV: str(keymap_path),
        si.NONCE_CACHE_ENV: str(Path(tmp) / "nonces.json"),
        "KAI_SERVICE_NAME": "cortex",
        "KAI_SERVICE_TOKEN": "a-valid-shared-token",
        "FF_CORTEX": "false",          # no background refresh during tests
    })

    import common.service_auth as sa
    sa.reset_identity_context()

    from fastapi.testclient import TestClient
    import cortex.app as cortex

    client = TestClient(cortex.app)
    PATH = "/observe_turn"

    def sign(priv, kid, body_bytes, *, method="POST", path=PATH,
             destination="cortex", **over):
        return si.signed_headers(key_id=kid, algorithm="ed25519",
                                 private_material=priv,
                                 destination=destination, method=method,
                                 path=path, body=body_bytes, **over)

    def body_of(message: str) -> bytes:
        # Must satisfy TurnObservation, or the endpoint returns 422 from
        # validation and every accept-path assertion fails for a reason
        # that has nothing to do with identity. That is what happened on
        # the first run of this file: session_id was missing.
        return json.dumps({"session_id": "s-1", "user_message": message},
                          separators=(",", ":"), sort_keys=True).encode()

    def accumulators():
        return (len(cortex._topic_history), len(cortex._tacit_msg_lengths),
                dict(cortex._tacit_hourly_counts))

    def post(raw: bytes, headers):
        sent = dict(headers)
        sent["content-type"] = "application/json"
        return client.post(PATH, content=raw, headers=sent)

    # ── the happy path ──
    raw = body_of("how is the deployment going")
    before = accumulators()
    response = post(raw, sign(agentic_priv, "agentic-v1", raw))
    check("a signed, granted caller is accepted", response.status_code == 200)
    check("PROVENANCE IS DERIVED FROM THE KEY",
          response.json().get("turn_source") == "agentic")
    check("and it is recorded in Cortex state",
          cortex._state.last_turn_source == "agentic")
    check("the turn was actually learned from",
          accumulators()[0] == before[0] + 1)

    # ── every refusal, each re-checking that nothing was learned ──
    def refuse(label: str, raw_body: bytes, headers, expect_status=None):
        before_state = accumulators()
        source_before = cortex._state.last_turn_source
        resp = post(raw_body, headers)
        refused = resp.status_code >= 400
        check(f"{label} -> {resp.status_code}",
              refused and (expect_status is None
                           or resp.status_code == expect_status))
        check(f"  ...and NOTHING was learned from it",
              accumulators() == before_state
              and cortex._state.last_turn_source == source_before)
        return resp

    raw2 = body_of("second turn")

    # 1. no credentials at all
    refuse("no credentials", raw2, {}, 401)

    # 2. shared token alone attempting class-B authority
    refuse("shared token alone attempting class-B authority", raw2,
           {"Authorization": "Bearer a-valid-shared-token"}, 401)

    # 3. service A attempting service B's identity: executor's key,
    #    agentic's key id
    refuse("service A signing as service B", raw2,
           sign(executor_priv, "agentic-v1", raw2), 401)

    # 4. valid identity, no route grant
    refuse("VALID IDENTITY BUT NO ROUTE GRANT", raw2,
           sign(executor_priv, "executor-v1", raw2), 403)

    # 5. forged identity headers alongside a valid ungranted signature
    forged = dict(sign(executor_priv, "executor-v1", raw2))
    forged.update({"X-Kai-Identity": "agentic", "X-Actor-Did": "agentic",
                   "X-Service-Name": "agentic", "actor_did": "agentic"})
    refuse("forged identity/actor_did headers cannot buy a grant", raw2,
           forged, 403)

    # 6. altered body
    refuse("altered body", body_of("SOMETHING ELSE ENTIRELY"),
           sign(agentic_priv, "agentic-v1", raw2), 401)

    # 7. wrong path in the signature
    refuse("signature bound to another path", raw2,
           sign(agentic_priv, "agentic-v1", raw2, path="/state"), 401)

    # 8. wrong method in the signature
    refuse("signature bound to another method", raw2,
           sign(agentic_priv, "agentic-v1", raw2, method="DELETE"), 401)

    # 9. wrong destination service
    refuse("signature bound to another service", raw2,
           sign(agentic_priv, "agentic-v1", raw2, destination="executor"),
           401)

    # 10. stale timestamp
    refuse("stale timestamp", raw2,
           sign(agentic_priv, "agentic-v1", raw2,
                timestamp=int(time.time()) - 10_000), 401)

    # 11. bad signature WITH a valid shared token present — must not
    #     downgrade to membership auth
    downgrade = dict(sign(agentic_priv, "agentic-v1", raw2))
    downgrade[si.SIGNATURE_HEADER] = downgrade[si.SIGNATURE_HEADER][:-4] + "0000"
    downgrade["Authorization"] = "Bearer a-valid-shared-token"
    refuse("BAD SIGNATURE + VALID SHARED TOKEN does not downgrade", raw2,
           downgrade, 401)

    # 12. replayed nonce — needs a request that succeeded first
    replayable = sign(agentic_priv, "agentic-v1", raw2)
    first = post(raw2, replayable)
    check("a fresh granted request succeeds", first.status_code == 200)
    refuse("REPLAYED NONCE", raw2, replayable, 401)

    # ── the grant table fails closed ──
    ungranted_map = Path(tmp) / "nogrants.json"
    ungranted_map.write_text(json.dumps({"keys": {
        "agentic-v1": {"identity": "agentic", "algorithm": "ed25519",
                       "public_key": agentic_pub.hex()}}}))
    ungranted_map.chmod(0o644)
    os.environ[si.KEYMAP_ENV] = str(ungranted_map)
    os.environ[si.NONCE_CACHE_ENV] = str(Path(tmp) / "nonces2.json")
    sa.reset_identity_context()
    raw3 = body_of("third turn")
    refuse("an ABSENT grant table denies rather than permits", raw3,
           sign(agentic_priv, "agentic-v1", raw3), 403)

    # A grant naming an identity with no key is a typo or a removed key
    # left authorised. Both read as a working grant; neither is one.
    try:
        si.KeyMap.from_text(json.dumps({
            "keys": {"agentic-v1": {"identity": "agentic",
                                    "algorithm": "ed25519",
                                    "public_key": agentic_pub.hex()}},
            "grants": {"cortex_observe_turn": ["agentic", "ghost-service"]}}))
        ok = False
    except si.IdentityError as exc:
        ok = "no key in this map" in str(exc)
    check("a grant naming an identity with no key is refused", ok)

    # ── GET /state is class A and must be UNCHANGED by all of this ──
    os.environ[si.KEYMAP_ENV] = str(keymap_path)
    sa.reset_identity_context()
    state = client.get("/state",
                       headers={"Authorization": "Bearer a-valid-shared-token"})
    check("GET /state still works on the shared token — class A is untouched",
          state.status_code == 200)
    check("and /state carries no signature requirement",
          client.get("/state").status_code in (401, 403))

    for key in (si.KEYMAP_ENV, si.NONCE_CACHE_ENV, "KAI_SERVICE_NAME",
                "KAI_SERVICE_TOKEN", "FF_CORTEX"):
        os.environ.pop(key, None)
    sa.reset_identity_context()

    print("=" * 66)
    print(f"/observe_turn identity slice: {PASSED} passed, {FAILED} failed")
    print(f"EXIT GATE: {'PASS' if FAILED == 0 else 'FAIL'}")
    return 1 if FAILED else 0


if __name__ == "__main__":
    print("/observe_turn — the first governed class-B endpoint")
    print("=" * 66)
    sys.exit(main())
