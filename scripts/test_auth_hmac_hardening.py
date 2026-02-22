#!/usr/bin/env python3
from __future__ import annotations

import os
import time

from common.auth import sign_gate_request, sign_gate_request_bundle, verify_gate_signature


def main() -> int:
    actor = "langgraph"
    session = "bootstrap-token-1"
    tool = "executor"
    nonce = "n-hardening"
    ts = time.time()

    os.environ["INTERSERVICE_HMAC_SECRET"] = "new-secret"
    os.environ["INTERSERVICE_HMAC_SECRET_PREV"] = "old-secret"
    os.environ["INTERSERVICE_HMAC_KEY_ID"] = "k-new"
    os.environ["INTERSERVICE_HMAC_KEY_ID_PREV"] = "k-old"

    bundle = sign_gate_request_bundle(actor_did=actor, session_id=session, tool=tool, nonce=nonce, ts=ts)
    assert len(bundle) == 2
    assert any(sig.startswith("k-new:") for sig in bundle)
    assert any(sig.startswith("k-old:") for sig in bundle)

    os.environ["INTERSERVICE_HMAC_STRICT_KEY_ID"] = "true"
    good = sign_gate_request(actor_did=actor, session_id=session, tool=tool, nonce=nonce, ts=ts, key_id="k-new")
    assert verify_gate_signature(actor_did=actor, session_id=session, tool=tool, nonce=nonce, ts=ts, signature=good)

    wrong_kid = "wrong:" + good.split(":", 1)[1]
    assert not verify_gate_signature(actor_did=actor, session_id=session, tool=tool, nonce=nonce, ts=ts, signature=wrong_kid)

    os.environ["INTERSERVICE_HMAC_REVOKED_IDS"] = "k-new"
    assert not verify_gate_signature(actor_did=actor, session_id=session, tool=tool, nonce=nonce, ts=ts, signature=good)

    print("test_auth_hmac_hardening: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
