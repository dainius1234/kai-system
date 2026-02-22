from __future__ import annotations

import os
import time

from common.auth import sign_gate_request, verify_gate_signature

ACTOR = "langgraph"
SESSION = "bootstrap-token-1"
TOOL = "executor"
NONCE = "rotation-drill"
TS = time.time()


def _assert(condition: bool, msg: str) -> None:
    if not condition:
        raise AssertionError(msg)


def main() -> None:
    # Phase 1: old key is active primary
    os.environ["INTERSERVICE_HMAC_SECRET"] = "old-secret"
    os.environ.pop("INTERSERVICE_HMAC_SECRET_PREV", None)
    sig_old = sign_gate_request(actor_did=ACTOR, session_id=SESSION, tool=TOOL, nonce=NONCE, ts=TS)
    _assert(verify_gate_signature(actor_did=ACTOR, session_id=SESSION, tool=TOOL, nonce=NONCE, ts=TS, signature=sig_old), "phase1 verify old failed")

    # Phase 2: rotate with overlap (new primary, old secondary)
    os.environ["INTERSERVICE_HMAC_SECRET"] = "new-secret"
    os.environ["INTERSERVICE_HMAC_SECRET_PREV"] = "old-secret"
    sig_new = sign_gate_request(actor_did=ACTOR, session_id=SESSION, tool=TOOL, nonce=NONCE, ts=TS)
    _assert(verify_gate_signature(actor_did=ACTOR, session_id=SESSION, tool=TOOL, nonce=NONCE, ts=TS, signature=sig_new), "phase2 verify new failed")
    _assert(verify_gate_signature(actor_did=ACTOR, session_id=SESSION, tool=TOOL, nonce=NONCE, ts=TS, signature=sig_old), "phase2 verify old overlap failed")

    # Phase 3: retire old key (new only)
    os.environ["INTERSERVICE_HMAC_SECRET"] = "new-secret"
    os.environ.pop("INTERSERVICE_HMAC_SECRET_PREV", None)
    _assert(verify_gate_signature(actor_did=ACTOR, session_id=SESSION, tool=TOOL, nonce=NONCE, ts=TS, signature=sig_new), "phase3 verify new failed")
    _assert(not verify_gate_signature(actor_did=ACTOR, session_id=SESSION, tool=TOOL, nonce=NONCE, ts=TS, signature=sig_old), "phase3 old should fail")

    print("hmac rotation drill passed")


if __name__ == "__main__":
    main()
