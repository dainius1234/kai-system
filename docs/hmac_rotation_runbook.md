# HMAC Rotation Runbook (Project Lead)

This runbook is the concrete operating procedure for `INTERSERVICE_HMAC_SECRET` rotation with zero downtime.

## Preconditions
- `tool-gate` is running with `REQUIRE_SIGNATURE=true`.
- `langgraph` signs requests through `common.auth.sign_gate_request`.
- Operators can deploy environment-variable changes in two steps.

## Rotation Procedure
1. **Current state**
   - `INTERSERVICE_HMAC_SECRET=old`
   - `INTERSERVICE_HMAC_SECRET_PREV` unset.

2. **Overlap window (safe cutover)**
   - Set `INTERSERVICE_HMAC_SECRET=new`.
   - Set `INTERSERVICE_HMAC_SECRET_PREV=old`.
   - Restart/redeploy services.
   - Verify both old and new signatures are accepted.

3. **Retirement window**
   - Keep `INTERSERVICE_HMAC_SECRET=new`.
   - Unset `INTERSERVICE_HMAC_SECRET_PREV`.
   - Restart/redeploy services.
   - Verify old signatures are rejected.

## Validation
- Run:
  - `make hmac-rotation-drill`
  - `make game-day-scorecard`
- Confirm no `invalid request signature` spikes in logs.

## Rollback
- If cutover fails in overlap:
  - Revert `INTERSERVICE_HMAC_SECRET` to `old`.
  - Unset `_PREV`.
  - Redeploy.


## Prepare-now controls (while staying on HMAC)
- `TOOL_GATE_DUAL_SIGN=true`: LangGraph sends both current + previous signatures in `signatures[]` during overlap windows.
- `INTERSERVICE_HMAC_KEY_ID_PREV`: explicit key-id label for previous secret.
- `INTERSERVICE_HMAC_STRICT_KEY_ID=true`: enforces key-id/secret binding during verification (recommended after all services send correct key IDs).

Suggested sequence:
1. Overlap with `TOOL_GATE_DUAL_SIGN=true` and strict mode off.
2. Verify no signature errors, then enable `INTERSERVICE_HMAC_STRICT_KEY_ID=true`.
3. Disable dual-sign once old secret is retired.

## Decision checkpoint for future
Keep HMAC for the current phase, then migrate in the next phase when **any 3 of these 7 triggers** are true:
- Service scale reaches `AUTH_SERVICES >= 8`.
- Ownership reaches `AUTH_TEAMS >= 3`.
- Rotation overhead reaches `HMAC_ROTATIONS_PER_QUARTER >= 4`.
- At least one auth incident in 90 days (`HMAC_INCIDENTS_90D >= 1`).
- At least one external verifier requires non-shared secrets (`EXTERNAL_VERIFIER_DEPENDENCIES >= 1`).
- Zero-trust mandate enabled (`ZERO_TRUST_TARGET=true`).
- Auditability pressure is high (`AUDITABILITY_SCORE >= 0.80`).

Use the advisor command to score readiness:
- `make hmac-migration-advice`

If migration is triggered, use phased rollout: dual-sign verify first, then enforce per-service asymmetric identity (mTLS SPIFFE/SPIRE or Ed25519 keypairs with key IDs).
