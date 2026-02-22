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

## Decision checkpoint for future
- If key-distribution complexity grows or multi-tenant boundaries tighten,
  evaluate migration to ed25519 per-service keys.
