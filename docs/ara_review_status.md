# Ara Review Follow-up Status

Implemented in this update:

1. Append-only checksummed episode spool fallback (`ChecksummedSpoolSaver`).
2. Error-budget-based dependency guards with warn/open thresholds.
3. HMAC revocation support + auto-rotation helper script.
4. Egress whitelist policy script updated for tool-gate + telegram bridge.
5. Chaos CI drill script with random service kill/restart and SLO gating.
6. Conviction override control in `kai_control` UI and consumption in `langgraph`.
7. Offline paper backup was already present before this update.

Notes:
- Rotation helper emits env values; integrating with Vault-managed secret pipelines is the next hardening step.
- Chaos script currently runs process-level chaos in local Python mode; container-level chaos can be added once Docker CI workers are available.
