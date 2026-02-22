# Ara Review Follow-up Status

Implemented in this update:

1. Append-only checksummed episode spool fallback (`ChecksummedSpoolSaver`).
2. Error-budget-based dependency guards with warn/open thresholds.
3. HMAC revocation support + auto-rotation helper script.
4. Egress whitelist policy script updated for tool-gate + telegram bridge.
5. Chaos CI drill script with random service kill/restart and SLO gating.
6. Conviction override control in `kai_control` UI and consumption in `langgraph`.
7. Offline paper backup was already present before this update.

8. Phase-1 Patch Set B catch-up: memu-core state key/value size guards, memory cap (`MAX_MEMORY_RECORDS`), and `/memory/diagnostics` endpoint.

Notes:
- Rotation helper emits env values; integrating with Vault-managed secret pipelines is the next hardening step.
- Chaos script currently runs process-level chaos in local Python mode; container-level chaos can be added once Docker CI workers are available.

9. Phase-1 Patch Set C completion: executor now returns `policy_context` and structured timeout 408 bodies.
10. Compose profile coupling fixed: core services no longer hard-depend on dev-profile `vault` service.
11. Added deterministic `phase1-closure` static gate (`scripts/phase1_closure_check.py`) to verify Patch Sets A-F closure criteria in-repo.

12. Chaos CI strengthened to include random hard-kill + recovery and memu degradation before SLO scorecard gating.
13. Error-budget guard alerting now pushes explicit Telegram message when dependency guard enters half_open/open state.
14. Monthly paper backup automation added (`scripts/monthly_paper_backup.py` + shell wrapper) for offline recovery artifacts.
15. Egress whitelist updated to allow optional Telegram HTTPS IP allowlist while keeping deny-by-default.
16. Weekly key rotation wrapper added (`scripts/weekly_key_rotation.sh`) to automate 7-day rotation cadence.
