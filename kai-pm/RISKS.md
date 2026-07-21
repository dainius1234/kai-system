# Kai Risk Register

Last reviewed: 2026-07-21

| ID | Description | Severity | Likelihood | Mitigation | Owner | Status |
|---|---|---|---|---|---|---|
| R1 | GPU procurement delay (RTX 5080) blocks Phase 1/2/4/5 | H | H | Track procurement weekly; keep CPU-safe work moving | @dainius1234 | Active |
| R2 | Doc/code drift between README, CHANGELOG, and PM brain | M | M | SESSION_BOOTSTRAP + DECISIONS log enforced; full audit pass done 2026-07-21 | @dainius1234 | Active (reduced) |
| R3 | Test coverage % is unverified / not automated in CI gates | M | H | Add explicit automated coverage gate in CI and track in metrics | @dainius1234 | Active |
| R4 | Single-maintainer bus-factor risk | H | M | Keep bootstrap/decisions/status current and reduce tacit process knowledge | @dainius1234 | Active |
| R5 | Stale `claude/*` branches accumulating | M | L | Friday-cleanup.yml auto-flags branches >30d; currently 0 stale (2026-07-21 audit) | @dainius1234 | Mitigated |
