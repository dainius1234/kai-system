# Kai Risk Register

Last reviewed: 2026-05-10

| ID | Description | Severity | Likelihood | Mitigation | Owner | Status |
|---|---|---|---|---|---|---|
| R1 | GPU procurement delay (RTX 5080) blocks Phase 1/2/4/5 | H | H | Track procurement weekly; keep Phase 0/CPU-safe work moving | @dainius1234 | Active |
| R2 | Doc/code drift between README, CHANGELOG, and PM brain | M | M | This cleanup PR + enforce diff-vs-README ritual before status claims | @dainius1234 | Active |
| R3 | Test coverage % is unverified / not automated in CI gates | M | H | Add explicit automated coverage gate in CI and track in metrics | @dainius1234 | Active |
| R4 | Single-maintainer bus-factor risk | H | M | Keep bootstrap/decisions/status current and reduce tacit process knowledge | @dainius1234 | Active |
| R5 | Stale `claude/*` branches accumulating | M | M | Periodic branch hygiene review and delete stale branches post-merge | @dainius1234 | Active (6 stale branches) |
