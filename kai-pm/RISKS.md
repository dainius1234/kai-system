# Kai Risk Register

Last reviewed: 2026-06-02

| ID | Description | Severity | Likelihood | Mitigation | Owner | Status |
|---|---|---|---|---|---|---|
| R1 | GPU procurement delay (RTX 5080) blocks Phase 1/2/4/5 | H | H | Track procurement weekly; keep Phase 0/CPU-safe work moving | @dainius1234 | Active |
| R2 | PM/status truth drifts after merges, leaving stale references in STATUS/backlog/risk docs | M | M | Re-run `make check-docs`, refresh PM docs in the same cleanup block, and treat `kai-pm/STATUS.md` as the first post-merge reconciliation target | @dainius1234 | Active |
| R3 | Test coverage % is unverified / not automated in CI gates | M | H | Add explicit automated coverage gate in CI and track in metrics | @dainius1234 | Active |
| R4 | Single-maintainer bus-factor risk | H | M | Keep bootstrap/decisions/status current and reduce tacit process knowledge | @dainius1234 | Active |
| R5 | Post-merge branch hygiene follow-through can stall after major consolidation PRs | M | M | After merged milestones (for example PR #46), explicitly review and delete the now-stale helper branches listed in the merged PR description | @dainius1234 | Active (5 stale branches still safe to delete) |
