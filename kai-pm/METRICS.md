# Kai PM Metrics

## Snapshot (2026-07-21 UTC)

| Metric | Current value | Notes |
|---|---|---|
| Test count | ~1,650+ tests | Last hard count ~1,620 pre-Phase-0.5; Phase 0.5 (D55–D59) added tests for financial awareness, behavioral scoreboard, LLM retry — exact count not re-measured |
| Test coverage % | ~60% (estimated) | Measurement not yet automated in PM workflow (R3 open risk) |
| Open PR count | 0 | All closed as of 2026-07-21 |
| Open issue count | Unknown | Not re-queried this session |
| Merged PRs (Phase 0.5) | 8 PRs (#77–#85) | All CPU-safe Phase 0.5 items shipped |
| Stale branches | 0 | 2 branches total: `main` + `claude/project-rework-plan-pgvp35` (both at same SHA) |
| Weekly CI | Active | weekly-report-card.yml fires Mon 09:00 UTC; behavioral scoreboard advisory |
| Friday cleanup | Active | friday-cleanup.yml fires Fri 09:00 UTC |

## Measurement gaps

- Automated test count / coverage still not piped into a PM metrics workflow (R3).
- Issue count not re-queried; use GitHub for live figures.
