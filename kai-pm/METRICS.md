# Kai PM Metrics

## Snapshot (2026-05-10 UTC)

| Metric | Current value | Notes |
|---|---|---|
| Test count | 1,620 tests | From `make sync-docs` (`scripts/sync_docs.py` scanner) |
| Test coverage % | ~60% (estimated) | Existing documented estimate; TBD — measurement not yet automated in PM workflow |
| Open PR count | 3 open PRs | #46, #54, #58 from GitHub API (`list_pull_requests state=open`) |
| Average PR age | ~11.5 days | Calculated from current open PR `created_at` timestamps (manual MCP-derived calculation) |
| Open issue count by label | 1 open issue (`pm`: 1, `tech-watch`: 1) | From GitHub API issue listing (issue #56) |
| Tech debt items logged | TBD — measurement not yet automated | Debt tracked in `docs/PROJECT_BACKLOG.md` (e.g., P6 parking lot) |
| Velocity (merged PRs/week, last 4 weeks) | 2.5 PR/week | 10 merged PRs in the last 28 days (`search_pull_requests is:merged merged:>=2026-04-12`) |

## Measurement gaps

- Automated PR-age and velocity dashboards are not wired yet (currently computed manually from GitHub API output).
- Coverage is documented but not yet piped into a PM metrics workflow.
