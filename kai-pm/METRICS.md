# Kai PM Metrics

## Snapshot (2026-06-02 UTC)

| Metric | Current value | Notes |
|---|---|---|
| Test count | 1,625 tests | From `make check-docs` (`scripts/sync_docs.py --check`) |
| Test coverage % | 78% on `common/` only | Verified 2026-06-01; repo-wide coverage is still not automated |
| Open PR count | 3 open PRs | #54, #67, #69 from GitHub API (`list_pull_requests state=open`) |
| Average PR age | ~13.0 days | Calculated from current open PR `created_at` timestamps (manual MCP-derived calculation) |
| Open issue count by label | 2 open issues (`pm`: 2, `tech-watch`: 2) | From GitHub API issue listing (issues #56, #61) |
| Tech debt items logged | TBD — measurement not yet automated | Debt tracked in `docs/PROJECT_BACKLOG.md` (e.g., P6 parking lot) |
| Velocity (merged PRs/week, last 4 weeks) | 3.25 PR/week | 13 merged PRs in the last 28 days (`search_pull_requests is:merged merged:>=2026-05-05`) |

## Measurement gaps

- Automated PR-age and velocity dashboards are not wired yet (currently computed manually from GitHub API output).
- Coverage is documented but not yet piped into a PM metrics workflow.
