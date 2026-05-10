# Reality Check — 2026-05-10 UTC

Purpose: reconcile README + PM brain facts with repository reality without changing project vision/scope.

## Metrics and facts checked

| Item | Source command/file | Old value | New value | Notes |
|---|---|---|---|---|
| README status date | `make check-docs` (`scripts/sync_docs.py --check`) | 21 April 2026 | 10 May 2026 | Auto-synced date target from docs scanner |
| Services count | `make sync-docs` scanner output (`count_services()` over `docker-compose.full.yml`) | 27 (table already current) + stale 26 in badges/quick refs | 27 everywhere in README metrics callouts | Badge + quick-reference drift fixed |
| Test targets count | `make sync-docs` scanner output (`count_test_targets()` from `Makefile test-core`) | Mixed 74 / 88 / 89 | 74 everywhere in README + PM metrics | Conflicting values removed |
| Individual test count | `make sync-docs` scanner output (`count_test_functions()`) | Mixed 1,587 / 1,618 / 1,624 | 1,620 everywhere in README + PM metrics | Conflicting values removed |
| Python LOC | `make sync-docs` scanner output (`count_python_loc()`) | ~42,537 (+ stale 41,107 badge) | ~42,613 | Badge and status values aligned |
| Compose file count | `make sync-docs` scanner output (`count_compose_files()`) | 3 | 3 | No change required |
| Milestones shipped | `make sync-docs` scanner output (`count_milestones()` from README DONE marks) | Mixed 31 / 32 | 32 everywhere in README metrics callouts | Badge + milestone summary drift fixed |
| Failures | README status table | 0 | 0 | No change |
| Malformed README badge HTML | `README.md` top badge block | `src=\"...\"` escaped quotes on tests badge | `src="..."` valid HTML | Rendering fix only |
| Open PR count | GitHub MCP `list_pull_requests` (`state=open`) | 1 | 3 | Open PRs: #46, #54, #58 |
| Average open PR age | GitHub MCP open PR data + manual timestamp calculation | TBD | ~11.5 days | Computed from `created_at` values at scan time |
| Open issue count by label | GitHub MCP `list_issues` (`state=OPEN`) | 0 open issues | 1 open issue (`pm`: 1, `tech-watch`: 1) | Issue #56 |
| Velocity (merged PRs/week, last 4 weeks) | GitHub MCP `search_pull_requests` (`is:merged merged:>=2026-04-12`) | ~1.0 PR/week | 2.5 PR/week | 10 merged PRs in last 28 days |
| Strategic correction flag P29 | `kai-pm/STRATEGIC_PLAN.md` | P29 placement TBD | P29 explicitly placed in Phase 3 | Rationale documented (memory/reflection, CPU-safe) |
| Sequence alignment | `kai-pm/SEQUENCE.md` | No explicit P29 placement note | P29 note aligned with STRATEGIC_PLAN | J-series correction ref kept (`CHANGELOG.md` 0.28.0 + `97a3a61`) |
| Risk R5 stale branches | `git ls-remote --heads origin 'refs/heads/claude/*' \| wc -l` | Active (no count) | Active (6 stale branches) | Added concrete count |
| Tech watch month header | `kai-pm/TECH_WATCH.md` | April 2026 | May 2026 | Month rolled forward to current review month |

## Validation run log

- `make check-docs` (pre-fix): **FAILED** due to stale README/PROJECT_BACKLOG metric values.
- `make sync-docs`: **PASSED**, updated README status table + `docs/PROJECT_BACKLOG.md`.
- `make check-docs` (post-fix): see current run in this PR (must pass before merge).

## Outstanding flags (still honest TBD)

- Test coverage percentage remains estimated (`~60%`) and is **TBD — measurement not yet automated** in PM metrics workflow.
- Tech debt item total remains **TBD — measurement not yet automated** (tracked qualitatively in backlog).
- PR-age/velocity are currently computed manually from GitHub API output; no dedicated automated PM dashboard pipeline yet.
