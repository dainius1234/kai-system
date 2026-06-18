# Kai PM Status Dashboard

- **Last updated (UTC):** 2026-06-18
- **Current phase:** Phase 0 — Pre-GPU Hardening
- **Current focus:** Decide merge order between `claude/project-rework-plan-pgvp35`
  (real, tested trust-loop + agentic-split work) and stale draft PRs #67/#69 (a
  different, unmerged file-reorg split of the same file). See
  [`REALITY_CHECK_2026-06-18.md`](REALITY_CHECK_2026-06-18.md) for the full picture
  — the 2026-06-02 "Cleanup Sprint Week 1/2" framing below this line is superseded.

## What's actually landed where

| Branch | Contains | Merged to `main`? |
|---|---|---|
| `main` | Keystone rename (`langgraph/`→`agentic/`), PM doc infra | — |
| `claude/project-rework-plan-pgvp35` | Trust-loop unification (D7), minimal-stack real spine (D8), agentic hot-path fix + `agentic-introspect` process split, sovereign-profile parity fix | **No** |
| PR #67 `copilot/cleanup-sprint-week-2-1-prompts` | `agentic/prompts.py` extraction | **No** — stale draft since 2026-06-02 |
| PR #69 `copilot/fix-core-tests-dockerfile` | `agentic/prompts.py` + `routes_identity/observability/ops/skills.py` | **No** — stale draft, conflicts with the branch above |

## Open PRs needing a decision

- [#46](https://github.com/dainius1234/kai-system/pull/46) — GPU Phase 0 consolidation (draft since 2026-04-21).
- [#54](https://github.com/dainius1234/kai-system/pull/54) — chassis polish C2/C5/C9 (draft since 2026-04-25).
- [#67](https://github.com/dainius1234/kai-system/pull/67), [#69](https://github.com/dainius1234/kai-system/pull/69) — see merge-order decision above.

Live list: https://github.com/dainius1234/kai-system/pulls

## Blocked items (GPU)

- Phase 1 — Local LLM Integration
- Phase 2 — Multi-Specialist Routing
- Phase 4 — Avatar / Voice / Multimodal
- Phase 5 — Production Hardening & Self-Improvement

Unlock condition: RTX 5080 procurement + provisioning + validation.

## Next 3 actions (priority order)

1. **Decide merge order** — land `claude/project-rework-plan-pgvp35` to `main`
   first (it's the tested, documented work), then re-evaluate #67/#69 against the
   new `agentic/app.py` shape.
2. **Live-verify** the minimal stack and the `agentic-introspect` split on a real
   Docker daemon — both are config/unit-test-validated only so far.
3. **Audit `memu-core` (~6,100 lines)** for the same hot/cold coupling `agentic`
   had, once the above lands.

## Sprint health signals (last hard measurement: 2026-06-02, not re-run this pass)

- Test baseline: 1,608 passed / 5 skipped on `main`'s green core (pre-dates the
  work on `claude/project-rework-plan-pgvp35`, which has its own clean
  `make test-core` run as of `fa18739`).
- Coverage on `common/`: 78% measured 2026-06-01. Repo-wide gate still unstarted.

## Source of truth pointers

- Latest reality check: [`REALITY_CHECK_2026-06-18.md`](REALITY_CHECK_2026-06-18.md)
- Resume layer: [`SESSION_BOOTSTRAP.md`](SESSION_BOOTSTRAP.md)
- Running log: [`../SESSION_BACKLOG.md`](../SESSION_BACKLOG.md)
- Decisions: [`DECISIONS.md`](DECISIONS.md)
