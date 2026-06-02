# Kai PM Status Dashboard

- **Last updated (UTC):** 2026-06-02
- **Current phase:** Phase 0 — Pre-GPU Hardening
- **Current focus:** Cleanup Sprint Week 1 → Week 2 transition. Keystone rename merged; Week 2 prep docs landing.

## In-flight PRs (Cleanup Sprint)

See live list: https://github.com/dainius1234/kai-system/pulls

- Week 1 housekeeping agent — close #59 + #60, label PRs, delete tempo tests
- Week 2.1 prep — `kai-pm/AGENTIC_APP_MAP.md` (not yet landed)
- Week 2.2 prep — `kai-pm/COMPOSE_DRIFT.md` (not yet landed)
- ✅ Week 2.3 prep — `kai-pm/MAKEFILE_AUDIT.md` **landed** 2026-06-02
- [#46](https://github.com/dainius1234/kai-system/pull/46) — GPU Phase 0 consolidation (draft since 2026-04-21 — needs land-or-close decision)

## Recently merged

- ✅ Week 1.4 KEYSTONE — `langgraph/` → `agentic/` rename merged

## Blocked items (GPU)

- Phase 1 — Local LLM Integration
- Phase 2 — Multi-Specialist Routing
- Phase 4 — Avatar / Voice / Multimodal
- Phase 5 — Production Hardening & Self-Improvement

Unlock condition: RTX 5080 procurement + provisioning + validation.

## Paused (until Cleanup Weeks 1–3 done)

- All feature PRs (J/P/H series)
- P29 Financial Awareness
- GPU readiness pre-wiring

## Next 3 actions (priority order)

1. **Review + merge** `AGENTIC_APP_MAP.md` and `COMPOSE_DRIFT.md` PRs when they land — the design specs for Week 2 splits.
2. **Dispatch first Week 2.1 split PR** — smallest leaf from `agentic/app.py` (likely `prompts/` extraction — pure data, no behavior).
3. **Land-or-close PR #46** — 6+ weeks stale; either rebase + merge or close and rebuild post-cleanup.

## Sprint health signals

- Test baseline (2026-06-02 00:15 UTC): 1,608 passed / 5 skipped on green core.
- 14 CI failures are all already on in-flight agent kill lists (12 tempo orphans + 1 §1.3 + 1 keystone-dependent — see `SESSION_BACKLOG.md`).
- Coverage on `common/`: 78% measured 2026-06-01. Repo-wide gate is Week 3 work.

## Source of truth pointers

- Live cleanup tracker: [`CLEANUP_TODO.md`](CLEANUP_TODO.md)
- Resume layer: [`SESSION_BOOTSTRAP.md`](SESSION_BOOTSTRAP.md)
- Running log: [`../SESSION_BACKLOG.md`](../SESSION_BACKLOG.md)
- Decisions: [`DECISIONS.md`](DECISIONS.md)
