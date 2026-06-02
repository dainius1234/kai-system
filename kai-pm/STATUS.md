# Kai PM Status Dashboard

- **Last updated (UTC):** 2026-06-02
- **Current phase:** Phase 0 — Pre-GPU Hardening
- **Current focus:** Cleanup Sprint Week 1 → Week 2 transition. Keystone rename merged; Week 2 prep docs landing.

## In-flight PRs (Cleanup Sprint)

See live list: https://github.com/dainius1234/kai-system/pulls

- Week 1 housekeeping agent — close #59 + #60, label PRs, re-triage tempo-test assumptions
- ✅ Week 2.1 prep — `kai-pm/AGENTIC_APP_MAP.md` landed 2026-06-02
- ✅ Week 2.2 prep — `kai-pm/COMPOSE_DRIFT.md` landed 2026-06-02
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

1. **Use the landed prep docs** — `AGENTIC_APP_MAP.md` and `COMPOSE_DRIFT.md` are now the design specs for Week 2 app and compose cleanup.
2. **Dispatch next Week 2.1 split PR** — follow `AGENTIC_APP_MAP.md` after the already-landed `prompts.py` split; next safest leaf is the low-risk admin/ops route surface.
3. **Land Makefile cleanup + decide PR #46** — keep `merge-gate` honest, trim alias noise, and make the stale-PR call.

## Sprint health signals

- Test baseline (2026-06-02 00:15 UTC): 1,608 passed / 5 skipped on green core.
- CI needs re-triage against current code reality: `scripts/test_tempo.py` now targets live `memu-core` tempo behavior, so the old "tempo orphan" assumption is stale.
- Coverage on `common/`: 78% measured 2026-06-01. Repo-wide gate is Week 3 work.

## Source of truth pointers

- Live cleanup tracker: [`CLEANUP_TODO.md`](CLEANUP_TODO.md)
- Resume layer: [`SESSION_BOOTSTRAP.md`](SESSION_BOOTSTRAP.md)
- Running log: [`../SESSION_BACKLOG.md`](../SESSION_BACKLOG.md)
- Decisions: [`DECISIONS.md`](DECISIONS.md)
