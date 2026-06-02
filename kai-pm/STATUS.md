# Kai PM Status Dashboard

- **Last updated (UTC):** 2026-06-02
- **Current phase:** Phase 0 — Pre-GPU Hardening
- **Current focus:** Cleanup Sprint Week 2.1 — low-risk `agentic/app.py` route split plus PM truth reconciliation.

## In-flight PRs (Cleanup Sprint)

See live list: https://github.com/dainius1234/kai-system/pulls

- Week 1 housekeeping agent — close #59 + #60, label PRs, re-triage tempo-test assumptions
- ✅ Week 2.1 prep — `kai-pm/AGENTIC_APP_MAP.md` landed 2026-06-02
- ✅ Week 2.2 prep — `kai-pm/COMPOSE_DRIFT.md` landed 2026-06-02
- ✅ Week 2.3 prep — `kai-pm/MAKEFILE_AUDIT.md` **landed** 2026-06-02
- Week 2.1 route split — move SOUL/AGENTS, skills, and observability leaves out of `agentic/app.py`

## Recently merged

- ✅ Week 1.4 KEYSTONE — `langgraph/` → `agentic/` rename merged
- ✅ [#46](https://github.com/dainius1234/kai-system/pull/46) — GPU Phase 0 consolidation merged 2026-06-01; `main` already contains the canonical Phase 0 baseline

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
2. **Finish the low-risk Week 2.1 split** — follow `AGENTIC_APP_MAP.md` after the already-landed `prompts.py` split; current leaf is the SOUL/skills/observability route surface.
3. **Land Makefile cleanup** — `merge-gate` composition is already honest; the remaining work is slimming the target surface and archiving non-live helpers.

## Sprint health signals

- Test baseline (2026-06-02 00:15 UTC): 1,608 passed / 5 skipped on green core.
- Baseline checks re-verified 2026-06-02: `make go_no_go` ✅, `make check-docs` ✅. `make test-core` is currently blocked in this sandbox until Python deps are installed (`fastapi` missing).
- CI needs re-triage against current code reality: `scripts/test_tempo.py` now targets live `memu-core` tempo behavior, so the old "tempo orphan" assumption is stale.
- Coverage on `common/`: 78% measured 2026-06-01. Repo-wide gate is Week 3 work.

## Source of truth pointers

- Live cleanup tracker: [`CLEANUP_TODO.md`](CLEANUP_TODO.md)
- Resume layer: [`SESSION_BOOTSTRAP.md`](SESSION_BOOTSTRAP.md)
- Running log: [`../SESSION_BACKLOG.md`](../SESSION_BACKLOG.md)
- Decisions: [`DECISIONS.md`](DECISIONS.md)
