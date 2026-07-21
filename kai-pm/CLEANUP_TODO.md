# Cleanup Sprint TODO — Live Tracker (superseded — see note)

> ⚠️ **2026-06-18: this tracker describes a plan that did not execute as written.**
> §2.1's routes/state/flows/providers/prompts split stalled in two open draft PRs
> (#67, #69) that never merged. A different approach (process-level failure-domain
> split) shipped on `claude/project-rework-plan-pgvp35` instead — not yet merged to
> `main` either. See [`REALITY_CHECK_2026-06-18.md`](REALITY_CHECK_2026-06-18.md)
> for the full picture before trusting any checkbox below as current.

> **Single source of truth (historical).** Tick boxes as work lands on `main`.

---

## Status legend
- [ ] not started
- [~] in progress (agent dispatched / PR open)
- [x] done, merged to `main`
- [!] blocked or needs Dainius decision

---

## Week 1 — Stop the Bleeding

### 1.1 / 1.2 / 1.5 / 1.6 — PR housekeeping + delete orphan tempo tests
- [x] `scripts/test_tempo.py` deleted (D70, 2026-07-21).
- [x] `test-tempo` Makefile target deleted and removed from `.PHONY` and `test-core` deps (D70, 2026-07-21).

### 1.3 Fix correction-memory ranking test
- [ ] Investigate `scripts/test_p3_organic_memory.py::test_correction_memory_gets_boost` after 1.2 lands.

### 1.4 Rename `langgraph/` → `agentic/` ⭐ KEYSTONE
- [x] Mechanical rename. ~30 file edits.
- [x] Update imports, compose files, Dockerfiles, Makefile, docs.
- [x] Remove sys.path hack in `scripts/agentic_integration_test.py`.
- [x] CI green. Merge.

---

## Week 2 — Untangle the Giant

### 2.1 Split `agentic/app.py`
- [ ] `AGENTIC_APP_MAP.md` — never landed, file does not exist in the repo.
- [!] Stalled as written. PR #67 (`prompts/` only) and PR #69 (`prompts/` +
      `routes_identity/observability/ops/skills.py`) are both open drafts, no
      activity since 2026-06-02, neither merged.
- [x] A different, process-level split (not a file reorg) shipped on
      `claude/project-rework-plan-pgvp35`: `agentic` keeps chat/run/checkpoints/
      skills, `agentic-introspect` is a new service owning dream/evolve/
      security-audit. **Not yet merged to `main`** — needs the merge-order
      decision in `REALITY_CHECK_2026-06-18.md` before #67/#69 are touched.

### 2.2 Reconcile docker-compose files
- [~] Diff minimal vs sovereign vs full → `kai-pm/COMPOSE_DRIFT.md` (agent dispatched 2026-06-01, not yet landed).
- [ ] Extract shared config to base.

### 2.3 Prune Makefile 100 → ~25 targets
- [x] Audit landed → [`MAKEFILE_AUDIT.md`](MAKEFILE_AUDIT.md) (2026-06-02).
- [x] 10 DELETE targets removed; `Makefile.archive` created with preserved definitions (D70, 2026-07-21).
- [ ] Slim `test-core` deps to KEEP-only list (blocked: needs honest merge-gate recomposition first).
- [ ] Update `merge-gate` to run the full live list per audit's "honest merge-gate" proposal.

---

## Week 3 — Honest Verification

- [ ] Run every surviving Makefile target. Categorise.
- [ ] Repo-wide coverage gate (currently only `common/` at 78%).
- [ ] `merge-gate` honesty.

---

## Week 4 — Resume Features (only if 1-3 done)

- [ ] Multi-backend LLM router (per Codex design).
- [ ] Skills templates, journal templates, CIS P29.

---

## Decisions log

| Date | Decision | Why |
|---|---|---|
| 2026-06-01 | Pause all feature PRs | Audit revealed 3 hot spots; foundation must stabilise |
| 2026-06-01 | Close PR #60 | Empty (only "Initial plan" commit, no code) |
| 2026-06-01 | Close PR #59 | Built on stale main; rebuild post-cleanup |
| 2026-06-01 | Rename langgraph → agentic | Self-inflicted PyPI shadowing |
| 2026-06-01 | langgraph → agentic rename merged | Unblocks Week 2 app.py split |
| 2026-06-01 | Delete tempo tests | Test attributes don't exist; orphans, not bugs |
| 2026-06-02 | MAKEFILE_AUDIT landed | Week 2.3 design spec ready; archival + merge-gate work unblocked |
| 2026-06-02 | First Week 2.1 split = `prompts/` | Pure data leaf, lowest risk, sets pattern for harder splits |

---

## Active agent sessions

| Dispatched | Scope | Status |
|---|---|---|
| 2026-06-01 | Week 1 housekeeping (close #59 + #60, label PRs, delete tempo tests) | running |
| 2026-06-01 | Week 1.4 KEYSTONE — `langgraph/` → `agentic/` rename | ✅ merged |
| 2026-06-01 | Week 2.1 prep — `AGENTIC_APP_MAP.md` | running (not landed) |
| 2026-06-01 | Week 2.2 prep — `COMPOSE_DRIFT.md` | running (not landed) |
| 2026-06-01 | Week 2.3 prep — `MAKEFILE_AUDIT.md` | ✅ merged 2026-06-02 |
