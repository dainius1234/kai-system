# Cleanup Sprint TODO — Live Tracker

> **Single source of truth.** Tick boxes as work lands on `main`.

---

## Status legend
- [ ] not started
- [~] in progress (agent dispatched / PR open)
- [x] done, merged to `main`
- [!] blocked or needs Dainius decision

---

## Week 1 — Stop the Bleeding

### 1.1 / 1.2 / 1.5 / 1.6 — PR housekeeping + delete orphan tempo tests
- [~] Coding agent dispatched 2026-06-01.
- Single PR will: close #59, close #60, label other open PRs, delete `scripts/test_tempo.py`, remove `test-tempo` Makefile target.

### 1.3 Fix correction-memory ranking test
- [ ] Investigate `scripts/test_p3_organic_memory.py::test_correction_memory_gets_boost` after 1.2 lands.

### 1.4 Rename `langgraph/` → `agentic/` ⭐ KEYSTONE
- [ ] Mechanical rename. ~30 file edits.
- [ ] Update imports, compose files, Dockerfiles, Makefile, docs.
- [ ] Remove sys.path hack in `scripts/agentic_integration_test.py`.
- [ ] CI green. Merge.

---

## Week 2 — Untangle the Giant

### 2.1 Split `agentic/app.py` (was `langgraph/app.py`)
- [ ] Map responsibilities → `kai-pm/AGENTIC_APP_MAP.md`.
- [ ] Split into routes / state / flows / providers / prompts. One PR per split.

### 2.2 Reconcile docker-compose files
- [ ] Diff minimal vs sovereign vs full → `kai-pm/COMPOSE_DRIFT.md`.
- [ ] Extract shared config to base.

### 2.3 Prune Makefile 100 → ~25 targets
- [ ] Audit. Archive dead targets to `Makefile.archive`.
- [ ] Update `merge-gate` to run the full live list.

---

## Week 3 — Honest Verification

- [ ] Run every surviving Makefile target. Categorise.
- [ ] Repo-wide coverage gate (currently only `common/`).
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
| 2026-06-01 | Delete tempo tests | Test attributes don't exist; orphans, not bugs |

---

## Active agent sessions

| Dispatched | Scope | Status |
|---|---|---|
| 2026-06-01 | Week 1: close #59 + #60, label other PRs, delete tempo tests | running |
