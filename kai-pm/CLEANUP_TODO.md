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
- [x] Investigated: test passes as-is (30/30 P3 tests green, 2026-07-21). Correction boost = +0.08 + importance advantage = sufficient to rank first with hash embeddings. No code change needed.

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
- [x] `kai-pm/COMPOSE_DRIFT.md` landed (D72, 2026-07-21) — 10 critical divergences documented, 11 inconsistencies, shared-block extraction candidates listed.
- [ ] Extract shared config to base (deferred — divergences D1/D2 should be resolved first).

### 2.3 Prune Makefile 100 → ~25 targets
- [x] Audit landed → [`MAKEFILE_AUDIT.md`](MAKEFILE_AUDIT.md) (2026-06-02).
- [x] 10 DELETE targets removed; `Makefile.archive` created with preserved definitions (D70, 2026-07-21).
- [ ] Slim `test-core` deps to KEEP-only list (deferred: no rush now that merge-gate is honest).
- [x] `merge-gate` recomposed: go_no_go → pypi-shadow-check → check-docs → quality_gate → dep-audit → test-core → test-integration → coverage (D71, 2026-07-21).

---

## Week 3 — Honest Verification

- [x] Run every surviving Makefile target. Categorise → `kai-pm/MAKEFILE_TARGETS.md` (D73, 2026-07-22). 1792/1794 tests pass offline; 2 env-specific failures (pyo3/live-API). 5 test isolation bugs fixed. 0 collection errors.
- [x] Repo-wide coverage gate (D75, 2026-07-22). Expanded from `common/` only to 5 modules: `common/`, `agentic/`, `memu-core/`, `letta-agent/`, `financial-awareness/`. Combined measured coverage: 62.67%. Threshold lowered to 60% (honest: `agentic/app.py` at 34% and `memu-core/app.py` at 53% are service-route-heavy files untestable offline). `.coveragerc`, Makefile, and `python-app.yml` all updated. `test_h3_coverage_gate.py` updated to parse multi-line Makefile target correctly.
- [x] `merge-gate` honesty — done (D71).

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
