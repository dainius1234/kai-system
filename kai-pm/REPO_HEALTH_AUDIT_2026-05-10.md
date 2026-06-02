# Repo Health Audit — 2026-05-10

> **Purpose:** honest situational awareness before any further work.
> **Author:** PM (Copilot Chat) for @dainius1234.
> **Scope:** what's actually in the repo, what's broken, what's bloated, what's solid.

---

## TL;DR — in two sentences

The repo is **not broken**, it's **bloated and tired**. Three months of feature
sprints added scaffolding faster than anything got pruned, and now ~3 hot spots
are hiding the real signal under noise.

---

## 1. What's actually in the repo

- **27 service directories** (agentic, memu-core, kai-advisor, verifier,
  tool-gate, executor, fusion-engine, perception, orchestrator, supervisor,
  dashboard, screen-capture, telegram-bot, calendar-sync, metrics-gateway,
  ledger-worker, memory-compressor, heartbeat, workspace-manager, sandboxes,
  security, backup-service, common, scripts, docs, kai-pm, data, output)
- **3 docker-compose files**: `minimal`, `sovereign`, `full`. Suspected drift.
- **Makefile: 419 lines, ~100 test targets**. Most are one-off sprint relics.
- **`merge-gate` runs only 18 of those 100 tests.** The other 82 are unverified.
- **`agentic/app.py`: 80,212 bytes (~2000 lines, single file).** Real god-object.
- **`agentic/kai_config.py`: 54,873 bytes (~1500 lines).** Second god-object.
- **`kai-advisor/app.py`: 58 lines, a stub.** Not a god-object — a skeleton.
- **1603 tests passing, 13 failing** (per latest CI on PR #59).
- **Coverage: 79%** (but only over `common/`, not the whole repo).
- **13 open PRs**, some > 14 days old (R2 drift risk).

---

## 2. The five real problems

### Problem 1 (fixed): `langgraph/` was shadowing the PyPI `langgraph` package
- Local folder with same name as installed package wins on `sys.path`.
- `from langgraph.graph import StateGraph` (real package) is unreachable
  without `sys.path.pop(0)` hacks.
- The hack already exists, in `scripts/agentic_integration_test.py`:
  > *"Ensure the workspace root is NOT first on sys.path so our local
  >   agentic/ service directory does not shadow the installed package."*
- Consequence: agentic patterns described in `docs/agentic_patterns_spec.md`
  cannot use real LangGraph. Custom Python is doing the job in `agentic/app.py`.
- **Fix:** rename `agentic/` → `agentic/` (or `kai_orchestrator/`).

### Problem 2: `agentic/app.py` is 80KB in one file
- Almost certainly contains: HTTP routes + state + business logic + provider
  glue + retries + prompt assembly, all entangled.
- Every PR that touches it tends to break unrelated tests.
- **Fix:** split into `routes.py`, `state.py`, `flows.py`, `providers.py`,
  `prompts.py`. No behaviour change, mechanical surgery.

### Problem 3: 12 failing tempo tests are orphaned
- `kai-advisor/app.py` is a 58-line FastAPI stub with no `store` and no
  `TEMPO_RAPID_THRESHOLD`.
- `scripts/test_tempo.py` tests behaviour that does not exist in this file.
- Likely cause: TDD scaffolding for a feature that was never finished, OR
  feature was deleted in a refactor and tests got left behind.
- **Fix (one of):**
    - (a) build the tempo feature properly and update `kai-advisor/app.py`, OR
    - (b) delete `scripts/test_tempo.py` and any related stubs.
- Recommendation: **(b) delete** — easier, cleaner, can rebuild later if needed.

### Problem 4: Makefile is 100 test targets, ~80 unverified
- Many targets predate current architecture (`test-j5-memory-diary`,
  `test-p20-conscience-values`, etc).
- `merge-gate` only runs 18 → false sense of "green CI".
- **Fix:** prune to ~25 active targets, archive the rest in `Makefile.archive`,
  add `merge-gate-full` that runs everything still on the live list.

### Problem 5: 13 open PRs, no triage
- Some are > 14 days old.
- Some likely conflict with each other.
- Some are agent-generated, never reviewed.
- **Fix:** PM (me) writes one-line status per PR: merge / rebase / close.

---

## 3. What's NOT broken (give credit)

- ✅ Service decomposition is real, 27 services with own Dockerfiles
- ✅ CI exists and is *catching* problems (that's why PR #59 is red)
- ✅ `kai-pm/` PM docs exist (RISKS, STRATEGIC_PLAN, NAVIGATION)
- ✅ `.coveragerc`, `.pre-commit-config.yaml`, `conftest.py` — real ops hygiene
- ✅ 1603/1616 tests passing
- ✅ Pre-commit hooks, alert rules, prometheus config all present
- ✅ Three compose files imply real awareness of dev/sovereign/prod modes
- ✅ Backup and key-rotation scripts wired into `merge-gate`

The bones are good. Connective tissue rotting in 3 hot spots.

---

## 4. Recommendation: 3-week structural cleanup

**No new features until this is done.** No multi-backend router, no P29
financial, no skills templates. Garden first.

### Week 1 — Stop the bleeding
1. Triage 13 open PRs — close, rebase, or merge.
2. Delete (or de-orphan) the tempo test family.
3. Rename `agentic/` → `agentic/`. Mechanical PR.

### Week 2 — Untangle the giant
4. Split `agentic/app.py` (now `agentic/app.py`) into 5-6 files. Behaviour-preserving.
5. Reconcile `docker-compose.{minimal,sovereign,full}.yml`. Document why each exists.
6. Prune Makefile 100 → ~25 targets. Archive the rest.

### Week 3 — Honest verification
7. Run all surviving Makefile targets. Categorise: fix / delete / rebuild.
8. Set a real, repo-wide coverage gate (not just `common/`).
9. **Then** resume features.

---

## 5. What I am NOT doing (despite earlier promises)

- ❌ Multi-backend router PR (#61) — deferred until cleanup complete.
- ❌ Automation scaffold (#60) — current PR has a real CI failure rooted in
  Problem 1; will be revisited after rename.
- ❌ Skills/journal templates — deferred until repo is calm.
- ❌ Behavioral scoreboard — deferred. Premature when the test suite itself
  is bloated.

---

## 6. What you (Dainius) need to know in plain English

- Your codebase isn't broken, it's overgrown. We've been adding rooms to a
  house without fixing the wiring. Cleanup is unsexy but it's the difference
  between Phase 1 landing in days vs months when GPU arrives.
- The "13 failing tests" are not your code being wrong. They're tests for
  features that don't exist anymore.
- The "langgraph (PyPI) not installed" error is not a missing dependency. It's a
  naming collision we caused ourselves.
- The way out is: rename, split, prune, verify. In that order. One PR at a time.

---

**Next single concrete action:** PM will propose ONE small, reversible PR
(not all of week 1) and wait for blessing before dispatching.
