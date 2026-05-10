# Phase 0.5 Backlog — CPU-Safe Work Before GPU Lands

> **Open this file when sitting down to work on Kai.** Pick the top unblocked item.
> All items here are CPU-only — no GPU required. They de-risk Phase 1 and ship value now.

**Status:** Active
**Created:** 2026-05-10
**Owner:** @dainius1234

---

## Why "Phase 0.5"?

Phase 0 = pre-GPU hardening (CI, docs, stability) — mostly done.
Phase 1 = local LLM integration — **blocked on RTX 5080 hardware arriving**.

Phase 0.5 = the months of legitimate engineering we can do *between* those two,
all of it CPU-testable, all of it making Phase 1 land cleaner when GPU arrives.

---

## The 5 items, in order

### 1. Behavioral test scoreboard ⭐ HIGHEST LEVERAGE
**What:** Add `@pytest.mark.gpu_required` marker. Write 50–100 behavioral test
*questions* Kai should get right (emotion detection accuracy, planning quality,
conviction calibration, refusal correctness). Skip them on CPU CI; run them
the moment GPU is online.

**Why first:** When GPU lands, you have an instant scoreboard to know if Kai
*actually* got smarter, instead of guessing. This is the only way to honestly
measure Phase 1 success.

**Deliverable:**
- `pytest.ini` registers `gpu_required` marker
- New `make test-behavioral-cpu` (skips gpu_required) and `make test-behavioral-gpu` (runs all)
- `tests/behavioral/` with at least 50 test stubs grouped by capability
- CI lane in `.github/workflows/core-tests.yml` runs CPU lane only
- Doc: `docs/BEHAVIORAL_TESTING.md` explains the contract

**Effort:** 1–2 sessions
**Blocked by:** Nothing

---

### 2. P29 Financial Awareness
**What:** Savings tracker, expense categorization, monthly summary.
Memory-centric, CPU-safe, ships actual user-facing value.

**Why:** Concrete deliverable, not infrastructure. Proves Phase 0.5 produces
real things. Already placed in Phase 3 in STRATEGIC_PLAN but has zero GPU dependency.

**Deliverable:**
- New service `financial-awareness/` or extension inside `memu-core/`
- Endpoints: `/finance/expense`, `/finance/summary`, `/finance/categorize`
- Storage in pgvector with category embeddings
- 20+ tests
- Dashboard tab (or extend Memory view)

**Effort:** 2–3 sessions
**Blocked by:** Nothing

---

### 3. Coverage gate in CI
**What:** `.coveragerc` exists; wire `--cov-fail-under=60` (current baseline)
into `make merge-gate`. Ratchet to 70% over next 4 weeks.

**Why:** Stops silent regression. Currently coverage is "estimated ~60%" with
no enforcement, which is the textbook drift pattern this repo just spent a PR cleaning up.

**Deliverable:**
- `Makefile` `coverage-gate` target
- `core-tests.yml` runs coverage gate
- `kai-pm/METRICS.md` shows real coverage %, not "estimated"

**Effort:** half a session
**Blocked by:** Nothing

---

### 4. Branch hygiene
**What:** Delete 6 stale `claude/*` branches and any merged `copilot/*` branches.
Update R5 in RISKS.md.

**Why:** RISKS.md flags this as Active. It's a 5-minute cleanup that closes a risk row.

**Deliverable:**
- Stale branches deleted
- `RISKS.md` R5 → Resolved (or Active with count = 0)
- Optional: `.github/workflows/branch-hygiene.yml` cron to flag stale branches

**Effort:** 15 minutes (manual) or half session (automated)
**Blocked by:** Nothing

---

### 5. Close PR #46 + PR #54
**What:** PR #46 (GPU Phase 0 consolidation) and PR #54 (chassis polish: stream
heartbeat, Ollama pre-flight, model warm-up) have been open for 15+ days.
Either merge, rebase, or close.

**Why:** Open PRs > 14 days old are R2 (drift) waiting to happen.

**Deliverable:** Both PRs in a terminal state.
**Effort:** depends on conflict count
**Blocked by:** Review attention

---

## What's NOT in Phase 0.5 (deliberately)

- Anything requiring real LLM quality (emotion *quality*, plan *quality*, etc.) — Phase 1.
- Multi-specialist *consensus quality* — Phase 2.
- Voice/avatar quality — Phase 4.
- Anything we'd be tempted to fake metrics for.

---

## Done definition for Phase 0.5

All 5 items shipped → mark Phase 0 truly closed in STRATEGIC_PLAN → ready to
flip to Phase 1 the day GPU arrives.
