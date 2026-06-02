# Kai System — Session Backlog

> Living scratchpad for session notes, open items, and next steps. Updated at the end of each significant work block.

---

## 2026-06-02 (00:15 UTC) — CI Failure Triage (no new bugs)

Pulled latest failing runs (Core Tests, Python application). All 14 failures map to known cleanup items:

- 12 × `scripts/test_tempo.py` failures → orphan tests, being deleted by Week 1 housekeeping agent (CLEANUP_TODO §1.5/§1.6)
- 1 × `test_correction_memory_gets_boost` (`AssertionError: 1 != 0`) → tracked as Week 1 §1.3, sequenced after §1.2
- 1 × `make test-conviction` FileNotFoundError on `langgraph/conviction.py` → resolves with keystone rename (§1.4)

Decision: NO new agent dispatched. All failures are already on in-flight agent kill lists; a parallel fix would cause merge conflicts.

Baseline: 1,608 passed / 5 skipped. Green core healthy. Reds are paper cuts.

Next: after housekeeping + keystone PRs merge, only §1.3 should remain red — dispatch focused agent then.

---

## 2026-06-02 (21:15 UTC) — Week 2.1 validation + PM truth reconciliation

- Re-ran branch baselines:
  - `make go_no_go` ✅
  - `make check-docs` ✅
- Installed Python dependencies in the sandbox using the same per-`requirements.txt` pattern as CI so targeted `agentic` tests could run locally.
- Targeted validation surfaced two realities:
  - **Directly caused by this branch:** `scripts/test_p4_personality.py` still imports `agentic.app._SYSTEM_PROMPTS`, so the compatibility shim in `agentic/app.py` must keep re-exporting that symbol.
  - **Pre-existing / unrelated:** several `scripts/test_p16_operational.py` struggle-detection assertions still return zero scores even after deps are installed; do not conflate that with the route split.
- Refreshed PM truth where it had drifted:
  - `SESSION_BOOTSTRAP.md` now points at the active Week 2.1 route-split + truth-reconciliation focus.
  - `METRICS.md` now reflects the current open PRs/issues instead of the old `#46/#58` snapshot.
  - `PHASE_0_5_BACKLOG.md` no longer treats merged PR #46 as still open.

---

## 2026-06-02 (20:35 UTC) — Week 2.1 route split + PM truth reconciliation

- Re-verified current reality before editing:
  - `make go_no_go` ✅
  - `make check-docs` ✅
  - `make test-core` blocked in this sandbox until Python deps are installed (`ModuleNotFoundError: fastapi`)
- Confirmed PR #46 is already merged on `main` (2026-06-01), so any PM note still saying "land or close PR #46" is stale and needed cleanup.
- Started the next Week 2.1 leaf split from `agentic/app.py`:
  - moved SOUL/AGENTS endpoints to `agentic/routes_identity.py`
  - moved skills endpoints to `agentic/routes_skills.py`
  - moved metrics/queue/models/logs endpoints to `agentic/routes_observability.py`
- Goal: reduce `agentic/app.py` surface without changing route behavior.
- Next: install the missing Python deps needed for sandbox validation, run targeted agentic route tests, then refresh PM docs/status to match the merged reality.

---

## 2026-06-01 (evening) — Cleanup Sprint Kickoff + Agent Fleet Dispatched

**Context.** Audit earlier today revealed three hot spots blocking healthy progress:
1. `langgraph/` folder shadows the upstream `langgraph` PyPI package (self-inflicted import hazard).
2. `langgraph/app.py` is **80,212 bytes** of mixed concerns — routes, state, flows, providers, prompts all in one file.
3. Makefile has ~100 targets with no honesty gate; `merge-gate` doesn't run the live list.

**Decision.** Pause all feature PRs. Run a 4-week cleanup sprint. Tracker: `kai-pm/CLEANUP_TODO.md`.

**Action taken this session.** Dispatched a fleet of cloud agents in parallel:

| Agent | Task | PR type |
|---|---|---|
| 1 | Week 1 housekeeping — close PR #59 + #60, label other PRs, delete orphan `scripts/test_tempo.py` + `test-tempo` Makefile target | code |
| 2 | **Week 1.4 KEYSTONE** — rename `langgraph/` → `agentic/` (mechanical rename, ~30 file edits, update imports/compose/Dockerfile/Makefile/docs, remove `sys.path` hack in `scripts/agentic_integration_test.py`) | code |
| 3 | Week 2.1 prep — produce `kai-pm/AGENTIC_APP_MAP.md` (responsibility map of `app.py`, proposed split into routes/state/flows/providers/prompts, ordered PR sequence) | docs |
| 4 | Week 2.2 prep — produce `kai-pm/COMPOSE_DRIFT.md` (diff of minimal/sovereign/full compose files, base extraction plan) | docs |
| 5 | Week 2.3 prep — produce `kai-pm/MAKEFILE_AUDIT.md` (every target categorized keep/archive/delete, honest merge-gate proposal) | docs |

**Why parallel.** All four prep tasks are read-only or scoped to non-overlapping paths, so they cannot collide with the keystone rename. By morning we should have full Week 2 plans ready to execute.

**Refreshed `kai-pm/SESSION_BOOTSTRAP.md`** so any future session (new chat, new agent, future-Dainius) can pick up in under 2 minutes from `CLEANUP_TODO.md` + `STATUS.md` + open PRs.

**Next session resume sequence:**
1. Read `kai-pm/SESSION_BOOTSTRAP.md`
2. Check open PRs — review the 4 dispatched ones
3. Tick Week 1 boxes in `CLEANUP_TODO.md` as PRs land
4. Dispatch first Week 2.1 split PR (smallest leaf — likely `prompts/` extraction — pure data, no behavior)

---

## 2026-04-21 — Backlog Reconciliation

- Audited backlog vs README: J1–J7 all marked DONE in README but still listed as "Open Items" here. Closed out.
- Removed stale "J2 recommended next build" — J2 wake-word shipped.
- Reset Open Items to reflect actual current state (P29, GPU track, coverage measurement, doc drift discipline).
- No code changes this session — pure doc hygiene to stop reality drift before resuming feature work.

---

## 2026-03-23 — Resilience/Narrative Integration & Recovery Log

- Implemented recovery log in memu-core: after every /recover, logs what was healed and what was learned to conscience/narrative system
- Updated README.md, PROJECT_BACKLOG.md, SESSION_BACKLOG.md to document new feature
- Validated patch and doc updates; all tests passing

## 2026-03-22 (cont.) — Quality Audit & Conscience Hardening

- Full audit of GPT-4.1 commits: found 2 bugs in recovery log (missing verdict → KeyError, missing conscience lock → race condition)
- Fixed recovery entry schema: added alignments, conflicts, alignment_score for full /conscience/audit compatibility
- Fixed pre-existing race condition: /memory/conscience/check now uses _conscience_lock
- Updated architecture.md and known_issues.md for recovery log
- Ran sync-docs to re-align README LOC count (~36,006)
- All 65 test targets passing (1 expected Codespace skip: test-agentic)

## 2026-03-22 (cont.) — Research Gap Close Sprint

- Closed all 5 research gaps from 2026 arXiv/GitHub:
  - Gap 3: Active Context Compression (memu-core, 44 tests, commit 39c677c)
  - Gap 1: Predictive Failure Modeling (supervisor/app.py)
  - Gap 2: Multi-Modal Sensory Input (audio emotion + camera frame analysis)
  - Gap 4: External World Anchor (calendar-sync rewrite)
  - Gap 5: Bio-inspired Self-Healing (4-phase ReCiSt in common/resilience.py)
- 123 new tests across 4 test files, all passing
- 4 new Makefile targets, docs synced (commit 4121d4b)

## 2026-03-22 (cont.) — J-Series Jewels Roadmap

- Added 7 new planned features (J1–J7) from 2026 research to PROJECT_BACKLOG
- Sources: OpenClaw (Live Canvas, SOUL.md, ClawHub), Jarvis (wake-word, PII, memory viewer), Proact-VL (low-latency voice)
- Updated What's Next priorities: J2 (wake-word) recommended as first build
- Full documentation sweep: PROJECT_BACKLOG, README, CHANGELOG, SESSION_BACKLOG, personality_and_proactive, unfair_advantages, next_level_roadmap, copilot-instructions
- Next: Start implementing J2 (wake-word detection + intent judge)

## 2026-03-22 — Engineering Maturity Gap-Close

- Added sync-docs/check-docs Makefile targets
- Updated copilot-instructions to require sync-docs after major changes
- Validated documentation freshness (README, PROJECT_BACKLOG, architecture, known issues)
- Ran go_no_go and merge-gate; fixed all doc/test/infra staleness
- Implemented test-core result caching (scripts/cache_test_core.py, test_core_results.json)
- Added 'What's Next' priority list to PROJECT_BACKLOG.md
- Created SESSION_BACKLOG.md for session notes
- All repo memory and docs are now up to date
- Next: Run make test-core, address Tier 1/2/3 hardening, improve session note workflow

---

## Open Items

### As of 2026-06-01 (evening)

**Active priorities (in order — Cleanup Sprint):**
1. **Land keystone rename** `langgraph/` → `agentic/` (Week 1.4) — unblocks all Week 2.
2. **Review 3 prep docs** when their PRs land: AGENTIC_APP_MAP, COMPOSE_DRIFT, MAKEFILE_AUDIT.
3. **Dispatch first Week 2.1 split** — smallest leaf from app.py.

**Paused until cleanup done:**
- P29 Financial Awareness
- GPU readiness pre-wiring
- Any new feature PRs

**Discipline / hygiene:**
- Run `make sync-docs` after every significant change.
- Update `kai-pm/CLEANUP_TODO.md` checkboxes the moment a PR merges.
- Keep `SESSION_BOOTSTRAP.md` and this file fresh — they are the resume-after-context-loss layer.

**Done since last reconciliation (now closed):**
- ✅ **Coverage measurement** — 78% on `common/` measured 2026-06-01 (1,616 tests). README updated with real number.
- ✅ J1 Live Canvas, J2 Wake-word + Intent Judge, J3 Auto-Redaction PII, J4 Proactive Low-Latency Voice, J5 Memory Viewer GUI, J6 SOUL.md + AGENTS.md, J7 Skills Auto-Install Hub
- ✅ H3 Context Budget Manager
- ✅ P1 Skill Security + TTL, P2 Multi-modal LLM Fusion, P3 World Anchor, P4 Debate Branching, P5 Deprecation Cleanup
