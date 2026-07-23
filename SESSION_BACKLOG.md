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
- Next: Run cache-test-core, address Tier 1/2/3 hardening, improve session note workflow

---


---

## 2026-07-22 — Auto-logged Session

- D72: conftest redis stub, COMPOSE_DRIFT.md, §1.3 verified clean
- refactor: honest merge-gate + remove orchestrator stub (D71)
- docs: housekeeping — delete test_tempo.py, sync README/CHANGELOG/CLEANUP_TODO/STUBS
- chore: CPU-safe tech-debt sweep — H1/H2/H3/H4/S9/cosmetic/Makefile cleanup
- fix: avoid sys.modules['app'] collision in test_financial_awareness
- fix: two CI failures — pii_redacted type and chassis httpx mock
- docs: expand STUBS_AND_PLACEHOLDERS with full codebase sweep
- docs: update README + add STUBS_AND_PLACEHOLDERS catalogue
- chore: add pytest coverage artifacts to .gitignore
- H3: Wire coverage gate (--cov-fail-under=65) into CI and Makefile
- J3: PII auto-redaction in memory write path + dashboard scanner
- J5: Memory Diary / Viewer — browse-recent, date groups, rich cards
- J1: Live Canvas with D3 v7 — force map, emotion timeline, plan flow
- J6: SOUL.md / AGENTS.md identity infrastructure — complete
- docs: full PM audit pass — eliminate stale info after Phase 0.5 completion
- D59: C3 LLM retry/backoff, behavioral scoreboard, Finance tab, PHONE_SETUP.md
- D58: automation infra, cloud LLM backends, PWA service worker, agentic financial wiring
- feat: P29 CIS Financial Awareness service (D57)
- chore: activate FF_GRAPH_INGEST + refresh SESSION_BOOTSTRAP (D56)
- feat: Letta agent memory controller (D55) — Steps 1–5
- docs: refresh docs layer — SESSION_BOOTSTRAP, LETTA plan, README repo structure, CHANGELOG
- Refresh README: date, qwen2.5 model, TurboVec, memu-graph, roadmap
- Activate TurboVec as default VECTOR_STORE in dev/CI compose stacks (D40)
- D53: raise ingest/query timeouts in test_graph_live.py for qwen2.5:3b on CPU
- D52: bypass Cognee pre-flight probe — qwen2.5:3b too slow for 30s timeout
- D51: bump memu-graph live-verify model to qwen2.5:3b
- D50: pre-install Ladybug/Kuzu JSON extension at build time in memu-graph
- Add D49 to DECISIONS.md: memu-graph live-verify model-quality fix
- D49: use a larger model for memu-graph live-verify CI step
- Re-trigger CI (no workflow run was created for 2a8f6a3 after 2+ hours)
- Fix memu-graph: pull a dedicated embedding model instead of reusing the chat model
- Fix memu-graph: EMBEDDING_ENDPOINT must point at /api/embed, not deprecated /api/embeddings
- Revert D45's EMBEDDING_MODEL prefix: OllamaEmbeddingEngine also expects bare tag
- Fix memu-graph embedding failures: restore ollama/ prefix on EMBEDDING_MODEL, fix dimension mismatch
- Fix memu-graph PermissionError: give app user a writable HF_HOME
- Fix memu-graph ImportError: add missing transformers dependency
- Fix memu-graph model 404s: strip ollama/ prefix from LLM_MODEL/EMBEDDING_MODEL
- Fix memu-graph crash: Cognee requires non-empty LLM_API_KEY for ollama
- Wire memu-graph live verification into core-tests.yml
- Swap default Ollama model to qwen2.5:0.5b, fix memu-core Postgres race
- Add D37 to DECISIONS.md: PR #77 CI bugs found/fixed, closes Phase 0.5 live-verify gap
- Add GitHub Models CI/tests-only LLM backend (Phi-4-mini default) (#77)
- Gitignore .claude/worktrees/ to stop flagging agent worktrees as untracked (#76)
- Add D12 to DECISIONS.md: memu-core hot-path fix + tool-gate CI fix (#75)
- Fix memu-core hot-path blocking: defer compress/persist/MARS-writes to background (#74)
- Fix tool-gate container crash: create /data dir before dropping to non-root user
- Add PyPI-shadow guard salvaged from orphaned pm-infra-main branch
- Fix two integration bugs surfaced by porting chassis-polish onto current main
- refactor: address code review — narrow exception catches, fix test comment, add debug log
- feat: C2/C5/C9 chassis polish — stream heartbeat, Ollama pre-flight, model warm-up
- Fix PM docs: PR #46 was already merged, not open
- PM brain reality check: front page was 5+ weeks stale, plan never landed
- Add agentic-introspect to sovereign profile, close the silent-gap
- Update README with Phase 0.5/A/B history and forward roadmap
- Split agentic-introspect out of agentic core (Phase B)
- agentic: move P13 performance snapshot off the /run hot path
- Phase 0.5: give minimal stack a real brain, fix broken full-stack HMAC wiring and IP collision
- Route proactive speech through the tool-gate; add signal-driven check
- Make memory's trust_tier actually affect retrieval ranking
- Unify trust scale and enforce irreversible-action floor in tool-gate
- Document Kai's unique vision and refined development principles
- Initialize bootstrap master plan
- Fix agentic/Dockerfile: replace stale config.py ref, copy all required modules
- Initial plan
- chore: tidy verifier compatibility typing
- fix: add langgraph compatibility dockerfile
- fix: tolerate missing pii helpers in verifier tests
- fix: restore langgraph compatibility paths
- chore: plan CI compatibility fix
- Initial plan
- docs(pm): refresh STATUS.md to reflect Cleanup Sprint reality (2026-06-02)
- docs(pm): refresh SESSION_BOOTSTRAP — keystone merged, MAKEFILE_AUDIT landed (2026-06-02)
- docs(pm): record MAKEFILE_AUDIT landing + full agent fleet in CLEANUP_TODO (2026-06-02)
- docs: log CI triage 2026-06-02
- docs: add Makefile audit plan
- rename langgraph/ → agentic/: update all imports, infra, and docs
- docs(pm): refresh SESSION_BOOTSTRAP for cleanup sprint (2026-06-01)
- docs: log 2026-06-01 cleanup sprint kickoff + 4 dispatched agents
- Initial plan
- Initial plan
- Merge branch 'origin/main' into copilot/fix-ci-on-pr-60
- fix(tests): fix CI failures in test_tempo and test_p3_organic_memory
- chore(coverage): record real coverage number — 78% on common/ (1616 tests)
- chore(coverage): record real coverage number — 78% on common/ (1616 tests)
- chore(pm): update STATUS.md — refresh date and reorder priorities
- Initial plan
- docs(pm): add live cleanup TODO tracker
- Initial plan
- docs(pm): repo health audit — honest situational awareness before next moves
- fix(ci): unblock test-agentic on PR #60 (install langgraph or honour skip)
- Initial plan
- docs(pm): rewrite PHASE_0_5_BACKLOG as 4-week PM-driven sprint with max automation
- Initial plan
- docs(pm): add PHASE_0_5_BACKLOG.md — concrete CPU-safe work order
- docs(pm): add NAVIGATION.md — "you are here" map so we stop getting lost
- docs(pm): reconcile README and kai-pm reality to 2026-05-10
- Initial plan
- Initial plan
- fix: replace unreachable yield with class-based async iterable in TTS test
- fix: TTS test offline, CVE dep bumps, H2.2 MEMU_MAX_CANDIDATES, PM doc refresh
- Initial plan
- Handle missing PortAudio runtime when importing sounddevice
- test: allow dev HMAC secret in tool-gate script tests
- ci: export HMAC_ALLOW_DEV_SECRET for test-core target
- chore: address workflow review feedback and finalize PM v2 updates
- docs: complete PM system v2 playbooks templates and automation
- Initial plan
- ci: harden python-app requirements install loop
- ci: fix workflow plumbing for tool-gate env and requirements install
- Initial plan
- docs: add J-series evidence reference in sequence correction
- fix ci E999 and replace fabricated PM artifacts
- Initial plan
- Post review comment for PR #48
- docs(pm): add kai-pm brain and github PM automation
- Initial plan
- test(j2): tighten wake intent routing coverage and thread safety
- fix: harden memu persist error handling and redis retry constants
- feat(j2): add wake-word and intent-judge service with dashboard/langgraph wiring
- feat: consolidate gpu phase0 and harden memu persistence validation
- docs: add Phase 0 GPU integration implementation summary
- feat(redis): H2.1 & H2.7 Redis persistence and reconnection hardening
- feat(gpu): add GPU detection, model registry expansion, and chassis optimizations
- Initial plan
- docs(backlog): reconcile session backlog with reality (2026-04-21)
- feat: chassis upgrade — model registry, tiktoken tokens, prompt templates, semantic fusion
- fix: honest audit — pgvector, auth hardening, dashboard redesign, behavioral tests, honest README
- feat: verifier semantic upgrade + context budget management
- chore: remove old build artifact kai_system1x.zip, add *.zip to gitignore
- harden: production polish — Dockerfiles, compose, deps, lint
- fix: clear all F401/F811/F841 lint issues (autoflake + manual)
- docs: update README — 30 milestones, 1,518 tests, 40,453 LOC, J-Series + P1-P5 shipped
- P1-P5: skill security scanner + unload/TTL, multi-modal LLM fusion, world anchor, debate branching, kill deprecation warnings


---

## 2026-07-23 — Auto-logged Session

- D75: repo-wide coverage gate — 5 modules, 60% floor
- D74: CI root-cause diagnosed, feature branch rebased onto main
- gitignore: add output/self_audit_log.json and health_scorecard.json
- D73: MAKEFILE_TARGETS.md + 5 test isolation fixes + J1 canvas assertions

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
