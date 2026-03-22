# Changelog

All notable changes to the Kai System project are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added
- Pre-commit hooks (flake8, trailing whitespace, secret detection, syntax check)
- CODEOWNERS file for automated review assignments
- Structured error codes (`common/errors.py`) with enumerated codes and HTTP status mapping
- Feature flags system (`common/feature_flags.py`) with env-based toggles
- Code coverage configuration (`.coveragerc`)
- Dependency vulnerability scanning (`pip-audit` in CI)
- Container image scanning (Trivy in CI)
- CHANGELOG.md (this file)

## [0.25.0] — 2026-03-22 — H3b LangGraph Checkpointing

### Added
- Checkpoint engine: create / list / load / diff / delete / restore full operational state
- Auto-checkpoint before `/recover` and after `/dream` cycle
- Manual save-points via `POST /checkpoint`
- Time-travel rollback via `POST /checkpoint/{id}/restore`
- Checkpoint diff via `GET /checkpoint/diff/{id_a}/{id_b}`
- 32 tests (`scripts/test_checkpoint.py`)

## [0.24.0] — 2026-03-22 — P24 Agent-Evolver Insight Engine

### Added
- `analyze_failures()` groups episodes by FailureClass, generates EvolutionSuggestions
- 8 fix templates mapped to failure classes with concrete remediation actions
- Priority assignment (critical/high/medium/low) based on frequency x severity
- Dream Phase 7 integration (`evolver_dream_phase()`)
- `POST /evolve/analyze` and `GET /evolve/suggestions` endpoints
- 34 tests (`scripts/test_agent_evolver.py`)

## [0.23.0] — 2026-03-22 — P23 SAGE Multi-Agent Critique

### Added
- Verifier self-critique (strategy 5): groupthink, thin-evidence, unsupported claims, contradictions
- Adversary self-review (challenge 7): false consensus, degraded groupthink, conflicting findings, over-optimism
- Both fire automatically before any action proposal
- 30 tests (`scripts/test_sage_critique.py`)

## [0.22.0] — 2026-03-22 — MARS Memory Consolidation

### Added
- Ebbinghaus stability parameter and retention formula R = e^{-τ/S}
- Conscience-filtered pruning during consolidation
- Nightly decay cycle with stability growth on rehearsal
- 35 tests (`scripts/test_mars_consolidation.py`)

## [0.21.0] — 2026-03-22 — H2 Self-Healing & Resilience

### Added
- Deep `/health` endpoints checking real dependencies on all core services
- `/recover` endpoints for self-healing without restart
- `resilient_call()` wrapper with retry, backoff, circuit breaker, and fallback
- TaskWatchdog for frozen async task detection
- Supervisor auto-recovery loop with cooldown
- 38 tests (`scripts/test_h2_self_healing.py`)

## [0.20.0] — 2026-03-22 — H1 Critical Hardening

### Fixed
- 7 security and stability issues across core services
- Injection detection hardening, error budget tuning, audit stream integrity

## [0.19.0] — 2026-03-21 — P22 Operator Model

### Added
- Echo engine, nudge escalation, cross-mode bridge, impact oracle, shadow branches
- 57 targets, 960 tests

## [0.18.0] — 2026-03-21 — P21 Proactive Agent Loop

### Added
- Scheduled tasks, reminders, briefings, action registry, agent summary
- Supervisor firing for proactive actions

## [0.17.0] — 2026-03-21 — P20 Conscience & Values Engine

### Added
- Emergent values, moral reasoning, loyalty memory, gratitude system

## [0.16.0] — 2026-03-21 — P19 Imagination Engine

### Added
- Counterfactual replay, theory of mind, creative synthesis, inner monologue

## [0.15.0] — 2026-03-21 — P18 Narrative Identity

### Added
- Life story engine, narrative context, identity coherence

## [0.14.0] — 2026-03-20 — P17 Emotional Intelligence

### Added
- Emotional memory, self-reflection, relationship timeline, epistemic humility, Soul dashboard

## [0.13.0] — 2026-03-20 — P16 Operational Intelligence

### Added
- Struggle detection, feedback loop, log aggregation, goals UI, memory browser

## [0.12.0] — 2026-03-20 — P4 Personality & Proactive

### Added
- Deep prompts, anti-annoyance, conversation holding, mode transitions

## [0.11.0] — 2026-03-20 — P3 Organic Memory

### Added
- Correction learning, category boost, spaced repetition, proactive engine, Ohana goals

## [0.10.0] — 2026-03-19 — Production Hardening

### Added
- Redis pubsub SSE, Docker secrets, backup service, HMAC rotation drill

## [0.9.0] — 2026-03-19 — HP2+HP4+HP5 Hardware Performance

### Added
- MoE model selector, CoT tree search with conviction pruning, priority queue

## [0.8.0] — 2026-03-18 — P8+P9+P15 Thinking & Security

### Added
- Dashboard thinking pathways, security self-hacking audit, Dream State consolidation

## [0.7.0] — 2026-03-18 — P7+P10+P11+P12+P13+P14

### Added
- Silence-as-signal, predictive pre-computation, operator tempo, self-deception detection
- Temporal self-model, self-improvement gate

## [0.6.0] — 2026-03-17 — P4+P5+P6 Cognitive Alignment

### Added
- Contradiction memory, GEM cognitive alignment, knowledge boundary awareness

## [0.5.0] — 2026-03-17 — Failure Taxonomy + SELAUR

### Added
- 9-class failure taxonomy, SELAUR scoring, metacognitive rule extraction

## [0.4.0] — 2026-03-16 — Agentic Framework Phase 2

### Added
- Specialist router, memory-driven planner, adversary challenge engine
- LangGraph, AutoGen, CrewAI, OpenAgents integration stubs

## [0.3.0] — 2026-03-15 — Quality Hardening

### Added
- Proactive nudges, correction learning, cross-session context, backup service
- Chat UI improvements (markdown, persist, stop, copy)

## [0.2.0] — 2026-03-14 — v7 Core Hardening

### Added
- Policy-as-code, verifier PASS/REPAIR/FAIL_CLOSED, evidence packs
- Circuit breakers, quarantine, rate limits, idempotency

## [0.1.0] — 2026-03-12 — Foundation

### Added
- 22 Docker services, initial compose stack
- Tool-gate, memu-core, LangGraph orchestration, heartbeat, supervisor
- HMAC inter-service auth, audit streams, error budgets
- GitHub Actions CI, flake8 lint, pytest suite
