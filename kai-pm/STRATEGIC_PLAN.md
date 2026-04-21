# Sovereign AI Strategic Plan

> Canonical 5-phase roadmap for Kai. This plan is grounded in current repository state and avoids fabricated performance claims.

## Executive summary

Kai is being delivered as a local-first sovereign intelligence system with Phase 0 hardening as the current priority: keep the platform honest, testable, and stable while GPU-gated phases remain blocked; unlock larger-model routing, multimodal latency work, and production self-improvement only after hardware and reliability gates are actually met.

## Phase 0 — Pre-GPU Hardening (current focus)

**Status:** ACTIVE  
**Why now:** This is the highest-leverage work that does not depend on GPU capacity.

### Exit criteria

- CI is consistently green on `main`.
- Core hardening workflows are stable (tests, health/recovery checks, docs sync).
- PM docs reflect reality (status, decisions, risks, metrics).
- Phase 1 entry prerequisites are explicitly verified, not assumed.

### In-flight items (rolling)

- CI reliability cleanup and honest failure triage.
- Documentation drift prevention and PM automation stabilization.
- CPU-friendly memory/reliability improvements.
- Cleanup of stale branches and stale PM metadata.

## Phase 1 — Local LLM Integration

**Entry criteria:** GPU online and validated on target host.  
**Target:** `qwen2.5:7b` validated as the practical local baseline.

### Phase goal

Bring local model runtime online with reproducible baseline behavior, stable latency envelope, and known fallback rules.

### Exit criteria

- Model runtime and routing path are stable for real operator sessions.
- Basic quality and latency checks pass on representative prompts.
- Failure/fallback behavior is documented and tested.

## Phase 2 — Multi-Specialist Routing

**Entry criteria:** Phase 1 stable.

### Specialists in scope

- code
- math
- creative

### Phase goal

Route tasks to specialist paths using measurable policy signals while preserving safety and auditability.

### Exit criteria

- Routing policy is deterministic and observable.
- Specialist quality beats or matches single-model baseline on representative tasks.
- Rollback path to single-route mode is tested.

## Phase 3 — Memory & Reflection Hardening

**Entry criteria:** Phase 0 foundation is stable.  
**Status note:** Partial; CPU-friendly portions are already active.

### Phase goal

Strengthen memory quality, retrieval consistency, and reflective loops without overfitting to one benchmark.

### Exit criteria

- Retrieval quality and reflection loops are measurable and repeatable.
- Corruption/fallback scenarios are handled safely.
- Memory policies are documented and aligned with operator expectations.

## Phase 4 — Avatar / Voice / Multimodal

**Entry criteria:** Phase 1 stable; sufficient GPU headroom available.

### Phase goal

Deliver practical multimodal interaction (voice and avatar paths) with controlled latency and clear operational boundaries.

### Exit criteria

- End-to-end voice/avatar paths are stable in real runs.
- Latency budget is acceptable for interactive use (TBD, unverified).
- Failure modes degrade gracefully to text-only operation.

## Phase 5 — Production Hardening & Self-Improvement

**Entry criteria:** Phases 1–4 stable.

### Phase goal

Consolidate reliability, observability, and controlled self-improvement loops into a production-ready operating cadence.

### Exit criteria

- Hardening gates are enforced and reproducible.
- Recovery playbooks are proven under fault injection.
- Self-improvement loops remain bounded by policy and validation.

## Anti-patterns (explicitly avoided)

- Do not chase models without a phase-aligned reason.
- Do not add features before hardening current foundations.
- Do not optimize prematurely without representative bottlenecks.
- Do not trust LLM output without validation.
- Do not skip testing for “small” changes.

## Known correction flags

- J1–J7 are DONE, not queued.
- RTX 5080 16GB VRAM cannot fit llama3.3:70b at usable quant — plan must reflect 8B/13B class for primary and use 70B only via remote/cloud fallback.
- Ollama speculative decoding support is partial / version-dependent — verify before claiming.
- P29 placement TBD — confirm phase.
- All cited external benchmarks need source links.
