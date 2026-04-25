# Decisions Log (Append-Only)

> This file is append-only. Never edit past entries; supersede with a new numbered entry.

## D1 — 2026-04-21 — Adopt `kai-pm/` as PM brain
**Context:** PR #48 merged `kai-pm/` and moved PM artifacts into a dedicated directory.
**Decision:** Keep `kai-pm/` as the durable project-management home.
**Rationale:** Centralizes status, sequencing, risks, and session bootstrap in one place.
**Consequences:** Root `PROJECT_STATUS.md` remains a pointer; PM operations run from `kai-pm/`.

## D2 — 2026-04-21 — Use Sovereign AI Strategic Plan as canonical roadmap
**Context:** `kai-pm/SEQUENCE.md` had a fabricated 11-step flow from PR #48.
**Decision:** Replace that with the canonical 5-phase Sovereign AI strategic model.
**Rationale:** Aligns PM artifacts with the real roadmap direction and removes fabricated sequencing.
**Consequences:** `SEQUENCE.md` and bootstrap references now point to `STRATEGIC_PLAN.md` as canonical roadmap location.

## D3 — 2026-04-21 — Treat J1–J7 as DONE
**Context:** Earlier commits/changelog already show J-series delivery completed (`97a3a61`, `223fc88`, README milestone status).
**Decision:** Mark J1–J7 as shipped, not queued.
**Rationale:** PM state must match delivered repo history.
**Consequences:** Sequence/status docs must not represent J1–J7 as pending work.

## D4 — 2026-04-21 — Defer GPU-dependent phases until RTX 5080 arrives
**Context:** Current hardware constraints still block GPU-heavy execution tracks.
**Decision:** Keep Phases 1, 2, 4, and 5 blocked until RTX 5080 procurement/provisioning is complete.
**Rationale:** Prevents planning drift and false in-flight reporting for hardware-gated work.
**Consequences:** Active delivery focus remains on CPU-safe Phase 0 and partially unblocked Phase 3 tasks.

## D5 — 2026-04-21 — Keep CI guardrails enabled and fix breakages immediately
**Context:** `main` CI failed on flake8 E999 in Python 3.11 workflow.
**Decision:** Preserve existing flake8 + pytest workflow checks and fix failures directly instead of weakening CI.
**Rationale:** CI catches real regressions; disabling checks would hide quality issues.
**Consequences:** Syntax/test breakages on `main` should be corrected immediately in follow-up PRs.


## D7 — 2026-04-25 — Adopt Ollama /api/tags pre-flight + warm-up + stream heartbeat as chassis policy
**Context:** Three small C-series chassis gaps (C2/C5/C9) were identified in `docs/PROJECT_BACKLOG.md`:
streaming had no stall protection, model routing didn't verify the model was pulled, and cold-start
latency was unmitigated.
**Decision:** Ship all three as a single "chassis polish" PR behind env-var feature flags with safe
defaults (`STREAM_HEARTBEAT_TIMEOUT=30`, `MODEL_TAGS_CACHE_TTL=60`, `LLM_WARMUP_ENABLED=true`,
`OLLAMA_AUTO_PULL=false`).  Implementation lives in `common/llm.py` (reusable by any service) with
a thin startup hook in `langgraph/app.py`.
**Rationale:** All three are low-risk, backward-compatible hardening wins that improve robustness
against streaming stalls, missing models, and cold-start latency — all without GPU hardware.
**Consequences:** Any service that imports `common/llm.py` inherits C2 and C5 automatically.  C9
must be wired per-service via the FastAPI startup hook pattern.
