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

## D6 — 2026-04-25 — CI green-again sweep
**Context:** PRs #49, #50, #51 merged but left `main` CI red: TTS test hitting live network, starlette/pillow/python-multipart CVEs unpatched, and PM docs drifted from reality.
**Decision:** Bundle regression fixes (TTS de-flake, dep CVE bumps, H2.2 retrieve_ranked cap) + PM brain refresh into a single green-again PR.
**Rationale:** Keeps `main` always-green discipline; clears outstanding hardening debt before resuming H2 backlog.
**Consequences:** TTS test is now offline + deterministic; three dep CVEs cleared; `retrieve_ranked()` capped at `MEMU_MAX_CANDIDATES` (default 500); PM docs reflect post-merge reality. H2.4 (`generate_embedding` executor) deferred — requires async cascade.
**PR:** https://github.com/dainius1234/kai-system/pull/52

## D7 — 2026-06-17 — Unify trust scale; PUB mode no longer absolute-blocks tool execution
**Context:** Audit found two disconnected trust numbers — `agentic`'s real 5-signal conviction score (0–10) was lossily squashed to a 0–1 "confidence" before being sent to `tool-gate`, which made the actual approve/deny decision against its own separate threshold. Separately, `tool-gate` enforced PUB mode as a hardcoded `if mode == PUB: deny everything` branch — a second, disconnected gate sitting next to the real one — and that mode check read a manually-set flag rather than the existing time-of-day schedule, so the WORK/PUB schedule never actually affected decisions. Irreversible (destructive/financial/public) actions had no real enforcement, only a prompt-text request to "double-check."
**Decision:** (1) `tool-gate`'s `GateRequest` now takes `conviction` on the same 0–10 scale `agentic/conviction.py` produces — one trust scale, end to end, no lossy conversion. (2) The hardcoded PUB-mode block is removed; PUB instead raises the required conviction by a large, configurable offset (`PUB_CONVICTION_OFFSET`, default 2.5, on top of `REQUIRED_CONVICTION` default 7.0) evaluated through the same gate logic as WORK mode — in practice this still means almost nothing executes while off-duty, but it is a real threshold, not a separate absolute rule. (3) The gate now resolves mode via the existing schedule-aware `_effective_mode()` helper instead of the static manual flag, so the WORK/PUB schedule is actually live. (4) A server-derived irreversible-action taxonomy (tool → destructive/financial/public, via `IRREVERSIBLE_TOOLS_JSON`) requires conviction ≥ `IRREVERSIBLE_MIN_CONVICTION` (default 9.0) **and** explicit operator cosign before approval, in either mode — confirmation alone never substitutes for the conviction floor.
**Rationale:** "PUB mode = zero execution, no matter what" was a previously-absolute safety guarantee. Replacing it with "PUB = extremely strict but real" is a safety-relevant behavior change, confirmed with the project owner before implementation (Phase 0 trust-loop plan). It closes the two-gates inconsistency and gives irreversible actions actual enforcement instead of prompt-only guidance.
**Consequences:** All callers of `/gate/request` must send `conviction` (0–10), not `confidence` (0–1) — `agentic/app.py` and all gate test scripts updated in the same change. Reason codes `LOW_CONFIDENCE`/`PUB_MODE` are replaced by `LOW_CONVICTION`/mode-aware `APPROVED`/`IRREVERSIBLE_REQUIRES_CONFIRMATION`/`IRREVERSIBLE_CONFIRMED`. Memory trust-tier weighting (Step D) and gate-routed proactive speech (Step E) build on this same single scale in follow-up changes.
