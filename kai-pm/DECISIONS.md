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

## D8 — 2026-06-18 — Minimal stack gets a real brain; fixed silently-broken HMAC auth and an IP collision in full stack
**Context:** Auditing `docker-compose.minimal.yml` against its own documentation found it had no `agentic` service at all — `dashboard`'s Chat view already defaulted to `http://agentic:8007`, so README's claim that Chat is "functional" in minimal was false; nothing was listening. `wake-service` (already in minimal) read `OLLAMA_URL` but `ollama` wasn't a service in minimal either. Tracing the fix surfaced two unrelated, more serious pre-existing defects in `docker-compose.full.yml`: (1) `tool-gate`, `agentic`, and (after Step E) `camera-service` all mount the `hmac_secret` Docker secret but none of them ever set `INTERSERVICE_HMAC_SECRET=/run/secrets/hmac_secret` — `common/auth.py`'s `_secret()` was silently falling back to the dev-secret default and then hard-raising `RuntimeError` on every signed gate request, because `HMAC_ALLOW_DEV_SECRET` was never set either. Inter-service HMAC auth in `full.yml` was non-functional as deployed. (2) `wake-service` and `orchestrator` both hardcoded `ipv4_address: 172.20.0.24` — a real network collision. Also: no service anywhere ever ran `ollama pull`, so `ollama` containers started with an empty model store and `common/llm.py`'s `LLMRouter` silently degraded to stub responses with no signal that the "brain" wasn't real.
**Decision:** (1) Add `ollama` (with a healthcheck, previously absent even in `full.yml`) and `agentic` to `docker-compose.minimal.yml`, with `HMAC_ALLOW_DEV_SECRET: "true"` set on `agentic` to match `tool-gate`'s existing dev-secret mode. (2) Add a one-shot `ollama-pull` init container (`full.yml` and `minimal.yml`) that pulls `qwen2:0.5b` and gates `agentic`/`wake-service` startup on `service_completed_successfully`, so the model is guaranteed present before anything queries it. (3) Fix `full.yml`'s HMAC wiring at the root cause — set `INTERSERVICE_HMAC_SECRET: /run/secrets/hmac_secret` on `tool-gate`, `agentic`, and `camera-service` (the last one newly mounting `hmac_secret` too). (4) Re-assign `orchestrator`'s IP to `172.20.0.32` to resolve the collision. (5) `docker-compose.sovereign.yml` is untouched — confirmed (this session and prior) to intentionally omit `agentic`/`ollama` for an external/Tailscale-routed LLM.
**Rationale:** A "minimal" stack that can't chat or reason isn't a usable spine, it's scaffolding — and bolting perception/execution/expansion onto a spine with broken inter-service auth underneath it would have made every future phase inherit a silent failure mode. Fixing the HMAC and IP-collision bugs in `full.yml` while in the same files, rather than filing them for later, follows the same "no afterthought connectors" standard the minimal-stack work was asked to meet.
**Consequences:** Minimal stack is now `agentic`/`ollama`-equipped end to end (chat → conviction → gate → memory). `full.yml`'s signed gate requests (from `agentic` and `perception/camera`) will now actually succeed instead of raising at call time — this is a functional fix, not just a docs correction, and should be called out if anyone previously worked around the broken HMAC path (e.g., by setting `HMAC_ALLOW_DEV_SECRET` manually in a local override). README's service tables/counts and `core-tests.yml`'s CI health-wait were updated to match.
