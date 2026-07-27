# Kai Code Audit — Swarm Orchestration Batch

Repository: `dainius1234/kai-system`  
Status: CONFIRMED — pending final consolidation  
Reviewed: 27 July 2026

Existing Moral Imagination findings in `CODE_AUDIT_BATCH_COGNITIVE_GOVERNANCE_FOUNDATIONS.md` are not duplicated here.

## Consolidated batch index

| ID | Severity | Finding |
|---|---|---|
| KAI-SWARM-001 | CRITICAL | A failed fact-check is followed by one gather pass and then bypassed without re-verification |
| KAI-SWARM-002 | CRITICAL | An adversary `block` recommendation does not block the swarm result |
| KAI-SWARM-003 | CRITICAL | Conviction-gate exceptions fail open with the prior confidence and COMPLETE status |
| KAI-SWARM-004 | HIGH | Socratic enriched queries are never used by later stages |
| KAI-SWARM-005 | HIGH | Socratic stage failure is reported as a COMPLETE handoff |
| KAI-SWARM-006 | HIGH | Gather confidence measures evidence/claim quantity rather than quality |
| KAI-SWARM-007 | HIGH | Untrusted memory and world text are inserted directly into teammate prompts |
| KAI-SWARM-008 | HIGH | Claim extraction accepts any JSON-looking substring from model output |
| KAI-SWARM-009 | HIGH | Debate always constructs a stock plan for the hard-coded DeepSeek specialist |
| KAI-SWARM-010 | HIGH | Counterargument content has no effect on consensus status |
| KAI-SWARM-011 | HIGH | Debate retries repeat materially identical reasoning and inflate reputation |
| KAI-SWARM-012 | HIGH | Fact-checking reuses the same memory/model loop rather than an independent verifier |
| KAI-SWARM-013 | HIGH | Model-supplied arbitrary verdict keys count as supported claims |
| KAI-SWARM-014 | HIGH | Only 40% supported verdicts are required to pass fact-check |
| KAI-SWARM-015 | HIGH | Re-gather after fact-check failure retains stale claims/verdicts and appends more state |
| KAI-SWARM-016 | HIGH | Causal analysis falls back to unsupported/uncertain claims or the raw query |
| KAI-SWARM-017 | HIGH | Unverified causal strings directly increase causal confidence |
| KAI-SWARM-018 | HIGH | Empty causal output is marked complete with confidence 5.0 |
| KAI-SWARM-019 | HIGH | DEGRADED causal status is ignored by the FSM |
| KAI-SWARM-020 | HIGH | Adversary reviews a generic replacement plan rather than the debate plan |
| KAI-SWARM-021 | HIGH | Swarm adversary history and calibration are disabled with `episodes=[]` |
| KAI-SWARM-022 | HIGH | Swarm adversary security challenge is omitted |
| KAI-SWARM-023 | HIGH | Conflict resolution ignores stage status and adversary recommendation |
| KAI-SWARM-024 | HIGH | Evidence count contributes 30% of final conviction without provenance/independence |
| KAI-SWARM-025 | HIGH | Causal-chain count contributes 25% despite no causal verification |
| KAI-SWARM-026 | HIGH | Missing verdicts and votes receive neutral positive scores |
| KAI-SWARM-027 | HIGH | Teammate reputation is built from each stage’s self-assigned confidence |
| KAI-SWARM-028 | HIGH | Successful handoffs are recorded even when no useful output was produced |
| KAI-SWARM-029 | HIGH | Reputation weights create a self-reinforcing conviction loop |
| KAI-SWARM-030 | MEDIUM | Reputation values and confidence inputs lack finite/range validation |
| KAI-SWARM-031 | MEDIUM | Corrupt reputation storage silently resets to empty state |
| KAI-SWARM-032 | MEDIUM | Reputation persistence is non-atomic and concurrency-unsafe |
| KAI-SWARM-033 | MEDIUM | Reputation saving performs synchronous filesystem work in the async request path |
| KAI-SWARM-034 | MEDIUM | “Schema-validated” handoffs are ordinary unvalidated dataclasses |
| KAI-SWARM-035 | MEDIUM | Swarm timeouts, thresholds and retry counts lack safe-range validation |
| KAI-SWARM-036 | MEDIUM | `_run_stage` catches timeouts but not ordinary stage-function exceptions |
| KAI-SWARM-037 | MEDIUM | No overall pipeline deadline bounds cumulative retries and stages |
| KAI-SWARM-038 | MEDIUM | Successful GATHER transition is missing from the FSM transition log |
| KAI-SWARM-039 | HIGH | Conviction rethink retries do not change the plan, evidence or challenge configuration |
| KAI-SWARM-040 | HIGH | Moral/causal handoff failure states do not prevent the conviction gate |
| KAI-SWARM-041 | MEDIUM | Shared context collections and stage outputs have no aggregate bounds |
| KAI-SWARM-042 | MEDIUM | Raw stage exception text enters halt reasons returned to callers |
| KAI-SWARM-043 | MEDIUM | Swarm API `passed` is derived only from halt state and numeric confidence |
| KAI-SWARM-044 | MEDIUM | Reputation/status endpoints expose internal teammate scoring without access control |
| KAI-SWARM-045 | MEDIUM | Session ID is caller-controlled and does not partition memory/evidence sources |

---

## Stage factories: `agentic/swarm_stages.py`

### KAI-SWARM-001 — CRITICAL — Failed fact-check is not re-run
**Issue:** when FACT_CHECK returns FAIL, the FSM runs GATHER once and, if that succeeds, proceeds directly to CAUSAL_CHECK. It never runs FACT_CHECK again on the new evidence/claims.  
**Risk:** a claim set explicitly rejected by the verification stage advances into causal reasoning and final conviction.  
**Recommendation:** loop back through a fresh fact-check and require a validated PASS before progression.  
**Status:** OPEN — immediate remediation required

### KAI-SWARM-002 — CRITICAL — Adversary block is advisory only
**Issue:** the conviction stage passes only `verdict.total_modifier` into `resolve_conflict`. `verdict.recommendation == "block"` and critical warnings are copied to metadata but never enforce a block. The API can still return `passed: true`.  
**Risk:** a plan the adversary explicitly blocks can be presented as approved.  
**Recommendation:** make block/critical warnings hard non-overridable gates for applicable risks.  
**Status:** OPEN — immediate remediation required

### KAI-SWARM-003 — CRITICAL — Conviction gate fails open
**Issue:** any conviction-gate exception returns a COMPLETE handoff with the incoming confidence unchanged.  
**Risk:** failure of adversary/conflict logic preserves a potentially high prior stage score and can satisfy the final threshold.  
**Recommendation:** return governance-unavailable/FAILED with zero trusted conviction and halt.  
**Status:** OPEN — immediate remediation required

### KAI-SWARM-004 — HIGH — Question decomposition is discarded
**Issue:** the Socratic stage writes `ctx.enriched_query`, but GATHER, DEBATE, FACT_CHECK and CAUSAL_CHECK continue using `ctx.query`.  
**Risk:** the advertised clarification/decomposition has no effect while logs show it completed.  
**Recommendation:** use a provenance-tracked approved enriched query or remove the false stage.  
**Status:** OPEN

### KAI-SWARM-005 — HIGH — Failed questioner reports COMPLETE
**Issue:** exceptions are logged as failed in `ctx.stage_log`, but the returned handoff has `status=COMPLETE`.  
**Risk:** the FSM and callers cannot detect that prerequisite decomposition failed.  
**Recommendation:** return DEGRADED/FAILED and define whether progression is permitted.  
**Status:** OPEN

### KAI-SWARM-006 — HIGH — Quantity becomes evidence confidence
**Issue:** gather confidence is `1.5 × evidence items + 0.5 × parsed claims`, capped at ten. Source trust, relevance, duplication and contradiction are ignored.  
**Risk:** numerous poisoned/duplicate memories create maximum confidence.  
**Recommendation:** score verified independent claim-level evidence.  
**Status:** OPEN

### KAI-SWARM-007 — HIGH — Stored prompt injection crosses stages
**Issue:** complete memory/world content is concatenated into the LLM user prompt with no provenance or instruction/data separation.  
**Risk:** poisoned sensory/memory content can control claim extraction and later reasoning.  
**Recommendation:** use bounded structured untrusted evidence fields and output verification.  
**Status:** OPEN

### KAI-SWARM-008 — HIGH — Permissive claim parser
**Issue:** the parser takes text from the first `[` through the last `]` and accepts any strings in the decoded array.  
**Risk:** unrelated/model-injected JSON can become authoritative claims; claim lengths are unbounded.  
**Recommendation:** require an exact typed response and bounded claim schema.  
**Status:** OPEN

### KAI-SWARM-009 — HIGH — Hard-coded stock planner/model
**Issue:** debate always calls `build_plan_fn(ctx.query, "DeepSeek-V4", chunks)` regardless of selected model, availability, swarm type or earlier decomposition.  
**Risk:** model routing/capability controls are bypassed and a generic plan is scored as task-specific.  
**Recommendation:** bind the exact ready approved specialist and preserve the produced plan in context.  
**Status:** OPEN

### KAI-SWARM-010 — HIGH — Counterargument cannot change consensus
**Issue:** the Sage reply is stored in `ctx.challenges`, but status is solely `CONSENSUS if conviction >= 6`. Whether the reply says CONTESTED, identifies fatal risk or is an error is ignored.  
**Risk:** the debate stage claims consensus based only on the pre-existing conviction score.  
**Recommendation:** parse and independently evaluate a strict debate verdict.  
**Status:** OPEN

### KAI-SWARM-011 — HIGH — Retry is repetition
**Issue:** debate retries reuse the same query/evidence and plan builder. Loop count itself can raise conviction under the scoring system, while each attempt records another success/reputation observation.  
**Risk:** repeated unchanged reasoning converts non-consensus into higher confidence and reputation.  
**Recommendation:** require material evidence/plan changes and do not reward retries.  
**Status:** OPEN

### KAI-SWARM-012 — HIGH — Fact-check is not independent
**Issue:** Doctor retrieves the same memory namespace and asks the same LLM abstraction to label claims. No Verifier service, external evidence or source independence is required.  
**Risk:** the system validates its own generated claims against the same poisoned/correlated data.  
**Recommendation:** use independent signed evidence and the authoritative verifier.  
**Status:** OPEN

### KAI-SWARM-013 — HIGH — Arbitrary supported keys count
**Issue:** the parsed verdict object is filtered only by value. Keys do not need to match the claims that were requested.  
**Risk:** the model can return unrelated keys marked supported and achieve a 10/10 fact-check score.  
**Recommendation:** require exactly one verdict for each immutable claim ID and reject extras/missing entries.  
**Status:** OPEN

### KAI-SWARM-014 — HIGH — Minority support passes
**Issue:** confidence is the supported fraction ×10 and PASS requires only confidence 4.0.  
**Risk:** two supported claims out of five pass despite three unsupported/uncertain claims.  
**Recommendation:** require task/risk-specific claim coverage and no unresolved critical claims.  
**Status:** OPEN

### KAI-SWARM-015 — HIGH — Re-gather accumulates stale verification state
**Issue:** after FACT_CHECK FAIL, GATHER extends existing evidence and claims; verdicts are not cleared and fact-check is not repeated.  
**Risk:** stale rejected claims/verdicts coexist with new claims and influence subsequent causal/conflict scoring.  
**Recommendation:** create a new versioned evidence/claim set and reverify it atomically.  
**Status:** OPEN

### KAI-SWARM-016 — HIGH — Unsupported claims feed causal analysis
**Issue:** if there are no supported verdicts, Oracle uses `ctx.claims[:3]` or the raw query.  
**Risk:** explicitly unsupported or unverified statements are treated as premises for consequences.  
**Recommendation:** return no-causal-analysis and halt/escalate when verified premises are absent.  
**Status:** OPEN

### KAI-SWARM-017 — HIGH — Generated narratives become causal evidence
**Issue:** any parsed causal-chain string increases confidence and later contributes to the final score. No graph, source or logical verification occurs.  
**Risk:** fluent fabricated consequences are treated as causal quality.  
**Recommendation:** verify causal propositions against a validated model/evidence authority.  
**Status:** OPEN

### KAI-SWARM-018 — HIGH — Empty causal reasoning is successful
**Issue:** zero parsed chains yields confidence 5.0 and COMPLETE.  
**Risk:** absence of consequence analysis contributes a neutral-positive stage rather than a failure.  
**Recommendation:** return DEGRADED/unavailable with zero causal evidence.  
**Status:** OPEN

### KAI-SWARM-019 — HIGH — Causal degradation is ignored
**Issue:** Oracle exceptions return DEGRADED confidence 3.5; the FSM does not inspect CAUSAL_CHECK status before moral/gate progression.  
**Risk:** missing consequence analysis is silently tolerated in high-stakes swarms.  
**Recommendation:** enforce stage requirements by swarm risk class.  
**Status:** OPEN

### KAI-SWARM-020 — HIGH — Adversary sees a different plan
**Issue:** conviction gate reconstructs a one-step `analyze` plan and does not receive the debate plan.  
**Risk:** adversarial findings do not apply to the actual candidate reasoning/action.  
**Recommendation:** preserve one immutable plan ID/body across stages.  
**Status:** OPEN

### KAI-SWARM-021 — HIGH — Historical safeguards are disabled
**Issue:** the adversary is always called with `episodes=[]`.  
**Risk:** history and calibration challenges always lack evidence regardless of available episode history.  
**Recommendation:** provide authenticated outcome history or explicitly block claims of historical challenge coverage.  
**Status:** OPEN

### KAI-SWARM-022 — HIGH — Security challenge omitted
**Issue:** no injection regex/sanitiser is passed to `challenge_plan`, so its security challenge is absent.  
**Risk:** the full swarm pipeline advertises adversarial review without running its security component.  
**Recommendation:** make required challenge configuration immutable by risk tier.  
**Status:** OPEN

### KAI-SWARM-023 — HIGH — Numeric aggregation replaces governance outcomes
**Issue:** resolve_conflict uses counts/votes/modifier only. FAILED/DEGRADED stage statuses, adversary recommendation and warnings do not participate.  
**Risk:** enough quantity-derived score overrides explicit control failures.  
**Recommendation:** apply hard status gates before any numeric synthesis.  
**Status:** OPEN

### KAI-SWARM-024 — HIGH — Evidence count dominates conviction
**Issue:** each evidence item adds 1.5 and the resulting score carries 30% weight, regardless of source quality or duplication.  
**Risk:** attacker-controlled memory volume materially raises final approval.  
**Recommendation:** use independent verified evidence weights.  
**Status:** OPEN

### KAI-SWARM-025 — HIGH — Causal count carries 25%
**Issue:** each generated chain adds two points and causal score carries 25%, with no truth/quality assessment.  
**Risk:** three fabricated strings produce 15% of final conviction.  
**Recommendation:** do not score unverified narrative count.  
**Status:** OPEN

### KAI-SWARM-026 — HIGH — Missing analysis receives positive defaults
**Issue:** absent verdicts and absent teammate votes both default to 5/10; zero adversary modifier also maps to 5.  
**Risk:** missing controls/evidence contribute positive conviction rather than uncertainty.  
**Recommendation:** missing required dimensions must score unavailable/zero and block applicable decisions.  
**Status:** OPEN

### KAI-SWARM-027 — HIGH — Reputation is self-assessed
**Issue:** stage functions call `record_success(slug, confidence)` using confidence they calculated from their own output/counts. No later correctness or operator outcome is used.  
**Risk:** verbose/high-scoring stages raise their authority irrespective of accuracy.  
**Recommendation:** update reputation only from independent verified outcomes.  
**Status:** OPEN

### KAI-SWARM-028 — HIGH — Empty output earns success
**Issue:** Gather records success with evidence but no claims; Oracle records success with no chains; other stages record completion without useful result validation.  
**Risk:** failed/empty behaviour improves reliability and future voting weight.  
**Recommendation:** define successful handoff contracts and verify them before reputation updates.  
**Status:** OPEN

### KAI-SWARM-029 — HIGH — Self-reinforcing authority loop
**Issue:** Sage’s conviction is stored as its vote and as reputation confidence; later conflict resolution weights the vote by that accumulated confidence/reliability.  
**Risk:** an initially inflated score increases future influence, creating runaway self-trust.  
**Recommendation:** separate prediction from independent outcome-calibrated reputation.  
**Status:** OPEN

---

## FSM and shared swarm state: `agentic/cognitive_fsm.py`, `agentic/swarm.py`, `agentic/app.py`

### KAI-SWARM-030 — MEDIUM — Reputation numerics are unvalidated
Confidence, counters and persisted totals accept NaN, infinity, negative/extreme values, producing invalid weights and comparisons.

### KAI-SWARM-031 — MEDIUM — Corruption resets reputation
Any load/deserialisation failure replaces the complete reputation map with `{}` without quarantine or readiness failure.

### KAI-SWARM-032 — MEDIUM — Reputation persistence races
Complete JSON is rewritten directly with no lock, atomic rename, revision or multi-worker coordination; save errors are suppressed.

### KAI-SWARM-033 — MEDIUM — Async request blocks on reputation write
`/chat/swarm` calls synchronous `save_reputation()` directly after the pipeline.

### KAI-SWARM-034 — MEDIUM — Handoffs are not schema-validated
`AgentHandoff` is a dataclass; stage names, payload, claims, loop count and confidence have no runtime validation despite the safety claim.

### KAI-SWARM-035 — MEDIUM — Configuration is unvalidated
Timeouts, thresholds and retry counts accept non-finite, negative, zero and contradictory values.

### KAI-SWARM-036 — MEDIUM — Ordinary stage exceptions escape the FSM
`_run_stage` converts only `TimeoutError`; unexpected exceptions from injected stage functions terminate the entire request outside the documented HALT contract.

### KAI-SWARM-037 — MEDIUM — No overall deadline
Each stage/retry has its own timeout, but cumulative pipeline duration has no cap and can be several minutes under custom/research settings.

### KAI-SWARM-038 — MEDIUM — Transition log is incomplete
The successful GATHER-to-DEBATE transition is never logged, contradicting “all transitions are logged”.

### KAI-SWARM-039 — HIGH — Rethink does not rethink
The conviction loop calls the same gate with the same context/plan/adversary configuration. Only `loop_count` changes on the handoff; the gate does not use it to produce new evidence/reasoning.  
**Risk:** repeated identical evaluation wastes resources and cannot legitimately improve conviction.  
**Status:** OPEN

### KAI-SWARM-040 — HIGH — Failure statuses do not control progression
After CAUSAL_CHECK and optional Moral Imagination, the FSM proceeds to conviction regardless of FAILED/DEGRADED/halt recommendations returned by those stages.  
**Risk:** prerequisite safety failures become advisory metadata.  
**Status:** OPEN

### KAI-SWARM-041 — MEDIUM — Context/output growth is unbounded
Evidence, claims, challenges, verdicts, chains and stage log have no aggregate bounds; repeated retries/gathers append state.

### KAI-SWARM-042 — MEDIUM — Raw exceptions are returned
Stage factories place `str(exc)` into halt reasons; `/chat/swarm` returns `halt_reason` publicly.

### KAI-SWARM-043 — MEDIUM — API pass criterion is incomplete
The API sets `passed` solely from `not halted` and confidence threshold. It ignores fact-check coverage, adversary recommendation, degraded stages and moral outcomes.

### KAI-SWARM-044 — MEDIUM — Internal reputation is publicly exposed
`/swarm/reputation` returns calls, success counts, confidence, reliability, weights and errors without an access check.

### KAI-SWARM-045 — MEDIUM — Session ID does not provide source isolation
Caller-controlled session IDs are returned/logged, but memory/world functions remain global and receive no session/principal binding.

---

## Batch totals

- Findings: **45**
- Critical: **3**
- High: **27**
- Medium: **15**
- Low: **0**

## Provisional repository totals after all logged batches

- Findings: **1,135**
- Critical: **99**
- High: **465**
- Medium: **568**
- Low: **3**

## Files materially reviewed in this batch

`agentic/swarm_stages.py`, `agentic/cognitive_fsm.py`, `agentic/swarm.py`, and `/chat/swarm` integration in `agentic/app.py`.
