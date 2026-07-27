# Kai Code Audit — Verifier Service Batch

Repository: `dainius1234/kai-system`  
Status: CONFIRMED — pending final consolidation  
Reviewed: 27 July 2026

## Consolidated batch index

| ID | Severity | Finding |
|---|---|---|
| KAI-VERIFY-001 | CRITICAL | The host-published verification authority has no authentication or authorisation |
| KAI-VERIFY-002 | CRITICAL | Caller-supplied evidence packs override authoritative retrieval and can fabricate PASS verdicts |
| KAI-VERIFY-003 | CRITICAL | Evidence scoring measures retrieval rank and word overlap, not whether evidence entails or contradicts the claim |
| KAI-VERIFY-004 | CRITICAL | Duplicate or same-source records are counted as independent strong evidence chunks |
| KAI-VERIFY-005 | CRITICAL | Caller-controlled plan structure can contribute a perfect consistency score without verifying truth or executability |
| KAI-VERIFY-006 | CRITICAL | Financial and configuration claims are excluded from material-claim checks by the default policy types |
| KAI-VERIFY-007 | CRITICAL | Anonymous verification reads and mutates global `keeper` memory ranking/state |
| KAI-VERIFY-008 | HIGH | Evidence-pack size, record depth and field sizes are unbounded and ignore `top_k` |
| KAI-VERIFY-009 | HIGH | Evidence rank, relevance and importance values are trusted without provenance or range/finiteness validation |
| KAI-VERIFY-010 | HIGH | One evidence chunk can support multiple unrelated material claims |
| KAI-VERIFY-011 | HIGH | Strong-chunk counting does not bind a chunk to a specific extracted claim |
| KAI-VERIFY-012 | HIGH | Poisoned, pinned, synthetic and caller-inflated memU records are accepted as verification evidence |
| KAI-VERIFY-013 | HIGH | Every retrieval is hard-coded to user `keeper` rather than an authenticated claimant/evidence scope |
| KAI-VERIFY-014 | HIGH | No-evidence/memU-outage verification returns REPAIR rather than FAIL_CLOSED for ordinary prose |
| KAI-VERIFY-015 | HIGH | Retrieval fallback hides evidence-pack endpoint failure and silently changes evidence semantics |
| KAI-VERIFY-016 | HIGH | Keyword overlap ignores negation, attribution, modality, quantities and causal direction |
| KAI-VERIFY-017 | HIGH | Memory evidence text is not compared proposition-by-proposition with material claims |
| KAI-VERIFY-018 | HIGH | A long arbitrary context string adds plausibility confidence without evidence |
| KAI-VERIFY-019 | HIGH | Hedging words are treated as evidence of plausibility |
| KAI-VERIFY-020 | HIGH | Claims with no recognised material pattern receive an automatic 0.7 low-risk score |
| KAI-VERIFY-021 | HIGH | Material-claim extraction misses most ordinary numbers, currencies, identifiers, dates and factual assertions |
| KAI-VERIFY-022 | HIGH | Material-claim regexes do not distinguish quoted, denied, hypothetical or attributed text |
| KAI-VERIFY-023 | HIGH | Claim and context material extractions are concatenated without global deduplication |
| KAI-VERIFY-024 | HIGH | Self-consistency checks only field presence and do not detect contradictory, unsafe or impossible steps |
| KAI-VERIFY-025 | HIGH | Absence of a plan receives a neutral 0.5 instead of an unavailable/not-applicable state |
| KAI-VERIFY-026 | HIGH | SAGE self-critique is derived from the same signals and is counted as an additional verification signal |
| KAI-VERIFY-027 | HIGH | The same memU ranking fields influence evidence score multiple times |
| KAI-VERIFY-028 | HIGH | Policy thresholds and minimum-evidence relationships are not schema/range validated |
| KAI-VERIFY-029 | HIGH | Policy-load failure silently activates built-in thresholds while health remains green |
| KAI-VERIFY-030 | HIGH | Policy changes are not reloaded and workers can enforce different revisions during rolling deployment |
| KAI-VERIFY-031 | HIGH | Claim, context and plan bodies are unbounded |
| KAI-VERIFY-032 | HIGH | Material-claim extraction and regex processing can generate an unbounded response |
| KAI-VERIFY-033 | HIGH | Verification has no rate limit, concurrency bound, caller quota or workload admission control |
| KAI-VERIFY-034 | HIGH | Repeated anonymous verification alters memU access counts and stability, changing future evidence ranking |
| KAI-VERIFY-035 | HIGH | Verdict results are not durably recorded with evidence IDs, actor, source or request digest |
| KAI-VERIFY-036 | HIGH | Claim identity hashes only the claim and omits context, plan, evidence pack and policy inputs |
| KAI-VERIFY-037 | HIGH | The 64-bit displayed claim hash is collision-prone for long-lived audit identity |
| KAI-VERIFY-038 | HIGH | Caller-provided `source` is unauthenticated and unused in scoring or audit |
| KAI-VERIFY-039 | HIGH | PII detection endpoint is unauthenticated and accepts arbitrary sensitive text |
| KAI-VERIFY-040 | HIGH | PII import failure silently classifies every input as clean and echoes it back |
| KAI-VERIFY-041 | HIGH | PII scanning is regex-limited and provides no purpose, retention or caller-identity boundary |
| KAI-VERIFY-042 | MEDIUM | Memory cross-reference creates new HTTP clients and may make two sequential memU calls per verification |
| KAI-VERIFY-043 | MEDIUM | Malformed evidence numeric fields can raise and fail the entire verification request |
| KAI-VERIFY-044 | MEDIUM | Malformed memU response types are silently interpreted as empty record dictionaries |
| KAI-VERIFY-045 | MEDIUM | Evidence hit ratio is caller-manipulable by adding or removing weak records |
| KAI-VERIFY-046 | MEDIUM | Material claim confidence values are fixed constants rather than calibrated probabilities |
| KAI-VERIFY-047 | MEDIUM | Plausibility uses substring matching and can trigger on words embedded inside unrelated terms |
| KAI-VERIFY-048 | MEDIUM | Absolute-language penalties and hedge bonuses are English-only style heuristics |
| KAI-VERIFY-049 | MEDIUM | Confidence is the arithmetic mean of heterogeneous, correlated heuristic scores |
| KAI-VERIFY-050 | MEDIUM | No evidence-source freshness, timestamp, independence or integrity is reported |
| KAI-VERIFY-051 | MEDIUM | Response evidence summary labels records as supporting without exposing directional evidence analysis |
| KAI-VERIFY-052 | MEDIUM | Full signal details and extracted claim fragments are returned without field minimisation |
| KAI-VERIFY-053 | MEDIUM | Verdict counters are process-local, restart-volatile and divergent across workers |
| KAI-VERIFY-054 | MEDIUM | Public metrics disclose verdict distribution, policy version and HTTP error ratios |
| KAI-VERIFY-055 | MEDIUM | Health exposes policy version/hash and does not test memU or effective policy validity |
| KAI-VERIFY-056 | MEDIUM | Health always reports `ok` even when memory evidence retrieval is unavailable |
| KAI-VERIFY-057 | MEDIUM | Verification timestamps use wall-clock time without request sequence or trusted clock provenance |
| KAI-VERIFY-058 | MEDIUM | Verdict is a free string rather than a strict enum in the response contract |
| KAI-VERIFY-059 | MEDIUM | PII requests and verification bodies have no byte-size limit |
| KAI-VERIFY-060 | MEDIUM | Redaction can return original clean text through an unnecessary sensitive-data echo endpoint |
| KAI-VERIFY-061 | MEDIUM | Middleware records only HTTP status ratios and no structured verification audit event |
| KAI-VERIFY-062 | MEDIUM | Service lifecycle has no shared client, graceful drain, policy watcher or multi-worker consistency contract |

---

## Critical verification-bypass findings

### KAI-VERIFY-001 — CRITICAL — Open verification authority
**Issue:** Compose publishes `8052:8052`. `/verify` and `/redact` require no caller identity, service authentication, delegated user or authorisation.  
**Risk:** Any reachable caller can manufacture verdicts, consume evidence resources, alter memU retrieval state and query sensitive text services.  
**Recommendation:** accept only authenticated, purpose-bound requests from approved services/principals and remove host publication.  
**Status:** OPEN — immediate remediation required

### KAI-VERIFY-002 — CRITICAL — Caller-provided evidence creates PASS
**Issue:** A non-empty `evidence_pack` completely bypasses memU retrieval. The caller supplies `rank_score`, `relevance`, `importance` and content. Two fabricated high-scoring records satisfy `MIN_STRONG_CHUNKS`; with normal plan/plausibility/material scores the arithmetic mean reaches PASS.  
**Risk:** The service labelled the single verification authority accepts evidence authored by the party requesting the verdict.  
**Recommendation:** accept only signed immutable evidence references resolved by Verifier from authorised stores; never caller-calculated ranking scores.  
**Status:** OPEN — immediate remediation required

### KAI-VERIFY-003 — CRITICAL — Support direction is never evaluated
**Issue:** A record is counted as supporting when its composite rank/keyword score crosses thresholds. The code never determines whether the record confirms, denies, contradicts or merely discusses the claim.  
**Risk:** “The permit is not approved” supports “the permit is approved” because the same words overlap.  
**Recommendation:** perform claim-level entailment/contradiction analysis with quoted source propositions and deterministic checks for material values.  
**Status:** OPEN — immediate remediation required

### KAI-VERIFY-004 — CRITICAL — Duplicate evidence counts independently
**Issue:** No record-ID, content-hash, source or causal independence deduplication exists. Repeating one record twice produces two strong chunks.  
**Risk:** One poisoned/source record becomes the minimum corroboration required for PASS.  
**Recommendation:** deduplicate semantically and require independent trusted sources for consequential claims.  
**Status:** OPEN — immediate remediation required

### KAI-VERIFY-005 — CRITICAL — Caller plan inflates verdict
**Issue:** The caller supplies `plan`. Any dictionary with steps containing `action` and a summary receives self-consistency 1.0. No action/effect/tool/evidence consistency is checked.  
**Risk:** Boilerplate formatting contributes a perfect factual-verification signal and helps fabricated evidence reach PASS.  
**Recommendation:** separate factual verification from plan quality and verify exact claims/actions against authoritative data.  
**Status:** OPEN — immediate remediation required

### KAI-VERIFY-006 — CRITICAL — Default policy omits financial/config claims
**Issue:** `_CLAIM_PATTERNS` defines `financial` and `config`, but the default `MATERIAL_CLAIM_TYPES` list excludes both. If the policy omits them, costs, payment, tokens, passwords, encryption, ports and firewall claims receive no material-claim scrutiny.  
**Risk:** High-stakes financial/security assertions are categorised as low-risk prose.  
**Recommendation:** require strict complete policy types and treat unknown/unclassified consequential claims as material.  
**Status:** OPEN — immediate remediation required

### KAI-VERIFY-007 — CRITICAL — Verification mutates global operator evidence
**Issue:** Verifier queries memU as `user_id=keeper`. memU retrieval increments access counts/stability and anonymous callers can repeatedly select records.  
**Risk:** Verification traffic changes future rankings, reflections and retention of the operator’s private evidence.  
**Recommendation:** use read-only immutable evidence snapshots with authenticated principal scope and no retrieval side effects.  
**Status:** OPEN — immediate remediation required

---

## High-severity evidence and scoring findings

### KAI-VERIFY-008 — HIGH — Unbounded evidence pack
`top_k` applies only to memU calls; a supplied list may contain unlimited deeply nested records/text.

### KAI-VERIFY-009 — HIGH — Untrusted evidence numbers
`rank_score`, relevance and importance are direct float conversions with no verified producer, finite range or schema.

### KAI-VERIFY-010 — HIGH — Evidence reused across claims
The total strong-chunk count is compared with the number of material claims; chunks are not mapped to particular claims.

### KAI-VERIFY-011 — HIGH — No claim-specific corroboration
A record strong for one part of a multi-claim paragraph can satisfy the requirement for unrelated numbers/dates/instructions.

### KAI-VERIFY-012 — HIGH — Poisoned synthetic memory accepted
Verifier does not independently exclude poisoned/quarantined records, verify trust tier/source, or reject reflections/preferences generated by the system/caller.

### KAI-VERIFY-013 — HIGH — Global keeper evidence scope
All internal retrieval ignores the caller/request’s actual user, session, tenant and evidence authority.

### KAI-VERIFY-014 — HIGH — Dependency outage returns REPAIR
With no records, ordinary prose receives nonzero consistency/plausibility/material scores and generally aggregates to REPAIR rather than FAIL_CLOSED. Downstream code already treats non-FAIL outcomes inconsistently.

### KAI-VERIFY-015 — HIGH — Silent retrieval-contract switch
Any evidence-pack endpoint error falls back to basic retrieval with a different schema/ranking contract; the response does not identify the degradation.

### KAI-VERIFY-016 — HIGH — Linguistically unsound overlap
Set-word overlap ignores negation, values, units, actor, target, sequence and modal language.

### KAI-VERIFY-017 — HIGH — No proposition matching
Extracted material claims are never matched to evidence text or verified individually.

### KAI-VERIFY-018 — HIGH — Context-length bonus
Any context longer than 20 characters adds 0.1 to plausibility, irrespective of provenance or content.

### KAI-VERIFY-019 — HIGH — Hedging earns confidence
Words such as “might”, “could” and “probably” increase plausibility to 0.7 even though they express uncertainty rather than evidence.

### KAI-VERIFY-020 — HIGH — Unrecognised claims receive 0.7
If regexes detect no material fragment, the material signal says “low-risk prose” and grants a strong positive score.

### KAI-VERIFY-021 — HIGH — Incomplete claim extraction
Most unqualified numbers, currencies with decimals/commas, names, legal/medical assertions, coordinates, URLs and natural-language facts are missed.

### KAI-VERIFY-022 — HIGH — Context/quotation blindness
Patterns classify text without determining whether it is quoted, denied, hypothetical, obsolete or attributed to another source.

### KAI-VERIFY-023 — HIGH — Duplicate extracted claims
`extract_material_claims()` deduplicates only within one text call. Claim and context lists are concatenated and may double-count the same fragment.

### KAI-VERIFY-024 — HIGH — Superficial consistency
The check verifies only step dictionary/action presence and plan summary; it does not detect mutually exclusive steps, impossible ordering, unsafe parameters or unsupported claims.

### KAI-VERIFY-025 — HIGH — Missing plan is positive-neutral
No plan produces 0.5 rather than an explicit unavailable signal excluded from factual aggregation.

### KAI-VERIFY-026 — HIGH — Self-critique double counts correlated signals
The critique is calculated from the first four signals and then appended as a fifth equal-weight signal, giving the same heuristic evidence another vote.

### KAI-VERIFY-027 — HIGH — Ranking fields counted repeatedly
memU `rank_score` already includes relevance/importance, then Verifier adds relevance and importance again to `chunk_score`.

### KAI-VERIFY-028 — HIGH — Unsafe threshold policy
PASS/REPAIR/strong thresholds and minimum counts are parsed without validating finiteness, ordering or safe ranges; zero/negative strong thresholds can make every record strong.

### KAI-VERIFY-029 — HIGH — Policy failure appears normal
Shared policy accessors return defaults on empty/corrupt policy; Verifier health still reports ok with a raw hash/version.

### KAI-VERIFY-030 — HIGH — Stale policy activation
Thresholds are loaded at import and never revalidated/reloaded; workers in a rolling deployment may enforce different policy versions.

### KAI-VERIFY-031 — HIGH — Unbounded claim/context/plan
No byte, nesting, step-count or aggregate body limits exist beyond `top_k`.

### KAI-VERIFY-032 — HIGH — Unbounded extracted response
Regex matches create one response object per unique fragment across unbounded claim/context.

### KAI-VERIFY-033 — HIGH — No admission control
Anonymous callers can issue concurrent regex scans, memU embedding retrievals and large response generation without quotas.

### KAI-VERIFY-034 — HIGH — Verification changes future truth ranking
Each memU read updates access/stability; repeated checks can make selected supporting or contradictory records more persistent/prominent.

### KAI-VERIFY-035 — HIGH — No durable verdict evidence
Only process counters and an info log remain. Evidence IDs/content hashes, actor, request body digest and verdict are not transactionally stored.

### KAI-VERIFY-036 — HIGH — Incomplete claim identity
`claim_hash` excludes context, plan, evidence pack, source, top_k and policy threshold state.

### KAI-VERIFY-037 — HIGH — Short displayed digest
Only 16 hexadecimal characters are retained for the response/log identity.

### KAI-VERIFY-038 — HIGH — Untrusted/unused source
`source` is accepted but does not constrain evidence, scoring, policy or logs.

### KAI-VERIFY-039 — HIGH — Open sensitive-text scanner
`/redact` accepts arbitrary text with no user/service identity or purpose limitation.

### KAI-VERIFY-040 — HIGH — PII detector failure fails open
If common detection imports fail, fallback functions return no findings and the endpoint echoes the original text as clean.

### KAI-VERIFY-041 — HIGH — PII policy is incomplete
Regex detection lacks structured-field context, broader identifiers/secrets, authenticated purpose, retention and deletion semantics.

---

## Medium-severity operational findings

### KAI-VERIFY-042 — MEDIUM — Connection churn/fallback latency
Each verification creates one client for evidence-pack and another after failure for retrieval.

### KAI-VERIFY-043 — MEDIUM — Malformed evidence crash
Direct float conversions and nested field assumptions can raise and return 500.

### KAI-VERIFY-044 — MEDIUM — Wrong response-type handling
A list containing non-dicts becomes empty dictionaries; malformed service responses produce misleading zero evidence rather than a schema failure.

### KAI-VERIFY-045 — MEDIUM — Caller-controlled denominator
Evidence score is hits divided by total supplied records, allowing the requester to tune the score through pack composition.

### KAI-VERIFY-046 — MEDIUM — Fixed extraction confidence
Regex categories receive 0.9/0.7 constants with no calibration.

### KAI-VERIFY-047 — MEDIUM — Substring plausibility
Absolute/hedging phrases are searched as substrings, not token/semantic expressions.

### KAI-VERIFY-048 — MEDIUM — English-only style scoring
Other languages, technical prose and legitimate absolute statements are mis-scored.

### KAI-VERIFY-049 — MEDIUM — Incomparable scores averaged equally
Memory coverage, formatting, writing style, material pattern count and self-derived critique are treated as equally calibrated probabilities.

### KAI-VERIFY-050 — MEDIUM — Missing evidence metadata
No source date, event date, trust revision, independence, retrieval mode or freshness appears in the decision.

### KAI-VERIFY-051 — MEDIUM — Misleading support summary
The response states records “support the claim” based solely on the threshold calculations.

### KAI-VERIFY-052 — MEDIUM — Excess response detail
Every signal and raw extracted fragment is returned rather than a minimised decision plus protected evidence reference.

### KAI-VERIFY-053 — MEDIUM — Volatile counters
Verdict metrics reset on restart and differ by worker.

### KAI-VERIFY-054 — MEDIUM — Public metrics
Verdict distribution, policy version and HTTP error ratio are unauthenticated.

### KAI-VERIFY-055 — MEDIUM — Health disclosure/readiness gap
Health exposes policy identity but checks neither memU nor policy schema/effective thresholds.

### KAI-VERIFY-056 — MEDIUM — False healthy dependency state
MemU may be unreachable while health stays ok.

### KAI-VERIFY-057 — MEDIUM — Wall-clock evaluation time
No monotonic sequence, trace ID or trusted timestamp authority accompanies the verdict.

### KAI-VERIFY-058 — MEDIUM — Free-string verdict schema
The response model does not constrain verdict to the documented enum.

### KAI-VERIFY-059 — MEDIUM — Unbounded PII workload
Redact text has no byte limit and scans the full content with multiple regexes.

### KAI-VERIFY-060 — MEDIUM — Sensitive echo behaviour
When no PII is detected, the endpoint returns the complete original text even though the caller already supplied it.

### KAI-VERIFY-061 — MEDIUM — No structured audit middleware
Only HTTP status ratios are recorded; verification source/evidence/actor/outcome is absent.

### KAI-VERIFY-062 — MEDIUM — Missing lifecycle/multi-worker contract
No shared HTTP client, startup policy validation, graceful drain, policy reload or distributed counter/audit authority exists.

---

## Batch totals

- Findings: **62**
- Critical: **7**
- High: **34**
- Medium: **21**
- Low: **0**

## Provisional repository totals after all logged batches

- Findings: **1,928**
- Critical: **177**
- High: **937**
- Medium: **811**
- Low: **3**

## Files materially reviewed

`verifier/app.py`, Verifier deployment in `docker-compose.full.yml`, shared policy/runtime code, and integration against memU, Agentic, Fusion and memory-promotion paths.
