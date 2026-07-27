# Kai Code Audit — Fusion Engine Extension Batch

Repository: `dainius1234/kai-system`  
Status: CONFIRMED — pending final consolidation  
Reviewed: 27 July 2026

This batch records only findings not already present in `CODE_AUDIT_BATCH_FUSION_ENGINE.md`. The existing 20 findings are not duplicated.

## Consolidated batch index

| ID | Severity | Finding |
|---|---|---|
| KAI-FUSIONX-001 | CRITICAL | The host-published Fusion service has no authentication or authorisation |
| KAI-FUSIONX-002 | CRITICAL | A single failed or circuit-blocked specialist response is assigned 100% agreement |
| KAI-FUSIONX-003 | HIGH | Verifier unavailability silently preserves the same consensus result |
| KAI-FUSIONX-004 | HIGH | Error specialists are removed from agreement without a required successful-coverage threshold |
| KAI-FUSIONX-005 | HIGH | Two surviving models can represent the full panel even when many requested specialists failed |
| KAI-FUSIONX-006 | HIGH | Pairwise averaging overweights correlated model variants and provider lineages |
| KAI-FUSIONX-007 | HIGH | Full model outputs are returned without sensitive-data minimisation |
| KAI-FUSIONX-008 | HIGH | Prompts and caller-controlled system context are sent to external backends without PII/secret controls |
| KAI-FUSIONX-009 | HIGH | Backend identity, model digest and claimed capability are never verified |
| KAI-FUSIONX-010 | HIGH | LLM backend requests have no service authentication or TLS identity requirement |
| KAI-FUSIONX-011 | HIGH | Semantic-agreement failure silently downgrades to lexical Jaccard |
| KAI-FUSIONX-012 | HIGH | Error-guard and verifier-breaker state is process-local and inconsistent across workers |
| KAI-FUSIONX-013 | HIGH | Fusion decisions have no durable actor/input/backend/evidence record |
| KAI-FUSIONX-014 | HIGH | Fusion has no rate limit, caller quota, cost budget or bounded request queue |
| KAI-FUSIONX-015 | HIGH | The backend-probing endpoint actively generates internal traffic for anonymous callers |
| KAI-FUSIONX-016 | HIGH | A configured backend may ignore the 512-token request cap and return an unbounded payload |
| KAI-FUSIONX-017 | HIGH | An empty live backend response remains a valid specialist vote and merge candidate |
| KAI-FUSIONX-018 | HIGH | Fusion output has no authenticated user, tenant, purpose or data-residency partition |
| KAI-FUSIONX-019 | HIGH | The configured memU breaker and URL are unused, so claimed multi-signal memory integration does not exist |
| KAI-FUSIONX-020 | MEDIUM | Stub identity uses only specialist name and the first 200 prompt characters |
| KAI-FUSIONX-021 | MEDIUM | Prompts sharing a 200-character prefix generate identical stub analyses |
| KAI-FUSIONX-022 | MEDIUM | Negative semantic similarity is clipped to zero, suppressing the magnitude of disagreement |
| KAI-FUSIONX-023 | MEDIUM | Verifier JSON is accepted without a strict response schema or policy-version check |
| KAI-FUSIONX-024 | MEDIUM | An open Verifier breaker produces `verification: null` without an explicit reason |
| KAI-FUSIONX-025 | MEDIUM | Static model name, temperature and token settings are not capability-validated |
| KAI-FUSIONX-026 | MEDIUM | Backend reachability accepts any HTTP 200 and does not verify model readiness |
| KAI-FUSIONX-027 | MEDIUM | Public metrics describe Fusion HTTP responses rather than model or consensus quality |
| KAI-FUSIONX-028 | MEDIUM | Evaluated time uses wall-clock time without a monotonic sequence or trusted timestamp |
| KAI-FUSIONX-029 | MEDIUM | The service has no shared client/model lifecycle, graceful cancellation or distributed-state contract |

---

### KAI-FUSIONX-001 — CRITICAL — Open consensus service
**Issue:** `docker-compose.full.yml` publishes `8053:8053`, while `fusion-engine/app.py` defines no inbound authentication, user/service identity or authorisation.  
**Risk:** Any reachable caller can consume parallel model resources, submit system prompts and obtain consensus-labelled output.  
**Recommendation:** remove host publication and require authenticated purpose-bound calls under a server-owned specialist policy.  
**Status:** OPEN — immediate remediation required

### KAI-FUSIONX-002 — CRITICAL — Failed single specialist equals unanimous consensus
**Issue:** `_measure_agreement()` returns 1.0 whenever the original response list has fewer than two entries. Filtering `source="error"` occurs only after this branch.  
**Risk:** A one-specialist request whose backend failed or whose circuit is open returns `agreement_score: 1.0` and `consensus: true`, while the merged response says no valid specialists existed.  
**Recommendation:** require at least two distinct successful live identities before calculating agreement.  
**Status:** OPEN — immediate remediation required

### KAI-FUSIONX-003 — HIGH — Verifier outage is non-enforcing
Transport/status/parsing failure records the breaker and leaves `verification=None`; consensus and merged output are unchanged.

### KAI-FUSIONX-004 — HIGH — Failed voters disappear
Agreement uses only responses whose source is not `error`; failures do not reduce the score or produce an insufficient-panel state.

### KAI-FUSIONX-005 — HIGH — No successful-coverage requirement
Two valid outputs can generate a normal consensus even if dozens of requested specialists failed.

### KAI-FUSIONX-006 — HIGH — Correlated-source inflation
Pairwise averaging assumes every model name/provider variant supplies independent evidence; shared training, prompts and backend routes are not modelled.

### KAI-FUSIONX-007 — HIGH — Full output disclosure
Every specialist response is returned, including potentially private, policy-sensitive or unsafe generated material.

### KAI-FUSIONX-008 — HIGH — Uncontrolled model egress
Prompt and caller-supplied system context are sent to all configured destinations without redaction, consent, residency or secret policy.

### KAI-FUSIONX-009 — HIGH — Model identity is an environment label
A name/URL pair is accepted as the claimed model without endpoint manifest, artefact digest, provider identity or capability probe.

### KAI-FUSIONX-010 — HIGH — Unauthenticated backend transport
Calls include no API credential, mTLS identity, request signature or response attestation.

### KAI-FUSIONX-011 — HIGH — Silent agreement downgrade
Any import/model/encoding exception in semantic agreement returns Jaccard without telling the caller or lowering decision authority.

### KAI-FUSIONX-012 — HIGH — Worker-local failure state
The global LLM guard and Verifier breaker are in-process objects; different workers admit different panels and verification behaviour.

### KAI-FUSIONX-013 — HIGH — No durable Fusion evidence
Results are not transactionally persisted with actor, request digest, exact responses, backend identities, agreement strategy and verifier outcome.

### KAI-FUSIONX-014 — HIGH — Missing workload governance
There is no request rate, backend concurrency, token/cost budget, principal quota or queue capacity.

### KAI-FUSIONX-015 — HIGH — Public active reconnaissance
`GET /backends` makes live health requests to every configured backend for each anonymous call.

### KAI-FUSIONX-016 — HIGH — Backend response cap is advisory
`max_tokens=512` is only a request field. Complete bytes/JSON/content from a non-compliant or malicious backend are materialised.

### KAI-FUSIONX-017 — HIGH — Empty live vote
An HTTP-success backend with missing/empty `choices[0].message.content` is marked `source="live"`, included in valid responses and may become a merge candidate.

### KAI-FUSIONX-018 — HIGH — Missing data partition
No authenticated user/tenant/purpose is carried to backends or result records.

### KAI-FUSIONX-019 — HIGH — Dead memory integration
`MEMU_URL` and `MEMU_BREAKER` are declared but never used; Fusion has no memory/evidence signal beyond the optional Verifier call.

### KAI-FUSIONX-020 — MEDIUM — Prefix-only stub fingerprint
Stub hash input excludes prompt content after character 200 and all context.

### KAI-FUSIONX-021 — MEDIUM — Stub collisions
Different tasks with the same first 200 characters produce the same specialist stub text.

### KAI-FUSIONX-022 — MEDIUM — Disagreement clipping
Negative cosine similarity is replaced with zero, losing information about opposing response directions.

### KAI-FUSIONX-023 — MEDIUM — Unvalidated Verifier response
Any successful JSON body is attached; verdict enum, confidence, policy version/hash and required fields are not validated.

### KAI-FUSIONX-024 — MEDIUM — Hidden breaker skip
When `VERIFIER_BREAKER.allow()` is false, no request is attempted and the result does not say verification was circuit-blocked.

### KAI-FUSIONX-025 — MEDIUM — Static generation contract
The caller-selected configured specialist name is sent as `model`, with fixed temperature/token settings regardless of backend support/task risk.

### KAI-FUSIONX-026 — MEDIUM — Weak reachability
The backend inventory labels any `/health` HTTP 200 reachable without checking semantic readiness, loaded model or identity.

### KAI-FUSIONX-027 — MEDIUM — Misleading metrics
ErrorBudget records only Fusion endpoint HTTP statuses, not backend success, panel coverage, verifier outcomes or calibrated consensus quality.

### KAI-FUSIONX-028 — MEDIUM — Untrusted wall clock
Evaluation time lacks a monotonic sequence, trace revision or trusted clock authority.

### KAI-FUSIONX-029 — MEDIUM — Missing lifecycle ownership
Clients and embedding models are constructed in request paths; there is no lifespan startup validation, bounded worker pool, cancellation drain or shared state.

---

## Batch totals

- Findings: **29**
- Critical: **2**
- High: **17**
- Medium: **10**
- Low: **0**

## Provisional repository totals after all logged batches

- Findings: **1,957**
- Critical: **179**
- High: **954**
- Medium: **821**
- Low: **3**

## Files materially reviewed

`fusion-engine/app.py`, existing Fusion audit findings, deployment configuration and integration against Verifier and configured LLM backends.
