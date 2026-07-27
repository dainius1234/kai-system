# Kai Code Audit — Architecture Interaction and System Invariants Batch

Repository: `dainius1234/kai-system`  
Status: CONFIRMED — pending final consolidation  
Reviewed: 27 July 2026

This phase records architectural root causes and missing invariants demonstrated repeatedly across the confirmed source and cross-service findings.

## Consolidated batch index

| ID | Severity | Architectural finding |
|---|---|---|
| KAI-ARCH-001 | CRITICAL | The system has no authoritative authenticated principal and delegation plane |
| KAI-ARCH-002 | CRITICAL | Policy decisions and action enforcement are separated, bypassable and inconsistently implemented |
| KAI-ARCH-003 | CRITICAL | Security-relevant requests are not bound to one immutable canonical operation digest |
| KAI-ARCH-004 | CRITICAL | There is no authoritative evidence/provenance model for facts, memories, outcomes or model claims |
| KAI-ARCH-005 | CRITICAL | Cross-service state changes lack atomic transaction, saga or compensating-recovery semantics |
| KAI-ARCH-006 | CRITICAL | Execution-capable services are not isolated by a real sandbox and network capability boundary |
| KAI-ARCH-007 | CRITICAL | Human approval is represented by reusable bearer tokens or caller assertions rather than a bound operator decision |
| KAI-ARCH-008 | CRITICAL | Self-generated outcomes and confidence recursively become evidence for future autonomy |
| KAI-ARCH-009 | CRITICAL | Personal/operator state is globally shared instead of partitioned by authenticated identity and purpose |
| KAI-ARCH-010 | CRITICAL | The architecture has no enforceable data-classification, privacy, retention and deletion model |
| KAI-ARCH-011 | HIGH | No single authoritative service, model, tool and capability registry exists |
| KAI-ARCH-012 | HIGH | “Conviction” conflates evidence quality, writing style, moral alignment, vote share and historical frequency |
| KAI-ARCH-013 | HIGH | Independent checks are correlated and double-count the same source or heuristic |
| KAI-ARCH-014 | HIGH | External content is not consistently separated from system instructions and trusted observations |
| KAI-ARCH-015 | HIGH | There is no canonical event envelope carrying actor, source, schema, digest, time, policy and outcome |
| KAI-ARCH-016 | HIGH | Error semantics are inconsistent and frequently return HTTP 200 for blocked, failed or partial operations |
| KAI-ARCH-017 | HIGH | Liveness, readiness, freshness, degraded and stub states are not defined consistently |
| KAI-ARCH-018 | HIGH | Multi-worker and multi-replica behaviour is not a supported architectural contract |
| KAI-ARCH-019 | HIGH | Single-writer ownership is absent for ledgers, vector indexes, queues, schedulers and personal state |
| KAI-ARCH-020 | HIGH | Time is treated as untrusted wall-clock strings/floats rather than canonical event time plus monotonic ordering |
| KAI-ARCH-021 | HIGH | Schema and policy versions are not negotiated or enforced across service boundaries |
| KAI-ARCH-022 | HIGH | Resource and cost governance is fragmented and cannot prevent fleet-wide workload amplification |
| KAI-ARCH-023 | HIGH | Recovery is not separated from security policy and can erase containment during incidents |
| KAI-ARCH-024 | HIGH | Audit evidence is local, optional, mutable and not externally anchored |
| KAI-ARCH-025 | HIGH | Secure defaults are systematically replaced by permissive fallback and fail-open behaviour |
| KAI-ARCH-026 | HIGH | Stub and fallback implementations use production-shaped success contracts |
| KAI-ARCH-027 | HIGH | Client displays and advisory reports are not bound to enforcement authorities |
| KAI-ARCH-028 | HIGH | Data deletion and supersession do not propagate through every derived store and index |
| KAI-ARCH-029 | HIGH | Model/backend identity is configuration text rather than a verified immutable artefact |
| KAI-ARCH-030 | HIGH | No machine-verifiable architecture manifest defines trust zones, data flows and safety invariants |

---

## Critical architectural failures

### KAI-ARCH-001 — CRITICAL — Missing principal/delegation authority
**Evidence:** Many services accept `user_id`, `session_id`, `requester`, `actor_did`, role or `keeper` as body strings; Dashboard and Agentic proxy anonymous callers as internal actors.  
**System effect:** Authentication, ownership, consent, audit attribution and cross-user isolation cannot be enforced consistently.  
**Required invariant:** Every operation must originate from a verified principal or service identity, with explicit delegated scopes and immutable actor chain.  
**Status:** OPEN — immediate architecture remediation required

### KAI-ARCH-002 — CRITICAL — Decision/enforcement split
**Evidence:** Tool Gate makes decisions but Executor accepts direct requests; Agentic can continue after low conviction/adversary block; Fusion/Verifier results are advisory; Dashboard has privileged bypass paths.  
**System effect:** Policy can be correct while the action still occurs through another route.  
**Required invariant:** The final side-effect boundary must cryptographically require one valid, unexpired, single-use policy capability for the exact operation.  
**Status:** OPEN — immediate architecture remediation required

### KAI-ARCH-003 — CRITICAL — No canonical operation binding
**Evidence:** HMAC excludes parameters/conviction; idempotency keys are checked before authentication; co-sign does not issue an exact-request grant; retries replay POST mutations.  
**System effect:** Approval, execution, audit and outcome can refer to different bodies or revisions.  
**Required invariant:** Canonicalise every security-relevant field into one digest used by authentication, policy, idempotency, execution and ledger records.  
**Status:** OPEN — immediate architecture remediation required

### KAI-ARCH-004 — CRITICAL — No evidence/provenance authority
**Evidence:** Verifier accepts caller ranking; memories omit source IDs; external feeds lack retrieval provenance; self-generated reflections become evidence; duplicate sources count independently.  
**System effect:** The system cannot distinguish observation, assertion, inference, generated text, operator decision and verified outcome.  
**Required invariant:** Immutable typed evidence objects with source identity, event time, content digest, trust class, independence and supersession links.  
**Status:** OPEN — immediate architecture remediation required

### KAI-ARCH-005 — CRITICAL — No cross-service transaction model
**Evidence:** file→memU→mapping, memory→graph, add→cognify, Gate→ledger→execute, notification→delivery→acknowledgement and task→Telegram→fire state all commit independently.  
**System effect:** Partial success is routine and retries create duplicates or irreversible divergence.  
**Required invariant:** Use durable operation state machines/sagas with idempotent steps, compensations and verified terminal outcomes.  
**Status:** OPEN — immediate architecture remediation required

### KAI-ARCH-006 — CRITICAL — No capability sandbox
**Evidence:** Executor has unrestricted network/filesystem and allowlisted arbitrary-code primitives; services share a flat network; browser/feed egress is unrestricted.  
**System effect:** One execution or SSRF compromise becomes a fleet pivot.  
**Required invariant:** Per-operation isolated workers with explicit filesystem, network, syscall, CPU/memory/time and credential capabilities.  
**Status:** OPEN — immediate architecture remediation required

### KAI-ARCH-007 — CRITICAL — Human approval is not a human decision object
**Evidence:** Any trusted token can co-sign/change mode; caller role strings can confirm wisdom; reasons/actors are assertions; approvals lack request digest and one-time execution binding.  
**System effect:** Service credentials and untrusted callers can impersonate operator intent.  
**Required invariant:** Strong operator authentication, explicit challenge/preview, immutable decision record and one-use capability bound to the reviewed operation.  
**Status:** OPEN — immediate architecture remediation required

### KAI-ARCH-008 — CRITICAL — Recursive self-certification
**Evidence:** blocked Gate responses are recorded as successful episodes; regex extraction confidence becomes alignment evidence; model heuristics become benchmark success; reflections and predictions re-enter memory.  
**System effect:** Fabricated success compounds over time and can increase trust/autonomy without external outcomes.  
**Required invariant:** Separate predictions/actions from independently observed outcomes; prohibit self-generated records from certifying their own quality.  
**Status:** OPEN — immediate architecture remediation required

### KAI-ARCH-009 — CRITICAL — Global personal-state namespace
**Evidence:** hard-coded `keeper`, shared feedback/emotion/value/conscience/session/preference stores and one operator fingerprint aggregate all callers.  
**System effect:** Privacy leakage, behavioural poisoning and incorrect personalisation are architectural defaults.  
**Required invariant:** Authenticated tenant/principal/purpose partition on every record, cache, query, model and derived state.  
**Status:** OPEN — immediate architecture remediation required

### KAI-ARCH-010 — CRITICAL — Missing data lifecycle model
**Evidence:** plaintext local files, indefinite localStorage, unbounded logs/queues, no retention classes, claimed deletion not propagated to vectors/graphs/reflections/backups.  
**System effect:** Sensitive data cannot be reliably located, minimised, expired or erased.  
**Required invariant:** Data inventory/classification, lawful purpose, encryption, retention, lineage and deletion propagation across every derivative.  
**Status:** OPEN — immediate architecture remediation required

---

## High-severity architectural failures

### KAI-ARCH-011 — HIGH — Registry fragmentation
Tool/model/service/capability identities are duplicated in Tool Gate, Executor, Agentic, memU, Model Council, selectors, Dashboard, Supervisor, Metrics and Compose.

### KAI-ARCH-012 — HIGH — Conviction semantic collapse
One scalar is used for factual confidence, plan quality, style, values, vote share and authority; callers also supply it directly.

### KAI-ARCH-013 — HIGH — Correlated checks double count
Fusion specialists, technical indicators, history/calibration challenges and memory records often share one underlying source but are summed/voted as independent evidence.

### KAI-ARCH-014 — HIGH — Instruction/data boundary absent
Email, RSS, browser, OCR, memory, graph and model outputs are inserted into prompt roles without one canonical untrusted-data representation.

### KAI-ARCH-015 — HIGH — Missing event envelope
There is no universal event ID/schema carrying principal, delegation, source, time, body digest, policy/version, causation, correlation and outcome.

### KAI-ARCH-016 — HIGH — Inconsistent failure contract
Services mix exceptions, `{ok:false}`, `{status:error}`, empty lists, stale values and HTTP 200, defeating reliable orchestration and retries.

### KAI-ARCH-017 — HIGH — Readiness vocabulary absent
Stub, degraded, stale, no-data, unavailable and not-initialised frequently report `ok`; downstream systems cannot enforce readiness.

### KAI-ARCH-018 — HIGH — Distributed execution unsupported
Process-local state and unsynchronised files mean multiple workers/replicas produce different security decisions and overwrite one another.

### KAI-ARCH-019 — HIGH — Missing single-writer ownership
Ledgers, TurboVec, JSON files, schedulers, queues and personal-state maps have multiple or ambiguous writers.

### KAI-ARCH-020 — HIGH — Weak temporal model
Caller timestamps, naive UTC strings, lexical date comparisons and wall-clock cooldowns control ordering and expiry.

### KAI-ARCH-021 — HIGH — No schema/policy negotiation
Callers accept arbitrary JSON and future/unknown enums; rolling deployments can enforce different policies and response shapes.

### KAI-ARCH-022 — HIGH — No fleet resource governor
Each service applies local timeouts/limits, but no authority budgets total inference, network, browser, graph, storage or subprocess work per principal/action.

### KAI-ARCH-023 — HIGH — Recovery/security coupling
Recovery endpoints reload tokens/nonces, close breakers, reset pools and delete files; Supervisor can invoke them from shallow health evidence.

### KAI-ARCH-024 — HIGH — Audit is not authoritative
Logs/ledgers are plaintext/local, writes fail silently, rotation removes evidence and no external signature/transparency anchor proves completeness.

### KAI-ARCH-025 — HIGH — Permissive fallback culture
Missing policy, unavailable verifier, failed security audit, absent Redis/model/backend and unknown modes frequently result in neutral/pass/WORK/stub success.

### KAI-ARCH-026 — HIGH — Production-shaped stubs
Workspace Manager, Orchestrator, Advisor, Fusion, forecasting, counterfactual and other stubs return healthy/success-shaped objects that consumers cannot reliably distinguish from real capability.

### KAI-ARCH-027 — HIGH — UI/advisory enforcement gap
Dashboard mode, go/no-go, health, consensus, risk and notification status can disagree with actual policy or delivery and do not enforce actions.

### KAI-ARCH-028 — HIGH — Derivative deletion gap
Deleting source memory/file does not reliably delete graph nodes, vectors, merged summaries, reflections, caches, archives and backups.

### KAI-ARCH-029 — HIGH — Unverified model identity
Model/backend names, quality, context, price and availability are static configuration claims rather than signed artefact/runtime proofs.

### KAI-ARCH-030 — HIGH — No executable architecture contract
Trust zones, allowed flows, authoritative stores, schemas, capabilities, recovery ownership and invariants are described in scattered code/comments rather than validated at build/deploy/runtime.

---

## Batch totals

- Findings: **30**
- Critical: **10**
- High: **20**
- Medium: **0**
- Low: **0**

## Provisional repository totals after all logged batches

- Findings: **2,319**
- Critical: **219**
- High: **1,167**
- Medium: **930**
- Low: **3**

## Evidence base

All source-level batches, the Cross-Service Attack Chains batch, current orchestration definitions and direct service-integration code.
