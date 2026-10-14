# D353 — Deep-Research Existing-Kai Maturation Implementation Reissue

> **STATUS: GOVERNANCE / RESEARCH CHECKPOINT — NOT ARCHITECTURE AUTHORITY, NOT IMPLEMENTATION AUTHORITY, NOT PROGRAMME EXECUTION AUTHORITY.**

## Trigger

Dainius asked Kai, after reviewing the first implementation research and DeepSeek's adversarial response, to continue deep searching and analysing both new external material and current repo findings until Kai could judge whether the research had reached the best practical coverage, then reissue it.

Kai performed a second adversarial research pass against current standards, mature production practice and the existing Kai repository/planning corpus.

## Reissued research subject

`kai-pm/KINGSMAN_EXISTING_KAI_MATURATION_IMPLEMENTATION_RESEARCH_REISSUE_2026-08-24.md`

Creation commit:

`80ba95aedd34747ee5a031485f7567b37248d4f6`

## Main conclusion

The second pass confirms D351/D352 rather than overturning them:

> **REFIT, HARDEN, RATIONALISE, MATURE — DO NOT REINVENT KAI.**

Most external production-grade recommendations are already represented in existing Kai design/planning work. The main programme risk is therefore **fragmentation and rediscovery**, not absence of good ideas.

The reissue makes these older plans first-class maturation inputs:

- `CODE_AUDIT_P1_SECURITY_FOUNDATION_PLAN.md`
- `CODE_AUDIT_P2_ISOLATION_AND_INTEGRITY_PLAN.md`
- `CODE_AUDIT_P3_RELIABILITY_AUDIT_PRIVACY_RECOVERY_PLAN.md`
- `CODE_AUDIT_P4_CAPABILITY_REQUALIFICATION_PLAN.md`
- `KAI_FINAL_PRODUCT_ARCHITECTURE_SPECIFICATION.md`
- `KAI_UNIFIED_HUNTER_ARCHITECTURE_AND_ROADMAP.md`
- `UH_PROGRESS_TRACKER.md`
- `HOUSE_IN_ORDER_PHASE2_PROFESSIONALISATION.md`
- `A4_SELF_DIAGNOSIS_EVOLUTION.md`
- D349/D351/D352 current governance corrections.

## Important rediscovery correction

Several items previously discussed as if newly discovered are already explicit design obligations in the earlier audit remediation plans.

Examples:

- P1 already requires exact audience-bound one-use capability enforcement and atomic consumption at the final side-effect endpoint.
- P2 already requires hostile-content isolation, controlled egress, untrusted-content boundaries, provenance and transactional memory/vector/graph lineage.
- P3 already requires typed unknown/partial-effect states, durable operation identities, leases/fencing, signed audit, lifecycle privacy and isolated restore qualification.
- P4 already requires exact model/runtime identity, no stub impersonation, proposition-level evidence, source independence, calibrated uncertainty and attack-chain capability release.

D349 KAI-REV-016/017/018 remain valid current-state findings; the correction is that some target principles predate D349 and should be recovered rather than reinvented.

## DeepSeek review integration

DeepSeek's prior review remains analysis input only.

Kai accepted these useful challenges after repo/programme reconciliation:

1. maturation instruments must consume Evidence Plane/A-4 contracts when frozen, without creating another truth system or changing programme order;
2. migration/readiness state is evidence, not authority;
3. reuse current service-identity/key registry rather than creating a parallel workload identity system;
4. side effect succeeds but receipt is lost → `OUTCOME_UNKNOWN`, requiring target reconciliation before retry;
5. E0 census instrument itself requires calibration/mutation/can-fail proof;
6. Mission Control must be generated from qualified machine evidence rather than manually maintained.

## New or materially under-specified obligations from second-pass research

### R353-01 — Secure update / anti-rollback / freeze resistance

**Classification:** NEW / UNDER-SPECIFIED LONG-HORIZON OBLIGATION.

Current release/provenance/backup work strongly addresses artifact identity and lineage but does not yet clearly establish the complete property:

> a restored or compromised Kai must not accept an obsolete vulnerable release merely because that release was once validly signed.

TUF-style security properties are retained as a requirements reference:

- monotonic release/version state;
- target hashes/sizes;
- expiry/freshness;
- rollback/freeze/mix-and-match resistance;
- compartmentalised/rotatable signing trust;
- explicit emergency rollback authority.

No TUF deployment is authorised by this decision.

### R353-02 — Cryptographic agility

**Classification:** NEW LONG-HORIZON REQUIREMENT / CURRENT MECHANISM RETAINED.

Ed25519 remains the correct current service-identity direction.

Long-lived identity/release/backup records must name algorithm/key/proof versions and permit controlled future cryptographic transitions instead of encoding Ed25519 as permanent product identity.

This follows current NIST CSWP 39upd1 crypto-agility guidance.

### R353-03 — Time authority semantics

**Classification:** EXISTING LESSONS / INTEGRATION GAP.

Current repo already contains correct monotonic-time use in the perception fencing lease and audit findings for wall-clock/caller-time failures elsewhere.

A common semantic contract should distinguish event time, observation/ingest time, authority/decision time and monotonic deadlines/leases.

No new time service is implied.

### R353-04 — Cross-organ resource budgets

**Classification:** EXISTING PARTIAL CONTROLS / INTEGRATION GAP.

Current rate-limit, prompt-budget, hardware/resource and audit work is fragmented.

Maturation should use a common vocabulary for time, tokens/context, RAM/GPU/NPU, network/parser bytes, retries, queue/in-flight work, interruption and consequence budgets, enforced by the organs that own those resources.

No central mega-scheduler is implied.

### R353-05 — Data-classification propagation

**Classification:** EXISTING P2/P3/FINAL-SPEC PRINCIPLE / INTEGRATION GAP.

Principal/purpose/classification/provenance metadata must travel through retrieval, prompt/context assembly, model/provider selection, tool proposals, capabilities and egress — not stop at storage metadata.

No new data-classification service is implied.

## External standards/guidance retained as references

The reissue cross-checked current authoritative/mature guidance including:

- W3C PROV;
- SLSA v1.2;
- in-toto;
- PostgreSQL 18 locking / `SKIP LOCKED` / `ON CONFLICT` semantics;
- RFC 9421 HTTP Message Signatures;
- RFC 9449 DPoP request/token proof-binding concepts;
- NIST CSWP 39upd1 crypto agility;
- TUF update security/metadata roles;
- NIST AI 600-1 Generative AI Profile;
- OWASP current LLM/Agent/RAG security guidance;
- Google SRE cascading-failure/retry-budget guidance;
- Docker Compose profiles;
- CycloneDX component/service/dependency evidence concepts;
- OpenTelemetry service identity conventions;
- AMD current Strix Halo/Ryzen AI/ROCm documentation.

Standards inform properties; they do not become automatic technology dependencies.

## Diminishing-return decision

Kai stops broad external research for this reissue because new searches are now predominantly confirming existing requirements or offering alternative implementations of already identified properties.

The remaining material uncertainty is Kai-specific current-state evidence, not lack of external best-practice coverage.

Therefore the next high-value step is **repo/current-runtime reconciliation**, not another broad architecture/research sweep.

## Programme relationship

D353 does not change current execution sequence.

Standing sequence remains subject to canonical D-numbered authority, including House-in-Order, KAI-GATE-048, Item 8, **ITEM 8 BEFORE A4**, `A-4 PROVENANCE`, Assurance/Evidence work and later professionalisation.

Kingsman maturation must consume the eventual Evidence/A-4 qualification contracts; it does not become a parallel truth programme and Evidence Plane does not become architecture authority.

## Programme authority unchanged

D353 authorises no H2 v1.1, House mutation, 048 scope change, Item8 build, A-4/Future-A4 implementation, runtime refactor, service merge/delete, secure-update implementation, crypto migration, succession, autonomous finance or uncontrolled self-modification.

## Recovery statement

The current implementation-research authority is the D353 reissue as **research input only**. The professional organism-first master architecture remains OPEN under D351/D352 and must be built from exact current-organism evidence rather than from this research document alone.
