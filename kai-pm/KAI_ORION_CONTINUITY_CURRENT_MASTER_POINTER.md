# KAI ↔ ORION — Current Master Recovery Pointer

> **STATUS: MANDATORY RECOVERY POINTER WHILE CANONICAL `DECISIONS.md` APPEND D344–D349 IS OPEN.**
>
> This file exists so a fresh Kai/Orion thread cannot stop at the last entry currently inside canonical `kai-pm/DECISIONS.md` and miss later architecture/governance work that is durably banked but awaiting a byte-safe canonical append.
>
> **READ THIS POINTER TOGETHER WITH `KAI_ORION_CONTINUITY.md`. IT DOES NOT REPLACE THE PROGRAMME AUTHORITY ORDER IN THAT FILE.**

## 1. Current recovery chain

Before consequential Kingsman/master-canon/Phase-2 architecture advice, recover:

1. `kai-pm/KAI_ORION_CONTINUITY.md` — standing protocol.
2. latest canonical D-numbered programme state in `kai-pm/DECISIONS.md` — execution/programme authority.
3. `kai-pm/DECISIONS_CANONICAL_APPEND_QUEUE_D344_D349.md` — pending governance not yet byte-safely appended.
4. D344 through D348 standalone checkpoints.
5. `kai-pm/D349_DEEPSEEK_V0_3_EXISTING_SYSTEM_REVIEW_RECONCILIATION.md`.
6. `kai-pm/KINGSMAN_EXISTING_KAI_EVOLUTION_MASTER_PLAN_V0_3.md` — current primary architecture candidate, not frozen.
7. `kai-pm/KAI_RECONCILIATION_DEEPSEEK_EXISTING_KAI_EVOLUTION_V0_3.md` — **mandatory repo-backed correction layer after DeepSeek review**.
8. `kai-pm/DEEPSEEK_REVIEW_PACKET_EXISTING_KAI_EVOLUTION_V0_3.md` — review prompt/history, not authority.
9. `kai-pm/KINGSMAN_ARCHITECTURE_DOCUMENT_AUTHORITY_INDEX_V0_3.md`.
10. `kai-pm/KINGSMAN_CANON_SYNCHRONISATION_REGISTER_V0_3.md`.
11. `kai-pm/UH_PROGRESS_TRACKER.md` and `KAI_UNIFIED_HUNTER_ARCHITECTURE_AND_ROADMAP.md` whenever runtime/control migration is discussed.
12. v0.2 master/manifest and standing doctrines as retained source inputs.

## 2. Current correct status

> **v0.3 EXISTING-KAI EVOLUTION REVIEWED BY DEEPSEEK AND RECONCILED BY KAI — v0.4 CHANGE SET / FINAL CANON NOT YET AUTHORED OR FROZEN — DOCUMENT/CANONICAL DECISION SYNCHRONISATION STILL OPEN.**

Do not reconstruct the state as:

- “nothing after D343 happened” — false;
- “DeepSeek's review is accepted verbatim” — false;
- “v0.2 is current review subject” — false;
- “Kai needs a fresh architecture from scratch” — false;
- “Unified Hunter final-hand cutover is complete” — false;
- “master canon is final/frozen” — false.

## 3. Current subject identities

- v0.3 Existing-Kai Evolution Master Plan: `98dc2560c0204dd9c58d823dbd8c07754704276a`
- v0.3 DeepSeek packet: `1d189de2cfd7b20499b7b7fc8bb8981915672350`
- D348: `c683a533fc984afa63cf890541ae7842c34f76b5`
- Kai DeepSeek reconciliation: `89fdd1b820e245550fe2574d26ef17e6651f4dec`
- D349 checkpoint: `8a949da867045a61563e59d608bf6d0f29521211`
- D344–D349 append queue: `f6713b50a4f829b81bc0f68b6568d33c763056b5`
- v0.3 architecture authority index: `343d7d4f736a89fb3944054051276f1d9da3def0`
- v0.3 synchronisation register: `c14232a8e93bc6d785de8adaf72cd6226a739497`

## 4. Mandatory D349 corrections

### KAI-REV-016 — final-hand capability blocker

Current central `ActuatorRegistry` consumes `ActionCapability`, but current mutating handlers send downstream parameters plus auth/signature rather than the exact one-use capability. The actual side-effecting service therefore does not yet atomically validate/consume the exact capability at the final hand.

### KAI-REV-017 — legacy closure blocker

An authenticated direct route is not equivalent to a dead legacy authority path. Current source verifier can call some routes closed once service authentication exists, while a shared-token holder may still bypass central one-use capability control.

### KAI-REV-018 — durable autonomy/preflight gap

Current autonomy grants are process-local and current preflight constructs a fresh authority. It cannot prove durable runtime grant readiness.

### Authority separation

`MEMBERSHIP != IDENTITY != AUTHORITY != ONE-TIME EXECUTION CAPABILITY != AUTONOMY DELEGATION`.

Manual operator-approved actions do **not** universally require `KAI_AUTONOMY_ENFORCE=true`; autonomous initiation does.

## 5. Architecture intent that survives

- Kai is the organism; models/frameworks/services/hardware are replaceable organs.
- Existing Kai is substantial; no blank-sheet redesign.
- Unified Hunter remains the migration skeleton, but its final-hand claim must be requalified against D349.
- **NO NEW BOX WITHOUT CURRENT-TO-TARGET LINEAGE.**
- Product capability may be consolidated but not silently deleted.
- Use existing House/Census/instrumentation as the basis for E0 rather than inventing another census system.
- Cortex/world-state fallback must become explicit COLD_START/DEGRADED/UNKNOWN, not silent steady-state legacy truth.
- Signed identity already has body/path/destination/timestamp/nonce/replay/revocation protections; finish migration instead of reimplementing them.
- Durable authority evolves behind existing Tool Gate-compatible interfaces.
- Postgres workflow/outbox is the first justified durable-workflow candidate, not an eternal technology prohibition.
- Feature flags remain selectors; evidence-bound migration/release state governs promotion.
- Egress/target constraints belong in policy/exact capability and final-hand enforcement.

## 6. Programme order remains separate

This pointer authorises no implementation/experiment.

Preserve latest valid canonical D-numbered sequence, including House-in-Order, KAI-GATE-048, Item8 separate frozen authority, **ITEM 8 BEFORE A4**, and `A-4 PROVENANCE` distinct from `FUTURE A4 SELF-DIAGNOSIS`.

No architecture/review document authorises H2 v1.1, runtime refactor, service merge/delete, succession, autonomous finance or uncontrolled self-modification.

## 7. Next architecture action

Prepare a **v0.4 delta/change set**, not a fresh architecture, carrying only repo-supported/corrected D349 findings. Before freezing physical service homes for memory/proactivity/model runtime, complete the exact current component/reader/writer/authority census and any discriminating spikes.

Retire this pointer only after D344–D349 are safely represented/appended in canonical `DECISIONS.md`, standing continuity is directly updated, and a later accepted/frozen master-canon pointer supersedes it.
