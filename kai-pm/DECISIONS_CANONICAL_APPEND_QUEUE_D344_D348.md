# Canonical `DECISIONS.md` Append Queue — D344 through D348

> **STATUS: ZERO-LOSS GOVERNANCE QUEUE — NOT A SUBSTITUTE FOR `kai-pm/DECISIONS.md`.**
>
> This supersedes `DECISIONS_CANONICAL_APPEND_QUEUE_D344_D347.md` as the current pending-append queue. The older queue is retained for history.

## Integrity rule

Do not reconstruct or overwrite the canonical ledger from truncated connector output.

Closure requires:

1. retrieve exact current `DECISIONS.md` bytes/blob;
2. verify terminal canonical entry;
3. append only;
4. prove mechanically that the entire old byte sequence is an unchanged prefix;
5. verify D344→D348 contents/references;
6. record resulting commit/blob;
7. only then mark canonical synchronization complete.

Until then:

> **D344–D348 ARE DURABLY BANKED GOVERNANCE RECORDS, BUT ARE NOT CLAIMED TO BE INSIDE CANONICAL `DECISIONS.md`.**

## Pending entries

### D344 — Primary mission / identity / lineage correction

Source: `D344_PRIMARY_MISSION_IDENTITY_LINEAGE_CANON_CORRECTION.md`

Essence: Kai's long-horizon survival/stewardship is root purpose; Kai is organism, models/frameworks/services/hardware are replaceable organs; no programme authority changed.

### D344A — Root canon alignment checkpoint

Source: `D344A_PRIMARY_MISSION_CANON_ALIGNMENT_CHECKPOINT.md`

Essence: mission/identity/lineage becomes Layer 0; core invariants/evolvable organs/learned state separated; proactivity and long-horizon survival become root constraints.

### D345 — Candidate architecture v0.1

Source: `D345_KINGSMAN_ARCHITECTURE_V0_1_REVIEW_CHECKPOINT.md`

Essence: first broad whole-system candidate; useful but later superseded as primary review subject.

### D346 — Engineering visual correction

Source: `D346_ENGINEERING_ARCHITECTURE_VISUAL_CORRECTION.md`

Essence: generative infographic rejected as architecture authority; deterministic evidence-bound engineering drawings required.

### D347 — Master architecture v0.2 consolidation / overclaim correction

Source: `D347_MASTER_ARCHITECTURE_CONSOLIDATION_CHECKPOINT.md`

Essence: v0.2 consolidated mission/target/professionalisation sources; prior “done” language corrected; v0.2 still not frozen.

### D348 — Existing-Kai evolution correction

Source: `D348_EXISTING_KAI_EVOLUTION_CORRECTION.md`

Essence:

- v0.2 target-first framing was still insufficient for DeepSeek because it could invite a blank-sheet redesign;
- repo comparison confirmed Unified Hunter's migration layer and multiple shims are already built/tested behind existing Kai;
- current architecture work must therefore be expressed as **evolution of current components**, not construction of generic new planes/services;
- current shims such as perception shadow/active, Cortex world-state adapter, LegacyTrustBridge, actuator migration/legacy verifier, service-auth→signed-identity transition, dashboard migration shims and feature flags are architecture assets;
- new review rule: `CURRENT → QUALIFY → PRESERVE INTENT → SHIM → TARGET → SHADOW/SOAK → VERIFIED CUTOVER → PROVE OLD PATH DEAD → RETIRE/REHOME`;
- `NO NEW BOX WITHOUT CURRENT-TO-TARGET LINEAGE`;
- current product capabilities including inner-life/identity, cognition, proactivity, growth and interfaces must not disappear during simplification;
- v0.3 becomes primary review subject; v0.2 retained as target-design/history input.

v0.3 master creation commit:

`98dc2560c0204dd9c58d823dbd8c07754704276a`

v0.3 DeepSeek packet creation commit:

`1d189de2cfd7b20499b7b7fc8bb8981915672350`

D348 standalone checkpoint creation commit:

`c683a533fc984afa63cf890541ae7842c34f76b5`

## Standing programme protections

Nothing in D344–D348 changes execution authority.

Preserve:

- current House-in-Order authority/frozen subjects;
- KAI-GATE-048 authority;
- Item 8 frozen design/authority;
- **ITEM 8 BEFORE A4**;
- distinction `A-4 PROVENANCE` vs `FUTURE A4 SELF-DIAGNOSIS`;
- no H2 v1.1 authorization from architecture work;
- no service consolidation/refactor merely because target design exists;
- no succession implementation;
- no autonomous finance;
- no uncontrolled self-modification/self-preservation.

## Queue closure

- [ ] exact canonical old blob/bytes retrieved
- [ ] old bytes verified as prefix
- [ ] D344 represented/appended
- [ ] D344A represented under accepted canonical convention
- [ ] D345 appended
- [ ] D346 appended
- [ ] D347 appended
- [ ] D348 appended
- [ ] resulting canonical commit/blob recorded
- [ ] standalone recovery records retained
- [ ] synchronization/continuity pointers updated to canonical completion

Until all applicable boxes close, this queue remains OPEN.
