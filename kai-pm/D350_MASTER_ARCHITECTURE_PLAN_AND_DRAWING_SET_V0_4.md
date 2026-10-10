# D350 — Master Architecture Plan & Engineering Drawing Set v0.4

> **STATUS: GOVERNANCE / ARCHITECTURE CHECKPOINT — v0.4 MASTER PLAN + DRAWINGS + MIGRATION MATRIX BANKED. NOT FINAL CANON. NOT IMPLEMENTATION AUTHORITY. NOT PROGRAMME EXECUTION AUTHORITY.**

## Trigger

Dainius authorised Kai to take the accumulated repo evidence, Kai analysis and reconciled DeepSeek review input and produce the actual professional plan and complete architecture schematics.

The requirement was explicitly not another conceptual description. The deliverable had to:

- start from existing Kai;
- preserve Unified Hunter's already-built migration machinery;
- incorporate D349 repo-backed corrections;
- preserve current capability families;
- distinguish current/transitional/target/new/unknown;
- show all major logical/control/data/failure/continuity/operator flows;
- include current→target shims and cutover evidence;
- avoid blank-sheet service invention;
- remain separate from current programme execution authority.

## Primary v0.4 master plan

`kai-pm/KINGSMAN_EXISTING_KAI_MASTER_ARCHITECTURE_PLAN_V0_4.md`

Creation commit:

`391f0b48eb3166888f0941cc61adb642e9197bde`

This document is now the **primary whole-system architecture candidate** for review. It supersedes v0.3 as the primary synthesis while retaining v0.3/D349 as review lineage and evidence.

Major additions/corrections:

- D349 authority semantics integrated directly;
- final-hand one-use capability made a mandatory architectural gate;
- manual-vs-autonomous authority lanes separated;
- runtime legacy-bypass proof made a retirement requirement;
- durable authority state behind existing Tool Gate-compatible APIs;
- explicit COLD_START/DEGRADED World-State fallback semantics;
- Postgres workflow/outbox around existing ActuatorRegistry as first implementation candidate;
- egress/target constraints attached to exact capability/final hand;
- model-runtime responsibility reduced to registry enrichment until stronger boundary is earned;
- memory/proactivity physical ownership deliberately left provisional pending E0;
- E0→E11 architecture dependency programme;
- zero-loss migration gates;
- Mission Control information architecture;
- freeze criteria.

## Complete deterministic drawing set

`kai-pm/KINGSMAN_ENGINEERING_ARCHITECTURE_DRAWING_SET_V0_4.md`

Creation commit:

`85df84ebd71c25a1e97305681edcc218316c8a18`

The drawing set contains deterministic Mermaid engineering views for:

A. whole rooted Kai organism;
B. current transitional legacy + built-UH runtime;
C. final manual/autonomous consequential-action authority lanes;
D. current final-hand gap;
E. perception→event→evidence→World State;
F. memory/identity/relationship architecture;
G. proactivity/Goal/Watch/Attention architecture;
H. cognition/Unified Hunter/model resources;
I. identity/authority semantic stack;
J. durable workflow state machine;
K. health/Doctor/contingency/recovery;
L. continuity/lineage/restore;
M. finance/sustainability boundary;
N. canonical deployment profile model;
O. current→target migration control loop;
P. E0→E11 roadmap;
Q. Mission Control operator view;
R. fault-containment map;
S. present/qualified/live/enforced evidence relationship;
T. final organic flow.

No generative image is architecture authority.

## Current→target migration matrix

`kai-pm/KINGSMAN_CURRENT_TO_TARGET_MIGRATION_MATRIX_V0_4.md`

Creation commit:

`9ee5c419ecb861305e00332da95bb12f967b232c`

The matrix maps existing component families to provisional dispositions, shims, target responsibilities, cutover proof and do-not-retire conditions.

It explicitly protects against bulk cleanup or capability loss.

## New/corrected mandatory joints now represented

- M17 Final-Hand Execution Capability — BLOCKER.
- M18 Runtime Legacy-Bypass Probe — BLOCKER.
- M19 Durable Authority State — MAJOR.
- M20 Scoped-Autonomy Grant Bootstrap — MAJOR.
- M21 Egress/Target Constraints — MAJOR.
- M22 Evidence-Bound Migration Record — MAJOR.

Existing shims E01–E08 and candidate M01–M16 remain retained/reconciled where useful.

## DeepSeek role preserved

DeepSeek remains Kai's external specialist analysis instrument. v0.4 incorporates only points that survived Kai's repo/history/philosophy reconciliation.

DeepSeek is not a co-authority or source of current repo truth.

## Current architecture status

Correct statement:

> **v0.4 PROFESSIONAL MASTER PLAN + ENGINEERING DRAWING SET + MIGRATION MATRIX BANKED — CURRENT COMPONENT CENSUS / PHYSICAL BOUNDARY QUALIFICATION / FINAL REVIEW / FINAL CANON FREEZE STILL OPEN.**

v0.4 is more complete than v0.3 but is **not yet `KINGSMAN_MASTER_CANON_v1`**.

## Open before freeze

1. E0 exact machine component/connection/reader/writer/state-owner/authority census.
2. E1 exact minimal/full/sovereign deployment-profile reconciliation.
3. physical service/module boundary decisions based on E0 rather than preference.
4. current memory authoritative/derived ownership proof.
5. current proactivity ownership/dependency proof.
6. exact final-hand capability transport/atomic-consumption design.
7. workflow/outbox exact design and discriminating tests.
8. operator Mission Control machine-state schema.
9. exact v0.4 zero-loss input manifest refresh.
10. Orion feasibility/current→target mapping against E0.
11. targeted DeepSeek attack only on unresolved architecture choices where useful.
12. Dainius final architecture review.
13. deterministic drawings updated from E0 exact subjects.
14. exact-byte `KINGSMAN_MASTER_CANON_v1` freeze/change control.
15. canonical `DECISIONS.md` append remains separately open.

## Programme authority unchanged

D350 authorises no H2 v1.1, House mutation, 048 scope change, Item8 execution, A-4/Future-A4 execution, runtime refactor, service merge/delete, succession, autonomous finance or uncontrolled self-modification.

Latest valid canonical D-numbered programme authority wins. **ITEM 8 BEFORE A4** remains standing; `A-4 PROVENANCE` remains distinct from `FUTURE A4 SELF-DIAGNOSIS`.

## THREAD RECOVERY BLOCK

**PRIMARY ARCHITECTURE SUBJECT:** `KINGSMAN_EXISTING_KAI_MASTER_ARCHITECTURE_PLAN_V0_4.md` @ `391f0b48eb3166888f0941cc61adb642e9197bde`.

**PRIMARY DRAWING SET:** `KINGSMAN_ENGINEERING_ARCHITECTURE_DRAWING_SET_V0_4.md` @ `85df84ebd71c25a1e97305681edcc218316c8a18`.

**MIGRATION MATRIX:** `KINGSMAN_CURRENT_TO_TARGET_MIGRATION_MATRIX_V0_4.md` @ `9ee5c419ecb861305e00332da95bb12f967b232c`.

**MANDATORY PREDECESSOR CORRECTION:** D349 + `KAI_RECONCILIATION_DEEPSEEK_EXISTING_KAI_EVOLUTION_V0_3.md`.

**CURRENT NEXT ARCHITECTURE ACTION:** E0 exact current machine/component/connection census and v0.4 verification, not implementation of target architecture.
