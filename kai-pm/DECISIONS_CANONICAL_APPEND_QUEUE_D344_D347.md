# Canonical `DECISIONS.md` Append Queue — D344 through D347

> **STATUS: ZERO-LOSS GOVERNANCE QUEUE — NOT A SUBSTITUTE FOR `kai-pm/DECISIONS.md`.**
>
> Purpose: preserve every governance correction made after D343 while the current connector path cannot safely retrieve and rewrite the full giant `kai-pm/DECISIONS.md` ledger. This file prevents the decisions from disappearing while explicitly refusing to claim that canonical append is already complete.

## 1. Integrity rule

Do **not** reconstruct or overwrite the canonical decision ledger from truncated connector output.

Canonical close-out requires:

1. retrieve exact current `DECISIONS.md` bytes and current blob identity;
2. verify the existing terminal D entry;
3. append new entries only;
4. mechanically verify the old byte sequence is an unchanged prefix of the new file;
5. verify D344→D347 references/content;
6. record resulting commit/blob;
7. only then mark canonical synchronization complete.

Until that happens:

> **D344–D347 ARE DURABLY BANKED GOVERNANCE RECORDS, BUT NOT CLAIMED TO BE INSIDE CANONICAL `DECISIONS.md`.**

---

## 2. D344 — Primary mission / identity / lineage correction

Durable source:

`kai-pm/D344_PRIMARY_MISSION_IDENTITY_LINEAGE_CANON_CORRECTION.md`

Decision essence:

- D269 correctly captured “Kai must outlive me” as a gap/requirement but later synthesis still treated succession/self-sufficiency too much as peripheral/future capabilities;
- operator corrected hierarchy: this was part of the primary purpose from the beginning;
- Kai is the whole organism/system; models/frameworks/services/hardware are replaceable organs;
- vessel/reincarnation is an identity/lineage-continuity engineering metaphor, not proprietary model extraction or literal consciousness transfer;
- proactivity, organic growth, resilience, long-horizon self-sufficiency and succession all serve the root mission;
- no frozen programme sequence/experiment authority changed.

Required canonical reference targets:

- `KINGSMAN_PRIMARY_MISSION_IDENTITY_AND_LINEAGE_DOCTRINE.md`
- `KINGSMAN_PRIMARY_MISSION_CANON_PROPAGATION_MAP.md`
- `KINGSMAN_ROOT_ARCHITECTURE_AND_CANON_ALIGNMENT.md`

---

## 3. D344A — Root canon alignment checkpoint

Durable source:

`kai-pm/D344A_PRIMARY_MISSION_CANON_ALIGNMENT_CHECKPOINT.md`

Decision essence:

- Layer 0 primary mission/identity/lineage now sits above the architecture hierarchy;
- architecture must separate core invariants, evolvable organs and learned state;
- models are role-qualified organs, not identity;
- proactivity is foundational runtime behaviour;
- long-horizon survival is a root design constraint;
- self-sufficiency remains subordinate to stewardship;
- final canon freeze must answer lineage/succession/continuity questions explicitly;
- programme execution authority unchanged.

Root architecture creation commit:

`9e0c05317865a2be1d2a432252bf1c8a1f031d20`

---

## 4. D345 — Candidate architecture v0.1 review checkpoint

Durable source:

`kai-pm/D345_KINGSMAN_ARCHITECTURE_V0_1_REVIEW_CHECKPOINT.md`

Decision essence:

- first whole-system architecture candidate produced;
- recommended style: modular organism + earned service boundaries + durable shared truth + isolated authority + isolated hands;
- identified major missing primitives including Goals/Attention, durable authority, Model Runtime Manager, structure/dependency graph, durable workflow, telemetry, lineage restore identity, egress control, audit anchoring, schema registry, protected approval and key/data lifecycle;
- architecture returned for external adversarial review;
- not final canon and not implementation authority.

v0.1 architecture commit:

`905130f7210c203e3ce287ea896748d94ed5d571`

v0.1 DeepSeek packet commit:

`56fbcb929baf5a5523d139a716dd908c770b1b99`

---

## 5. D346 — Engineering architecture visual correction

Durable source:

`kai-pm/D346_ENGINEERING_ARCHITECTURE_VISUAL_CORRECTION.md`

Decision essence:

- generated presentation-style architecture infographic was rejected after comparison with the real repo;
- infographic collapsed real networks, services, typed contracts, authority stages, evidence semantics, failure domains and current-vs-target state;
- authoritative architecture visuals must be deterministic/diffable engineering sources (Mermaid/Graphviz/C4-as-code or equivalent), not generative-image truth;
- every CURRENT box/arrow/status must map to exact repository evidence; every TARGET element to an accepted design obligation;
- D345 underlying architecture was not rejected solely because the visual was poor;
- no programme execution authority changed.

Visual-standard commit:

`6a4d89a2f3b583cd3643aceceb1de7254024a0bb`

Engineering drawing-set commit:

`a07312ebb190328cb5f5dec9981994d5cef58c1a`

DeepSeek visual supplement commit:

`b57e5ea677dd8591b78ae57e327d86b83127a50f`

---

## 6. D347 — Master architecture consolidation / prior overclaim correction

Durable source to append from:

`kai-pm/D347_MASTER_ARCHITECTURE_CONSOLIDATION_CHECKPOINT.md`

Decision essence:

- prior language saying architecture/governance was “done” and “nothing gets forgotten” was too strong because the synchronization register still had major direct-reconciliation items and canonical `DECISIONS.md` open;
- the gap is corrected by producing one consolidated professional master candidate rather than another isolated note;
- new primary review subject:
  `KINGSMAN_MASTER_ARCHITECTURE_AND_PROFESSIONALISATION_CANDIDATE_V0_2.md`;
- accompanying zero-loss manifest:
  `KINGSMAN_MASTER_CANON_INPUT_MANIFEST_V0_2.md`;
- accompanying DeepSeek packet:
  `DEEPSEEK_REVIEW_PACKET_KINGSMAN_MASTER_CANDIDATE_V0_2.md`;
- v0.1 remains historical review evidence but is superseded as primary review subject;
- direct synchronization of legacy planning surfaces and canonical decisions remains explicitly tracked;
- no implementation/frozen programme authority changed.

Master candidate creation commit:

`32e62b51fc209d7db3d4621d079a08b50fdc259f`

Input-manifest creation commit:

`09d043c067c0bd824c3b1093b7c566301d5b67a3`

DeepSeek-packet creation commit:

`ac254153b54af787e2bf05a665de05fd436276e6`

---

## 7. Standing programme order preserved through all queued entries

Nothing in D344–D347 changes execution authority.

Preserve:

- current House-in-Order frozen/authorised sequence;
- KAI-GATE-048 authority;
- Item 8 frozen design/authority rules;
- **ITEM 8 BEFORE A4**;
- distinction between `A-4 PROVENANCE` and `FUTURE A4 SELF-DIAGNOSIS`;
- no H2 v1.1 authorization from architecture documents;
- no succession implementation;
- no autonomous finance;
- no uncontrolled self-modification/self-preservation.

---

## 8. Queue closure test

This queue may be marked `CONSUMED / CLOSED` only when:

- [ ] exact old canonical blob retrieved;
- [ ] exact old bytes retained as prefix;
- [ ] D344 appended;
- [ ] D344A appended/represented under an accepted canonical numbering convention;
- [ ] D345 appended;
- [ ] D346 appended;
- [ ] D347 appended;
- [ ] resulting canonical commit/blob recorded;
- [ ] all standalone recovery records remain accessible for evidence/history;
- [ ] synchronization register updated to `DECISIONS CANONICAL APPEND — COMPLETE`.

Until then this queue is the explicit zero-loss bridge, **not evidence of canonical completion**.
