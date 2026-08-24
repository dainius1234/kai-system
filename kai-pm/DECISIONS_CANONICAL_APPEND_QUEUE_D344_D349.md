# Canonical `DECISIONS.md` Append Queue — D344 through D349

> **STATUS: ZERO-LOSS GOVERNANCE QUEUE — NOT A SUBSTITUTE FOR `kai-pm/DECISIONS.md`.**
>
> Supersedes the D344–D348 queue as the current pending-append surface. Older queues remain history.

## Integrity rule

Do not reconstruct or overwrite canonical `DECISIONS.md` from truncated connector output.

Closure requires exact current bytes/blob, append-only change, mechanical proof old bytes remain an unchanged prefix, verification of D344–D349 entries/references, and resulting commit/blob recording.

Until then:

> **D344–D349 ARE DURABLY BANKED GOVERNANCE RECORDS BUT ARE NOT CLAIMED TO BE INSIDE CANONICAL `DECISIONS.md`.**

## Pending entries

- D344 — primary mission / identity / lineage correction.
- D344A — root canon alignment checkpoint under accepted numbering convention.
- D345 — candidate architecture v0.1.
- D346 — engineering visual correction.
- D347 — master architecture v0.2 consolidation / overclaim correction.
- D348 — existing-Kai evolution correction; v0.3 becomes primary review subject.
- D349 — DeepSeek v0.3 existing-system adversarial review reconciled against repo.

## D349 essence

DeepSeek's `APPROVE WITH CHANGES` review is accepted as review input, not authority.

Kai independently verified/corrected it and recorded:

- 9 materially supported findings;
- 5 supported with correction/narrowing;
- 1 phasing refinement;
- KAI-REV-016 BLOCKER: central ActuatorRegistry consumes ActionCapability but downstream actual side-effecting service does not yet validate/atomically consume that exact one-use capability at the final hand;
- KAI-REV-017 BLOCKER: authenticated direct path is not equivalent to dead legacy authority path; source closure can prove the wrong property;
- KAI-REV-018 MAJOR: autonomy grant/preflight state is process-local and preflight cannot inspect durable runtime grant state.

Critical conceptual correction:

`AUTONOMY DELEGATION != EXECUTION AUTHORITY`.

Manual operator-approved mutation requires exact policy/approval/capability/final-hand validation. Autonomous mutation requires the same path **plus** valid scoped autonomy delegation.

Reconciliation artifact:

`KAI_RECONCILIATION_DEEPSEEK_EXISTING_KAI_EVOLUTION_V0_3.md`
commit `89fdd1b820e245550fe2574d26ef17e6651f4dec`.

D349 checkpoint:

`D349_DEEPSEEK_V0_3_EXISTING_SYSTEM_REVIEW_RECONCILIATION.md`
commit `8a949da867045a61563e59d608bf6d0f29521211`.

## Programme protections

Nothing in D344–D349 changes implementation/experiment authority.

Preserve current House authority, KAI-GATE-048, Item8 separate authority, **ITEM 8 BEFORE A4**, A-4 provenance vs FUTURE A4 distinction, and all no-go rules for unapproved refactor/succession/autonomous finance/self-modification.

## Queue closure

- [ ] exact canonical old bytes/blob retrieved
- [ ] old bytes verified unchanged prefix
- [ ] D344 represented/appended
- [ ] D344A represented under accepted canonical convention
- [ ] D345 appended
- [ ] D346 appended
- [ ] D347 appended
- [ ] D348 appended
- [ ] D349 appended
- [ ] resulting canonical commit/blob recorded
- [ ] standalone recovery records retained
- [ ] synchronization/continuity pointers updated
