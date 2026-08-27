# HOUSE_H2 v1.1 — ACCEPTANCE CONTRACT, PRECOMMITTED

**Written and hashed BEFORE any v1.1 classifier or package code exists.**
Kai's D360 scope §7: *"Before modifying classifier/package code, freeze
the exact v1.1 acceptance contract and test population/rules in the
execution evidence. Do not change acceptance criteria after seeing
results merely to obtain green."*

Any defect discovered later may cause repair or restart. **These criteria
remain visible and unamended.**

---

## 0. SUBJECT AND SOURCES — parameters, never assumptions

```
subject commit  d8aac4d49e6ba997e3eb38062c0917186ee3f197
subject tree    3abc9e9d8ca11966a6f996d5f0af68072ee5b117
population      272 tracked .md at that tree
```

* subject checkout, history source, subject SHA and output location are
  **explicit parameters**. No user-home path, no scratchpad path, no
  hard-coded repository.
* the runner **verifies** the subject commit AND tree in the subject
  source, and **verifies the subject commit is present in the history
  source**, before measuring anything (R11).
* the history source is recorded by identity: shallow flag, oldest
  reachable commit, oldest date, observation-window span, ancestry depth
  at the subject. **Every history-derived number carries which window
  produced it.**
* history capability failure ⇒ `UNMEASURED` / `UNKNOWN`. **Never a
  plausible count.**

## 1. THE ONTOLOGY RULING (D360 §5) IS THE CONTRACT

Evidence facts are emitted as a **separate field**. They are never
lifecycle verdicts.

| evidence fact | earned by |
|---|---|
| `MAINTENANCE_OBSERVED` | commits > 1 at the subject in the declared history window |
| `SELF_ASSERTS_CURRENT` | a structured self-claim of currentness, subject-bound |
| `CONSUMED_AT_SUBJECT` | a proven executable read of the document at the subject |

**None of these earns `ACTIVE`.**

## 2. STATE DISPOSITION — MACHINE-READABLE, NOT COMMENTS

Every declared ontology value carries exactly one disposition:

* `H2_EMITTABLE` — earnable at H2, with a fixture proving it
* `H2_NOT_EARNABLE` — declared, but no qualified rule exists at H2
* `DEFERRED_TO_H3` — ownership belongs to a later stage

Precommitted dispositions:

| axis | value | disposition |
|---|---|---|
| LIFECYCLE | `SUPERSEDED` | `H2_EMITTABLE` |
| LIFECYCLE | `HISTORICAL` | `H2_EMITTABLE` |
| LIFECYCLE | **`ACTIVE`** | **`H2_NOT_EARNABLE`** |
| LIFECYCLE | `UNKNOWN` | `H2_EMITTABLE` |
| FUNCTION | all 11 incl. `REFERENCE`, `OTHER` | `H2_EMITTABLE` |
| AUTHORITY | `AUTHORITATIVE` `VERIFIED_DERIVED` `ADVISORY` `NON_AUTHORITY` | `DEFERRED_TO_H3` |
| AUTHORITY | `UNKNOWN` | `H2_EMITTABLE` |
| GENERATION | `MANUAL` `PARTIAL_DERIVED` `FULL_DERIVED` | `H2_NOT_EARNABLE` |
| GENERATION | `UNKNOWN` | `H2_EMITTABLE` |
| VALIDITY_BINDING | all 5 | `H2_EMITTABLE` |
| SCOPE | `WHOLE_FILE` | `H2_EMITTABLE` |
| SCOPE | `HEADING` `TABLE` `MANAGED_REGION` | `H2_NOT_EARNABLE` |

A meta-check must prove each `H2_EMITTABLE` value is reachable and each
`H2_NOT_EARNABLE` / `DEFERRED_TO_H3` value is **not emitted**.

## 3. REQUIRED HOSTILE FIXTURES — all six must pass

| # | fixture | required result |
|---|---|---|
| F1 | abbreviated date `2 Mar 2026`, present-tense | v1.0 rule ⇒ `CURRENT_TREE` (**fails**); v1.1 ⇒ `TIME_BOUND`. Boundary pair: an undated present-tense doc must still yield `CURRENT_TREE` |
| F2 | self-current claim only | `SELF_ASSERTS_CURRENT` present; `LIFECYCLE=ACTIVE` **forbidden** |
| F3 | code reader only | `CONSUMED_AT_SUBJECT` present; `SELF_ASSERTS_CURRENT` absent; `ACTIVE` **forbidden** |
| F4 | no maintenance, no self-claim, no reader | none of the three facts manufactured |
| F5 | attempted emission of an `H2_NOT_EARNABLE` value | qualification **fails** |
| F6 | discovery/import/traversal failure | `FAIL`/`UNKNOWN`; **silent denominator shrink with green is forbidden** |

Plus: `REFERENCE` and `OTHER` each reached by a fixture; the D340
known-positive/known-negative and mutation fixtures retained.

## 4. BLIND HOLDOUT — SELECTION RULE FIXED NOW

```
key   = sha256("H2V11:" + path).hexdigest()
order = ascending by key
take  = first 24 of the 272
```

Deterministic, unseeded by any result, and **different from D340's set**
by construction (distinct salt). The selection is generated and
committed **before** classification output is inspected.

**I DO NOT SELF-ADJUDICATE THE HOLDOUT.** The 24 rows are produced and
returned for Kai's independent blind adjudication. Any agreement figure
in my own report is descriptive only and carries no acceptance weight.

## 5. ACCEPTANCE CRITERIA

1. all six hostile fixture classes pass;
2. **fail-old / pass-new** demonstrated for every repaired defect class;
3. every `H2_EMITTABLE` value reachable; every `H2_NOT_EARNABLE` and
   `DEFERRED_TO_H3` value proven **not emitted**;
4. denominator reconciliation: `discovered = admitted + rejected +
   errored`, and `errored > 0` ⇒ `FAIL`/`UNKNOWN`, never green;
5. population 272 == classified 272, asserted not assumed;
6. `UNKNOWN` / `UNMEASURED` preserved; no abstention converted to a
   verdict to improve coverage;
7. **no adjudicated false positive surviving in any emitted
   non-abstention state within the declared admission populations** —
   a bounded claim over those populations, *not* a universal claim;
8. explicit limitations and restrictions carried in the artefact;
9. fresh-environment reproduction **executed**, not hashed;
10. machine-readable H2→H3 admission contract emitted.

## 6. WHAT IS EXPLICITLY NOT SUFFICIENT

* a higher agreement percentage than v1.0's 93.3%;
* a lower `UNKNOWN` percentage — **v1.1 is expected to report MORE
  `UNKNOWN` than v1.0**, because 29 of v1.0's 43 `ACTIVE` verdicts rest
  on a self-claim that no longer earns a verdict;
* "all ontology values emittable" — several must **not** be;
* passing tests. Admission is Kai's ruling, not a test result.

## 7. PREDICTED OUTCOME, STATED IN ADVANCE

Recorded so it cannot be retrofitted:

* `LIFECYCLE=ACTIVE` ⇒ **0** (not earnable);
* v1.0's 43 ACTIVE redistribute to `UNKNOWN`, except any earning
  `SUPERSEDED`/`HISTORICAL` by their own witnesses;
* `LIFECYCLE=UNKNOWN` **rises materially above 216**;
* `MAINTENANCE_OBSERVED` ⇒ ~60 under full history (43 recovered + ~15
  already >1), exact figure to be measured;
* `FUNCTION=UNKNOWN` **falls**, as `REFERENCE`/`OTHER` and nomination
  gaps are repaired.

**If UNKNOWN falls instead of rising, that is a red flag, not a
success.**

## 8. OUT OF SCOPE

No H3. No H4/H5/H6. No Census successor or modification. No H2 v1.0
modification. No ACTIVE heuristic invented for coverage. No manual
repair of the 173 or the 12. No programme-order CI mechanism. No 048 /
Item 8 / A-4_PROVENANCE work.
