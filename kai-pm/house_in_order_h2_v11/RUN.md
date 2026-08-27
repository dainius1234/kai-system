# HOUSE-IN-ORDER-H2-CLASSIFIER v1.1

Built under Dainius's authorisation within Kai's D360 scope. **HOUSE_H2
v1.0 is untouched** and remains the historical benchmark. Census v1.1
(`eb7aad7c…fa0e`) is untouched and consumed read-only.

> **STATUS: CANDIDATE. NOT FROZEN. NOT ADMITTED TO HOUSE_H3.**
> Admission is a governing decision, never a test result. This package
> supplies evidence and approves nothing about itself.

---

## The contract this implements

**D360 §5 — EVIDENCE FACTS ARE NOT VERDICTS.** v1.0 awarded `ACTIVE`
from `commits > 1` or from a present-tense self-claim. Neither proves
current lifecycle. **67% of v1.0's ACTIVE verdicts (29 of 43) rested on
a self-claim alone.**

| evidence fact | says | does NOT say |
|---|---|---|
| `MAINTENANCE_OBSERVED` | the document was changed in the window | that it governs anything now |
| `SELF_ASSERTS_CURRENT` | the document asserts currentness | that the assertion is true |
| `CONSUMED_AT_SUBJECT` | code reads it at the subject | that it is live — stale artefacts are read too |

The separation is **structural**: `classify2.lifecycle()` does not
receive the evidence facts, so it cannot convert one into a verdict even
by accident. A comment would not have prevented v1.0's defect — v1.0's
docstring already said authority was earned at H3 while its LIFECYCLE
branch awarded ACTIVE from a self-claim.

## Running it

Nothing is hard-coded; v1.0's `pass_a.py` hard-coded a home directory
and a session scratchpad while its RUN.md advertised generic inputs.

```sh
python3 passa.py --subject-repo <exact checkout> \
                 --history-repo <FULL-history repo> \
                 --subject <sha> \
                 --census-package <frozen Census v1.1> --out passA.json

python3 run_h2_v11.py --subject-repo <...> --history-repo <...> \
                 --subject <sha> --census-package <...> \
                 --passa passA.json --out result.json

python3 qualify_h2.py --result result.json      # exits non-zero on findings
python3 cal_fixtures.py                         # 33 hostile assertions
python3 holdout.py --result result.json --out holdout.json
```

**The subject is verified before measurement** (R11): the subject repo's
`HEAD` must equal the subject, and the subject commit must be present in
the history source, or the run aborts.

**The history source must be full-history.** The active repository is
shallow with a 21-day window; under it `commits==1` for 94% of documents
— an artefact of truncation, not a document property. The observation
window's identity travels with every history-derived number.

## Results — subject `d8aac4d4`, tree `3abc9e9d`, 272 documents

History window `2025-06-18 → 2026-08-07`, non-shallow, ancestry 986.

| axis | v1.0 | v1.1 |
|---|---|---|
| `LIFECYCLE ACTIVE` | 43 | **0** |
| `LIFECYCLE UNKNOWN` | 216 | **259** |
| `LIFECYCLE HISTORICAL` | 13 | 13 |
| `FUNCTION UNKNOWN` | 69 | **50** |
| `FUNCTION REFERENCE` | 0 | **8** |
| `FUNCTION OTHER` | 0 | **3** |
| `FUNCTION USER_GUIDE` | 3 | 10 |
| `VALIDITY CURRENT_TREE` | 8 | 7 |
| `VALIDITY TIME_BOUND` | 22 | 23 |

**UNKNOWN rose, as predicted before the build.** That is the contract
working, not a regression.

Evidence facts: `MAINTENANCE_OBSERVED` 71 · `SELF_ASSERTS_CURRENT` 6 ·
`CONSUMED_AT_SUBJECT` 5.

**`SELF_ASSERTS_CURRENT` is 6 where v1.0's `present_tense` flag was 39.**
The raw flag was six times broader than an actual subject-bound
currentness claim — which is why it should never have earned a verdict.

### The D340 false positive is repaired

`docs/agentic_patterns_spec.md` — `Version: 1.0 — 2 Mar 2026`:
`CURRENT_TREE` → **`TIME_BOUND`**. Proven by a fail-old/pass-new pair:
v1.0's pattern is reproduced verbatim in `cal_fixtures.py` and must
*fail* the input v1.1 passes.

### One D340 gap NOT repaired — stated, not curated

`docs/wake_intent_j2.md` remains `FUNCTION=UNKNOWN`; Kai's blind
adjudication said `REFERENCE`. `kai-pm/NAVIGATION.md` *is* now
`REFERENCE`. **I did not add a rule to capture the remaining one.** Its
correct answer is known to me from D340, and tuning to match a known
adjudication is fitting to the holdout, not repair.

## Qualification

* hostile fixtures **33/33**, covering all six precommitted classes plus
  `REFERENCE`/`OTHER` reachability and the three values unobserved on
  this subject;
* state-disposition qualification **0 findings** — every
  `H2_NOT_EARNABLE` and `DEFERRED_TO_H3` value emitted **zero** times;
  every `H2_EMITTABLE` value reachable;
* population 272 == 272, asserted;
* fresh-environment reproduction **executed** from an unrelated
  directory: all 272 rows and the admission contract identical.

Acceptance contract precommitted at
`fa1069103a721cf5911641cbe6447360069eb9f2a3873a4296531ae280f4258e`
before any v1.1 code existed, and **unamended**.

## Blind holdout

24 of 272 by the precommitted rule `sha256("H2V11:" + path)` ascending —
fixed in `PRECOMMIT.md` §4 before the code existed, and distinct from
D340's set by salt. **Not self-adjudicated.** Emitted in
`h2v11-holdout.json` for independent blind adjudication; any agreement
figure I compute would carry no acceptance weight.

## Restrictions carried in the artefact

* `LIFECYCLE=ACTIVE` is **`H2_NOT_EARNABLE`**;
* `AUTHORITY` states are **`DEFERRED_TO_H3`**;
* `GENERATION` verdicts and `SCOPE` region overrides are
  **`H2_NOT_EARNABLE`**;
* `UNKNOWN`/`UNMEASURED` are abstentions — per D340 §7 and D358 they may
  **never** be used as negative evidence or an exclusion criterion;
* history-derived facts are void outside the declared window.

## Known confound, stated so it is not misattributed

v1.0 consumed **Census v1.0**, which D341 proved defective; v1.1
consumes **frozen Census v1.1**. Any v1.0↔v1.1 delta may originate in
the Census change as well as the classifier change. The two are **not
separable by comparison alone**.
