# HOUSE_H2 CONSOLIDATED REPAIR CONTRACT — D367

**PRE-IMPLEMENTATION. NO CLASSIFIER CODE EXISTS UNDER THIS CONTRACT.**

This document is frozen **before** the first repair byte, in its own
durable commit, and hashed. That ordering is the direct remedy for the
D361 defect in which `PRECOMMIT.md` landed *together with* the
implementation, so its chronology could only ever be an execution claim.
Here the ordering is attested by git.

Authority: Dainius approves; Kai adjudicates semantics and admission;
Orion implements **only after Kai verifies this contract and Dainius
separately authorises the consolidated repair.**

> **Nothing in this document authorises implementation.**

---

## 1. Frozen identity

Every item below is independently recomputed, not copied.

| | |
|---|---|
| subject commit | `d8aac4d49e6ba997e3eb38062c0917186ee3f197` |
| subject tree | `3abc9e9d8ca11966a6f996d5f0af68072ee5b117` |
| population | **272** tracked `.md` documents |
| Census dependency | frozen Census v1.1, aggregate `eb7aad7c1a565cb25fcf6a7e250133e95d210f3e8ceb8765489046e3d945fa0e` |
| failed candidate | HOUSE_H2 v1.1 `be37a0aa5d56255a151c31361d93e8b4be94ab912ec9441c8ac3535a84fbf133` |
| contract-freeze commit | D366 `86a1399e6e31477ba67cd38c12d22627a8b4d6ef` |

**History-source requirement.** Non-shallow, containing the subject,
ancestry **986** at the subject, oldest reachable
`e8c3209c0131e1401f6002ab984c0728a40424e4` dated `2025-06-18`, newest
`2026-08-07`.

The active repository is **shallow** (ancestry 280, oldest 2026-08-05)
and **must not be used as a history source**. It does not fail on these
queries — it returns its graft boundary as a plausible date
(`TECH_WATCH.md`: `2026-08-05` shallow, `2026-07-24` true). Every
history-consuming instrument **must abort** on a shallow source.

Census v1.1 and HOUSE_H2 v1.0 are **untouched and remain so.** The
failed candidate is preserved as evidence, not discarded.

---

## 2. Demonstrated defect-class register — the complete pre-freeze set

Admission criterion for latent defects, per Kai's Q2 ruling:

> Every **demonstrated** false-output defect in the current H2 execution
> path known before this contract freezes must be closed **and calibrated
> fail-old / pass-new** in the repaired candidate.
>
> *Demonstrated* means a concrete input causes the current code to emit
> the wrong evidence fact or verdict. A merely imaginable defect with no
> demonstrated failing input is **not** automatically an admission
> blocker.

This register is **closed at freeze.** Nothing may be added to the
admission bar after implementation begins without a new decision entry.

| # | defect | axis | demonstrated by | banked |
|---|---|---|---|---|
| **D1** | evidence token → whole-file verdict | `VALIDITY` | `SEQUENCE.md` cites `97a3a61` for one J-series statement → `EXACT_SNAPSHOT` | D364 |
| **D2** | witness kind assumed from shape, never verified | `VALIDITY` | 7 of 26 witnesses do not resolve as commits: `ed25519` ×2, `1700000000`, 3 run ids, `b5e68a3` from `sha256:b5e68a3…` | D364 |
| **D3** | self-claim never checked against available evidence | `VALIDITY` | 9 of 23 dated documents changed **after** the date they claim (`TECH_WATCH` +94d) | D364 |
| **D4** | detector cannot fire; empty population reported as measured | `VALIDITY` | `RUN` class `[ :#]*` excludes `*` and backtick, so `**Last run:** 31570714150` misses and `SHA` takes it | D364 |
| **D5** | raw flag earns a verdict the qualified fact refuses | `VALIDITY` | `CURRENT_TREE` 7/7, `SELF_ASSERTS_CURRENT` 0/7 | D364 |
| **D6** | default emitted as verdict | `SCOPE` | no `def scope()`; `WHOLE_FILE` the only reachable value across 153 mutations; 267 rows witness `default` | D365 |
| **D7** | internal scope contradiction | `SCOPE` | 5 rows witness `REGION DETERMINATION REQUIRED` and emit `WHOLE_FILE` | D365 |
| **D8** | term matched inside a different word | `FUNCTION` | `Audit`⊂`Auditor`, `Plan`⊂`Plane`, `Plan`⊂`Planning`; sibling control: `doctor/oracle/sage/scout` all `UNKNOWN` | D365 |
| **D9** | closed-vocabulary negative over a degenerate capture | `FUNCTION` | `OTHER` ×3 tested against `'This file'` / `'records'`; one from prose about a **source file**, one at a **line-wrap boundary** | D365 |
| **D10** | never-executed branch | `FUNCTION` | `purpose_statement` fires **0 times in 272** | D365 |
| **D11** | candidate order decides the verdict | `FUNCTION` | 4 rows where two candidates' terms both match; `PATH_NOMINATION` order picks | D365 |
| **D12** | subject misbinding — bare pronoun bound to SELF | `AUTHORITY` | `README.md`: `It` refers to `UH_PROGRESS_TRACKER.md`, named in the previous sentence | D366 |
| **D13** | polarity inversion | `AUTHORITY` | `'This document is non-authoritative.'` → `SELF_ASSERTS_AUTHORITY` (hyphen is a word boundary; `AUTH_NEG` needs the word `not`) | D366 |
| **D14** | undeclared evidence truncation | `AUTHORITY` | `[:6]` cap; `DECISIONS.md` 43 claims, 1 SELF-bound, **0 SELF rows in the artefact** | D366 |
| **D15** | artefact cannot adjudicate its own cell | `VALIDITY`, `FUNCTION`, `AUTHORITY` | static `"cites a commit sha"`; aggregate `"144 members; …"`; silent `[:6]` | D364–D366 |
| **D16** | ontology omits a governing value | `SCOPE` | `UNKNOWN` absent from the only alphabet of six; `emittable('SCOPE','UNKNOWN')=False`; injection → `CONTRACT VIOLATION … disposition None` | D366 |
| **D17** | meta-check cannot detect a missing alphabet value | qualification | removing `UNKNOWN` from `VALIDITY`'s alphabet leaves qualification at **0 findings** while **216 documents emit it** | D367 §2 |

**D17 is demonstrated here and is the reason the other sixteen could
coexist with a green gate.** `qualify_h2.qualify()` iterates
`ont.ALPHABETS`, so a value *absent* from the alphabet lies outside its
denominator entirely. R5: the check's scope was defined by a list rather
than by the tree.

Each of D1–D17 requires a **fail-old / pass-new** control: a fixture
that **fails against the pre-repair implementation** and passes against
the repaired one. A repair proved only on the corrected case can
silently destroy the property it was protecting, so each pair must also
prove the opposite side.

---

## 3. Ontology corrections authorised for implementation

**Corrective amendment, not new semantics.** The governing text
(`kai-pm/house_in_order_instrument/AUTHORITY_ONTOLOGY.md:44`) already
states `UNKNOWN` is *"First-class on EVERY axis, independently."* The
executable ontology contradicts it. The change brings the machine into
conformance with an invariant already ratified.

1. **`SCOPE` gains `UNKNOWN`**, with an explicit disposition, emittable
   as an abstention — matching the other five axes.
2. `SCOPE=WHOLE_FILE` ceases to be a **default**. It becomes an earned
   determination or it is not emitted.
3. No region-scoped `VALIDITY` state is invented. Standing ruling
   preserved: H2 does not add a state to retain positive counts.

Authority: Dainius approves the amendment · Kai defines the exact
contract · Orion implements after explicit authorisation. **No
standalone `SCOPE` candidate or release.**

---

## 4. Scope and applicability semantics

**Row-level `SCOPE` is not the automatic scope of every cell.** One
document may carry whole-file `FUNCTION` evidence, one-citation
`VALIDITY` evidence, one-declaration `AUTHORITY` evidence and
managed-region `GENERATION` evidence simultaneously.

* a document-level `SCOPE` positive must itself be **earned**; if
  unearned, `SCOPE=UNKNOWN`;
* `SCOPE` **must not widen any other axis**;
* every evidence fact, claim and verdict carries **its own applicability
  scope**;
* region/citation evidence may be recorded **with its own selector**
  while the whole-file verdict for that axis abstains.

---

## 5. Evidence / witness trace schema

Every **positive evidence fact** and every **non-abstention verdict**
must carry a source-bound witness sufficient for independent
adjudication:

| field | meaning |
|---|---|
| `witness_type` | what kind of evidence this is |
| `witness_value` | the **exact** token or value matched — never a description |
| `source_path` | the document it came from |
| `source_selector` | line/span, or an equivalent **stable** selector |
| `local_context` | surrounding text sufficient to judge the match |
| `applicability_scope` | what the witness binds — whole document, or a region with its selector |
| `evidence_total` | how many candidate rows existed |
| `evidence_shown` | how many are carried here |
| `truncated` | explicit `true`/`false` |

Oversized evidence may live in a **bound sidecar** referenced by hash.
**Silent truncation is forbidden.** The evidence actually responsible for
the emitted cell must always be recoverable from the candidate package
**without guessing which source fragment mattered.**

This is **step 1** of the implementation order and closes D14 and D15.

---

## 6. Per-axis repair semantics

### `VALIDITY` (closes D1–D5)

Observations are demoted to evidence facts that earn nothing:
`CITES_COMMIT` · `CITES_RUN` · `CARRIES_DATE_STAMP` ·
`SELF_ASSERTS_CURRENT` · `BINDING_CONTRADICTION`.

A positive whole-file verdict requires **all** of:

* explicit **document-level binding** whose subject is the document as a
  whole — a requirement that is **semantic, not a Markdown-header layout
  rule**, so that a phrase-list defect is not replaced by a formatting
  defect;
* **verified witness kind** — a commit claim requires the token to
  **resolve as a commit in the declared history source**; a token
  preceded by `sha256:` is a digest; an 8+ digit decimal is not hex by
  accident. Discrimination is **by kind, not by rule precedence** —
  fixing the `RUN` regex alone would be the instance, not the class;
* **applicability scope = whole document**;
* **no unresolved material contradiction** — `last` is already in the
  row; a currency self-claim the history contradicts stays an assertion
  plus a `BINDING_CONTRADICTION`, and the verdict abstains.

Otherwise `UNKNOWN`.

### `FUNCTION` (closes D8–D11)

* term matching must respect **word boundaries**, distinguishing a
  plural of the term from a different word;
* the `OTHER` negative must be computed over a **substantive purpose
  capture**, never a trigger phrase;
* a branch that cannot fire is a defect, not a spare capability;
* **candidate order must not decide a verdict.** Where two nominations
  are corroborated, that is ambiguity to be reported, not resolved by
  list position.

**Path + title are not independent corroboration.** They are created as
part of the same document by the same author. `PATH says audit + TITLE
says audit` is not two proofs of function. They may support an evidence
fact such as `NOMINAL_FUNCTION` / `SELF_ASSERTS_FUNCTION`, which is
**not** actual function.

**Static Census reader evidence does not prove `FUNCTION=RUNTIME_INPUT`.**
It establishes static/proven reader evidence only. A runtime-input
function claim requires evidence that the **live contract actually
consumes the document in that role**, excluding: test readers, dead
paths, tooling/migration readers, generic scanners, conditionally
inactive readers, and unrelated reads. **Current reader lists must not be
used to rescue `FUNCTION` positives.**

The 5 objective `MARKER` cases remain **candidate-proven, subject to
final qualification**.

### `AUTHORITY` extractor (closes D12–D13)

* self-binding requires a resolvable subject. A **bare pronoun with no
  antecedent resolution must not bind to SELF**;
* polarity must be evaluated on meaning, not on a word boundary that
  happens to fall inside a negation.

Minimum hostile boundary set: `authoritative` · `non-authoritative` ·
`not authoritative` · `not a source of truth` · an authoritative
statement **about another document** · a **quoted** authority statement ·
a controlled-field self-declaration.

### `GENERATION` and `AUTHORITY` verdict invariants — retained

Both remain abstention-only at H2. Proven by a **known-positive**:
injecting a forbidden value through the real code path is rejected by
the contract self-check **while the clean row still passes**. Absence of
emission alone never establishes non-emittability (D340 §7 / D358).

---

## 7. Ontology / meta-check invariants (closes D16–D17)

Mechanically encoded, derived from the axis list, never hand-maintained:

```
for EVERY declared axis:
    UNKNOWN ∈ alphabet
    UNKNOWN has an explicit disposition
    UNKNOWN is emittable as an abstention
```

**Calibration, required:** synthetically remove `UNKNOWN` from one axis →
**qualification MUST fail.** That is fail-old / pass-new for the ontology
defect itself, and it closes the exact class that let `SCOPE`'s governing
abstention requirement disappear while the gate stayed green.

The qualification denominator must be derived such that a value **absent
from an alphabet but present in output** is detectable. A value emitted
by any row and unknown to the ontology is a finding.

---

## 8. Qualification criteria

1. every declared `H2_EMITTABLE` value **reachable**, established by
   runtime observation — never by grepping for the literal;
2. every `H2_NOT_EARNABLE` / `DEFERRED_TO_H3` value emitted **zero**
   times, with the injection known-positive proving the guard fires;
3. population reconciles, denominator printed;
4. all **17** demonstrated defect classes closed, each with its
   fail-old / pass-new pair, the fail-old half run against the **actual
   committed** pre-repair implementation;
5. ontology invariants of §7 pass, with the removal calibration proving
   they can fail;
6. **runtime module identity** — every loaded module's `__file__`
   resolves under the candidate directory and its **source bytes hash to
   the manifest entry**;
7. **fresh reproduction** from an unrelated directory: all rows and the
   admission contract byte-identical;
8. every emitted positive carries the §5 witness trace.

---

## 9. Final blind holdout — selection and evaluation, both frozen here

**Size: 40 documents**, all six axes plus consequential evidence facts.
An independent surprise detector on top of the mechanical fixtures — not
a claimed statistical guarantee.

**Selection rule**, fixed now, resolvable only after implementation
completes:

```
key = sha256(
    "H2FINAL-D367:"
    + "86a1399e6e31477ba67cd38c12d22627a8b4d6ef"
    + ":" + FINAL_CANDIDATE_AGGREGATE
    + ":" + path
)
sort ascending, select the first 40
```

`FINAL_CANDIDATE_AGGREGATE` does not exist until the repair is complete,
so **the sample cannot be known during implementation** while the rule
itself is frozen before any code. If a candidate fails and code changes,
the new candidate identity deterministically yields a **new** sample;
previously revealed rows become regression evidence only.

**Evaluation rule**, frozen here so success cannot be reinterpreted
afterwards:

| finding | disposition |
|---|---|
| incorrect non-abstention verdict | **BLOCKER** |
| false evidence fact | **BLOCKER** |
| unsupported scope widening | **BLOCKER** |
| forbidden or undeclared state emitted | **BLOCKER** |
| determining witness absent or silently truncated | **BLOCKER** |
| genuinely ambiguous source evidence | `UNRESOLVED` — never forced into agreement |
| `UNKNOWN` | abstention — **never** negative evidence |
| independently adjudicated earnable positive emitted as `UNKNOWN` | `OVER_ABSTENTION` / coverage finding — not automatically a safety blocker unless systematic or contrary to a separately precommitted coverage requirement |

**Kai independently adjudicates all 40 documents across all six axes.
Orion computes no acceptance agreement figure.**

The D363 24-row holdout is partially revealed **regression evidence
only** and is not to be inspected further. No separate per-axis holdouts.

---

## 10. Independence

> **ORION SELF-AUDIT RESULTS HAVE ZERO FINAL ADMISSION WEIGHT.**

They are defect-discovery evidence, repair-design evidence and
calibration input. They are **not** independent qualification.

The record supporting this: `TIME_BOUND` was initially graded 83% sound
and was not; the 144 `CODE_AUDIT_BATCH_` rows were initially called
`PROVEN` and were not; `AUTHORITY` was reported sound and carried a false
evidence fact; and the `AUTHORITY`/`GENERATION` deferral was claimed
*proven by mutation* when the sweep had merely failed to reach the
state. **Every self-assessment error ran in the same direction.**

Final admission requires: mechanical package qualification · known-defect
fail-old/pass-new controls · fresh reproduction · exact evidence
traceability · **Kai's independent six-axis holdout adjudication.**

---

## 11. Utility metrics — reported, never optimised

Positives must **not** be preserved to keep classification rates
attractive. After repair, report per axis: positive verdict coverage ·
evidence-fact coverage · `UNKNOWN` rate · `UNMEASURED` rate · deferred
rate.

Two separate gates:

* **qualification** asks *is the instrument truthful?*
* a later **operator decision** asks *is the truthful instrument
  sufficiently discriminating to feed HOUSE_H3?*

A technically truthful candidate may pass qualification and still be
judged operationally unfit for progression. If H2 proves to be
principally an evidence-and-abstention instrument, **report that** rather
than manufacturing classifications.

---

## 12. Implementation order

1. evidence / witness trace completeness
2. `SCOPE` ontology and foundation
3. `VALIDITY` evidence-vs-verdict, under corrected scope
4. `FUNCTION` evidence-vs-verdict and witness
5. `AUTHORITY` claim-extractor defects
6. retain `GENERATION` / `AUTHORITY` abstention controls
7. recompute all six axes
8. one full qualification
9. one fresh independent six-axis blind holdout

**One consolidated candidate. No standalone axis release. No
implementation authority exists under this contract.**

---

## 13. What happens next

After this contract is banked: **RETURN.** Implementation begins only
when Kai has independently verified this contract **and** Dainius has
separately authorised the consolidated repair.
