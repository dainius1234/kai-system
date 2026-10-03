# CROSS-AXIS SEMANTIC AUDIT — READ-ONLY EVIDENCE

Banked under **D365, evidence only**. No repair authority, no ontology
implementation authority, no new candidate, no holdout, no HOUSE_H3.

| | |
|---|---|
| candidate audited | `be37a0aa5d56255a151c31361d93e8b4be94ab912ec9441c8ac3535a84fbf133` |
| subject | `d8aac4d49e6ba997e3eb38062c0917186ee3f197` |
| subject tree | `3abc9e9d8ca11966a6f996d5f0af68072ee5b117` |
| population | 272 |
| axes in scope | FUNCTION, AUTHORITY, GENERATION, SCOPE |
| audited elsewhere | LIFECYCLE D361–D363 · VALIDITY D364 |

> **Adjudication classes are an AUTHOR NOMINATION carrying no admission
> weight.** They are derived by declared rule from mechanical signals, so
> they are reproducible — but the rules are mine, on an instrument I
> wrote. Kai adjudicates independently after the final candidate returns.

## Result

| axis | N | PROVEN | UNSUPPORTED | FALSE_POS | AMBIG | ABSTENTION | DEFERRED |
|---|---|---|---|---|---|---|---|
| `FUNCTION` | 272 | 149 | 63 | 6 | 4 | 50 | 0 |
| `AUTHORITY` | 272 | 0 | 0 | 0 | 0 | 0 | 272 |
| `GENERATION` | 272 | 0 | 0 | 0 | 0 | 0 | 272 |
| `SCOPE` | 272 | 0 | 272 | 0 | 0 | 0 | 0 |
| **TOTAL** | **1088** | **149** | **335** | **6** | **4** | **50** | **544** |

Of 494 cells that are neither abstention nor deferred: **149 PROVEN (30%) · 335 UNSUPPORTED_POSITIVE (68%) · 6 FALSE_POSITIVE · 4 AMBIGUOUS.**

## A correction to the first pass of this audit

The first version of this instrument reported `FUNCTION` as
`DEFERRED_BY_DESIGN`. That is false — `FUNCTION` emits 222 verdicts on
the corpus. The cause: the mutation sweep used a neutral row that never
reaches a `FUNCTION` verdict, and **"the sweep found nothing" was read
as "nothing is reachable"**.

That is `UNKNOWN` used as negative evidence — the abstention invariant
(D340 §7 / D358) — broken inside the instrument built to detect exactly
that. It also means the claim made to Kai that `AUTHORITY` and
`GENERATION` deferral was *proven by mutation* **was not proven**: the
same sweep produced the same silence for an axis that is not deferred at
all.

Deferral is now decided by three legs, and only the third is proof:

1. every declared non-abstention value is `H2_NOT_EARNABLE` or
   `DEFERRED_TO_H3` — a *declaration*;
2. the corpus emits none of them — an *observation* that cannot
   distinguish forbidden from not-encountered;
3. **injecting a forbidden value through the real code path is rejected
   by the contract self-check, while the clean row still passes** — a
   known-positive with its known-negative.

Leg 3 was run. `AUTHORITY=ADVISORY` and `GENERATION=FULL_DERIVED`
injected through `_unknown()` are both rejected; the clean row is still
accepted. `_unknown` is poisoned rather than `classify()` — patching
`classify()` would bypass the guard and prove nothing, and patching
`ont.emittable` would be patching the check instead of testing it.

## SCOPE — a default presented as a finding, 272 of 272

There is **no `def scope()`**. The value is a literal, and `WHOLE_FILE`
is the only value reachable by 153 mutations *or* observed on the corpus.
267 rows carry the witness `default`.

**Five carry `default_pending_region`, witness
`"proven writer present: REGION DETERMINATION REQUIRED"` — and emit
`WHOLE_FILE` anyway.** The instrument states the determination is
required and returns the verdict without it. Those five are listed in
the JSON.

This is load-bearing: every `VALIDITY` verdict is scoped `WHOLE_FILE` by
this default, and that scope is what made the D364 defect harmful.

## AUTHORITY — the axis that gets it right, and the model for the repair

Self-claims are computed and recorded as a **row-level evidence field**
— `SELF_ASSERTS_AUTHORITY` 5, `SELF_ASSERTS_NON_AUTHORITY` 1,
`NO_SELF_CLAIM` 266 — while the verdict abstains 272/272. Evidence
recorded, verdict withheld.

That is precisely what `VALIDITY` failed to do with `present_tense`.
**The correct pattern already exists inside the package**, implemented
on one axis and not on two others, in the same file, by the same author.

Residual risk unchanged: the claim sits in the artefact, so a careless
H3 consumer could read it as authority. That is the D358 consumer gate —
**specified but unbuilt**.

## FUNCTION — the most disciplined axis, and one source counted twice

`FUNCTION` requires **path nomination + a corroborating title witness**,
genuinely stronger than any `VALIDITY` rule. But path and title are both
authored by the same person at the same time, so by the rule this
programme already banked — *repetition is not corroboration; common
provenance across copies adds no authority* — **the two-source design is
one source counted twice**. Hence 63 `UNSUPPORTED_POSITIVE`: the
verdicts are mostly right, but what is proven is what the document
*says* its role is.

**PROVEN (149).** 144 from the `CODE_AUDIT_BATCH_*` family rule — an
`all()` over the family derived from the tree, so if it fires every
member individually carries the title evidence, and no member carries a
non-evidence title. Plus 5 `MARKER` on byte count and path suffix.
*Fragility noted: if one member failed, all 144 fall through silently.*

**FALSE_POSITIVE (6)** — two mechanisms:

| document | emitted | mechanism |
|---|---|---|
| `data/teammates/auditor.md` | `EVIDENCE` | matched term lies inside a different word, not a plural of it |
| `docs/known_issues.md` | `OTHER` | closed-vocabulary negative computed over a trigger-only capture: satisfied by construction, not by the document |
| `kai-pm/CODE_AUDIT_PLANNING_PACKAGE_QA.md` | `PLAN` | matched term lies inside a different word, not a plural of it |
| `kai-pm/EVIDENCE_PLANE_RESEARCH_LINEAGE.md` | `PLAN` | matched term lies inside a different word, not a plural of it |
| `kai-pm/SERVICE_IDENTITY_STATE.md` | `OTHER` | closed-vocabulary negative computed over a trigger-only capture: satisfied by construction, not by the document |
| `kai-pm/WAYPOINTS.md` | `OTHER` | closed-vocabulary negative computed over a trigger-only capture: satisfied by construction, not by the document |

Witness detail: `Audit` ⊂ **`Auditor`**, `Plan` ⊂ **`Plane`**,
`Plan` ⊂ **`Planning`** — none a plural of the term. (`Decision` ⊂
`Decisions` and `Operating rule` ⊂ `Operating rules` **are** plurals and
land correctly; the rule distinguishes them.)

`data/teammates/auditor.md` has a control that did not need constructing:
`doctor.md`, `oracle.md`, `sage.md` and `scout.md` are the same kind of
document in the same directory and are **all `UNKNOWN`**. The only
difference is that one role is *named* auditor.

The three `OTHER` verdicts fail differently: `OTHER` means *a stated
purpose matching no declared function term* — a closed-vocabulary
negative — but `PURPOSE` captures only the **trigger phrase**, so the
negative is tested against `'This file'` and `'records'`, strings that
could never contain a function term. One capture is prose about a
**source file** under `## Common Mistakes`; another matched at a
**line-wrap boundary**. 18 of 28 `PURPOSE` matches use trigger-only
alternatives, and the `purpose_statement` corroboration branch fires
**0 times in 272 documents** — never-executed code (R8), the D341 class.

**AMBIGUOUS (4).** 26 paths nominate more than one function; in these
the terms of two candidates both match, so `PATH_NOMINATION` **order**
picks the winner. A different ordering emits a different verdict on
identical evidence — the `RUN` vs `SHA` precedence shape.

## Divergence from the hand adjudication, stated

The hand pass reported 5 FALSE_POSITIVE / 5 AMBIGUOUS. The derived rule
reports **6 / 4**: `CODE_AUDIT_PLANNING_PACKAGE_QA.md` moves from
AMBIGUOUS to FALSE_POSITIVE, because `Plan` inside `Planning` is a
word-boundary violation with a non-plural extension. **The derived
result is the one to keep** — it is reproducible; the hand call was not.

## The consolidated defect population

| defect | axis | population |
|---|---|---|
| default emitted as verdict | `SCOPE` | **272** |
| self-description counted twice | `FUNCTION` | **63** |
| evidence token → whole-file verdict | `VALIDITY` (D364) | **50** |
| self-claim never checked against available evidence | `VALIDITY` | 9 |
| witness kind assumed from shape, never verified | `VALIDITY` | 7 |
| candidate order decides the verdict | `FUNCTION`, `VALIDITY` | 4 + `RUN`/`SHA` |
| substring match, no word boundary | `FUNCTION` | 3 |
| closed-vocabulary negative over a degenerate capture | `FUNCTION` | 3 |
| population empty because the detector cannot fire | `VALIDITY` | 0 measured |
| never-executed branch | `FUNCTION` | 0 fired |

**One shape underneath all of them:** an observation is recorded at one
scope and reported at a wider one. `SCOPE` is the purest case — the
widening *is* the entire rule.

## Reproduction

```sh
python3 kai-pm/cross_axis_semantic_audit.py \
    --subject-repo <exact checkout at the subject> \
    --subject d8aac4d49e6ba997e3eb38062c0917186ee3f197 \
    --package kai-pm/house_in_order_h2_v11 --out audit.json
```

Output is byte-identical across runs. The instrument aborts on a mutable
ref, a mismatched subject repo, or a module loaded from outside the
named package. It is **not** part of the HOUSE_H2 package and must never
be imported by a classifier.

## What this audit does NOT establish

* The 335 `UNSUPPORTED_POSITIVE` cells are **not shown to be wrong**.
  Most are probably right. They are not proven at the scope claimed.
* No blind holdout was used. Every finding is the author's, on an
  instrument the author wrote — and a self-audit graded one `VALIDITY`
  population 83% sound before an independent held-out sample corrected
  it. These counts should carry no admission weight for that reason.
* `LIFECYCLE` and `VALIDITY` were not re-adjudicated here.
