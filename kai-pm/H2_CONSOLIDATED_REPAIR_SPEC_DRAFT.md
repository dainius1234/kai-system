# HOUSE_H2 — CONSOLIDATED REPAIR SPECIFICATION / DESIGN OPTIONS

**STATUS: DRAFT FOR REVIEW. NOT AUTHORISED. NO IMPLEMENTATION.**

Produced under the Kai ruling *"DISCOVERY CLOSED. PROCEED TO
CONSOLIDATED REPAIR-SPEC / DESIGN REVIEW. NO CODE."*

This document proposes mechanisms. It decides nothing. Every
`RECOMMENDED` line below is a proposal to Kai and Dainius, not a
selection. No repair, fixture mutation, candidate generation or
repository change to the H2 package has been made or is authorised.

| | |
|---|---|
| repository state at authorship | `f35f890`, tree `2c791833`, worktree clean |
| candidate under repair | HOUSE_H2 v1.2, aggregate `ba2b16d4…de4a` |
| subject | `d8aac4d49e6ba997e3eb38062c0917186ee3f197`, tree `3abc9e9d…b117`, 272 documents |
| governing contract | `kai-pm/H2_REPAIR_CONTRACT_D367.md`, sha256 `0ce5792e…00bb`, 389 lines |
| working obligations | 12 |
| new D-numbers created | 0 |
| new admission classes | 0 |

---

## 0. Units — declared once, binding on every count in this document

Kai §1: *"state the unit for every inventory count."* Counts below are
never interchangeable and are labelled at every use.

| unit | meaning | corpus total |
|---|---|---|
| `DOCUMENT` | a subject document | 272 |
| `AXIS CELL` | one of six axes on one document | 1632 |
| `EMITTED WITNESS RECORD` | one witness object as emitted | 491 |
| `UNIQUE §5 IDENTITY` | distinct nine-field witness identity | 480 |
| `SOURCE OCCURRENCE` | a physical token occurrence in source | 1967 distinct |
| `§5 SUBJECT` | a thing owing a §5 trace = verdict + positive fact | 659 (343 + 316) |
| `POSITIVE EVIDENCE FACT` | a `true` evidence-fact boolean | 316 |

Where a population below was measured in one unit and not another, that
is stated. Where it was not measured at all, it says `NOT MEASURED`.

---

## 1. The twelve obligations, separated by type

Kai §8: the twelve are **not at identical implementation stages** and
must not be presented as if they were.

### GROUP A — LOGIC / SEMANTIC REPAIR (9)

`D2` · `M1` · `M2` · `M3` · `D14` · `E1` · `E2` · `D12-residual` ·
`D13-residual`

These change what the analyser *concludes* from a source document. Each
has a demonstrated wrong output or a demonstrated contract violation, a
known invariant, and an open mechanism.

### GROUP B — QUALIFICATION / PACKAGE INTEGRITY (3)

`Q1b` · `Q1a` · `I1`

These change what the package can *prove about itself*. They do not
alter a single verdict. They determine whether any verdict can be
trusted to have come from the code we tested.

**The distinction matters for sequencing.** Group B failures are why a
Group A repair cannot be self-certified: today the qualification step
would pass a package whose §5 subject population was never validated
(`Q1b`) and whose blind precommit identity excludes the evidence (`I1`).
Repairing Group A first and Group B second would produce a candidate we
could not honestly qualify.

---

# GROUP A — LOGIC / SEMANTIC REPAIR

## A1 · `D2` — token kind decided by character class, not by resolution

**DEMONSTRATED FAILURE.** `passa.py:163 classify_token_kind` carries the
docstring *"D2: DISCRIMINATE the kind. Never assume it from the
character class."* The function then does exactly that at the residual
branch: `if tok.isdigit(): return "DECIMAL_TOKEN", False`. An all-digit
token is refused resolution before any resolution is attempted.

**CURRENT AFFECTED POPULATION.** 1 `SOURCE OCCURRENCE`. Token `9904486`
resolves against the history repository to
`9904486d1cf133556b5c5496a6317caf59f0a3c4`,
*"feat(UH-5): Policy, human approval and capability bridge"*. It is
emitted as `DECIMAL_TOKEN` and is never resolved.

**GOVERNING INVARIANT.** Kind is established by what the token *is*,
proved by resolution, never by which characters it happens to contain.

**DESIGN OPTIONS.**

* **A1-i — resolve-before-classify.** Move the resolution probe ahead of
  the `isdigit` branch. Any token of length ≥7 that is hex-valid (all
  decimal digits are hex-valid) is offered to `git cat-file -e`. Kind
  follows the answer.
* **A1-ii — dual-candidate emission.** Emit both candidate kinds with
  the resolution result as the discriminating witness.
* **A1-iii — narrow probe.** Keep the `isdigit` branch but probe first
  when the token is not preceded by a `RUN_NEAR` / `RUN_URL` context.

**RECOMMENDED: A1-i.** It is the mechanism the docstring already
declares, and it removes the class rather than the instance (R6).

**FAIL-OLD CONTROL.** `9904486` at its source occurrence classifies
`DECIMAL_TOKEN`, `resolved=False`.
**PASS-NEW CONTROL.** Same token classifies as a resolved commit with
the full 40-character SHA as `witness_value`.
**KNOWN-NEGATIVE (mandatory).** A genuine decimal that is *not* a commit
— a run id, a byte count, a large ordinary number — must remain
`DECIMAL_TOKEN` and must not be promoted by an accidental hex collision.
Without this control A1-i trades one wrong output for a class of them.

**INTERACTION RISKS.** Adds git probes to a previously short-circuited
path; cost is bounded by the decimal-token population and must be
measured, not assumed. Interacts with `D14`: if the extraction window
changes, the token population entering this branch changes.

**QUALIFICATION PROOF REQUIRED.** The emitted record for the repaired
token must carry a complete §5 trace including the resolved SHA.

---

## A2 · `M1` — a review event promoted to a temporal validity binding

**DEMONSTRATED FAILURE.** A date appearing beside a review or inspection
predicate (`Reviewed:`, `Last reviewed`) is promoted to
`VALIDITY = TIME_BOUND`. Kai's formulation, which is the precise one:
*a review event is not a validity/currency binding.* Someone having
looked at a document on a date does not bind the document's validity to
that date.

**CURRENT AFFECTED POPULATION.** 145 of 272 `AXIS CELL` (VALIDITY),
corpus-wide, measured. 19 of these are `AXIS CELL`s inside the frozen
40-row holdout, adjudicated as blockers.
**PROTECTED POSITIVES: 10.** Ten further `TIME_BOUND` rows were
adjudicated by Kai as genuine document-state / version / report
bindings. They are protected **on the merits of their source contexts**,
not because of their labels. Measured and confirmed: none of the ten
carries a binding contradiction.

**GOVERNING INVARIANT.** `TIME_BOUND` requires a predicate that binds
the document's *state* to the date. An inspection event does not.

**DESIGN OPTIONS.**

* **A2-i — binding-predicate discriminator.** Classify the predicate
  governing the date into `STATE_BINDING` (updated / valid until /
  version / effective) vs `INSPECTION_EVENT` (reviewed / checked /
  audited / inspected). `TIME_BOUND` requires the former.
* **A2-ii — abstain unless positively bound.** Emit `UNKNOWN` for any
  date whose binding predicate is not affirmatively a state binding.
* **A2-iii — new non-promoting observation.** Retain the review date as
  an emitted observation with its own value, promoting nothing.

**RECOMMENDED: A2-i combined with A2-iii.** A2-ii alone is the trade Kai
explicitly rejected — it would convert 145 false positives into an
unknown number of false negatives, and would very likely take the ten
protected positives with it. A2-iii keeps the evidence visible without
letting it decide.

**FAIL-OLD CONTROL.** The 19 holdout `AXIS CELL`s, plus a sample from
the 145, classify `TIME_BOUND`.
**PASS-NEW CONTROL.** All ten adjudicated positives still classify
`TIME_BOUND`. **This is the load-bearing control**: it is the one that
proves the repair discriminated rather than merely suppressed.

**INTERACTION RISKS.** **`M1` must be repaired after `M3`.** The
determining VALIDITY witness is selected using `applicability_scope`;
`M3` changes `applicability_scope` on 48 `EMITTED WITNESS RECORD`s.
Repairing `M1` first would tune a discriminator against a witness
selection that is about to change. D367 §12 already orders these
correctly — scope at step 2, validity at step 3 — and that ordering is
now load-bearing rather than stylistic.

**QUALIFICATION PROOF REQUIRED.** Every surviving `TIME_BOUND` cell
carries a §5 trace whose `local_context` contains the state-binding
predicate that earned it.

---

## A3 · `M2` — the audited tree's lifecycle substituted for the document's

**DEMONSTRATED FAILURE.** An audit document that records the commit it
audited has that commit routed into its own `LIFECYCLE`, producing
`HISTORICAL` for the *document* on evidence about a *different subject*.
`passa.py:92` states the predicate's rationale as *"the document states
the snapshot it audits"* — the code then uses that snapshot as the
document's own lifecycle evidence. The invariant is written in the
package and contradicted by it.

**CURRENT AFFECTED POPULATION.** 3 of 272 `AXIS CELL` (LIFECYCLE),
COMMIT-routed `HISTORICAL`. 1 of the 3 is in the frozen holdout
(`CODE_AUDIT_MASTER`).

**GOVERNING INVARIANT.** The subject of a `LIFECYCLE` verdict is the
document. Evidence about an artefact the document *describes* is
evidence about that artefact.

**DESIGN OPTIONS.**

* **A3-i — subject-binding gate on the lifecycle witness.** A COMMIT
  witness may determine `LIFECYCLE` only if its binding predicate makes
  the document itself the subject.
* **A3-ii — forbid COMMIT-routed `HISTORICAL` entirely.** Blunt; would
  remove any genuine case with it.
* **A3-iii — require an explicit lifecycle predicate** (superseded by /
  archived / retired), routing everything else to `UNKNOWN`.

**RECOMMENDED: A3-i.** It repairs the substitution rather than deleting
the route.

**FAIL-OLD CONTROL.** The 3 COMMIT-routed rows classify `HISTORICAL`.
**PASS-NEW CONTROL — MANDATORY, ALREADY DIRECTED BY KAI.** A *synthetic
same-route positive*: a document whose own lifecycle is genuinely
evidenced by a commit must still classify `HISTORICAL`. Without it, A3-i
is indistinguishable from A3-ii and the repair cannot be shown to have
discriminated. `REPO_HEALTH_AUDIT` stands as a route-independent
regression control.

**INTERACTION RISKS.** Shares the subject-binding machinery with `D12`.
A change to `bind_subject` semantics touches both.

**QUALIFICATION PROOF REQUIRED.** Each surviving COMMIT-routed
`HISTORICAL` cell carries a §5 trace showing the document as subject.

---

## A4 · `M3` — applicability scope under-assigned

**DEMONSTRATED FAILURE.** `applicability_scope` is emitted without being
derived from what the witness actually binds. A witness in a
document-level header field binds the whole document; a witness in body
prose binds its span. The emitted value does not reliably distinguish
them.

**CURRENT AFFECTED POPULATION.** Measured on the closed M3 truth table:
491 `EMITTED WITNESS RECORD`; expected 214 `WHOLE_FILE` and 277 `SPAN`;
**48 wrong** (47 under-assigned, 1 over-assigned); 443 unchanged.
Dual-unit reconciliation: 480 `UNIQUE §5 IDENTITY`, 47 wrong. Locator
rows affected: 44 of 272 `DOCUMENT`.

**GOVERNING INVARIANT.** `applicability_scope` states what the witness
binds and must be derived from the witness's structural position, not
defaulted.

**DESIGN OPTIONS.**

* **A4-i — structural derivation.** Front-matter / controlled-field /
  document-header position → `WHOLE_FILE`; in-body occurrence → `SPAN`
  with its selector.
* **A4-ii — explicit predicate table** mapping binding predicates to
  scope.
* **A4-iii — abstain when undeterminable**, emitting neither.

**RECOMMENDED: A4-i with A4-iii as the residual.** A4-ii alone is a
hand-maintained list beside the thing it governs — the exact shape R5
forbids; the table must be derived from the binding-predicate set the
package already computes, not written out again.

**FAIL-OLD CONTROL.** The 48 wrong `EMITTED WITNESS RECORD`s.
**PASS-NEW CONTROL.** The 443 unchanged records remain unchanged, and
the 1 over-assigned record is corrected downward — a repair that only
moves records upward has not been shown to discriminate.

**INTERACTION RISKS.** **`M3` is upstream of `M1` and of every
determining-witness selection.** It is D367 §12 step 2 and must land
before validity and function work. Emission conservation applies: the
repair-stable emission key `path+start+end+detector` is unique over all
491 records and `detector` is load-bearing; count changes must be
source-anchored controlled deltas, never a changed denominator.

**QUALIFICATION PROOF REQUIRED.** 491 records in, 491 keys out, with
every scope change individually attributable.

---

## A5 · `D14` — evidence bisected at the extraction boundary

**DEMONSTRATED FAILURE.** `passa.py:190 head = text[:HEAD_BYTES]` with
`HEAD_BYTES = 6000`, **a character index despite its name**. A token
straddling that index is cut, and a *prefix* is emitted as the
`witness_value` with `truncated=False`. The record asserts completeness
of something it truncated.

**CURRENT AFFECTED POPULATION.** 2 `SOURCE OCCURRENCE`, both `HEX`,
across all detector families, source-verified, geometry independently
reproducible without my instruments. The known instance:
`CODE_AUDIT_CONTINUATION_LOG.md`, token at character offsets 5971–6011,
emitted as a 29-character prefix.

**A NOTE ON THE DENOMINATOR.** An earlier detector reported this
population as 1 of 272. That detector required the *truncated* form to
still match the recogniser, so any token cut below the recogniser
minimum was outside its denominator by construction — R5. The population
above is measured over source occurrences, not over surviving matches.

**GOVERNING INVARIANT — D367 §5, verbatim.** *"Silent truncation is
forbidden. The evidence actually responsible for the emitted cell must
always be recoverable from the candidate package without guessing which
source fragment mattered."*

**DESIGN OPTIONS.**

* **A5-i — remove the extraction cap.** Scan the full document text for
  tokens. 272 documents is not a scale that needs a 6000-character
  heuristic; the cost must be measured before this is chosen.
* **A5-ii — boundary-aware extension.** Keep the cap, but extend the
  window to the end of any token straddling it, so no token is ever cut.
* **A5-iii — declare the truncation.** Keep the cap, emit
  `truncated=true` with the full token in a hash-bound sidecar.

**RECOMMENDED: A5-i, with A5-ii as the fallback if measured cost
forbids it.** A5-iii satisfies the letter of §5 while leaving the
analyser blind past character 6000, which is a reporting fix for an
extraction defect.

**FAIL-OLD CONTROL.** Both boundary-crossing tokens emit prefixes with
`truncated=False`.
**PASS-NEW CONTROL.** Both emit full 40-character values. Plus a
synthetic token placed deliberately astride the boundary.

**INTERACTION RISKS — THE LARGEST IN THIS DOCUMENT.** `HEAD_BYTES`
governs three separate consumers: token scanning (`passa.py:190`),
`says_supersedes` (`passa.py:277`, `F3`), and — at the same literal
value 6000 — `PURPOSE` matching (`classify.py:171`, `F2`). Removing or
widening the cap **widens what every one of them sees.**

> **Kai's F2 constraint binds here and must be carried into the design:
> widening visibility must not widen subject applicability.** The F2
> measurement already showed that widening `PURPOSE`'s window surfaced
> 9 additional matches of which the semantically correct disposition was
> that both apparent "changes" came from false `PURPOSE` matches on
> domain prose. A5-i without a subject-applicability guard would import
> that failure mode across the corpus.

**QUALIFICATION PROOF REQUIRED.** Every `EMITTED WITNESS RECORD` whose
value changed is attributable to a named boundary-crossing occurrence,
and the total record count is conserved or its delta source-anchored.

---

## A6 · `E1` — positive evidence facts without a compliant §5 trace

**DEMONSTRATED FAILURE.** Positive evidence facts are emitted as bare
booleans. §5 requires **every positive evidence fact** to carry a
source-bound witness sufficient for independent adjudication.

**CURRENT AFFECTED POPULATION.** 81 of 316 `POSITIVE EVIDENCE FACT`
subjects violate the frozen §5 trace contract.
**Correcting my own earlier transmission:** I once reported this as
"316 positive facts carry none." 235 of 316 carry the full nine-field
trace in the package-bound sidecar. The violating population is 81.

**GOVERNING INVARIANT — Kai's strengthened form.** *Every positive
evidence fact must carry a semantically truthful §5 trace sufficient for
independent adjudication.* "Semantically truthful" is the operative
word: a trace that is present but does not contain the evidence that
determined the fact does not satisfy this.

**A KNOWN SUB-POPULATION ALREADY IN HAND.** The `F5` measurement found
40 clipped `local_context` fields of which **10 do not contain their own
`witness_value`** — all `TECH_WATCH` — every one carrying
`truncated=False`. These 10 are concrete, named fail-old controls for
this obligation and are also carried into `Q1b`'s regression basis.

**DESIGN OPTIONS.**

* **A6-i — emit the determining witness with each positive fact.** The
  producer that sets the boolean also carries the witness that set it.
* **A6-ii — refuse to emit a positive fact with no compliant trace**,
  abstaining instead.
* **A6-iii — sidecar-only**, referencing by hash.

**RECOMMENDED: A6-i with A6-ii as the residual.** A6-iii alone repeats
the current shape, where the trace exists somewhere but is not bound to
the fact it explains.

**FAIL-OLD CONTROL.** The 81 violating subjects, including the 10
`TECH_WATCH` cases where the context does not contain its own value.
**PASS-NEW CONTROL.** The 235 already-compliant traces are unchanged.

**INTERACTION RISKS.** `E1` and `Q1b` are logically distinct and Kai has
already fixed the sequence: a repaired `Q1b` **makes the old `E1`
population visible and failing**; it does not repair it. `E1` also
depends on `D14`/`F5`: a trace cannot be semantically truthful if the
extraction that produced it clipped the determining evidence.

**QUALIFICATION PROOF REQUIRED.** All 316 positive facts pass a §5
compliance check that itself validates content, not merely presence.

---

## A7 · `E2` — a static textual reference reported as runtime consumption

**DEMONSTRATED FAILURE.** `run_h2_v12.py:84`
`f["CONSUMED_AT_SUBJECT"] = bool(row["readers"])`. `readers` is a static
textual reference set. The emitted fact asserts *consumption at the
subject* — a runtime property — on static evidence.

**CURRENT AFFECTED POPULATION.** 5 of 272 `DOCUMENT` carry
`CONSUMED_AT_SUBJECT = true`.

**GOVERNING INVARIANT.** A fact must name what it measured. Static
reference and runtime consumption are different properties.

**DESIGN OPTIONS.**

* **E2-i — rename with an explicit scope record.**
  `REFERENCED_AT_SUBJECT`, carrying an `ANALYSIS_SCOPE` field stating
  that the evidence is static. This follows the precedent Kai already
  accepted for `F4` (`NO_WRITER_WITHIN_ANALYZED_SCOPE`): renamed, not
  removed, with closure rules unrelaxed.
* **E2-ii — require runtime evidence** for the existing name, emitting
  `UNKNOWN` in its absence.
* **E2-iii — two facts**, static and runtime, independently traced.

**RECOMMENDED: E2-i.** It is the minimum change that makes the fact
true, and it preserves the 5 observations rather than discarding
evidence to fix a label.

**FAIL-OLD CONTROL.** The 5 rows emit `CONSUMED_AT_SUBJECT`.
**PASS-NEW CONTROL.** The same 5 rows emit `REFERENCED_AT_SUBJECT` with
a declared static `ANALYSIS_SCOPE`, and no row claims runtime
consumption anywhere in the package.

**INTERACTION RISKS.** Low. Renaming an evidence fact changes the §5
subject population's key set — `Q1b` must enumerate the new name.

**QUALIFICATION PROOF REQUIRED.** No emitted fact name asserts a
property stronger than the evidence class behind it. This is checkable
mechanically and should be.

---

## A8 · `D12-residual` — a bare pronoun with no antecedent binds to SELF

**DEMONSTRATED FAILURE, AT THREE LAYERS.** This is the only obligation
in this document that is contradicted at contract, code and fixture
simultaneously.

1. **Contract.** D367 §6 forbids it.
2. **Code.** `subjectbind.py:110` returns
   `"SELF", "bare pronoun with no nearer antecedent"`.
3. **Fixture.** `cal_fixtures.py:285-287` **asserts the forbidden
   behaviour as a passing check** —
   `check("D12c KNOWN-NEGATIVE: a bare pronoun with NO antecedent is SELF", … == "SELF")`.

The calibration fixture currently *defends* the defect. Any repair that
does not also correct the fixture will fail its own gate.

**CURRENT AFFECTED POPULATION.** Executed population **1**
(`README.md`, character 3162), which binds correctly to `OTHER`.
**Current false SELF bindings: 0.**
**On the invalid denominator I previously reported:** I once described
this as "70 F8 flips." That figure was wrong — `bind_claims` runs
`polarity_of()` and `continue`s on `None` *before* `bind_subject` is
reached, so 69 of those 70 sentences never enter the binding path at
all. The executed population is 1.

**GOVERNING INVARIANT.** Absence of an antecedent is not evidence of
self-reference. It is absence of evidence.

**DESIGN OPTIONS.**

* **A8-i — return `AMBIGUOUS` / `UNRESOLVED`** when no antecedent is
  found.
* **A8-ii — widen the antecedent search** (`ANTECEDENT_WINDOW = 400`,
  `F8`) and keep the SELF default beyond it. *Rejected in analysis: this
  moves the boundary without removing the unearned default.*
* **A8-iii — abstain on the axis** when a determining claim rests on an
  unbound pronoun.

**RECOMMENDED: A8-i.** It is what §6 requires, and it converts an
unearned assertion into a declared abstention.

**FAIL-OLD CONTROL.** A bare pronoun with no antecedent binds `SELF`.
**PASS-NEW CONTROL.** The same input binds `AMBIGUOUS`/`UNRESOLVED`, and
the one executed case (`README.md` char 3162) still binds `OTHER`.
**FIXTURE CORRECTION REQUIRED — FLAGGED FOR EXPLICIT AUTHORISATION.**
`cal_fixtures.py:285-287` must be inverted. This is a fixture mutation
and is **not** authorised by this document. It is named here because a
design that omits it would be undeliverable, and because a fixture that
asserts a contract violation is itself a finding about our calibration
discipline, not merely a line to edit.

**INTERACTION RISKS.** Shares `bind_subject` with `M2` and shares the
`bind_claims` execution path with `D13`. See the coupling note in A9.

**QUALIFICATION PROOF REQUIRED.** Zero SELF bindings in the corpus rest
on an unbound pronoun, proved by enumeration rather than by the absence
of a complaint.

---

## A9 · `D13-residual` — polarity decided by character distance

**DEMONSTRATED FAILURE.** `subjectbind.py:42`:

```
NEG_WORD = re.compile(r"\b(?:not|never|no longer|isn't|is not|aren't)\b"
                      r"[^.;]{0,40}?(?=authoritative|source of)", re.I)
```

Negation is recognised only within 40 characters of the authority term.
Beyond that the sentence scores `POSITIVE`.

**SYNTHETIC HOSTILE CONTROL, ALREADY RUN AGAINST THE FROZEN
`polarity_of()`.** Gap 36 → `NEGATIVE` (correct). Gap 70 →

> *"This file is not, in any circumstance whatsoever that anyone has yet
> described to me, authoritative."* → **`POSITIVE`. Wrong.**

**CURRENT AFFECTED POPULATION — STATED PRECISELY.**
266 authority-bearing sentences corpus-wide. **3 long-gap polarity
inputs** (gap 94, 94, 60) in `CODE_AUDIT_FINAL_REPORT.md`,
`CODE_AUDIT_REMEDIATION_BACKLOG.md`, `DECISIONS.md`.
**0 proven false SELF authority facts.** Two of the three describe other
components; one is instructional. Per Kai's ruling these are **not**
counted as three false facts, and only SELF-bound claims determine
`SELF_ASSERTS_AUTHORITY`. The synthetic failure alone establishes the
obligation.

**GOVERNING INVARIANT.** Semantic negation determines polarity — not
whether the negative word happens to fall within N characters.

**DESIGN OPTIONS.**

* **A9-i — clause-scoped negation.** Negation anywhere within the same
  clause, bounded by clause punctuation rather than by a character
  count.
* **A9-ii — unbounded within the sentence.** Simplest; **carries a real
  false-negative risk**, below.
* **A9-iii — explicit polarity-predicate set** with a declared
  closed-world justification.

**RECOMMENDED: A9-i.**

> **R12 — THE RISK IN THE OBVIOUS FIX, FLAGGED UNASKED.** A9-ii is the
> tempting repair and it is the wrong one. Removing the bound makes
> *"This document is not a draft but is authoritative"* score
> `NEGATIVE` — trading a false-positive class for a false-negative
> class, which is precisely the trade Kai rejected on `M1`. **A
> contrastive-negation known-negative of exactly that shape is
> mandatory** in the regression set, whichever option is chosen. It is
> not currently in it.

**HOSTILE CONTROLS REQUIRED — Kai's list, plus the one above.**
`non-authoritative` · `not authoritative` · `not a source of truth` ·
long separated negation beyond the historical 40-character bound ·
genuine positive control · quoted / OTHER-subject control ·
**contrastive negation ("not X but authoritative")**.

**INTERACTION RISKS — COUPLED TO `D12`.** `bind_claims` calls
`polarity_of()` and skips the sentence when it returns `None`; only
surviving sentences reach `bind_subject`. `D13` changes polarity
outcomes on the same sentence set whose binding `D12` changes. **The two
must be calibrated together against one combined regression set**, or
each will be measured against a population the other has moved. This is
the coupling that produced the invalid "70 flips" denominator, and it
will produce another one if the repairs are calibrated separately.

**QUALIFICATION PROOF REQUIRED.** All seven hostile controls pass, and
the 3 long-gap corpus inputs are individually adjudicated after the
repair — not assumed to be correct because they were correct before.

---

# GROUP B — QUALIFICATION / PACKAGE INTEGRITY

## B1 · `Q1b` — qualification does not validate the §5 subject population

**DEMONSTRATED FAILURE.** `qualify.py:178-181`, criterion [5], iterates
`ont.ALPHABETS` and checks non-abstention **axis cells** for a
`witness_value`. `evidence_facts` is not in `ont.ALPHABETS`. The
qualification therefore never inspects the positive evidence facts at
all — the 81 `E1` violations and the 10 malformed `TECH_WATCH` traces
sit entirely outside its denominator.

**CURRENT AFFECTED POPULATION.** The check's denominator excludes 316 of
659 `§5 SUBJECT`s — **48% of the population it is named for.**

**GOVERNING INVARIANT — R5.** A check's scope is defined by the data it
traverses. `ALPHABETS` is the axis alphabet, not the §5 subject
population; using it as the denominator is a scope smaller than the
name.

**DESIGN OPTIONS.**

* **B1-i — enumerate the §5 subject population from the contract.**
  343 verdicts + 316 positive facts = 659, derived, not listed.
* **B1-ii — add `evidence_facts` to the iteration.** Fixes this
  instance; leaves the class — the next subject type added will be
  missed the same way.

**RECOMMENDED: B1-i.** B1-ii is the R6 failure: fixing the instance and
declaring the class closed.

**FAIL-OLD CONTROL.** The current package qualifies with 81 `E1`
violations present and undetected.
**PASS-NEW CONTROL.** The same package **fails** qualification, naming
all 81 — and the 10 `TECH_WATCH` traces specifically, per Kai §12.
**KNOWN-NEGATIVE.** A package with compliant traces qualifies.

**INTERACTION RISKS.** `Q1b` must land *before* `E1` is judged repaired,
so that `E1`'s repair is verified by an instrument that can see it.
`Q1b`'s denominator changes if `E2` renames a fact.

**QUALIFICATION PROOF REQUIRED.** The check prints its denominator and
is calibrated with a known-positive and a known-negative (I-8). A
denominator that *shrinks when a defect is repaired* is the failure mode
to test for explicitly.

---

## B2 · `Q1a` — package-integrity qualification absent

**DEMONSTRATED FAILURE.** Nothing in the qualification step proves that
the modules which produced the results are the modules that were
qualified. `envelope.py:85` checks
`truncated != (evidence_shown < evidence_total)` — a count relation
over witnesses, not an integrity property of the package.

**CURRENT AFFECTED POPULATION.** Package-level. Not a per-document
count.

**GOVERNING INVARIANT.** The qualified package must be provably the
package that ran.

**DESIGN OPTIONS.**

* **B2-i — extend `PACKAGE.sha256` to a complete manifest** covering
  every module, every input artefact, and every emitted artefact, and
  verify it at qualification time as a gate that can fail.
* **B2-ii — per-file digest manifest** without a single aggregate.

**RECOMMENDED: B2-i.** A single aggregate that decomposes to per-file
digests gives both the one-line identity and the attribution.

**FAIL-OLD CONTROL — MANDATORY.** A package with one byte altered in one
module must **fail**. This must be executed, not asserted (R2): a
contingency that has not been run is a hypothesis with good
presentation.
**PASS-NEW CONTROL.** The unmodified package passes.

**INTERACTION RISKS.** Every Group A repair changes module digests, so
`B2` must be the last thing built and the first thing run.

**QUALIFICATION PROOF REQUIRED.** The gate demonstrably fails on a
deliberately corrupted package.

---

## B3 · `I1` — blind precommit identity excludes the evidence

**DEMONSTRATED FAILURE.** The blind precommit identity binds the
candidate but not the evidence artefacts that the holdout is evaluated
against. The commitment can therefore be satisfied by a package whose
evidence differs from the one adjudicated.

**CURRENT AFFECTED POPULATION.** Package-level.

**GOVERNING INVARIANT — D367 §9 and §10.** The blind holdout's integrity
depends on the evidence being committed before evaluation, not merely
the code.

**DESIGN OPTIONS.**

* **B3-i — extend the precommit identity to cover the evidence
  artefacts** (`passA.json`, classification, holdout selection) as well
  as the modules.
* **B3-ii — a separate evidence commitment** published alongside.

**RECOMMENDED: B3-i.** One identity, one moment, nothing outside it.

**FAIL-OLD CONTROL.** The current precommit identity is unchanged when
an evidence artefact is altered.
**PASS-NEW CONTROL.** Altering any evidence artefact changes the
identity. Executed, not asserted.

**INTERACTION RISKS.** `Q1a` and `I1` overlap and should share one
manifest mechanism rather than two. **Both must target a NEW candidate**
— per Kai's standing ruling, nothing is repaired in place.

**QUALIFICATION PROOF REQUIRED.** A fresh six-axis blind holdout under
D367 §9, selected after the identity is committed.

---

# 2. Repair-design constraints that are NOT matrix rows

Kai §9: these stay visible so the chosen repairs do not recreate the
same bounded-information failure elsewhere. **None is an obligation.**

| id | constraint | current population |
|---|---|---|
| `PURPOSE {12,160}` | semantic purpose recognition must not silently depend on an arbitrary 160-character body cap | 0 of 36 `PURPOSE` matches reach the cap |
| `F2` | **widening visibility must not widen subject applicability** | binds `D14`/A5 directly |
| `F5` | `_context` `[:200]` clips determining evidence while declaring `truncated=False` | 40 clipped, 10 lacking their own `witness_value` |
| `F6` | `binding_contradiction.context` `[:120]` is a *second* truncation of an already-bounded field | 5 records, 0 currently clipped |
| `F7` | `RUN_NEAR` `{0,10}` | 10 tokens, 0 classification deltas at 24/48/96/200/400/whole-head |
| `F3` | `says_supersedes` emitted 272 times, no consuming package path found | 3 true / 269 false |
| `F4` | `bind_claims(head_bytes=None)` dormant at all 4 call sites | 0 |

**`F6` carries one design question for the review, per Kai §6:** is
`binding_contradiction.context` **evidence** or **presentation**? If
evidence, it obeys §5 trace preservation. If presentation, any
shortening must be labelled an excerpt and must not masquerade as
complete evidence. It is currently neither labelled nor preserved.

**`F3`/`F4` produce repair invariants only:** an emitted-but-unconsumed
field must not later acquire a consumer without a bound justification,
and `head_bytes` must not be reintroduced.

---

# 3. Sequencing

D367 §12 already fixes the implementation order, and this analysis makes
that order load-bearing rather than stylistic:

| §12 step | obligations | why the position matters |
|---|---|---|
| 1 · evidence/witness trace completeness | `D14`, `E1`, `F5` | everything downstream reads these records |
| 2 · `SCOPE` ontology and foundation | `M3` | `M1`'s witness selection depends on it |
| 3 · `VALIDITY` under corrected scope | `M1` | must follow `M3` |
| 4 · `FUNCTION` evidence-vs-verdict | `PURPOSE` constraint | no matrix row |
| 5 · `AUTHORITY` claim-extractor | `D12-res`, `D13-res`, `M2`, `D2`, `E2` | `D12`/`D13` calibrated **together** |
| 6 · retain abstention controls | — | no relaxation |
| 7 · recompute all six axes | — | one consolidated candidate |
| 8 · one full qualification | `Q1b`, `Q1a`, `I1` | Group B must be able to see Group A |
| 9 · fresh independent blind holdout | — | §9 selection, §10 independence |

**One consolidated candidate. No standalone axis release.**

---

# 4. What this document does not decide

* It does not select a mechanism. Every `RECOMMENDED` is a proposal.
* It creates no D-number, no admission class, no closure.
* It does not authorise the `cal_fixtures.py:285-287` correction, which
  is a fixture mutation and is flagged in A8 for explicit decision.
* It does not reopen discovery. Under the Kai hard-stop ruling,
  discovery reopens **only** if repair-design evidence demonstrates that
  a material assumption supporting the repair basis is false or
  incomplete enough to make a proposed repair incorrect.
* **Bounded-negative discipline, carried from the closed gate.** The
  information-bound inventory earned only: *"No additional information
  bounds found within the frozen declared mechanical search grammar."*
  The reachability analysis earned only: *"No dynamic/reflection route
  was found within the declared static search grammar."* Anything
  outside those grammars is **unproven, not disproven**. Neither
  sentence may be strengthened in any design built on this document.

---

# THREAD RECOVERY BLOCK

```
DOCUMENT          H2 CONSOLIDATED REPAIR SPECIFICATION / DESIGN OPTIONS
STATUS            DRAFT. NOT AUTHORISED. NO IMPLEMENTATION.
AUTHORED          2026-08-29, by Orion, under Kai ruling
                  "DISCOVERY CLOSED — PROCEED TO DESIGN REVIEW, NO CODE"
REPOSITORY        f35f890, tree 2c791833, worktree clean at authorship
CANDIDATE         HOUSE_H2 v1.2, aggregate ba2b16d4…de4a — UNMODIFIED
SUBJECT           d8aac4d4…f197, tree 3abc9e9d…b117, 272 documents
CONTRACT          H2_REPAIR_CONTRACT_D367.md, sha256 0ce5792e…00bb
OBLIGATIONS       12 — Group A 9 logic/semantic, Group B 3 qualification
                  A: D2 M1 M2 M3 D14 E1 E2 D12-residual D13-residual
                  B: Q1b Q1a I1
NEW D-NUMBERS     0
NEW ROWS          0
DISCOVERY         CLOSED by Kai ruling. Narrow reopen condition only.
NOT AUTHORISED    repair · fixture mutation · design implementation ·
                  candidate generation · D375 · ledger append
HOLD              H2 HOLD remains until Dainius authorises implementation
NEXT              Kai attacks this design; simplifies; checks interactions;
                  brings a recommendation to Dainius. Dainius decides.
```
