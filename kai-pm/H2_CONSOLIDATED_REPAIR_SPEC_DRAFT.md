# HOUSE_H2 — CONSOLIDATED REPAIR SPECIFICATION / DESIGN OPTIONS

**REVISION 2 — incorporating Kai Design Review Round 1.**
**STATUS: DRAFT FOR REVIEW. NOT AUTHORISED. NO IMPLEMENTATION.**

Revision 1 (commit `3782662`) proposed mechanisms. Kai reviewed it and
issued fourteen rulings. This revision marks every mechanism
`ACCEPTED` / `ACCEPTED WITH KAI MODIFICATION` / `REJECTED — SUPERSEDED`,
incorporates the two-stage identity architecture, and records one
source-confirmed finding that arose while implementing Kai's correction.

This document still decides nothing. No repair, fixture mutation,
candidate generation or change to the H2 package has been made.

| | |
|---|---|
| repository state at authorship | `3782662` + this revision, worktree otherwise clean |
| candidate under repair | HOUSE_H2 v1.2, aggregate `ba2b16d4…de4a` — **UNMODIFIED** |
| subject | `d8aac4d49e6ba997e3eb38062c0917186ee3f197`, tree `3abc9e9d…b117`, 272 documents |
| governing contract | `kai-pm/H2_REPAIR_CONTRACT_D367.md`, sha256 `0ce5792e…00bb`, 389 lines |
| working obligations | 12 — unchanged |
| new D-numbers created | 0 |

---

## 0. Units — declared once, binding on every count in this document

Counts below are never interchangeable and are labelled at every use.

| unit | meaning | corpus total |
|---|---|---|
| `DOCUMENT` | a subject document | 272 |
| `AXIS CELL` | one of six axes on one document | 1632 |
| `EMITTED WITNESS RECORD` | one witness object as emitted | 491 |
| `UNIQUE §5 IDENTITY` | distinct nine-field witness identity | 480 |
| `SOURCE OCCURRENCE` | a physical token occurrence in source | 1967 distinct |
| `§5 SUBJECT` | a thing owing a §5 trace = verdict + positive fact | 659 measured today |
| `POSITIVE EVIDENCE FACT` | a `true` evidence-fact boolean | 316 |

**Kai §11 condition, binding on this table.** 659, 343 and 316 are
**today's measured population, not the definition.** No qualification
mechanism may hard-code them. This table is a baseline for regression
comparison only.

---

## 0.1 DISPOSITION SUMMARY — Kai Design Review Round 1

| obligation | rev-1 recommendation | disposition |
|---|---|---|
| `D2` | A1-i resolve-before-classify | **ACCEPTED WITH KAI MODIFICATION** — context *and* resolution |
| `M1` | A2-i + A2-iii | **ACCEPTED, RULE TIGHTENED** — no whitelist of any kind |
| `M2` | A3-i subject-binding gate | **ACCEPTED** |
| `M3` | A4-i structural derivation | **ACCEPTED WITH KAI MODIFICATION** — structure alone is insufficient |
| `D14` | A5-i remove the cap | **ACCEPTED WITH KAI MODIFICATION** — decouple consumers, do not remove globally |
| `E1` | A6-i + A6-ii | **ACCEPTED** |
| `E2` | E2-i rename | **ACCEPTED WITH KAI MODIFICATION** — stronger name |
| `D12-res` | A8-i unresolved | **ACCEPTED, SEQUENCING ADDED** |
| `D13-res` | A9-i clause scope | **ACCEPTED WITH KAI MODIFICATION** — conservative, abstain on complex prose |
| `Q1b` | B1-i derive the population | **ACCEPTED WITH ONE CONDITION** — derive, never hard-code |
| `Q1a` | B2-i single manifest | **REJECTED — SUPERSEDED** by two-stage identity |
| `I1` | B3-i "one identity, one moment" | **REJECTED — SUPERSEDED** by two-stage identity |
| `F6` | open question | **DECIDED BY KAI** — presentation, `context_excerpt` |

---

# ⚠ SECTION 0.2 — SOURCE-CONFIRMED FINDING ARISING FROM KAI §1

**This is not reopened discovery.** It falls inside the narrow reopen
condition Kai defined: *repair-design evidence demonstrating that a
material assumption supporting the repair basis is false or incomplete.*
It arose while verifying Kai's cycle argument against source, and it
changes how `I1` must be characterised.

## The cycle Kai forbade us to build **already exists in the live candidate.**

Kai's ruling §1 warns that putting evidence inside the selecting
aggregate would create:

```
candidate identity → holdout selection → evidence → changed identity
                  → changed holdout selection
```

I opened the artefacts to build Stage A correctly. **That loop is
already closed in HOUSE_H2 v1.2.**

**MEASURED, three facts, each independently checkable:**

1. `MANIFEST.sha256` line 4 is
   `eb50452d…0ad2fd  h2v12-classification.json`
   — an **evidence artefact produced by executing the candidate**, listed
   inside the manifest alongside the ten modules.
2. `sha256(MANIFEST.sha256)` = `ba2b16d400aafd2b694890149bbaae1d1369d3771c25ce809d2f752d5248de4a`.
3. `h2v12-holdout.json` records `candidate_aggregate` =
   `ba2b16d400aafd2b694890149bbaae1d1369d3771c25ce809d2f752d5248de4a`
   — **the same value**, and D367 §9 makes that value the sole variable
   input selecting the blind 40.

**Therefore: altering one byte of classification output changes the
aggregate, which changes which 40 documents are selected.** The property
§9 relies on — that the sample is determined by the candidate and not by
its results — does not currently hold.

**SECOND, INDEPENDENT COUPLING.** `holdout.py:49-50`:

```python
by_path = {r["path"]: r for r in res["rows"]}
chosen  = select(sorted(by_path), aggregate)
```

The **selection universe** is derived from the candidate's own output
rows, not from the subject tree. A candidate that emitted a different
set of paths would draw its 40 from a different population, independently
of the aggregate.

**SCOPE OF THIS CLAIM (R17).** I measured the manifest membership, the
aggregate equality, and the selection universe expression. I have **NOT**
established that this was ever exploited, that the adjudicated 40 are
wrong, or that any verdict is affected. The 240-cell result
(208 CLEAN / 20 BLOCKER / 12 OVER_ABSTENTION / 0 UNRESOLVED) stands
unchallenged by this finding.

**WHAT I DELIBERATELY DID NOT MEASURE, AND WHY.** I could size the impact
by computing which 40 a code-only aggregate would select and diffing it
against the adjudicated 40. **I did not, and I recommend against it.**
Doing so would reveal a *second* set of 40 documents. Under §9,
previously revealed rows become regression evidence only — so
pre-emptively revealing another 40 would permanently shrink the pool of
unrevealed documents available to the final blind holdout. That is a
contamination cost paid to satisfy curiosity about a defect we have
already confirmed structurally. **Kai's decision, not mine.**

**EFFECT ON THE MATRIX: NONE. NO NEW D-NUMBER. NO NEW ROW.**
`I1` is already the obligation covering blind precommit identity. What
changes is its characterisation: `I1` is not *"the identity excludes the
evidence"*. It is **"the identity includes evidence it must not include,
and excludes evidence it must include."** Kai's two-stage architecture
repairs both halves; it now has a confirmed defect to point at rather
than a hypothetical one to avoid.

---

## 1. The twelve obligations, separated by type

### GROUP A — LOGIC / SEMANTIC REPAIR (9)
`D2` · `M1` · `M2` · `M3` · `D14` · `E1` · `E2` · `D12-residual` ·
`D13-residual` — these change what the analyser *concludes*.

### GROUP B — QUALIFICATION / PACKAGE INTEGRITY (3)
`Q1b` · `Q1a` · `I1` — these change what the package can *prove about
itself*. They alter no verdict, and they are why a Group A repair cannot
be self-certified.

---

# GROUP A — LOGIC / SEMANTIC REPAIR

## A1 · `D2` — token kind decided by character class, not by resolution
### DISPOSITION: **ACCEPTED WITH KAI MODIFICATION**

**DEMONSTRATED FAILURE.** `passa.py:163` carries the docstring *"D2:
DISCRIMINATE the kind. Never assume it from the character class."* The
residual branch then does exactly that:
`if tok.isdigit(): return "DECIMAL_TOKEN", False`.

**CURRENT AFFECTED POPULATION.** 1 `SOURCE OCCURRENCE`. `9904486`
resolves to `9904486d1cf133556b5c5496a6317caf59f0a3c4`,
*"feat(UH-5): Policy, human approval and capability bridge"*.

**GOVERNING INVARIANT — as amended by Kai §2.**
**Resolution proves the object EXISTS. It does not prove the source used
the token AS a commit.** A decimal can accidentally be a valid hex
abbreviation. Kind requires **context AND resolution**, not resolution
alone.

**REJECTED — SUPERSEDED.** Rev-1's A1-i as written
("offer every hex-valid token ≥7 to `git cat-file`") is superseded. It
defines the class as *resolves ⇒ commit*, which is the collision hazard
Kai names.

**MECHANISM, as amended.** A two-factor discriminator, in this order:

| source context | resolution | kind |
|---|---|---|
| digest context | — | `DIGEST_FRAGMENT` |
| RUN context (`RUN_NEAR` / `RUN_URL`) | — | `RUN_ID` |
| explicit commit / SHA / revision context | resolves | **`COMMIT`** |
| explicit commit / SHA / revision context | does not resolve | abstain / non-promoting evidence |
| ordinary numeric context | — | `DECIMAL_TOKEN` |
| ambiguous | — | abstain / non-promoting evidence |

**`9904486` must become `COMMIT` for the right reason** — because its
source context presents it as a revision *and* it resolves — not because
every decimal is offered promotion.

**FAIL-OLD.** `9904486` classifies `DECIMAL_TOKEN`, `resolved=False`.
**PASS-NEW.** It classifies `COMMIT` with the full 40-character SHA as
`witness_value`, and the emitted record names both the context predicate
and the resolution as its evidence.
**KNOWN-NEGATIVE — MANDATORY, ACCIDENTAL COLLISION.** A decimal in
ordinary numeric context that *happens* to resolve must stay
`DECIMAL_TOKEN`. Without this the repair trades one wrong output for a
class of them.

**INTERACTION RISKS.** Adds git probes on a previously short-circuited
path — cost bounded by the decimal population, to be measured not
assumed. Coupled to `D14`: changing the extraction window changes the
token population reaching this branch.

**QUALIFICATION PROOF.** The repaired record carries a complete §5 trace
including the resolved SHA and the context predicate that qualified it.

---

## A2 · `M1` — a review event promoted to a temporal validity binding
### DISPOSITION: **ACCEPTED, RULE TIGHTENED (Kai §3)**

**DEMONSTRATED FAILURE.** A date beside a review or inspection predicate
is promoted to `VALIDITY = TIME_BOUND`. *A review event is not a
validity/currency binding.*

**CURRENT AFFECTED POPULATION.** 145 of 272 `AXIS CELL` (VALIDITY),
corpus-wide. 19 are holdout `AXIS CELL`s adjudicated as blockers.
**PROTECTED POSITIVES: 10**, adjudicated on the merits of their source
contexts as document-state / version / report bindings. Measured: none
of the ten carries a binding contradiction.

**GOVERNING INVARIANT — as tightened by Kai §3.** `TIME_BOUND` requires
a **positively established DOCUMENT-STATE binding**. An inspection event
does not supply one.

**EXPLICITLY FORBIDDEN MECHANISMS.** Path whitelist · document-name
whitelist · label alone treated as truth. **The ten protected positives
must all survive through the SAME general deterministic rule.**

> **Kai §3 escalation clause, carried verbatim into the design:** if a
> protected positive cannot be described by the deterministic rule,
> **surface it during final spec revision — do not special-case its
> path.** I record this as a live obligation on me, not a footnote: the
> temptation at that moment will be to add one exception and call the
> rule general. That is the R5 defect (a list kept beside the check) and
> it would be invisible in a passing test suite.

**MECHANISM.** A2-i binding-predicate discriminator
(`STATE_BINDING`: updated / valid until / version / effective —
vs `INSPECTION_EVENT`: reviewed / checked / audited / inspected),
combined with A2-iii, which retains the review date as a **non-promoting
emitted observation**. Evidence stays visible; it stops deciding.

**REJECTED — SUPERSEDED.** A2-ii ("abstain unless positively bound",
alone) — the trade Kai rejected; it converts 145 false positives into an
unmeasured false-negative population and would likely take the ten with
it.

**FAIL-OLD.** The 19 holdout cells plus a sample of the 145.
**PASS-NEW — LOAD-BEARING.** All ten adjudicated positives still
classify `TIME_BOUND`, **each one explained by the general rule.** This
is the control that proves discrimination rather than suppression.

**INTERACTION RISKS. `M1` must follow `M3`.** The determining VALIDITY
witness is selected using `applicability_scope`, which `M3` changes on
48 `EMITTED WITNESS RECORD`s. D367 §12 already orders scope (step 2)
before validity (step 3); that ordering is load-bearing, not stylistic.

**QUALIFICATION PROOF.** Every surviving `TIME_BOUND` cell carries a §5
trace whose `local_context` contains the state-binding predicate that
earned it.

---

## A3 · `M2` — the audited tree's lifecycle substituted for the document's
### DISPOSITION: **ACCEPTED (Kai raised no modification)**

**DEMONSTRATED FAILURE.** An audit document that records the commit it
audited has that commit routed into its own `LIFECYCLE`, producing
`HISTORICAL` for the *document* on evidence about a *different subject*.
`passa.py:92` states the rationale as *"the document states the snapshot
it audits"* — and the code then uses that snapshot as the document's own
lifecycle evidence. The invariant is written in the package and
contradicted by it.

**CURRENT AFFECTED POPULATION.** 3 of 272 `AXIS CELL` (LIFECYCLE),
COMMIT-routed `HISTORICAL`; 1 in the frozen holdout
(`CODE_AUDIT_MASTER`).

**GOVERNING INVARIANT.** The subject of a `LIFECYCLE` verdict is the
document. Evidence about an artefact the document *describes* is
evidence about that artefact.

**MECHANISM — A3-i.** A COMMIT witness may determine `LIFECYCLE` only if
its binding predicate makes the document itself the subject.

**REJECTED.** A3-ii (forbid the route entirely) — removes genuine cases
with the defective ones. A3-iii alone — too narrow.

**FAIL-OLD.** The 3 COMMIT-routed rows classify `HISTORICAL`.
**PASS-NEW — MANDATORY SYNTHETIC SAME-ROUTE POSITIVE.** A document whose
own lifecycle is genuinely evidenced by a commit must still classify
`HISTORICAL`. Without it A3-i is indistinguishable from A3-ii and cannot
be shown to have discriminated. `REPO_HEALTH_AUDIT` remains a
route-independent regression control.

**INTERACTION RISKS.** Shares `bind_subject` with `D12`.

**QUALIFICATION PROOF.** Each surviving COMMIT-routed `HISTORICAL` cell
carries a §5 trace showing the document as subject.

---

## A4 · `M3` — applicability scope under-assigned
### DISPOSITION: **ACCEPTED WITH KAI MODIFICATION**

**DEMONSTRATED FAILURE.** `applicability_scope` is emitted without being
derived from what the witness actually binds.

**CURRENT AFFECTED POPULATION.** 491 `EMITTED WITNESS RECORD`; expected
214 `WHOLE_FILE` / 277 `SPAN`; **48 wrong** (47 under-assigned, 1
over-assigned); 443 unchanged. Dual-unit: 480 `UNIQUE §5 IDENTITY`, 47
wrong. Locator rows: 44 of 272 `DOCUMENT`.

**REJECTED — SUPERSEDED.** Rev-1's A4-i as written
("header/control field ⇒ `WHOLE_FILE`"). **Kai §4 is right and the
reason matters:** a header-like field can belong to a *register entry*,
and this repository is full of registers. Rev-1 would have replaced one
layout heuristic with another layout heuristic and called it a
derivation.

**GOVERNING INVARIANT — as amended.** `WHOLE_FILE` requires **BOTH**:
structural document-level binding **AND** subject = the document as a
whole. Repeated or per-entry controlled fields remain `SPAN`.

**MECHANISM, four inputs, all derived:**
parsed structural position · predicate uniqueness vs repetition ·
explicit subject · enclosing entry / table / register context.

**RESIDUAL AMBIGUITY RULE.** `SPAN` where clearly local; otherwise
`UNKNOWN` / abstention — **never a guessed `WHOLE_FILE`.** Over-assigned
scope is the failure mode §9 calls *unsupported scope widening*, a
BLOCKER.

**FAIL-OLD.** The 48 wrong records.
**PASS-NEW.** The 443 unchanged remain unchanged, **and the 1
over-assigned record is corrected downward** — a repair that only moves
records upward has not discriminated.

**INTERACTION RISKS.** Upstream of `M1` and of every determining-witness
selection; D367 §12 step 2. Emission conservation: the repair-stable key
`path+start+end+detector` is unique over all 491 and `detector` is
load-bearing. Count changes must be source-anchored controlled deltas,
never a changed denominator.

**QUALIFICATION PROOF.** 491 records in, 491 keys out, every scope change
individually attributable.

---

## A5 · `D14` — evidence bisected at the extraction boundary
### DISPOSITION: **ACCEPTED WITH KAI MODIFICATION**

**DEMONSTRATED FAILURE.** `passa.py:190 head = text[:HEAD_BYTES]`,
`HEAD_BYTES = 6000`, **a character index despite its name**. A token
straddling that index is cut and a *prefix* is emitted as
`witness_value` with `truncated=False`.

**CURRENT AFFECTED POPULATION.** 2 `SOURCE OCCURRENCE`, both `HEX`,
source-verified, geometry independently reproducible without my
instruments. Known instance: `CODE_AUDIT_CONTINUATION_LOG.md`, offsets
5971–6011, emitted as a 29-character prefix.

**A NOTE ON THE DENOMINATOR.** An earlier detector reported 1 of 272. It
required the *truncated* form to still match the recogniser, so any token
cut below the recogniser minimum was outside its denominator by
construction — R5. The population above is over source occurrences.

**GOVERNING INVARIANT — D367 §5 verbatim.** *"Silent truncation is
forbidden. The evidence actually responsible for the emitted cell must
always be recoverable from the candidate package without guessing which
source fragment mattered."*

**MECHANISM — A5-i for token/witness extraction, WITH KAI'S DECOUPLING.**
`HEAD_BYTES` is **not** removed globally. Three consumers are separated
so that repairing token loss cannot silently widen the others:

| consumer | current | after |
|---|---|---|
| lexical evidence scanner | `text[:6000]` | **full source** |
| `PURPOSE` recogniser (`classify.py:171`, `F2`) | `text[:6000]` | unchanged unless separately authorised |
| supersession recogniser (`passa.py:277`, `F3`) | `txt[:6000]` | unchanged unless separately authorised |

**ORDERING INVARIANT (Kai §5).** *Recognise complete lexical evidence
first; then apply any authorised semantic/application boundary.*

> This is the F2 constraint made structural. Rev-1 named the risk —
> *widening visibility must not widen subject applicability* — and then
> recommended a global removal that would have realised it. Kai's
> decoupling is the correct form: the earlier F2 measurement showed that
> widening `PURPOSE`'s window surfaced 9 additional matches whose correct
> disposition was that both apparent changes came from **false** `PURPOSE`
> matches on domain prose.

**WITNESS CONTEXT INVARIANT.** Witness context must **always contain the
exact witness value**. For oversized surrounding context: stable
selector · exact value · truthful excerpt/truncated flag · bound sidecar
where necessary. **No 200→400 or 6000→12000 patch** — moving a boundary
is not repairing one.

**FAIL-OLD.** Both boundary-crossing tokens emit prefixes with
`truncated=False`.
**PASS-NEW.** Both emit full 40-character values, plus a synthetic token
placed deliberately astride the old boundary.
**KNOWN-NEGATIVE.** `PURPOSE` and `says_supersedes` outputs are
**byte-identical** before and after — the mechanical proof that the
decoupling held.

**INTERACTION RISKS.** Cost of full-source scanning to be measured before
selection, not assumed.

**QUALIFICATION PROOF.** Every changed `EMITTED WITNESS RECORD` is
attributable to a named boundary-crossing occurrence; total count
conserved or its delta source-anchored.

---

## A6 · `E1` — positive evidence facts without a compliant §5 trace
### DISPOSITION: **ACCEPTED — A6-i + A6-ii (Kai §6)**

**DEMONSTRATED FAILURE.** Positive evidence facts are emitted as bare
booleans; §5 requires every one to carry a source-bound witness
sufficient for independent adjudication.

**CURRENT AFFECTED POPULATION.** 81 of 316 `POSITIVE EVIDENCE FACT`
subjects violate the frozen §5 trace contract.
**Correcting my own earlier transmission:** I once reported this as
"316 positive facts carry none." **235 of 316 carry the full nine-field
trace** in the package-bound sidecar. The violating population is 81.

**GOVERNING INVARIANT — Kai's strengthened form, plus §6.** Every
positive evidence fact must carry a **semantically truthful** §5 trace
bound **directly to the fact it determines**. *A positive with no
compliant trace is not allowed to survive merely because a related trace
exists elsewhere.*

**KNOWN SUB-POPULATION IN HAND.** The `F5` measurement found 40 clipped
`local_context` fields of which **10 do not contain their own
`witness_value`** — all `TECH_WATCH` — every one carrying
`truncated=False`. Named fail-old controls, also carried into `Q1b`'s
regression basis per Kai §12 of the previous round.

**MECHANISM.** A6-i — the producer that sets the boolean carries the
witness that set it. A6-ii — refuse to emit a positive fact with no
compliant trace; abstain instead.
**REJECTED.** A6-iii (sidecar-only) — repeats the current shape, where
the trace exists somewhere but is not bound to the fact it explains.

**FAIL-OLD.** The 81 violating subjects, including the 10 `TECH_WATCH`
cases whose context does not contain its own value.
**PASS-NEW.** The 235 already-compliant traces unchanged.

**INTERACTION RISKS.** `E1` and `Q1b` are logically distinct: a repaired
`Q1b` **makes the old `E1` population visible and failing**; it does not
repair it. `E1` also depends on `D14`/`F5` — a trace cannot be
semantically truthful if extraction clipped the determining evidence.

**QUALIFICATION PROOF.** `Q1b` must be able to **enumerate and reject
such records mechanically** (Kai §6), validating content, not presence.

---

## A7 · `E2` — a static textual reference reported as runtime consumption
### DISPOSITION: **ACCEPTED WITH KAI MODIFICATION — stronger name**

**DEMONSTRATED FAILURE.** `run_h2_v12.py:84`
`f["CONSUMED_AT_SUBJECT"] = bool(row["readers"])`. `readers` is a static
textual reference set; the fact asserts consumption at the subject — a
runtime property.

**CURRENT AFFECTED POPULATION.** 5 of 272 `DOCUMENT`.

**GOVERNING INVARIANT.** A fact must name what it measured. **No
runtime-consumption statement without runtime evidence.**

**REJECTED — SUPERSEDED.** Rev-1's `REFERENCED_AT_SUBJECT`. Kai §7 is
right: it can be re-read as runtime behaviour, which is the same defect
one word quieter.

**MECHANISM.** `STATIC_REFERENCE_AT_SUBJECT`, carrying an explicit
`ANALYSIS_SCOPE` field naming the evidence class as static. Follows the
`F4` precedent Kai accepted (`NO_WRITER_WITHIN_ANALYZED_SCOPE`): renamed,
not removed, closure rules unrelaxed.

**FAIL-OLD.** The 5 rows emit `CONSUMED_AT_SUBJECT`.
**PASS-NEW.** The same 5 emit `STATIC_REFERENCE_AT_SUBJECT` with a
declared static `ANALYSIS_SCOPE`, and **no fact name anywhere in the
package asserts a property stronger than its evidence class.** That last
clause is mechanically checkable and should be a gate.

**INTERACTION RISKS.** Renaming changes the §5 subject key set — `Q1b`
must derive it, not hold a list (see B1).

---

## A8 · `D12-residual` — a bare pronoun with no antecedent binds to SELF
### DISPOSITION: **ACCEPTED — A8-i, SEQUENCING ADDED (Kai §8)**

**DEMONSTRATED FAILURE, AT THREE LAYERS** — the only obligation here
contradicted at contract, code and fixture simultaneously.

1. **Contract.** D367 §6 forbids it.
2. **Code.** `subjectbind.py:110` returns
   `"SELF", "bare pronoun with no nearer antecedent"`.
3. **Fixture.** `cal_fixtures.py:285-287` **asserts the forbidden
   behaviour as a passing check** —
   `check("D12c KNOWN-NEGATIVE: a bare pronoun with NO antecedent is SELF", … == "SELF")`.

**CURRENT AFFECTED POPULATION.** Executed population **1**
(`README.md`, character 3162), which binds correctly to `OTHER`.
**Current false SELF bindings: 0.**
**On the invalid denominator I previously reported:** I once described
this as "70 F8 flips". `bind_claims` runs `polarity_of()` and `continue`s
on `None` *before* `bind_subject` is reached, so 69 of those 70 never
enter the binding path. The executed population is 1.

**GOVERNING INVARIANT.** Absence of an antecedent is not evidence of
self-reference. It is absence of evidence.

**MECHANISM — A8-i.** Return `AMBIGUOUS` / `UNRESOLVED`; let the H2 axis
abstain.
**REJECTED.** A8-ii (widen `ANTECEDENT_WINDOW`) — moves the boundary
without removing the unearned default.

**MANDATORY SEQUENCING — Kai §8, and this is the part I would have got
wrong.**

```
1. bank / freeze the CORRECTED EXPECTED CONTRACT
2. prove the FROZEN OLD CANDIDATE FAILS it
3. only then implement the repair
```

**Do not change fixture and implementation in one opaque step.** A
simultaneous change makes the green result unfalsifiable: it cannot
distinguish *the repair works* from *the expectation was moved to meet
the code*. That is I-8 — the source of the expected answer must not be
the thing under test — and the current inverted fixture is what happens
when it is violated.

**FAIL-OLD.** A bare pronoun with no antecedent binds `SELF`.
**PASS-NEW.** The same input binds `AMBIGUOUS`/`UNRESOLVED`; the one
executed case (`README.md` char 3162) still binds `OTHER`.
**FIXTURE CORRECTION.** Required, inside the authorised repair sequence
above. Still not authorised by this document.

---

## A9 · `D13-residual` — polarity decided by character distance
### DISPOSITION: **ACCEPTED WITH KAI MODIFICATION — conservative semantics**

**DEMONSTRATED FAILURE.** `subjectbind.py:42`:

```
NEG_WORD = re.compile(r"\b(?:not|never|no longer|isn't|is not|aren't)\b"
                      r"[^.;]{0,40}?(?=authoritative|source of)", re.I)
```

**SYNTHETIC HOSTILE CONTROL, run against the frozen `polarity_of()`.**
Gap 36 → `NEGATIVE` (correct). Gap 70 →
*"This file is not, in any circumstance whatsoever that anyone has yet
described to me, authoritative."* → **`POSITIVE`. Wrong.**

**CURRENT AFFECTED POPULATION — stated precisely.** 266 authority-bearing
sentences corpus-wide. **3 long-gap polarity inputs** (gaps 94, 94, 60).
**0 proven false SELF authority facts** — two describe other components,
one is instructional, and only SELF-bound claims determine
`SELF_ASSERTS_AUTHORITY`. Per Kai's ruling these are **not** three false
facts. The synthetic failure alone carries the obligation.

**GOVERNING INVARIANT.** Semantic negation determines polarity — not
whether the negative word falls within N characters.

**REJECTED — SUPERSEDED.** A9-ii (unbounded sentence regex). And Kai §9
goes further than rev-1 did, correctly: **do not replace the
40-character heuristic with a more elaborate heuristic that forces every
prose sentence into POSITIVE/NEGATIVE.** Rev-1 flagged the contrastive
risk but still proposed a mechanism that would classify everything. The
amendment is that *declining to classify* is a legitimate output.

**MECHANISM — conservative deterministic semantics.**

| form | disposition |
|---|---|
| explicit attached negation (`non-authoritative`) | polarity earned |
| explicit simple same-clause negation | polarity earned |
| explicit controlled-field declaration | polarity earned |
| genuine simple positive declaration | polarity earned |
| **contrastive / structurally complex / semantically unresolved prose** | **`UNKNOWN` / unresolved — never guessed** |

**H2 is not required to solve unrestricted natural-language semantics**
(Kai §9). The AUTHORITY verdict remains abstention-only at H2 regardless.

**MANDATORY HOSTILE CONTROLS — Kai's eight.**
`non-authoritative` · `not authoritative` · `not a source of truth` ·
long separated negation beyond the historical 40-character bound ·
genuine positive · quoted / OTHER-subject ·
**`"not X but authoritative"`** · **`"authoritative but not X"`**.

> The eighth is Kai's addition and it closes the symmetric hole. Rev-1
> supplied only the first direction. A rule that handles
> *"not a draft but authoritative"* and mishandles
> *"authoritative but not a draft"* has learned the example, not the
> class — the R4 calibration failure.

**INTERACTION RISKS — COUPLED TO `D12`, one combined regression
population (Kai §9).** `bind_claims` calls `polarity_of()` and skips on
`None`; only survivors reach `bind_subject`. `D13` moves the polarity
outcome on the same sentence set whose binding `D12` changes. Calibrated
apart, each is measured against a population the other just moved —
**the exact mechanism that produced the invalid "70 flips" denominator.**

**QUALIFICATION PROOF.** All eight hostile controls pass, and the 3
long-gap corpus inputs are individually adjudicated *after* the repair —
not assumed correct because they were correct before.

---

# GROUP B — QUALIFICATION / PACKAGE INTEGRITY

## B1 · `Q1b` — qualification does not validate the §5 subject population
### DISPOSITION: **ACCEPTED WITH ONE CONDITION (Kai §11)**

**DEMONSTRATED FAILURE.** `qualify.py:178-181`, criterion [5], iterates
`ont.ALPHABETS` and checks non-abstention **axis cells** for a
`witness_value`. `evidence_facts` is not in `ont.ALPHABETS`. The 81 `E1`
violations and the 10 malformed `TECH_WATCH` traces sit entirely outside
its denominator.

**CURRENT AFFECTED POPULATION.** The check excludes 316 of 659
`§5 SUBJECT`s — **48.0% of the population it is named for.**

**GOVERNING INVARIANT — R5.** A check's scope is defined by the data it
traverses. `ALPHABETS` is the axis alphabet, not the §5 subject
population.

**MECHANISM — B1-i, with Kai's condition binding.** The qualifier
**DERIVES** its §5 subject population from actual emitted output and
schema: every non-abstention verdict, every positive evidence fact — and
**reports the resulting denominator.**

**HARD-CODING FORBIDDEN.** Not `343`. Not `316`. Not `659`. Not the
current fact names. **659 is a baseline population, not the algorithm.**

**REJECTED.** B1-ii (add `evidence_facts` to the iteration) — fixes the
instance, leaves the class; the next subject type added is missed the
same way. R6.

**FAIL-OLD.** The current package qualifies with 81 `E1` violations
present and undetected.
**PASS-NEW.** The same package **fails**, naming all 81 and the 10
`TECH_WATCH` traces specifically.
**CALIBRATION — Kai §11, mandatory.** Prove that **adding or corrupting a
positive fact cannot leave that fact outside the qualifier's
denominator.** Plus the I-8 failure mode to test explicitly: **a
denominator that shrinks when a defect is repaired** makes progress and
absence indistinguishable.

**INTERACTION RISKS.** `Q1b` must land *before* `E1` is judged repaired,
so `E1`'s repair is verified by an instrument that can see it. `Q1b`'s
derived key set absorbs the `E2` rename automatically — which is the
point of deriving it.

---

## B2 + B3 · `Q1a` and `I1` — TWO-STAGE IDENTITY
### DISPOSITION: **REV-1 RECOMMENDATION REJECTED — SUPERSEDED BY KAI §1**

**REV-1 WAS WRONG.** It recommended *"one identity, one moment, nothing
outside it"*, extending a single manifest over code **and** evidence.
Kai §1 rejects it, and §0.2 above shows the cycle is not merely a risk
to avoid in a future design — **it is already closed in the live
candidate.**

**DEMONSTRATED FAILURE, as now characterised.**
`Q1a`: nothing in qualification proves the modules that produced the
results are the modules qualified. `envelope.py:85` checks
`truncated != (evidence_shown < evidence_total)` — a count relation over
witnesses, not an integrity property.
`I1`: the blind precommit identity **includes evidence it must not
include** (`h2v12-classification.json` inside `MANIFEST.sha256`, whose
hash selects the holdout) **and excludes evidence it must include**
(Pass A, holdout, qualification outputs are not bound to the adjudicated
result).

**GOVERNING INVARIANT — D367 §9.** The blind sample is selected from
`FINAL_CANDIDATE_AGGREGATE`, which *"does not exist until the repair is
complete, so the sample cannot be known during implementation."* That
property requires the aggregate to be a function of the **candidate**,
never of the candidate's **results**.

### MECHANISM — TWO-STAGE BINDING (Kai §1, REQUIRED)

```
STAGE A — CANDIDATE PRECOMMIT IDENTITY          frozen FIRST
    code · candidate modules · governed static inputs ·
    contract / manifest dependencies
    → THIS is the identity D367 §9 uses to derive the blind 40
    → contains NO artefact produced by executing the candidate

STAGE B — EVIDENCE BUNDLE IDENTITY              frozen AFTER execution
    exact Pass A output · exact classification output ·
    selected holdout identity/list ·
    consequential evidence artefacts · qualification output

ADMISSION RECORD                                binds both
    candidate_precommit_identity
    evidence_bundle_identity
    subject identity
    holdout derivation
    qualification identity
```

**MUTATION CONTROLS.**
Alter a candidate byte → **candidate identity changes.**
Alter an evidence byte → **evidence identity changes.**
**Altered evidence MUST NOT retroactively redefine which candidate
selected the holdout.**

`Q1a` and `I1` may share manifest machinery but **MUST NOT create a
self-referential aggregate.**

### ADDITIONAL DESIGN REQUIREMENT ARISING FROM §0.2

The two-stage split fixes the aggregate. It does **not**, on its own,
fix the second coupling: `holdout.py:49-50` derives the **selection
universe** from `res["rows"]` — the candidate's own output. Under
Stage A the universe must be pinned to the **subject tree** (the 272
documents of `3abc9e9d…b117`), not to whatever the candidate emitted, or
a candidate that drops a row still moves its own holdout.

**FAIL-OLD CONTROLS — BOTH MANDATORY, BOTH EXECUTED NOT ASSERTED (R2).**
1. A package with one byte altered in one module must **fail**.
2. **An evidence artefact altered after Stage A must NOT change the
   Stage A identity, and therefore must not change the selected 40.**
   This is the control that would have caught the current defect.
**PASS-NEW.** The unmodified package passes both stages, and the
admission record reproduces the holdout selection from Stage A alone.

**INTERACTION RISKS.** Every Group A repair changes module digests, so
Stage A is the last thing built and the first thing run. Stage B cannot
exist until Stage A has executed.

**QUALIFICATION PROOF.** A fresh six-axis blind holdout under D367 §9,
selected from a Stage A identity that provably contains no execution
output.

---

# 2. Repair-design constraints that are NOT matrix rows

Kai: these stay visible so the chosen repairs do not recreate the same
bounded-information failure elsewhere. **None is an obligation.**

| id | constraint | current population |
|---|---|---|
| `PURPOSE {12,160}` | semantic purpose recognition must not silently depend on an arbitrary 160-character body cap | 0 of 36 `PURPOSE` matches reach the cap |
| `F2` | **widening visibility must not widen subject applicability** | binds `D14`/A5; now structural via decoupling |
| `F5` | `_context` `[:200]` clips determining evidence while declaring `truncated=False` | 40 clipped, 10 lacking their own `witness_value` |
| `F6` | **DECIDED — see below** | 5 records, 0 currently clipped |
| `F7` | `RUN_NEAR` `{0,10}` | 10 tokens, 0 classification deltas at 24/48/96/200/400/whole-head |
| `F3` | `says_supersedes` emitted 272 times, no consuming package path found | 3 true / 269 false |
| `F4` | `bind_claims(head_bytes=None)` dormant at all 4 call sites | 0 |

## `F6` — DESIGN DECISION MADE BY KAI (§10)

`binding_contradiction.context` is **PRESENTATION, not determining
evidence.** Represent it as **`context_excerpt`**. If shortened: label it
an excerpt and carry truthful truncation state. **The determining
contradiction evidence remains separately source-bound under §5.**

This removes the 120-character field from any implied evidentiary
authority — which is the right outcome, because it is a *second*
truncation of a field already bounded to 200 by `F5`, and today it is
neither labelled nor preserved.

`F3`/`F4` produce **repair invariants only**: an emitted-but-unconsumed
field must not later acquire a consumer without a bound justification,
and `head_bytes` must not be reintroduced.

---

# 3. Sequencing

D367 §12 fixes the order; this analysis makes it load-bearing.

| §12 step | obligations | why the position matters |
|---|---|---|
| 1 · evidence/witness trace completeness | `D14`, `E1`, `F5` | everything downstream reads these records |
| 2 · `SCOPE` ontology and foundation | `M3` | `M1`'s witness selection depends on it |
| 3 · `VALIDITY` under corrected scope | `M1` | must follow `M3` |
| 4 · `FUNCTION` evidence-vs-verdict | `PURPOSE` constraint | no matrix row |
| 5 · `AUTHORITY` claim-extractor | `D12-res`, `D13-res`, `M2`, `D2`, `E2` | `D12`/`D13` calibrated **together**, contract frozen before code |
| 6 · retain abstention controls | — | no relaxation |
| 7 · recompute all six axes | — | one consolidated candidate |
| 8 · one full qualification | `Q1b`, then Stage A `Q1a`/`I1` | Group B must be able to see Group A |
| 9 · fresh independent blind holdout | — | §9 selection from Stage A only, §10 independence |

**One consolidated candidate. No standalone axis release.**

---

# 4. Test strategy (Kai §12) and scope discipline (Kai §13)

**THE FROZEN ADMISSION CONTRACT IS NOT REPLACEABLE.** Required proof
remains: every demonstrated fail-old population · protected pass-new
controls · mutation / known-positive controls · full mechanical candidate
qualification · fresh reproduction · the D367 final blind 40 · Kai
independent adjudication.

**No generic 20-document sample substitutes for D367 qualification.**
Property-based tests are welcome as *supplementary* hostile testing —
particularly for `D12`/`D13` and boundary handling — and have **ZERO
authority** to replace the frozen contract.

**NO BROAD TYPE-SYSTEM REWRITE (Kai §13).** Enum / dataclass / Result
objects have merit for future hardening. Stronger types are introduced
**only** where they directly enforce one of the 12 obligations or prevent
an identified illegal promotion. We fix the trust boundaries first; we do
not enlarge this repair into a general H2 rewrite for elegance.

---

# 5. What this document does not decide

* It selects no mechanism. Every disposition above records **Kai's**
  ruling or a proposal awaiting one.
* It creates no D-number, no admission class, no closure. Matrix: 12.
* It does not authorise the `cal_fixtures.py:285-287` correction.
* It does not reopen discovery. §0.2 arose inside the narrow reopen
  condition — repair-design evidence about a material assumption — and
  adds no row.
* **Bounded-negative discipline, carried unstrengthened.** The
  information-bound inventory earned only *"No additional information
  bounds found within the frozen declared mechanical search grammar."*
  The reachability analysis earned only *"No dynamic/reflection route was
  found within the declared static search grammar."* Outside those
  grammars: **unproven, not disproven.** Neither may be strengthened in
  any design built on this document.

---

# THREAD RECOVERY BLOCK

```
DOCUMENT          H2 CONSOLIDATED REPAIR SPECIFICATION / DESIGN OPTIONS
REVISION          2 — incorporates Kai Design Review Round 1 (14 rulings)
STATUS            DRAFT. NOT AUTHORISED. NO IMPLEMENTATION.
AUTHORED          2026-08-29, by Orion
PRIOR REVISION    3782662 (revision 1)
CANDIDATE         HOUSE_H2 v1.2, aggregate ba2b16d4…de4a — UNMODIFIED
SUBJECT           d8aac4d4…f197, tree 3abc9e9d…b117, 272 documents
CONTRACT          H2_REPAIR_CONTRACT_D367.md, sha256 0ce5792e…00bb
OBLIGATIONS       12 — unchanged
                  A: D2 M1 M2 M3 D14 E1 E2 D12-residual D13-residual
                  B: Q1b Q1a I1
DISPOSITIONS      4 ACCEPTED · 6 ACCEPTED WITH KAI MODIFICATION ·
                  2 REJECTED/SUPERSEDED (Q1a, I1 → two-stage identity) ·
                  1 DECIDED BY KAI (F6 = presentation)
NEW FINDING       §0.2 — the identity cycle Kai forbade ALREADY EXISTS:
                  h2v12-classification.json is inside MANIFEST.sha256,
                  whose sha256 ba2b16d4…de4a IS the holdout-selecting
                  aggregate. Second coupling: holdout.py:49-50 draws the
                  selection universe from candidate OUTPUT rows.
                  NO NEW D-NUMBER. NO NEW ROW. Recharacterises I1.
NOT MEASURED      the alternative 40 under a code-only aggregate —
                  deliberately not computed; it would reveal a second 40
                  and permanently contaminate the blind pool. KAI DECIDES.
NEW D-NUMBERS     0
DISCOVERY         CLOSED by Kai ruling. Narrow reopen condition only.
NOT AUTHORISED    repair · fixture mutation · design implementation ·
                  candidate generation · D375 · ledger append
HOLD              H2 HOLD remains until Dainius authorises implementation
NEXT              Kai reviews revision 2; final design to Dainius for the
                  implementation-authority decision.
```
