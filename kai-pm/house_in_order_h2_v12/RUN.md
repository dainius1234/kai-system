# HOUSE-IN-ORDER-H2-CLASSIFIER v1.2

Built under Dainius's authorisation, strictly within **frozen D367**
(`kai-pm/H2_REPAIR_CONTRACT_D367.md`, sha256
`0ce5792e…00bb`). The contract was **not** altered during
implementation. HOUSE_H2 v1.0 and v1.1 are untouched; Census v1.1
(`eb7aad7c…fa0e`) is untouched and consumed read-only.

> **STATUS: CANDIDATE. NOT FROZEN. NOT ADMITTED TO HOUSE_H3.**
> Admission is a governing decision, never a test result. This package
> supplies evidence and approves nothing about itself.

**Candidate aggregate** (instrument + result):
`ba2b16d400aafd2b694890149bbaae1d1369d3771c25ce809d2f752d5248de4a`

---

## The one architectural change

Seventeen registered defect classes were one shape: **an observation
recorded at one scope, reported at a wider one.** Remembering not to do
that failed seventeen times, so it is now mechanism.

Pass A no longer emits booleans. `has_sha=True` threw away the token,
its position, its kind and its scope — everything a verdict needs in
order not to over-claim — and six of the seventeen defects live in that
single design decision. v1.2 emits **`Witness` objects** carrying the
nine fields of D367 §5, and every verdict is built through
`envelope.claim()`, which refuses to widen subject, scope, polarity,
certainty or temporal applicability without a **declared** promotion:

```python
SCOPE_ORDER     = ("SPAN", "SECTION", "WHOLE_FILE")
CERTAINTY_ORDER = ("ASSERTED", "OBSERVED", "VERIFIED")
DECLARED_PROMOTIONS = {("SPAN", "WHOLE_FILE"): "DOCUMENT_LEVEL_BINDING: …"}
```

Each dimension is **ordered**, so "wider" is *computed*, not judged. A
rule requiring judgement at the call site gets applied to the case in
front of the author and to nothing else — which is how R5 was breached
four times inside one instrument. A promotion absent from the table
cannot be performed at all, and an invented justification string is
refused.

## Running it

```sh
python3 passa.py --subject-repo <exact checkout> \
                 --history-repo <NON-SHALLOW repo> \
                 --subject <40-hex> \
                 --census-package <frozen Census v1.1> --out passA.json

python3 run_h2_v12.py --subject-repo <...> --passa passA.json \
                 --out h2v12-classification.json

python3 qualify.py --result h2v12-classification.json --manifest MANIFEST.sha256
python3 cal_fixtures.py
python3 holdout.py --result h2v12-classification.json --manifest MANIFEST.sha256 \
                 --out h2v12-holdout.json
```

The history source **must be non-shallow**. It does not fail on these
queries — it returns its graft boundary as a plausible date
(`TECH_WATCH.md`: `2026-08-05` shallow, `2026-07-24` true), so Pass A
aborts rather than measuring.

## Results — subject `d8aac4d4`, tree `3abc9e9d`, 272 documents

| axis | positive | UNKNOWN | v1.1 positive |
|---|---|---|---|
| `LIFECYCLE` | 11 `HISTORICAL` | 261 | 13 |
| `FUNCTION` | 5 `MARKER` | 267 | 222 |
| `AUTHORITY` | 0 | 272 | 0 |
| `GENERATION` | 0 | 272 | 0 |
| `VALIDITY` | 161 (155 `TIME_BOUND`, 6 `EXACT_SNAPSHOT`) | 111 | 56 |
| `SCOPE` | 166 `WHOLE_FILE` | 106 | 272 |

**Independent corroboration.** `EXACT_SNAPSHOT` is **exactly the 6
documents Kai adjudicated from source** in D364 — no extras, no misses.
That set was derived by a human reading documents; this one by a rule
derived from the contract. Neither could excuse the other.

**`FUNCTION` falls from 222 to 5, and that is correction E working.**
Path and title are created as part of the same document by the same
author, so `PATH says audit + TITLE says audit` is one source counted
twice. Self-description now earns the evidence fact `NOMINAL_FUNCTION`
(207 documents) and the verdict abstains. Only the objective `MARKER`
case — byte count and path role — earns `FUNCTION` at H2.

**`VALIDITY` rises from 56 to 161, which needs saying plainly.** v1.1
required `has_date AND present_tense`; the defective `present_tense`
conjunct was suppressing 133 documents that *do* carry a document-level
date stamp. v1.1 was not more careful, it was accidentally narrower via
a broken conjunct. 155 of those stamps sit in a header block, are
unique in their document, and survive the history check.

**`SCOPE` falls from 272 to 166** because it is now earned rather than
defaulted.

Evidence facts (never verdicts): `CARRIES_DATE_STAMP` 206 ·
`MAINTENANCE_OBSERVED` 71 · `CITES_COMMIT` 21 · `CONSUMED_AT_SUBJECT` 5 ·
`BINDING_CONTRADICTION` 5 · `SELF_ASSERTS_AUTHORITY` 4 · `CITES_RUN` 3 ·
`SELF_ASSERTS_NON_AUTHORITY` 1.

## Qualification

* **78 hostile fixtures, 0 failures.** One fail-old/pass-new pair per
  registered class D1–D17. **The fail-old half runs the actual committed
  v1.1 in a subprocess** — its modules import each other by bare name and
  three collide with v1.2's, so an in-process import would let one
  version shadow the other and a contaminated fail-old proves nothing.
* **Three denominators, from three different places.** The axis set (is
  the governing invariant satisfied?), the alphabet (dispositions,
  reachability, forbidden values), and **the output** (did any row emit a
  value the ontology has never heard of?). Each is blind in a direction
  the others are not — D17 existed because v1.1 had only the second.
* **Removal calibration runs in the qualification itself**, not only in
  the fixtures: `UNKNOWN` is removed from each axis in turn and the
  invariant must fire. A gate that has never been shown to fail is an
  untested instrument, and v1.1 reported 40/40 green the entire time
  seventeen defects were live.
* **Runtime module identity** — all five modules resolve under the
  candidate directory with source bytes matching the manifest.
* **Fresh reproduction executed** from an unrelated directory: all 272
  rows, every tally and the admission contract byte-identical.
* 0 qualification findings.

## Two changes made outside the letter of the contract, both declared

**1. A uniqueness requirement on binding predicates.** D367 §6 requires
a binding "whose subject is the document as a whole". A predicate
appearing *twice* cannot have the whole document as its subject in both
places — `kai-pm/WAYPOINTS.md` carries `**Date:**` per waypoint record,
and its L79 entry would otherwise have bound the whole file to one
entry's date. This discriminates **1 document of 162**; the denominator
is stated because a rule justified by one instance deserves suspicion.
It implements the contract's wording rather than extending it, and it
fails toward abstention.

**2. `LIFECYCLE` was ported, not redesigned.** It was qualified under
D361–D363 and was not in the repair scope — but its *inputs* changed
underneath it, since `has_sha` no longer exists. v1.1's two `HISTORICAL`
rules are carried onto the repaired evidence, and the snapshot rule is
now **strictly stronger**: a commit witness verified by resolution and
bound at document scope, where v1.1 accepted any hex-shaped token
anywhere. Silently losing 13 verdicts would have been a scope change I
had no authority to make.

**11 of 13 `HISTORICAL` retained. The 2 dropped are justified by the
repaired evidence:** `CODE_AUDIT_BATCH_HMAC_ROTATION_DRILL.md` earned it
from `1700000000` — a **unix timestamp**, named as one in its own line —
and `CODE_AUDIT_CONTINUATION_LOG.md` from a real commit cited
**mid-document at SPAN scope**.

## A defect I introduced and caught

The first draft of `bind_claims` imposed a 6000-byte window by analogy
with Pass A. v1.1 had no such window on that extractor. The window
silently destroyed two authority claims **Kai had adjudicated CORRECT** —
`PHASE1_READINESS.md`'s self-declaration at byte 18,877 and
`DECISIONS.md`'s at byte 287,981. Both fell to `NO_SELF_CLAIM`.

It was caught by asserting the five correct rows unchanged, which is the
only reason it is not in this candidate. Those assertions are now
permanent fixtures.

## Blind holdout — 40 documents, selection and evaluation both frozen

Selected by the D367 §9 rule under the candidate aggregate, which did not
exist when the rule was frozen:

```
sha256("H2FINAL-D367:" + "86a1399e…d6ef" + ":" + CANDIDATE_AGGREGATE + ":" + path)
```

`MANIFEST.sha256` covers the **instrument and its result**, fixing the
aggregate *before* the holdout exists; `PACKAGE.sha256` is the full
inventory including the holdout. Without that separation the selection
would depend on its own output.

**5 rows overlap the D363 24-row holdout, and `SEQUENCE.md` — the only
row of that set ever adjudicated — is not among them.** Contamination is
zero.

**Not self-adjudicated. No agreement figure is computed anywhere in this
package**, and any figure I computed would carry no admission weight
(D367 §10).

## Utility, reported and never optimised

`UNKNOWN` dominates four of six axes. That is the measured truth, not a
target and not a failure. Per D367 §11 and Kai's stop rule, two separate
gates follow: **qualification** asks whether the instrument is truthful;
a later **operator decision** asks whether the truthful instrument is
discriminating enough to feed HOUSE_H3. If H2 proves to be principally an
evidence-and-abstention instrument, that is to be reported rather than
engineered away.

## Restrictions carried in the artefact

* `LIFECYCLE=ACTIVE` is `H2_NOT_EARNABLE`;
* `AUTHORITY` states are `DEFERRED_TO_H3`; `GENERATION` verdicts and
  `SCOPE` region overrides are `H2_NOT_EARNABLE` — each proven by an
  injection known-positive with its known-negative;
* `UNKNOWN`/`UNMEASURED` are abstentions and may **never** be used as
  negative evidence or an exclusion criterion (D340 §7 / D358);
* history-derived facts are void outside the declared window.

## Known confound, stated so it is not misattributed

v1.1 and v1.2 both consume **frozen Census v1.1**, so the Census is not a
confound between them. The v1.0↔v1.1 confound recorded in D361 stands
unchanged and is not resolved here.
