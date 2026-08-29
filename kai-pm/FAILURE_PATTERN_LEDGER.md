# Failure pattern ledger

**AUTHORITATIVE. APPEND-ONLY.** Governed under engineering doctrine rule
49, directed by Dainius on 2026-08-29 and banked in D374.

This file is **not** `kai-pm/ORION_FIELD_NOTES.md`. The field notes are
non-authoritative working memory and create no programme state. This
ledger is the governed record of verified incidents and confirmed
mechanisms, and its `current_control` / `recurred_after_control` fields
are the evidence behind any escalation decision.

**Its purpose is retrieval before reasoning, not archival after it.** A
ledger that is written beautifully and never consulted has improved our
post-mortems and nothing else. The target is a system that becomes
progressively harder to fool in ways that have already fooled it.

---

## Rules that govern this file

* **Append-only.** A correction is a new entry. Nothing is edited to make
  a producer look consistently right (doctrine 23, doctrine 49.7).
* **Incident ≠ mechanism.** Wording similarity, shared file paths and
  routing signatures are **locators** (doctrine 37). A mechanism is
  earned with evidence, never asserted from resemblance.
* **A producer may propose a pattern; a producer may not self-certify
  causal equivalence.** Promotion to `PATTERN_CONFIRMED` is an
  adjudication.
* **Everyone is in the denominator** — operator, Orion, Kai, DeepSeek,
  subagents, scripts, classifiers, gates, and anything autonomous built
  later. No role is exempt because it adjudicates others.
* **Third confirmed occurrence forces escalation** (doctrine 49.6). A
  prose-only control is presumed insufficient at that point unless
  evidence shows otherwise.

## Pattern states

```
INCIDENT_ONLY -> PATTERN_CANDIDATE -> PATTERN_CONFIRMED -> CONTROLLED
                                                  |
                                                  +-> RECURRED_AFTER_CONTROL
                                                  +-> MECHANISED
```

## Control types

| type | meaning |
|---|---|
| `MANUAL` | a written rule a producer must remember to apply |
| `STRUCTURAL` | the failure is made difficult or impossible by construction |
| `MACHINE` | an executing check that can fail, with hostile calibration |

`MANUAL` is the weakest and is the state most likely to be found present
at the moment a failure occurs — 14 of 17 H2 defects were covered by
written doctrine when they happened (doctrine 44).

## Incident schema

Every material incident records at minimum:

```
INCIDENT_ID
date
producer
subject/version
false_or_faulty_output
corrected_output
detection_method
evidence
affected_scope
downstream_impact
mechanism_status
mechanism_id
related_incidents
recurrence_count
stop_signal
current_control
control_type            MANUAL | STRUCTURAL | MACHINE
control_introduced_at
recurred_after_control
owner/stage
status
```

---

# MECHANISMS

## `M-SCOPE-WIDEN` — bounded measurement → unbounded transmitted claim

**State: `PATTERN_CONFIRMED`.** Adjudicated by Kai across three exchanges
on 2026-08-29; directed into doctrine by Dainius the same day.

**Mechanism.** A producer runs a correctly-bounded measurement, obtains a
locally correct result, and then transmits a sentence whose universe,
subject or certainty is wider than what was measured. The failure is not
in the measurement and not in the knowledge. It occurs at the
**transmission** step, typically within one or two lines of a correctly
qualified intermediate.

**Why it is one mechanism and not three coincidences.** In all three
incidents the correct qualifier was present in the producer's own working
output — a table header, a search command, a field label — and was absent
from the prose built on it. The subject matter differed completely
(a source file, a filesystem, a JSON artefact); the reasoning step was
identical.

**Recurrence count: 3.** All within a single working session.

**Controls in force at the time of every occurrence:** doctrine 33
(derivation travels), 35 (no unearned promotion), 46 (memory is a
locator), 47 (source opened ≠ source read), CLAUDE.md R1, R13, R16. All
`RULE_BANKED`, none `CONTROL_OPERATIONALISED`. Doctrine 47 — which
governs exactly this shape for *reading* — was quoted by the producer in
the same message as incident 2.

**Control after escalation:** doctrine rule 48 + CLAUDE.md R17 + six R0
stop-signals. `control_type: MANUAL`, with a machine hook specified
(reconcile extraction count against transmission count; refuse a
consequential claim whose declared scope exceeds its measurement
universe). **The escalation obligation under doctrine 49.6 is discharged
by specifying the machine hook and its owner, NOT by the prose. Until
that hook exists this mechanism remains controlled only manually, and
that is the honest state.**

**Stop-signal (in-flight):** *I searched X and am about to write nowhere
/ none / nothing / all without naming X.* Also: *I am describing a
snippet as though it were the source.* Also: *my table carried a
qualifier and my sentence does not.*

---

### `INC-2026-08-29-01` — truncated instrument output described as source corruption

```
date                    2026-08-29
producer                Orion
subject/version         kai-pm/CODE_AUDIT_CONTINUATION_LOG.md at subject
                        d8aac4d49e6ba997e3eb38062c0917186ee3f197;
                        HOUSE_H2 v1.2 candidate ba2b16d4…de4a
false_or_faulty_output  "the document itself carries a truncated SHA",
                        "with no closing backtick" — reported as OBS-B in a
                        relay to the adjudicator
corrected_output        The source is intact. L164 reads
                        `2d830f25d569baa5ce955dd8d17e8f0744239876` — 40
                        characters, closing backtick present, byte-identical
                        between `git show` and the working tree
                        (sha256 ef81ac69…3f54). Pass A bisects the token at
                        HEAD_BYTES=6000 and emits a 29-character prefix.
detection_method        Kai opened the frozen GitHub source, found the full
                        SHA, refused to accept OBS-B, and demanded a raw
                        reconciliation block before any explanation
evidence                git show d8aac4d4…:kai-pm/CODE_AUDIT_CONTINUATION_LOG.md
                        | nl -ba | sed -n '160,168p'  · token char offsets
                        5971..6011 vs the 6000-byte window · Pass A witness
                        record with truncated=false, evidence_shown=1,
                        evidence_total=1, certainty=VERIFIED
affected_scope          1 occurrence / 1 document / 272-document population
downstream_impact       None reached a verdict — the row is UNKNOWN on all six
                        axes and is not in the 40-row holdout. The false
                        SOURCE claim did reach the adjudicator
mechanism_status        PATTERN_CONFIRMED
mechanism_id            M-SCOPE-WIDEN
related_incidents       INC-2026-08-29-02, INC-2026-08-29-03
recurrence_count        1 of 3
stop_signal             I am describing an instrument's output field as though
                        it were the artefact it came from
current_control         doctrine 48 / CLAUDE.md R17
control_type            MANUAL
control_introduced_at   2026-08-29 (D373)
recurred_after_control  NO — the control postdates the incident
owner/stage             evidence-plane tooling
status                  CLOSED as an incident. The DEFECT it exposed is open:
                        residual D14, BLOCKER on D368 admission
```

**Secondary finding, recorded because it is a distinct defect and not a
restatement of the incident:** the truncation is silent. `truncated` is
`false` and `evidence_shown == evidence_total == 1`, because the envelope
invariant counts **witnesses**, not **bytes of a witness** — the window
cuts below the granularity the invariant guards. The 29-character prefix
then resolves in git, because git resolves unique prefixes, so the
mutilated token is stamped `VERIFIED`. Adjudicated by Kai as residual
**D14** (`bounded extraction window → lexical token bisected → partial
source evidence emitted as complete`), not a new defect class. The class
invariant, as Kai stated it: *a bounded extraction must never silently
terminate inside a recognised evidence token.* **No repair mechanism has
been chosen; none is authorised.**

---

### `INC-2026-08-29-02` — bounded filesystem search transmitted as universal absence

```
date                    2026-08-29
producer                Orion
subject/version         HOUSE_H2 v1.2 candidate package at f196366
false_or_faulty_output  "passA.json was not preserved anywhere on disk. I
                        looked." — and, built on it, "you cannot audit any
                        witness from the frozen package" and "you are checking
                        my re-run against my re-run"
corrected_output        kai-pm/house_in_order_h2_v12/passA.json exists, is
                        tracked at f196366 at 359,173 bytes, and is bound by
                        PACKAGE.sha256 as
                        0ea78096887ddbc60d3af147c1f808ded69faea825968643a470c0969d082d42.
                        `sha256sum -c PACKAGE.sha256` returns all 14 entries OK
detection_method        Kai independently opened the path and quoted the
                        PACKAGE.sha256 line
evidence                The original search was
                        `find /tmp/tmp.6xNl2hBs2V /tmp/claude-0 -name "passA*.json"`
                        — the subject checkout and the scratchpad. THE
                        REPOSITORY WAS NEVER SEARCHED
affected_scope          The claim's stated universe was "disk"; the measured
                        universe was two /tmp paths
downstream_impact       Would have established a false programme narrative that
                        the candidate's evidence was never preserved — the
                        largest single overclaim of the session. Caught before
                        it entered any decision entry
mechanism_status        PATTERN_CONFIRMED
mechanism_id            M-SCOPE-WIDEN
related_incidents       INC-2026-08-29-01, INC-2026-08-29-03
recurrence_count        2 of 3
stop_signal             I searched X and am about to say "nowhere" without
                        naming X
current_control         doctrine 48 / CLAUDE.md R17
control_type            MANUAL
control_introduced_at   2026-08-29 (D373)
recurred_after_control  NO — the control postdates the incident
owner/stage             evidence-plane tooling
status                  CLOSED. Claim withdrawn in full and replaced with the
                        adjudicator's formulation
```

**What survived the correction, and matters independently:** the sidecar
is hash-**bound** by `PACKAGE.sha256` and hash-**referenced** by nothing
the pipeline consumes, and `MANIFEST.sha256` — whose own digest is the
candidate aggregate that seeds the frozen holdout — does not list it.
Recorded by Kai as **I1, HOLDOUT EVIDENCE PRECOMMIT INCOMPLETE**, BLOCKER
on the blind-holdout admission path. A proposed remedy of hashing
`PACKAGE.sha256` into the selection rule was **rejected as circular**,
because `PACKAGE.sha256` lists `h2v12-holdout.json`.

---

### `INC-2026-08-29-03` — artefact-scoped measurement transmitted as package-scoped absence

```
date                    2026-08-29
producer                Orion
subject/version         h2v12-classification.json, sha256 eb50452d…0ad2fd
false_or_faulty_output  "All 316 positive evidence facts carry none — they're
                        bare booleans." The producer's own table said
                        "carrying a witness in the RESULT: 0"; the prose one
                        line later dropped "in the RESULT"
corrected_output        235 of 316 positive evidence facts carry a full
                        nine-field D367 §5 Witness in package-bound passA.json.
                        81 do not. ZERO subjects have no support at all.
                        Breakdown of the 81: MAINTENANCE_OBSERVED 71 (a scalar
                        commits_in_window), CONSUMED_AT_SUBJECT 5 (a reader
                        path list), SELF_ASSERTS_AUTHORITY 4 +
                        SELF_ASSERTS_NON_AUTHORITY 1 (a five-field determining
                        record: polarity, selector, subject, subject_reason,
                        text)
detection_method        Kai read run_h2_v12.py L83-91, established that
                        CITES_COMMIT / CITES_RUN / CARRIES_DATE_STAMP derive
                        directly from Pass A witness buckets, and ordered a
                        bounded per-subject extraction over all 316
evidence                316 subjects · 235 at 9/9 · 81 at 0/9 · uniform failure
                        within each fact type (71/71, 5/5, 4/4, 1/1) ·
                        reconciliation 206+21+5+3 = 235, 71+5+4+1 = 81, sum 316
affected_scope          The claim's stated universe was the candidate package;
                        the measured universe was one JSON artefact
downstream_impact       Overstated a real finding by 235 subjects. Caught before
                        adjudication
mechanism_status        PATTERN_CONFIRMED
mechanism_id            M-SCOPE-WIDEN
related_incidents       INC-2026-08-29-01, INC-2026-08-29-02
recurrence_count        3 of 3 — ESCALATION THRESHOLD REACHED (doctrine 49.6)
stop_signal             My table carried a qualifier and my sentence does not
current_control         doctrine 48 / CLAUDE.md R17
control_type            MANUAL
control_introduced_at   2026-08-29 (D373)
recurred_after_control  NO — the control postdates the incident
owner/stage             evidence-plane tooling
status                  CLOSED. The underlying findings stand and are separately
                        registered
```

**The two findings that survived, kept apart deliberately:**

* **E1 — candidate §5 positive-fact trace noncompliance, 81 of 316**,
  unit `path × positive fact`. Kai's ruling, recorded verbatim in spirit:
  **these are not 81 false evidence facts.** The underlying propositions
  may be true. The failure is that D367 §5 requires every positive
  evidence fact to carry a source-bound witness in the defined schema,
  and a scalar count, a reader-path list and a five-field authority
  record do not satisfy it as emitted.
* **Q1 — qualification denominator failure.** `qualify.py` criterion [5]
  iterates axis cells only and never reaches `evidence_facts`, so the
  frozen §8(8) criterion (*every emitted positive carries the §5 witness
  trace*) is not mechanically established. Demonstrated a second way by
  calibration in temp copies: with `passA.json` **deleted**, and again
  with it **gutted and unbound**, qualification returned `FINDINGS: 0,
  EXIT 0` both times, while `sha256sum -c PACKAGE.sha256` returned exit 1
  on the same trees. No executable file in the candidate references
  `PACKAGE.sha256`.

E1 and Q1 **may later deduplicate under one evidence-chain root cause and
are not to be counted as independent root causes yet.**

---

## `M-PRODUCER-CURATION` — unexpected result → tidied handoff

**State: `PATTERN_CANDIDATE`.** Recorded on 2026-08-29 at Kai's
instruction and **kept distinct from `M-SCOPE-WIDEN` unless and until
evidence shows the mechanisms are the same.** Resemblance is a locator,
not a cause (doctrine 37).

**Proposed mechanism.** A producer executes a declared extraction, the
result contains something awkward — a duplicate, a counterexample, a
malformed record, a row that weakens the story — and the producer omits,
deduplicates, reorders or "cleans" it on its way out. Unlike
`M-SCOPE-WIDEN`, the transmitted claim's *scope* is honest; the
*population* behind it has been silently edited.

**Why it is not yet confirmed:** the H2 session produced the shape as a
recognised temptation and as a design principle Kai articulated, but the
counterexamples that would confirm a recurring causal mechanism in this
programme have not been assembled. Promoting it on resemblance to
`M-SCOPE-WIDEN` would be exactly the error doctrine 49.2 forbids.

**Evidence held against it so far — all negative, i.e. the discipline
held:** the 44-row / 238-witness M3 locator extraction was transmitted
complete, in deterministic original order, with duplicates left in place
(`EMBEDDING_BACKEND_STATE.md` RUN_ID ×2, `RUNTIME_TOPOLOGY_CENSUS.md`
RUN_ID ×2, `TECH_WATCH.md` DATE_STAMP ×3 twice, `WAYPOINTS.md` COMMIT
×2), with the `ed25519` row retained after it was ruled **not** an M3
defect, and with per-chunk coverage reconciliation. Duplicate population:
corpus 5 rows / 9 groups / 11 excess; 44-row locator 4 / 5 / 7.

**Control:** doctrine 48 clauses 5, 6, 7 and 8. `control_type: MANUAL`.

**Stop-signal:** *the predicate returned an awkward row and I am
considering leaving it out.*

---

## Open escalation obligations

| mechanism | recurrences | control now | obligation |
|---|---|---|---|
| `M-SCOPE-WIDEN` | 3 | `MANUAL` (doctrine 48 / R17) | doctrine 49.6 triggered at occurrence 3. Machine hook specified and owned; **not built.** Until it exists, do not report this mechanism as prevented |
| `M-PRODUCER-CURATION` | 0 confirmed | `MANUAL` (doctrine 48.5-8) | earn or refute the mechanism before promoting it |

**Nothing in this file authorises implementation.** The H2 hold is
absolute: no classifier repair, no Pass A repair, no fixture, no new
holdout until a repaired candidate exists.

---

# INCIDENTS NOT ASSIGNED TO A MECHANISM

Appended under D374 authority at Kai's ruling of 2026-08-29, the first
operational enforcement of rule 49 — and it lands on the change that
created rule 49.

**Both carry `mechanism_status: INCIDENT_ONLY` deliberately.** No
mechanism id is invented in order to file them, and neither is merged
into `M-SCOPE-WIDEN` or `M-PRODUCER-CURATION`. They resemble both.
Resemblance is a locator, not causal equivalence (rule 49.2 / rule 37),
and the recurrence evidence that would earn a mechanism does not exist.

**Both were corrected before `fb2892b`, so committed downstream impact is
NONE. That does not make them immaterial.** A near-miss that exposes a
reusable failure mode is learning evidence — which is the whole reason
rule 49 exists rather than a note in a commit message.

---

### `INC-2026-08-29-04` — doctrine grammar collision

```
date                    2026-08-29
producer                Orion
subject/version         kai-pm/ENGINEERING_DOCTRINE.md, while authoring
                        rules 48 and 49 under D373/D374
false_or_faulty_output  Rules 48 and 49 were drafted with numbered
                        SUBORDINATE invariants written as `    N. **text**`.
                        The doctrine parser in
                        scripts/security/check_doctrine_integrity.py is
                        `^\s*(\d+)\.\s+\*\*(.+?)\*\*` — `^\s*` admits leading
                        whitespace — so 19 of my subordinate items were read
                        as top-level RULE IDS
corrected_output        Subordinate items re-delimited as `N)`, which the
                        parser cannot match. Gate returns:
                        "PASS: contiguous, unduplicated, every rule has
                        provenance", inspected 49 rules across 49 provenance
                        entries
detection_method        make policy-check, stage
                        scripts/security/check_doctrine_integrity.py:
                        "FAIL: DUPLICATE: rule id(s) [1,2,3,4,5,6,7,8,9,10]
                        appear more than once. A split or a paste, either way
                        ambiguous"
evidence                Indented numbered-bold items at HEAD: 0.
                        After my draft: 19 (later 21 counting the two items
                        beginning with a backtick rather than bold).
                        Baseline established from an independent tree
                        extracted with `git archive HEAD`
intended_population     21 subordinate items across rules 48 and 49
actual_population       21 — the AUTHORING scope was correct; the defect is
                        that the chosen GRAMMAR collided with the parser's
                        rule-id grammar
reached_committed_branch NO
control_that_caught_it  the doctrine integrity gate itself
                        (control_type MACHINE — it fired, correctly, and named
                        the exact ambiguity)
recurrence_previously_established  NO
mechanism_status        INCIDENT_ONLY
mechanism_id            (none — not invented to file this)
related_incidents       INC-2026-08-29-05 (same authoring session, different
                        failure; NOT asserted to share a mechanism)
related_controls        rule 5 / R5 — a checker's scope is defined by the data
                        it traverses. A document grammar and its parser's
                        grammar are one namespace, and I authored in it
                        without reading the parser first
downstream_impact       NONE
status                  CLOSED as an incident. Remains available as
                        recurrence evidence if a mechanism is later earned
```

**Worth preserving:** this is a case where a `MACHINE` control existed,
fired, named the defect precisely, and cost minutes. It is the
counter-example to `M-SCOPE-WIDEN`, whose control is `MANUAL` and which
recurred three times before an adjudicator caught it. **That contrast is
the argument for rule 49.6, and it should not be lost.**

---

### `INC-2026-08-29-05` — transformation-scope overrun

```
date                    2026-08-29
producer                Orion
subject/version         kai-pm/ENGINEERING_DOCTRINE.md, repairing
                        INC-2026-08-29-04
false_or_faulty_output  The repair applied
                        `re.subn(r"(?m)^(\s+)(\d+)\. ", r"\1\2) ", text)`.
                        In Python `\s` INCLUDES `\n`, so `^` matched at the
                        start of a BLANK line, `\s+` consumed the newline, and
                        the pattern went on to match a COLUMN-0 item on the
                        following line. The transformation crossed line
                        boundaries
corrected_output        Reversed with `(?m)^(\d+)\) ` -> `\1. `, which is
                        line-anchored at column 0 and cannot cross a newline.
                        27 restored, 21 retained. Final diff to HEAD:
                        0 deletions, 3 pure insertion hunks
detection_method        The substitution PRINTED ITS OWN COUNT — 48 — against
                        an intended 21. The discrepancy was visible in the
                        tool output before any further step
evidence                48 substitutions reported · 21 intended ·
                        27 unintended · 27 restored · 27 + 21 = 48 reconciles
intended_population     21 subordinate items in rules 48 and 49
actual_population       48 items. THE 27 UNINTENDED, MEASURED RATHER THAN
                        INFERRED: 26 governed doctrine rules + 1 item in the
                        section-0 procedural list (L80,
                        "1. state what you observed;"). Every one was a
                        column-0 numbered item whose preceding line was blank
reached_committed_branch NO
control_that_caught_it  R4 step 3 — count the population; the instrument
                        printing its own denominator
                        (control_type MANUAL)
recurrence_previously_established  NO
mechanism_status        INCIDENT_ONLY
mechanism_id            (none — not invented to file this)
related_incidents       INC-2026-08-29-04
related_controls        R4 / doctrine 13 — measure the population before
                        applying a rule to it; the count is what exposed the
                        scope overrun.
                        R17 / rule 48 — a transformation whose actual
                        population exceeds its declared one is the mutation-
                        side analogue of a claim wider than its measurement.
                        RECORDED AS RELATED, NOT AS MEMBERSHIP: this is NOT
                        filed as an instance of M-SCOPE-WIDEN or
                        M-PRODUCER-CURATION
downstream_impact       NONE
status                  CLOSED as an incident
```

**The reusable shape, stated without promoting it to a mechanism:** a
transformation is a claim about a population. `\s` in a multiline regex
is a silent scope widener because it crosses the boundary the author is
reasoning in terms of. The defence that worked was not knowing that fact
— it was **making the instrument print its own count and reading it
against the intended one.**

**A correction applied to this very record, under rule 48.** My commit
message and the adjudicator's ruling both say *"27 pre-existing rules"*.
Measured, it is **26 rules + 1 non-rule procedural item**. The false
figure stays here beside its correction (rule 49.7). It changes no
disposition; it is recorded because a record of a scope error must not
itself contain one.

---

## Ledger state after this append

| id | mechanism_status | mechanism | control that caught it | reached branch |
|---|---|---|---|---|
| `INC-2026-08-29-01` | `PATTERN_CONFIRMED` | `M-SCOPE-WIDEN` | adjudicator (Kai) | NO |
| `INC-2026-08-29-02` | `PATTERN_CONFIRMED` | `M-SCOPE-WIDEN` | adjudicator (Kai) | NO |
| `INC-2026-08-29-03` | `PATTERN_CONFIRMED` | `M-SCOPE-WIDEN` | adjudicator (Kai) | NO |
| `INC-2026-08-29-04` | `INCIDENT_ONLY` | none assigned | MACHINE gate | NO |
| `INC-2026-08-29-05` | `INCIDENT_ONLY` | none assigned | MANUAL (R4 count) | NO |

**Observation offered as a candidate signal, NOT as a finding:** the
three occurrences of the confirmed mechanism were all caught by a person
downstream; the two incidents whose controls were a machine gate and an
explicit population count were caught by the producer, before transmission.
One session is not a denominator. Recorded so it can be tested when there
is one, not relied upon now.

---

# MATERIALITY BOUNDARY

Added under D374 authority at Kai's ruling, 2026-08-29. **Without this,
rule 49 becomes noise and the ledger stops being read — which is the one
failure mode that makes the whole control worthless (rule 49.9).**

**Log an incident where a verified error:**

* could alter admission, closure, authority or repair scope;
* materially misstates evidence, identity or coverage;
* could cause an unsafe or incorrect mutation; or
* provides recurrence evidence for a known or candidate mechanism.

**Do not log:** normal hypothesis → evidence → revised hypothesis. That
is engineering, not failure. A changed opinion is not an incident.

**And the distinction that matters most for the adversarial role:**
**being wrong is not the same as doing adversarial reasoning.** A
reviewer whose job is to attack the architecture must attack it
aggressively and will sometimes be wrong; that is the role working, not
failing. What is logged is a **specific, disproven, consequential
technical assertion or proposal** — never *"DeepSeek challenged us"* and
never a rejected challenge as such.

The system we want is: **challenge freely, correct visibly, retain proven
failure mechanisms, become progressively less likely to repeat them.** A
ledger that makes producers challenge less would damage the programme
more than the failures it records.

---

# ADJUDICATOR AND REVIEWER INCIDENTS

Rule 49.8: **everyone is in the denominator, and no role is exempt
because it adjudicates others.** These three entries exist because that
clause is load-bearing rather than decorative. All were directed by Kai
against Kai and DeepSeek after Orion's correction reached him.

---

### `INC-2026-08-29-06` — adjudicator propagated an unverified producer label

```
date                    2026-08-29
producer                KAI
subject/version         kai-pm/ENGINEERING_DOCTRINE.md at fb2892b; the
                        INC-2026-08-29-05 incident record
false_or_faulty_output  "27 pre-existing rules" — repeated in an adjudicator
                        ruling. The unit was not independently checked
corrected_output        27 unintended column-0 numbered items = 26 GOVERNED
                        DOCTRINE RULES + 1 SECTION-0 PROCEDURAL ITEM
                        (L80, "1. state what you observed;"). The COUNT was
                        correct; the LABEL was wide by one
detection_method        Orion applied rule 48 to his own incident record and
                        measured the population against the actual file rather
                        than repeating the figure
evidence                column-0 numbered items preceded by a blank line: 27 ·
                        inside the governed rules section: 26 · outside it: 1
origin_of_the_claim     Orion's commit message for fb2892b. The label
                        originated with Orion; PROPAGATING IT WITHOUT
                        CHECKING THE UNIT IS A SEPARATE PRODUCER EVENT
affected_scope          one label in one ruling
downstream_impact       NONE — no disposition, count, admission or repair
                        scope turned on it
mechanism_status        INCIDENT_ONLY
mechanism_id            (none)
NOT_ASSIGNED_TO         M-SCOPE-WIDEN. Explicitly. Kai did not perform a
                        bounded measurement and then widen it; Kai accepted
                        and forwarded another producer's classification
                        without independent unit verification. Different step,
                        different producer, different failure. Assigning it
                        for resemblance is exactly rule 49.2
related_controls        doctrine 33 (a derived claim travels with its
                        derivation — including its UNIT) · 46 (no cascaded
                        memory authority; another producer's recollection is
                        not my verified premise) · 48 (claim ⊆ measurement) ·
                        49 (the ledger applies to adjudicators)
recurrence_previously_established  see the candidate below
status                  CLOSED as an incident
```

---

### `P-ADJUDICATOR-PROPAGATION` — candidate locator, NOT a mechanism

**State: `PATTERN_CANDIDATE`. Two occurrences preserved. Causal
equivalence NOT asserted.**

**Proposed shape:** *a producer-derived consequential claim is accepted
or propagated by the adjudicator without independent source or unit
verification.*

| # | occurrence | what was propagated | what disproved it |
|---|---|---|---|
| 1 | the D359 incident, recorded in doctrine 46's earned row and banked in D371 | six specific absence claims about D359, produced without opening it. **All six false.** The adjudicator accepted the summary and designed a governance remedy on the false premise | operator scrutiny forced primary-source inspection, before any repository mutation |
| 2 | `INC-2026-08-29-06` | the label "27 pre-existing rules", propagated without checking the unit | the originating producer measured it against the file: 26 rules + 1 procedural item |

**Why this is a locator and not yet a mechanism.** Two occurrences
separated by a day, in different subject matter, with materially
different consequence — one nearly drove a governance remedy, the other
changed nothing. That is enough to **preserve the candidate**; it is not
enough to declare the causal step identical (rule 49.2, doctrine 37).
**Two is not three, and resemblance is not cause.**

**What would earn it:** a third independently confirmed occurrence, or
evidence that the same reasoning step — *accepting a producer's derived
claim as a premise without re-deriving it* — is what produced both.

**If it is ever confirmed, note that rule 49.6 would fire immediately**,
because the manual control (doctrine 46's no-cascaded-memory-authority
clause) was already banked and cited at the time of occurrence 2.

---

### `INC-2026-08-29-07` — reviewer mis-scoped a frozen qualification obligation

```
date                    2026-08-29
producer                DEEPSEEK
subject/version         kai-pm/H2_REPAIR_CONTRACT_D367.md (frozen,
                        0ce5792e…00bb) §8(8); qualify.py at ee4e1824…
false_or_faulty_output  "This is not a bug in qualify.py. qualify.py was
                        scoped to verify the classification, not the evidence."
corrected_output        Frozen §8 is titled "Qualification criteria" and item
                        8 reads verbatim: "every emitted positive carries the
                        §5 witness trace." It is a QUALIFICATION obligation,
                        not a packaging one. A qualification implementation
                        that cannot observe that denominator fails to
                        establish its frozen criterion
detection_method        Kai rejected the framing and had the frozen contract
                        opened and §8 read to the claim boundary; Orion
                        returned L277 verbatim and measured the denominator
                        (343 non-abstention verdicts + 316 positive evidence
                        facts = 659 §5 subjects; qualify.py [5] reaches only
                        the 343)
impact_if_accepted      repair responsibility would have been displaced
                        wholly into packaging, and Q1 — the qualification
                        coverage defect — could have been missed entirely
downstream_impact       NONE — rejected before it entered any ruling
mechanism_status        INCIDENT_ONLY
mechanism_id            (none — not invented for a single occurrence)
related_controls        doctrine 14 · 46 · 47 — the governing frozen contract
                        is opened, and read to the claim boundary, BEFORE
                        responsibility for a defect is assigned
status                  CLOSED as an incident
```

**Recorded with the role protected.** The logged item is the specific
disproven technical assertion, not the act of challenging. DeepSeek's
adversarial review in the same exchange is what widened Q1 correctly and
forced the §5 denominator to be measured at all.

---

### `INC-2026-08-29-08` — reviewer proposed a circular holdout identity

```
date                    2026-08-29
producer                DEEPSEEK
subject/version         holdout.py at 4c5c06ac… ; PACKAGE.sha256 ;
                        MANIFEST.sha256 (candidate ba2b16d4…de4a)
false_or_faulty_output  proposed remedy for I1: make blind-holdout selection
                        incorporate / hash PACKAGE.sha256
corrected_output        REJECTED AS CIRCULAR. PACKAGE.sha256 lists
                        h2v12-holdout.json (26168dbb…362046) among its 14
                        entries, and holdout.py L47-50 derives the sample from
                        sha256(--manifest). Feeding the package inventory in
                        yields:
                            holdout -> package identity -> holdout selection
                        The correct requirement remains a PRE-HOLDOUT EVIDENCE
                        IDENTITY binding instrument, ontology/envelope/
                        qualification bytes, the classification result, the
                        evidence sidecar or its root digest, subject
                        commit/tree and dependency identity — and EXCLUDING
                        the as-yet-unselected holdout
detection_method        Kai identified the circularity; Orion verified from
                        the artefacts that PACKAGE.sha256 does list
                        h2v12-holdout.json and that holdout.py consumes
                        sha256(MANIFEST.sha256)
impact_if_accepted      would have reintroduced precisely the self-dependence
                        D367 §9 was frozen to prevent
downstream_impact       NONE — rejected before any design or mutation
mechanism_status        INCIDENT_ONLY
mechanism_id            (none)
related_controls        circularity / self-observation checks (doctrine 32,
                        R9, I-8 — an instrument must not observe itself) ·
                        rule 46 primary-source inspection
status                  CLOSED as an incident
```

**Again, the role is not the incident.** DeepSeek identified the real
structural gap in the blind-sample chain of custody, which is why I1 was
upgraded to BLOCKER at all. A wrong counterproposal from an adversarial
reviewer is the role functioning; the logged item is the specific
disproven proposal.

---

## Ledger state after this append

| id | producer | mechanism_status | assigned mechanism | reached branch / ruling |
|---|---|---|---|---|
| `INC-2026-08-29-01` | Orion | `PATTERN_CONFIRMED` | `M-SCOPE-WIDEN` | NO |
| `INC-2026-08-29-02` | Orion | `PATTERN_CONFIRMED` | `M-SCOPE-WIDEN` | NO |
| `INC-2026-08-29-03` | Orion | `PATTERN_CONFIRMED` | `M-SCOPE-WIDEN` | NO |
| `INC-2026-08-29-04` | Orion | `INCIDENT_ONLY` | none | NO |
| `INC-2026-08-29-05` | Orion | `INCIDENT_ONLY` | none | NO |
| `INC-2026-08-29-06` | **Kai** | `INCIDENT_ONLY` | none — locator only | reached a ruling; no disposition turned on it |
| `INC-2026-08-29-07` | **DeepSeek** | `INCIDENT_ONLY` | none | NO |
| `INC-2026-08-29-08` | **DeepSeek** | `INCIDENT_ONLY` | none | NO |

| candidate | occurrences | state |
|---|---|---|
| `M-PRODUCER-CURATION` | 0 confirmed | `PATTERN_CANDIDATE` |
| `P-ADJUDICATOR-PROPAGATION` | 2 preserved | `PATTERN_CANDIDATE` |

**Producers represented: Orion 5 · Kai 1 · DeepSeek 2.** Recorded because
a ledger containing only one producer's failures would be evidence about
who writes the ledger, not about the system (rule 49.8).
