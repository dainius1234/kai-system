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

---

# QUEUED INCIDENTS, BANKED 2026-08-29

Kai directed these three to be held until the repair matrix closed, then
**reversed that instruction**: an incident that exists only in chat is
not durable engineering memory, and rule 49.1 says material errors are
logged, not merely corrected. Banked under existing D374 authority.
Ledger only. No D-number. No doctrine change. No H2 artefact touched.

All three `INCIDENT_ONLY`. No mechanism id invented for any of them.

---

### `INC-2026-08-29-09` — a control asserted as a current output, never checked

```
date                    2026-08-29
producer                Orion
subject/version         the M1 row of the repair-spec matrix; committed
                        h2v12-classification.json eb50452d…0ad2fd
false_or_faulty_output  "same-family positive: `Last updated:` on
                        kai-pm/STATUS.md L3 MUST still earn it
                        [TIME_BOUND]" — offered as the control proving an
                        M1 repair had not over-fired
corrected_output        kai-pm/STATUS.md emits UNKNOWN ON ALL SIX AXES.
                        Its VALIDITY observed reads "witness kinds
                        present: ['DATE']"; its SCOPE observed reads "no
                        witness whose applicability is the document as a
                        whole". Its date witness is SPAN. It is row [35]
                        of the 44-row M3 locator and appears in Kai's own
                        list of 29 M3-AFFECTED documents. IT IS NOT A
                        TIME_BOUND POSITIVE AND NEVER WAS
detection_method        Kai rejected the control on SEMANTICS — that a
                        metadata label does not earn temporal validity
                        under D367. Orion then opened the artefact and
                        found the underlying FACTUAL error, which is the
                        worse of the two
evidence                the six committed cells for kai-pm/STATUS.md;
                        locator membership True
affected_scope          one control in one matrix row
downstream_impact       NONE — the matrix was rejected before any repair
                        or fixture was designed against it
mechanism_status        INCIDENT_ONLY
mechanism_id            (none)
NOT_ASSIGNED_TO         M-SCOPE-WIDEN. That mechanism is measure-then-
                        widen: a correct bounded measurement transmitted
                        at a wider scope. THIS IS ASSERT-WITHOUT-
                        MEASURING — no measurement was taken at all.
                        Adjacent, and NOT demonstrated identical.
                        Assigning it on resemblance is rule 49.2
related_controls        R1 (do not assert what you have not run) ·
                        R17 / rule 48 · doctrine 46
why_it_matters          the assertion was label-shaped: STATUS.md carries
                        `Last updated:`, the predicate family under
                        adjudication, and the producer inferred the
                        verdict from the label instead of reading the
                        cell. The M1 repair must not degenerate into a
                        label whitelist — and the matrix specifying that
                        repair contained a label-shaped assumption
status                  CLOSED as an incident
```

---

### `INC-2026-08-29-10` — reproducibility reported as independence

```
date                    2026-08-29
producer                Orion
subject/version         frozen Census v1.1, aggregate eb7aad7c…fa0e;
                        subject d8aac4d4… / tree 3abc9e9d…
false_or_faulty_output  "INDEPENDENT COUNT CHECK, not a restatement:
                        frozen census-worldA disposition_tally
                        RESOLVED_READ : 8 / my live re-run : 8"
corrected_output        NOT INDEPENDENT. The re-run imported the Census's
                        OWN frozen opscan/docgraph/claims modules, so
                        both legs trace to one authority and neither can
                        excuse the other. Correct label: SAME-INSTRUMENT
                        REPRODUCIBILITY CHECK AGAINST THE IDENTICAL
                        IMMUTABLE SUBJECT
detection_method        Kai rejected the wording and supplied the actual
                        independent leg himself — opening the subject
                        sources directly, without the Census parser, and
                        confirming the reads at auto_changelog.py L55 and
                        L112 and the other six
evidence                the re-run's own import line:
                        `import docgraph as G, opscan as O, claims as C`
                        from ../house_in_order_census_v11
affected_scope          one label on one result
downstream_impact       NONE — corrected before the selector ruling
                        depended on it. The 8==8 result itself is sound;
                        only its independence status was wrong
mechanism_status        INCIDENT_ONLY
mechanism_id            (none)
related_controls        doctrine 39 — NAME THE AUTHORITY BEHIND EVERY
                        INDEPENDENCE CLAIM. Rule 33's
                        `independence_status` field exists for exactly
                        this. CLAUDE.md R0 carries the tell: "I am
                        writing 'independent' — name the authority behind
                        each leg"
                        THE CONTROL EXISTED, WAS BANKED, AND DID NOT
                        FIRE. Recorded because that is the kind of fact
                        rule 49.5 asks us to notice
CONSEQUENT_SCOPE_LIMIT  the wider 1186 / 1143 / 41-duplicate-group result
                        from the same re-run is likewise ORION SAME-
                        INSTRUMENT MEASUREMENT, not a hash-bound property
                        of census-worldA.json, which records the 1186
                        denominator and the RESOLVED_READ count but does
                        NOT preserve the raw (src,line) population. It is
                        not load-bearing for E1
status                  CLOSED as an incident
```

---

### `INC-2026-08-29-11` — adjudicator supported a subset ruling with corpus-wide evidence

```
date                    2026-08-29
producer                KAI
subject/version         the ten non-M1 candidate TIME_BOUND rows;
                        committed h2v12-classification.json
false_or_faulty_output  cited "the committed classification contains
                        `Last updated` cases whose own history
                        contradicts the claimed date" as support for
                        rejecting or protecting those ten rows
corrected_output        TRUE OF THE CORPUS, NOT OF THE SUBSET. None of
                        the ten carries a binding contradiction — all ten
                        have binding_contradiction = null. The five
                        contradiction rows corpus-wide are
                        docs/PROJECT_BACKLOG.md 32d ·
                        docs/unfair_advantages.md 90d ·
                        kai-pm/ORION_FIELD_NOTES.md 7d ·
                        kai-pm/STUBS_AND_PLACEHOLDERS.md 3d ·
                        kai-pm/UH_PROGRESS_TRACKER.md 1d — and ALL FIVE
                        ALREADY EMIT VALIDITY=UNKNOWN. The contradiction
                        check is working; the contradicting documents are
                        not among the positives
                        Correct statement: 5 contradiction rows exist
                        corpus-wide; 0 occur in this 10-row subset
detection_method        Orion measured binding_contradiction across both
                        populations rather than accepting the premise,
                        and reported the mismatch to the adjudicator
affected_scope          one evidential leg of one ruling
downstream_impact       NONE. THE SEMANTIC CONCLUSION SURVIVED ON ITS
                        OWN — a review event is not a validity binding,
                        which stands without the contradiction evidence.
                        Only the supporting leg was out of scope
mechanism_status        INCIDENT_ONLY
mechanism_id            (none)
related_controls        rule 48 (CLAIM_SCOPE ⊆ MEASURED_SCOPE) applied to
                        a SUBSET rather than a search universe · rule 33
                        (the derivation, including its population,
                        travels with the claim)
note                    directed into the ledger by Kai against Kai.
                        Rule 49.8: no role is exempt because it
                        adjudicates others
status                  CLOSED as an incident
```

---

## Ledger state after this append

| id | producer | mechanism_status | assigned mechanism |
|---|---|---|---|
| `INC-2026-08-29-01` | Orion | `PATTERN_CONFIRMED` | `M-SCOPE-WIDEN` |
| `INC-2026-08-29-02` | Orion | `PATTERN_CONFIRMED` | `M-SCOPE-WIDEN` |
| `INC-2026-08-29-03` | Orion | `PATTERN_CONFIRMED` | `M-SCOPE-WIDEN` |
| `INC-2026-08-29-04` | Orion | `INCIDENT_ONLY` | none |
| `INC-2026-08-29-05` | Orion | `INCIDENT_ONLY` | none |
| `INC-2026-08-29-06` | Kai | `INCIDENT_ONLY` | none — locator only |
| `INC-2026-08-29-07` | DeepSeek | `INCIDENT_ONLY` | none |
| `INC-2026-08-29-08` | DeepSeek | `INCIDENT_ONLY` | none |
| `INC-2026-08-29-09` | Orion | `INCIDENT_ONLY` | none |
| `INC-2026-08-29-10` | Orion | `INCIDENT_ONLY` | none |
| `INC-2026-08-29-11` | Kai | `INCIDENT_ONLY` | none |

**Producers: Orion 7 · Kai 2 · DeepSeek 2.** The counts carry no fairness
or bias inference; they are the currently recorded population and nothing
more.

**A withdrawal recorded here rather than elsewhere.** Orion reported the
two-witness line at `AUTHORITY_ONTOLOGY.md:L2` as a possible
line-inheritance false positive. Kai opened the source and adjudicated
both tokens — commit `9d15bcd` and tree `627104d6` — as legitimate
components of one whole-document subject declaration:
`Subject: QUALIFICATION_SUBJECT 9d15bcd / tree 627104d6`. **The concern
is withdrawn; no defect is established there, and no `167 + 1 exception`
regression contract is to be created.** The per-line implementation
remains structurally risky as an unadjudicated property. That risk is not
an incident and is not logged as one.

---

# APPEND 2026-08-29 — acquisition-side contamination

Authorised by Kai under existing D374 ledger authority. **No D-number, no
doctrine edit, no `CLAUDE.md` edit, no H2 artefact change.** Banked
*before* M3 resumes, because this file's stated purpose is *retrieval
before reasoning, not archival after it* — the failure mode has to be
available before the next contamination-sensitive search, not after.

## `M-QUERY-OVERREACH` — lower-information question, higher-information instrument

**State: `PATTERN_CANDIDATE`.** Named by Orion, adjudicated by Kai on
2026-08-29. **Explicitly NOT `PATTERN_CONFIRMED`: one material incident.
The recurrence threshold is not met and no general pattern is proven.**

**Proposed mechanism.** A query is run to answer a *lower-information*
question — does this exist, how many are there, is it non-empty — but the
instrument chosen returns *higher-information* content, in a context
where **merely seeing that content changes or contaminates the
experiment**. The failure is complete at the moment of reading. There is
no later step at which it can be caught, and no transmission is required
for the damage to be done.

**Why it is kept distinct from `M-SCOPE-WIDEN`.** That mechanism fails at
**TRANSMISSION**: a correctly bounded measurement is described in a
sentence wider than the measurement. This one fails at **ACQUISITION**:
the measurement itself was wider than the question, and the producer's
own knowledge is the thing contaminated. The two are adjacent and will
look alike in a post-mortem. Resemblance is a locator, not a cause
(doctrine 37), and promoting this into the confirmed pattern on
similarity would be the error doctrine 49.2 forbids.

**Stop-signal (in-flight).**

> *I only need to know WHETHER something exists. Can the query I am about
> to run reveal WHAT it contains?*

**Manual control.** Before a contamination-sensitive search, classify the
information need as `EXISTENCE` / `METADATA` / `CONTENT`, and use the
least-revealing operation that answers it. If `EXISTENCE` is sufficient,
do not retrieve substantive content.

**`control_type: MANUAL`.** Kai's ruling is explicit that this is not
promoted into doctrine or `CLAUDE.md` in this commit, and that **it is
not claimed to prevent recurrence.** A future recurrence is evidence
against the adequacy of this manual control, not against the producer
alone.

---

### `INC-2026-08-30-12` — an existence question answered with the answer sheet

> **Id/date note, recorded rather than silently corrected.** Kai
> allocated the id `INC-2026-08-30-12`. The verified container date at
> banking is **2026-08-29**, and every prior id encodes the incident
> date. The allocated id is used verbatim because id allocation is the
> adjudicator's, and the discrepancy is recorded here instead of being
> resolved unilaterally.

```
INCIDENT_ID             INC-2026-08-30-12
date                    2026-08-29 (see id/date note above)
producer                Orion
subject/version         Step-2 M3 blind/reference calibration at 0f3da09
                        STEP2_M3_RECORD_MANIFEST.tsv
                        manifest sha256
                        8343760003c18abd7a1787d697cdab861c2571285dfd9a2c81005f75ecb2efd0
false_or_faulty_output  A query intended to answer the EXISTENCE question
                        "does the historical per-record M3 answer set
                        exist on disk?" was run with a content-extracting
                        regex over past adjudication text. It returned
                        substantive adjudication content. Reading it
                        exposed approximately 15-20 expected rows,
                        INCLUDING THE SOLE WHOLE_FILE -> SPAN CORRECTION,
                        and therefore compromised the intended
                        builder-blind Step-2 M3 comparison.
corrected_output        The original 48-row blind claim is PERMANENTLY
                        WITHDRAWN. The 48 are retained only as
                        REFERENCE / REGRESSION CORRECTNESS EVIDENCE WITH
                        ZERO BLIND WEIGHT. No further historical-ruling
                        extraction is authorised. The final D367 section
                        9 candidate-derived blind 40 remains untouched.
detection_method        Producer self-disclosure, before any M3
                        implementation and before any evaluation
evidence                HEAD 0f3da09 with no M3 code; manifest sha256
                        unchanged; the exposing search and its output are
                        in the session record
affected_scope          The intermediate Step-2 M3 blind control. One
                        control, not a candidate artefact
downstream_impact       The M3 blind-control design lost its blind
                        status. NO M3 code had been written. NO result
                        had been produced. NO candidate verdict changed.
                        The final D367 independent holdout is intact
mechanism_status        PATTERN_CANDIDATE
mechanism_id            M-QUERY-OVERREACH
related_incidents       none confirmed. M-SCOPE-WIDEN is ADJACENT, not
                        assigned: that one fails at transmission, this at
                        acquisition
recurrence_count        1 of 1. Threshold for escalation NOT met
stop_signal             I only need to know WHETHER something exists --
                        can this query reveal WHAT it contains?
current_control         classify the need EXISTENCE / METADATA / CONTENT
                        and use the least-revealing operation
control_type            MANUAL
control_introduced_at   2026-08-29, this append
recurred_after_control  NO -- the control postdates the incident
owner/stage             Step-2 M3 calibration
status                  CLOSED as an incident. The CONTROL it damaged is
                        withdrawn, not repaired: builder blindness for
                        the 48 cannot be restored
```

**What had no control at all.** Nothing in the doctrine, in `CLAUDE.md`
or in R0 said *before searching, ask whether SEEING the answer is itself
the harm.* Every banked rule to date governs what is measured, how far it
is read, and how it is described. **None governs whether the act of
looking is itself destructive.** That gap is the finding; the incident is
the instance.

## Ledger state after this append

| id | producer | mechanism_status | assigned mechanism |
|---|---|---|---|
| `INC-2026-08-29-01` … `-03` | Orion | `PATTERN_CONFIRMED` | `M-SCOPE-WIDEN` |
| `INC-2026-08-29-04` `-05` | Orion | `INCIDENT_ONLY` | none |
| `INC-2026-08-29-06` | Kai | `INCIDENT_ONLY` | none — locator only |
| `INC-2026-08-29-07` `-08` | DeepSeek | `INCIDENT_ONLY` | none |
| `INC-2026-08-29-09` `-10` | Orion | `INCIDENT_ONLY` | none |
| `INC-2026-08-29-11` | Kai | `INCIDENT_ONLY` | none |
| `INC-2026-08-30-12` | Orion | `PATTERN_CANDIDATE` | `M-QUERY-OVERREACH` |

**Producers: Orion 8 · Kai 2 · DeepSeek 2.** The counts carry no fairness
or bias inference; they are the currently recorded population and nothing
more.

**Mechanisms: `M-SCOPE-WIDEN` `PATTERN_CONFIRMED` · `M-PRODUCER-CURATION`
`PATTERN_CANDIDATE` · `P-ADJUDICATOR-PROPAGATION` `PATTERN_CANDIDATE` ·
`M-QUERY-OVERREACH` `PATTERN_CANDIDATE`.** Three of the four are
candidates. None is `CONTROL_OPERATIONALISED`.

---

# APPEND 2026-09-12 — a fabricated admissibility input, and a common-source
#                      independence claim about its repair

Two incidents, one producer, one session. The first is the defect the repair
addressed; the second is a false claim made about the evidence for that
repair, caught by external review before this append and before merge.

Both are PATTERN_CANDIDATE with no mechanism assigned. Resemblance is a
locator; equivalence has not been earned.


### `INC-2026-09-12-13` — a gate's refusal branch made unreachable by a
###                       fabricated input, and tested through a different door

```
INCIDENT_ID             INC-2026-09-12-13
date                    2026-09-12 (defect present since the gate was
                        authored; first measured consequence 2026-09-10)
producer                Orion — the gate, its floors file, its test suite and
                        the workflow that calls it were authored together in
                        one stint (DECISIONS.md:3588). The same producer wrote
                        the check and the evidence that it worked
subject/version         scripts/security/check_assertion_floors.py at 1009f31
                        .github/workflows/unified-hunter.yml at 1009f31

false_or_faulty_output  On 1009f31 `make test-uh` exited 2, halting at
                        prerequisite 61 of 78 (Makefile:970,
                        test-container-proof-harness). The gate's report step
                        published:
                          "missing": ["Depends-On Readiness Tests",
                                      "Healthcheck Runnable Tests",
                                      "Policy Loader Tests",
                                      "Shipped Package Deps Tests",
                                      "Suite Floor Tests"]
                        Those five are targets 74-78. They did not erode; they
                        never started. A run that STOPPED was published as a
                        surface that ERODED — two different problems with two
                        different fixes

corrected_output        `--from-log PATH` now derives admissibility from the
                        bound `PATH.status`. A non-zero aggregate status
                        refuses adjudication BEFORE floor comparison and omits
                        the semantic finding buckets entirely — fallen,
                        missing, unrecorded and drifted are absent from the
                        payload rather than empty, because an empty list
                        asserts an adjudicated finding of none. A zero,
                        admissible status still executes the erosion
                        comparison and still exposes the five never-run suites
                        as `missing`. The refusal gates the detector; it does
                        not switch it off

mechanism (proposed,    `--from-log` bound the aggregate's exit status to the
 NOT self-certified)    literal 0:
                            output, status = args.from_log.read_text(...), 0
                        so the `status != 0` refusal branch was UNREACHABLE on
                        the only path CI used. That branch's prose had never
                        executed in CI.
                        The second half is why it survived a suite written to
                        stop this gate eroding:
                        test_a_failing_aggregate_run_is_reported_as_itself
                        monkey-patched `run_suites`, which always carried the
                        true status. THE TEST AND THE CI CALL SITE USED
                        DIFFERENT DOORS. The gate's own docstring lists five
                        checks that quietly stopped checking and three rules
                        written to stop it becoming case 6. This is case 6,
                        and those rules did not see it because they guard the
                        comparison, not the admissibility of its input

detection_method        Investigation of an unrelated CI failure (RC-7),
                        during which the report-step payload was read rather
                        than assumed. Not detected by any gate, test or review

evidence                EVIDENCE INDEPENDENCE — STATED EXPLICITLY.
                        Runner step conclusion and the .status sidecar agreed
                        on aggregate status 2. These are TWO MANIFESTATIONS OF
                        THE SAME status-capture mechanism, not independent
                        producers: both descend from one shell `status`
                        variable, which feeds the sidecar and `exit "$status"`
                        alike. Their agreement is arithmetic, not
                        corroboration. See INC-2026-09-12-14.

                        Producer correctness is supported SEPARATELY, by
                        controlled calibration of the capture itself —
                        removing `pipefail` against a failing subject yields
                        sidecar 0 AND a green step, exposing the common-mode
                        dependency; removing `|| status=$?` yields no sidecar
                        at all — and by subject-failure evidence present in
                        the log body upstream of the status variable:
                          setpriv: setresuid failed: Operation not permitted
                          Container proof: 3 passed, 13 failed
                          EXIT GATE: FAIL — status remains UNKNOWN
                          subprocess.TimeoutExpired: ...
                          make: *** [Makefile:970: ...] Error 1
                        THE PRODUCTION GATE DOES NOT CONSUME THOSE WITNESSES.
                        They corroborate producer truth retrospectively for
                        evidential review; they are NOT an operating
                        cross-check.

                        The status 2/0 injection pair on one log demonstrates
                        a DIFFERENT property — consumer gating with the
                        erosion detector retained. It does not validate
                        producer status truth. `"Assertion Floor Tests": 80`
                        in the CI counts shows the current gate code executed
                        on the runner; it does not authenticate the status.

                        1009f31: run 34491294245 / job 102918401237 — the five
                          published as `missing` against exit 2
                        3d84d2d: run 34631061383 / job 103367725885 — same
                          output; the status was recorded, nothing read it
                        0545536: run 34699510366 / job 103568657812 — suite
                          step exit 2, job RED, report step SUCCESS, payload
                          aggregate_status 2 / admissible false /
                          adjudicated false / refusal.code
                          aggregate_incomplete, and NONE of missing, fallen,
                          unrecorded, drifted

affected_scope          The `--from-log` adjudication path only. Verified from
                        the diff, not asserted: no hunk touches parse_counts,
                        load_floors, compare, check_determinism,
                        SUITE_TARGETS or run_suites. assertion_floors.json
                        unchanged across 1009f31 -> 0545536. No floor value
                        modified

downstream_impact       No programme decision is known to have been taken on
                        the false `missing` output. It was published to the CI
                        log on at least three runs and read by Orion, Kai and
                        Dainius during the RC-7 investigation before being
                        identified

mechanism_status        PATTERN_CANDIDATE
mechanism_id            none assigned
                        Candidate hypothesis preserved: an evidence /
                        admissibility input was fabricated at the caller
                        boundary, making the refusal branch unreachable, while
                        the test intended to protect that behaviour exercised
                        another entry path carrying the true status.
                        Equivalence to I-8, to the self-consuming-guard
                        family, or to any existing mechanism is NOT
                        established. Those remain LOCATORS ONLY

related_incidents       INC-2026-09-12-14 — the false independence claim made
                        about this incident's own closure evidence
                        INC-2026-08-29-09 (a control asserted as a current
                        output, never checked) — EXPLICITLY QUALIFIED LOCATOR.
                        No equivalence implied

recurrence_count        1 measured occurrence of this candidate shape

stop_signal             "the refusal text exists, therefore the refusal
                        happens." The tell: prose in a branch, with no
                        execution path from the caller that matters

current_control         W2.1 3d84d2d — the CI step records the aggregate's
                          exit status beside the log, and `exit "$status"`
                          leaves the step's own conclusion unchanged
                        W2.2 0545536 — `--from-log PATH` derives PATH.status
                          and refuses without it. Five states: S0 adjudicate ·
                          S1 aggregate_incomplete · S2
                          required_input_missing · S3 status_unreadable ·
                          S4 status_authority_conflict. `--status N` is an
                          alternate authority, never a fallback, and cannot
                          overrule the sidecar

control_type            MACHINE — 8 permanent scenarios, each with a
                        known-negative, plus a frozen hand-written fixture of
                        the recovered incident. The load-bearing calibration
                        is the status-2 / status-0 pair on one log: it proves
                        the control gates admissibility WITHOUT disabling the
                        detector it gates. EXPECTED_SCENARIOS 28, derived by
                        running the suite and reading the population back
control_introduced_at   3d84d2d (producer) · 0545536 (consumer + calibration)
recurred_after_control  no

THREAT-MODEL BOUNDARY (Kai, 2026-09-12, verbatim — this closure may not be
read more widely than this):
  "WF-2 closes the fabricated-status defect for the same-job CI
  accidental-failure model. The log/status pair is process-bound evidence
  generated by the same workflow step; W2.2 does not claim cryptographic
  co-identity or protection against deliberate post-production tampering."

NOT CLOSED BY THIS ENTRY — separate findings, not interchangeable:
  WF-2R  `--from-log --update-floors` retains the legacy unbound capability.
         Containment: no production repository caller exercises that
         combination (Makefile:402 uses run_suites()). OPEN. No D-number
  WF-3   The population is inferred lexically. False admission measured
         ("Container proof": 3, the subject's own tally); false exclusion
         measured (`Service identity (ed25519) tests` 80, `/observe_turn
         identity slice` 43 — in neither bucket). 15 suites print tallies
         with no floor. OPEN, DESIGN ONLY. The 15 must not be auto-floored
  RC-7   Why the aggregate actually fails. 7a privilege-transition
         assumption · 7b timeout-boundary control defect · 7c scenario does
         not establish its named precondition. Reproduces identically at
         0545536. OPEN
  DOC-1  sync_docs.py backlog suffix self-amplification. OPEN

owner/stage             Orion (execution) · Kai (adjudication) · Dainius
                        (consequential authority)
status                  WF-2 ORIGINAL IMPLEMENTATION DEFECT CLOSED, bounded as
                        above. Mechanism remains a candidate
```

---

### `INC-2026-09-12-14` — common-source observations reported as independent
###                       evidence

```
INCIDENT_ID             INC-2026-09-12-14
date                    2026-09-12
producer                Orion
subject/version         WF-2 closure evidence for 0545536; run 34699510366 /
                        job 103568657812

false_or_faulty_output  "The binding is from two independent producers."
                        Earlier instance of the same error in the W2.2 CI
                        evidence return: "the two halves of the evidence unit
                        agree, from two different producers." Both described
                        the GitHub step conclusion and the .status sidecar as
                        independent corroboration of aggregate status 2

corrected_output        NOT INDEPENDENT. Both descend from a single computed
                        node:
                          make test-uh | tee  ->  pipeline status
                                              ->  status variable
                                              ->  (a) .status sidecar
                                              ->  (b) exit "$status"
                                                      -> step conclusion
                        Their agreement is arithmetic, not corroboration.
                        Separate retrospective corroboration of
                        O_subject = FAIL does exist, from a different
                        authority: the subject's own stderr and tally, and
                        make's own error line, all written to the log body
                        upstream of the status variable. The production gate
                        does not consume them

detection_method        External adversarial review reconstructed the causal
                        graph and applied Orion's OWN destructive calibration
                        as the falsifier. Not detected by Orion, who had
                        measured the falsifier, recorded it in the W2.1 commit
                        message as the reason the line is load-bearing, and
                        then published the independence claim regardless

evidence                Common-source graph above.
                        Falsifier, measured: remove `set -o pipefail` against
                        a failing subject — `tee` returns 0, captured status
                        becomes 0, sidecar reads 0, the script exits 0, and
                        the GitHub step reports GREEN. Both alleged
                        independent observations agree, and both are wrong
                        about the failing subject.
                        Independent subject-truth witnesses, upstream of
                        $status, present in job 103568657812:
                          setpriv: setresuid failed: Operation not permitted
                          Container proof: 3 passed, 13 failed
                          EXIT GATE: FAIL
                          subprocess.TimeoutExpired: ...
                          make: *** [Makefile:970: ...] Error 1

affected_scope          WF-2 closure evidence WORDING only. No production code
                        path, no gate behaviour, no test

downstream_impact       Could have overstated the strength of the closure
                        evidence in an append-only authoritative record.
                        Caught before ledger banking and before merge. No
                        programme decision rests on the false wording

mechanism_status        PATTERN_CANDIDATE
mechanism_id            none assigned
                        Two preserved occurrences of the
                        independence / common-authority shape now exist
                        (INC-2026-08-29-10 and this). Causal equivalence is
                        NOT declared from resemblance. THIRD-OCCURRENCE
                        ESCALATION HAS NOT FIRED

related_incidents       INC-2026-08-29-10 — reproducibility reported as
                        independence. Same governing stop-signal. That entry
                        already records "THE CONTROL EXISTED, WAS BANKED, AND
                        DID NOT FIRE." This is the second time that same
                        banked control did not fire
                        INC-2026-09-12-13 — the incident whose closure
                        evidence carried this false claim

recurrence_count        2 preserved occurrences of the candidate shape

stop_signal             "I am writing 'independent' — name the authority
                        behind each leg." Present in CLAUDE.md R0 at the time
                        of the failure, and quoted by Orion in the reviewer
                        brief that solicited this very challenge

current_control         doctrine 39 — NAME THE AUTHORITY BEHIND EVERY
                        INDEPENDENCE CLAIM; rule 33's `independence_status`
                        field; CLAUDE.md R0 tell as above
control_type            MANUAL
control_introduced_at   banked prior to 2026-08-29 (see INC-2026-08-29-10)
recurred_after_control  YES

owner/stage             Orion (producer) · Kai (adjudication) · Dainius
                        (consequential authority)
status                  CLOSED as an incident. Implementation unaffected — no
                        code change is authorised or required by this
                        incident. WF-2 remains CLOSED, narrowly scoped
```

---

## Ledger state after this append

| id | producer | mechanism_status | assigned mechanism |
|---|---|---|---|
| `INC-2026-08-29-01` … `-03` | Orion | `PATTERN_CONFIRMED` | `M-SCOPE-WIDEN` |
| `INC-2026-08-29-04` `-05` | Orion | `INCIDENT_ONLY` | none |
| `INC-2026-08-29-06` | Kai | `INCIDENT_ONLY` | none — locator only |
| `INC-2026-08-29-07` `-08` | DeepSeek | `INCIDENT_ONLY` | none |
| `INC-2026-08-29-09` `-10` | Orion | `INCIDENT_ONLY` | none |
| `INC-2026-08-29-11` | Kai | `INCIDENT_ONLY` | none |
| `INC-2026-08-30-12` | Orion | `PATTERN_CANDIDATE` | `M-QUERY-OVERREACH` |
| `INC-2026-09-12-13` | Orion | `PATTERN_CANDIDATE` | none assigned |
| `INC-2026-09-12-14` | Orion | `PATTERN_CANDIDATE` | none assigned |

**Producers: Orion 10 · Kai 2 · DeepSeek 2. Total incidents 14.** The counts
carry no fairness, quality or producer-reliability inference; they are the
currently recorded population and nothing more.

**Mechanisms: `M-SCOPE-WIDEN` `PATTERN_CONFIRMED` · `M-PRODUCER-CURATION`
`PATTERN_CANDIDATE` · `P-ADJUDICATOR-PROPAGATION` `PATTERN_CANDIDATE` ·
`M-QUERY-OVERREACH` `PATTERN_CANDIDATE`.** Unchanged by this append. No
confirmed mechanism's recurrence count is altered: `-13` and `-14` are
candidate locators and are assigned to no mechanism.

**Escalation state.** INC-2026-08-29-10 and INC-2026-09-12-14 are two
preserved occurrences of an independence / common-authority locator shape
under the same MANUAL control, which existed and did not fire on both
occasions. Causal equivalence has NOT been adjudicated; these therefore do
not yet constitute two confirmed occurrences of one mechanism for doctrine
49.6. Third-occurrence escalation has NOT fired. If these incidents are
later adjudicated as occurrences of one mechanism, a subsequent third
independently confirmed occurrence of that same mechanism would trigger
doctrine 49.6: the prose/manual control is presumed insufficient and must
escalate to machine enforcement, structural constraint, or an explicit
accepted-risk decision.

---

# APPEND 2026-09-14 — two shadow-implementation faults in the WF-3 runner

Both were caught before any production path invoked the runner, and neither
altered a CI result or a programme verdict. They are recorded because of what
each would have done had it been cut over: the first would have let
unsupported Make invocation modes pass a guard that reported its context
valid; the second would have given a future consumer two plausible answers to
"did the aggregate complete?".

A verified bug is not automatically a material failure-pattern incident. Two
other real defects found in the same correction -- MALFORMED being documented
but unreachable, and an external calibration plan crashing on
`relative_to(REPO)` -- are deliberately NOT recorded here. Neither changed an
admission, closure or authority outcome in this shadow state; they live in the
implementation evidence and the commit history, which is where they belong.

Both incidents below are INCIDENT_ONLY with NO mechanism assigned.


### `INC-2026-09-14-15` — an environment representation written from
###                       intuition instead of measured from its producer

```
INCIDENT_ID             INC-2026-09-14-15
date                    2026-09-14
producer                Orion
subject/version         WF-3 shadow runner during development leading to
                        859526c28f890eda26f882506e86ce75f870e570

false_or_faulty_output  The initial MAKEFLAGS guard encoded hostile Make
                        options using intuitive DASHED spellings -- "-k",
                        "--keep-going", "-i", "-n". GNU Make 4.3 does not
                        export them that way. Measured:

                          make -k            -> MAKEFLAGS=[k]
                          make --keep-going  -> MAKEFLAGS=[k]
                          make -i            -> MAKEFLAGS=[i]
                          make -n            -> MAKEFLAGS=[n]
                          make -kj2          -> MAKEFLAGS=[k -j2 --jobserver-auth=3,4]
                          make -j4           -> MAKEFLAGS=[ -j4 --jobserver-auth=3,4]

                        Single-letter options are PACKED UNDASHED into the
                        first word; only argument-bearing options keep a
                        dash. The guard therefore matched nothing for five of
                        nine hostile invocations -- -k, -i, -n, --keep-going,
                        --ignore-errors -- catching only -j, while REPORTING
                        THE CONTEXT ACCEPTABLE.

corrected_output        The runner parses the measured representation: the
                        packed undashed first token, dashed argument-bearing
                        forms, and long forms. Hostile letters j, i, k, n and
                        their long options are refused. Known-negative forms
                        were measured separately -- s, rR, w,
                        --warn-undefined-variables must be ALLOWED -- so the
                        guard discriminates rather than refusing all
                        MAKEFLAGS. 13 cases, zero mismatches.

detection_method        DYNAMIC CALIBRATION against a real GNU Make 4.3
                        exporting MAKEFLAGS for each invocation. MEASUREMENT
                        FOUND THIS DEFECT. Doctrine did not, static reasoning
                        did not, and review had not yet seen it. The
                        implementation contract's instruction to verify actual
                        GNU Make behaviour rather than guess is the only
                        reason it surfaced before cutover.

affected_scope          Shadow WF-3 runner only. No production path invoked
                        the runner at any point.

downstream_impact       NONE. No production decision, CI result or programme
                        verdict was affected. Had it been cut over
                        uncorrected, unsupported invocation modes -- keep-
                        going, ignore-errors, dry-run -- could have weakened
                        the serial/fail-fast evidence model while the runner
                        reported its context valid. A guard whose scope is
                        narrower than its name is why this is ledger-material
                        despite being caught pre-production.

mechanism_status        INCIDENT_ONLY
mechanism_id            none assigned
related_incidents       none cited. Resemblance to existing mechanisms is not
                        asserted; no locator is useful enough here to risk
                        implying equivalence.
recurrence_count        1 measured occurrence of this shape

stop_signal             "I am writing an environment or protocol
                        representation from memory instead of measuring what
                        its producer actually emits."

current_control         The corrected MAKEFLAGS parser in the shadow runner,
                        plus measured positive and negative calibration.
control_type            STRUCTURAL -- AND THE MATURITY IS QUALIFIED. The
                        correction exists in SHADOW CODE. No executing
                        regression test is banked for it yet, so this is not
                        a MACHINE control, and production operationalisation
                        is NOT established.
control_introduced_at   859526c (guard present) / 8fc5bc (re-verified after
                        the ownership correction)
recurred_after_control  NO measured recurrence

owner/stage             Orion (execution) · Kai (adjudication) · Dainius
                        (consequential authority)
status                  CLOSED AS AN IMPLEMENTATION INCIDENT.
                        WF-3 itself remains OPEN. No production
                        control-operationalised claim is made. CI semantics
                        remain UNVERIFIED.
```

---

### `INC-2026-09-14-16` — a second representation built for a fact whose
###                       authoritative producer already existed

```
INCIDENT_ID             INC-2026-09-14-16
date                    2026-09-14
producer                Orion
subject/version         WF-3 shadow runner.
                        before: 859526c28f890eda26f882506e86ce75f870e570
                        after:  8fc5bc085b1e124bb25e9aca0dcd6809d90853f9

false_or_faulty_output  At 859526c the shadow runner did all three of:
                          1. opened and wrote EVIDENCE_ROOT/run.log
                          2. wrote EVIDENCE_ROOT/run.log.status from its own
                             computed status
                          3. published a manifest field `aggregate_status`
                        Each duplicated an authority WF-2 had already assigned
                        to the workflow shell: the TOP-LEVEL `make test-uh`
                        pipeline status captured under pipefail. The result
                        was two structurally different observations that a
                        future consumer could plausibly read as
                        aggregate-completion authority.

corrected_output        ONE canonical aggregate completion authority:
                          workflow shell -> `make test-uh 2>&1 | tee
                          "$ROOT/run.log"` under pipefail -> shell status ->
                          canonical run.log.status sidecar.
                        The runner now owns only plan traversal, per-target
                        sub-make execution observations, target result
                        observations, the atomic results.json, and stdout
                        forwarding so the outer tee remains the single
                        producer of the canonical log. Its process exit is a
                        signal upward to Make, documented as NOT a status.
                        `aggregate_status` is REMOVED, NOT RENAMED, so no
                        future code can mistake it for alternate truth.
                        Measured after correction: the evidence root contains
                        results.json AND NOTHING ELSE; no run.log; no
                        run.log.status; the only remaining run.log strings in
                        the source are comments naming whose file it is.

detection_method        Kai architecture/review boundary check against frozen
                        WF-2 authority semantics, followed by source
                        correction and Orion's local boundary calibration.
                        NOT found by Orion.

evidence                before 859526c: runner wrote run.log, wrote
                        run.log.status, manifest carried aggregate_status.
                        after 8fc5bc: none of the three; manifest carries
                        schema, plan_digest, plan_path, plan_scope,
                        evidence_root, population, slots -- and no aggregate
                        status field under any name.

affected_scope          WF-3 shadow runner only. No production code invoked
                        it. The Makefile, the production workflow, the
                        existing assertion-floor consumer and the floor
                        registry all remained on the old production path
                        throughout.

downstream_impact       NONE. No production verdict or programme decision was
                        affected. Had the shadow design been cut over, the
                        floor consumer could have faced competing sources for
                        aggregate completion and potentially selected
                        runner-derived status over the WF-2-authoritative
                        workflow sidecar. It is an AUTHORITY-BOUNDARY defect,
                        which is why it is ledger-material despite being
                        caught before cutover.

mechanism_status        INCIDENT_ONLY
mechanism_id            none assigned
related_incidents       INC-2026-09-12-13 and INC-2026-09-12-14 are recorded
                        here as QUALIFIED LOCATORS ONLY. The current evidence
                        does NOT establish causal equivalence with the
                        fabricated-status incident or with the
                        common-authority independence incident, and none is
                        asserted.
recurrence_count        1 measured occurrence of this exact candidate shape

stop_signal             "I am about to create a second representation of a
                        fact whose authoritative producer already exists --
                        name which component owns the fact before writing
                        another one."

current_control         Structural ownership separation at 8fc5bc: the
                        workflow shell holds aggregate log and status
                        authority; the runner holds target and result
                        evidence only.
                        REMAINING ACCEPTANCE REQUIREMENT, recorded so it is
                        not lost: the future floor consumer MUST consume the
                        authoritative workflow status and MUST NEVER use
                        manifest-derived state as a fallback for an absent or
                        non-zero authoritative status.
control_type            STRUCTURAL -- SHADOW IMPLEMENTATION STRUCTURE, NOT
                        YET CONTROL_OPERATIONALISED IN PRODUCTION.
control_introduced_at   8fc5bc085b1e124bb25e9aca0dcd6809d90853f9
recurred_after_control  NO measured recurrence

owner/stage             Orion (execution) · Kai (adjudication) · Dainius
                        (consequential authority)
status                  CLOSED AS A SHADOW IMPLEMENTATION INCIDENT.
                        WF-3 remains OPEN. WF-2 remains CLOSED within its
                        existing bounded threat model. No claim is made that
                        WF-3 production control is operational until cutover
                        and CI evidence exist.
```

---

## Ledger state after this append

| id | producer | mechanism_status | assigned mechanism |
|---|---|---|---|
| `INC-2026-08-29-01` … `-03` | Orion | `PATTERN_CONFIRMED` | `M-SCOPE-WIDEN` |
| `INC-2026-08-29-04` `-05` | Orion | `INCIDENT_ONLY` | none |
| `INC-2026-08-29-06` | Kai | `INCIDENT_ONLY` | none — locator only |
| `INC-2026-08-29-07` `-08` | DeepSeek | `INCIDENT_ONLY` | none |
| `INC-2026-08-29-09` `-10` | Orion | `INCIDENT_ONLY` | none |
| `INC-2026-08-29-11` | Kai | `INCIDENT_ONLY` | none |
| `INC-2026-08-30-12` | Orion | `PATTERN_CANDIDATE` | `M-QUERY-OVERREACH` |
| `INC-2026-09-12-13` | Orion | `PATTERN_CANDIDATE` | none assigned |
| `INC-2026-09-12-14` | Orion | `PATTERN_CANDIDATE` | none assigned |
| `INC-2026-09-14-15` | Orion | `INCIDENT_ONLY` | none assigned |
| `INC-2026-09-14-16` | Orion | `INCIDENT_ONLY` | none assigned |

**Producers: Orion 12 · Kai 2 · DeepSeek 2. Total incidents 16.** The counts
carry no fairness, quality or producer-reliability inference; they are the
currently recorded population and nothing more.

**Mechanisms: `M-SCOPE-WIDEN` `PATTERN_CONFIRMED` · `M-PRODUCER-CURATION`
`PATTERN_CANDIDATE` · `P-ADJUDICATOR-PROPAGATION` `PATTERN_CANDIDATE` ·
`M-QUERY-OVERREACH` `PATTERN_CANDIDATE`.** Unchanged by this append. No
confirmed mechanism's recurrence count is altered: `-15` and `-16` are
INCIDENT_ONLY and are assigned to no mechanism.

**Escalation state.** Doctrine 49.6 fires on the third independently
confirmed occurrence of ONE mechanism. Neither incident in this append is
assigned to a mechanism, so neither contributes to any mechanism's count and
neither advances an escalation. The independence / common-authority locator
shape recorded at `INC-2026-08-29-10` and `INC-2026-09-12-14` stands at two
preserved occurrences, unchanged here; `INC-2026-09-14-16` is NOT counted
toward it, because its equivalence to that shape has not been adjudicated and
resemblance is not a mechanism.

---

# APPEND 2026-09-14 (second) — a verification attached to the wrong tree,
#                              and a contract the gate never checked

Both found in the WF-3 shadow floor-consumer tranche. Neither reached
production. The first misstated verification evidence for a banked commit;
the second would have let floor counts be adjudicated without proving they
belonged to the contracted result.


### `INC-2026-09-14-17` — a verification run against the working tree,
###                       transmitted as verification of the committed tree

```
INCIDENT_ID             INC-2026-09-14-17
date                    2026-09-14
producer                Orion
subject/version         commit 009cfadd27c0f47f149c841934055556a39f3098

false_or_faulty_output  The commit message states, under VERIFIED HERE:
                        "check-docs exit 0".
                        Against that committed tree, `make check-docs` exits
                        2: "PROJECT_BACKLOG.md stale -- tests 4759 -> 4783".

corrected_output        The successful measurement was taken against a
                        DIFFERENT SUBJECT: the working tree, which contained
                        a generated change to docs/PROJECT_BACKLOG.md that
                        was then omitted from an explicit `git add` file
                        list. The commit added a test file; sync_docs.py
                        patches the generated test count in TWO files; only
                        one was staged. The transmitted verification claim
                        was therefore attached to the wrong tree identity.
                        Corrected at 79cd2dbc420741b3d65073d4f62ec83519e17c13.

detection_method        Post-push `git status` showed a leftover modification,
                        followed by an explicit check of the committed tree
                        via stash. NOT caught by the verification that was
                        supposed to catch it.

affected_scope          WF-3 shadow documentation consistency only.

downstream_impact       No production verdict. No programme decision. Caught
                        before Kai accepted the tranche. The impact is
                        bounded by those facts -- and they do not make the
                        false exact-tree verification claim disappear, which
                        is why it is recorded rather than left in the commit
                        history alone.

mechanism_status        INCIDENT_ONLY
mechanism_id            none assigned
related_incidents       `M-SCOPE-WIDEN` is cited here as a QUALIFIED LOCATOR
                        ONLY: the measured subject and the transmitted
                        subject differ, which is that mechanism's shape. The
                        incident is NOT assigned to it. That equivalence
                        would alter recurrence and escalation state for a
                        PATTERN_CONFIRMED mechanism, and is an adjudication,
                        not a producer's self-certification.
recurrence_count        1 measured occurrence of this shape

stop_signal             "I verified a working tree and am about to claim the
                        commit passed -- prove the tested tree ID equals the
                        tree I am transmitting."

current_control         For WF-3 tranche commits: verify the COMMITTED tree
                        after commit and before push or acceptance. Applied
                        immediately at 53389986aa3f9d067be0bd4d429b6038e5dc2e1f,
                        where check-docs and the full suite were both run
                        against the committed tree.
control_type            MANUAL -- a producer discipline, not an executing
                        check. NOT operationalised programme-wide, and no
                        such claim is made: it currently binds WF-3 tranche
                        commits and nothing else.
control_introduced_at   53389986aa3f9d067be0bd4d429b6038e5dc2e1f
recurred_after_control  NO measured recurrence

owner/stage             Orion (execution) · Kai (adjudication) · Dainius
                        (consequential authority)
status                  CLOSED AS AN IMPLEMENTATION INCIDENT. WF-3 remains
                        OPEN. This is NOT the DOC-1 plus-run amplification;
                        that is separate and still open. The incident here is
                        VERIFIED SUBJECT != COMMITTED SUBJECT.
```

---

### `INC-2026-09-14-18` — the consumer adjudicated counts it never bound to
###                       the contracted result

```
INCIDENT_ID             INC-2026-09-14-18
date                    2026-09-14
producer                Orion
subject/version         shadow plan-aware floor consumer
                        scripts/security/uh_floor_gate.py and its calibration
                        scripts/test_uh_floor_gate.py at
                        009cfadd27c0f47f149c841934055556a39f3098

false_or_faulty_output  The consumer accepted a green manifest whose slots
                        carried fabricated result labels -- literally
                        f"{make_target} label" -- unrelated to the
                        result_label values declared in the canonical
                        execution plan, and adjudicated floor counts from
                        them.
                        It validated schema, evidence root, plan scope, plan
                        digest, target count, uniqueness, foreign and missing
                        targets, order, execution-state shape, and that
                        result_observation was RESOLVED -- and then read
                        slot["result"]["passed"] directly. It never checked
                        result_label, ordinal, the completed target's
                        exit_code, or that passed and failed were sane
                        non-boolean non-negative integers with failed == 0.
                        THE FALSIFIER WAS INSIDE MY OWN FIXTURE. Every
                        "positive" case in the calibration carried wrong
                        labels and adjudicated cleanly, so the suite was
                        proving a weaker protocol than the one being shipped.

corrected_output        reconcile() now validates each slot per ordinal
                        against the canonical plan entry: position,
                        make_target, result_label, execution_state COMPLETED,
                        exit_code 0, result_observation RESOLVED, result an
                        object, passed and failed both present, both integers,
                        neither bool, neither negative, failed == 0 under a
                        zero authoritative status. Counts reach the floor
                        comparison only through it, and a malformed field
                        produces a governed refusal rather than a KeyError or
                        TypeError. The plan is validated by the consumer
                        itself; manifest population and plan_path must agree
                        with the canonical plan. The fixture now derives
                        (make_target, result_label) PAIRS from the plan, so
                        the known-positive genuinely satisfies the contract.

detection_method        Kai independent source review of uh_floor_gate.py and
                        test_uh_floor_gate.py. NOT found by Orion, and not
                        found by the calibration suite -- which contained the
                        falsifier and passed anyway.

affected_scope          Shadow floor consumer only. No production caller. The
                        Makefile, workflow, legacy consumer and live floor
                        registry remained on the old production path.

downstream_impact       NONE realised. Had it been cut over, floor counts
                        could have been adjudicated without proving the count
                        was bound to the plan-declared exact result label, and
                        without validating the reported result structure --
                        which could alter admission and floor findings.

mechanism_status        INCIDENT_ONLY
mechanism_id            none assigned
                        This is a compound contract-validation omission until
                        evidence establishes anything stronger. No mechanism
                        is invented for it.
related_incidents       none asserted. I-8 -- evidence for a check coming from
                        the same place as the thing it checks -- describes the
                        fixture's shape, but it is doctrine, not a banked
                        mechanism, and no equivalence is claimed.
recurrence_count        1 measured occurrence of this shape

stop_signal             "My positive fixture is half-derived from the
                        authority and half invented -- and the invented half
                        is the field the contract turns on."

current_control         Full per-ordinal manifest/plan reconciliation, plus 13
                        new destructive scenarios covering fabricated labels
                        on one target and on all 78, wrong ordinal, COMPLETED
                        with non-zero exit, missing and invalid passed and
                        failed, a resolved result reporting failures,
                        population and plan_path lying, and the plan validator
                        against nine malformed documents. 217 assertions, 0
                        failures, 37 scenarios, with the original
                        known-negatives re-run and unweakened.
control_type            STRUCTURAL -- SHADOW IMPLEMENTATION STRUCTURE WITH AN
                        EXECUTING CALIBRATION SUITE, but NOT YET
                        CONTROL_OPERATIONALISED: no production caller invokes
                        the consumer, and the suite is not yet wired into any
                        make target or CI workflow.
control_introduced_at   53389986aa3f9d067be0bd4d429b6038e5dc2e1f
recurred_after_control  NO measured recurrence

owner/stage             Orion (execution) · Kai (adjudication) · Dainius
                        (consequential authority)
status                  CLOSED AS A SHADOW IMPLEMENTATION INCIDENT. WF-3
                        remains OPEN. Production cutover, CI integration, the
                        real-78 traversal and the floor migration all remain
                        held.
```

---

## Ledger state after this append

| id | producer | mechanism_status | assigned mechanism |
|---|---|---|---|
| `INC-2026-08-29-01` … `-03` | Orion | `PATTERN_CONFIRMED` | `M-SCOPE-WIDEN` |
| `INC-2026-08-29-04` `-05` | Orion | `INCIDENT_ONLY` | none |
| `INC-2026-08-29-06` | Kai | `INCIDENT_ONLY` | none — locator only |
| `INC-2026-08-29-07` `-08` | DeepSeek | `INCIDENT_ONLY` | none |
| `INC-2026-08-29-09` `-10` | Orion | `INCIDENT_ONLY` | none |
| `INC-2026-08-29-11` | Kai | `INCIDENT_ONLY` | none |
| `INC-2026-08-30-12` | Orion | `PATTERN_CANDIDATE` | `M-QUERY-OVERREACH` |
| `INC-2026-09-12-13` | Orion | `PATTERN_CANDIDATE` | none assigned |
| `INC-2026-09-12-14` | Orion | `PATTERN_CANDIDATE` | none assigned |
| `INC-2026-09-14-15` | Orion | `INCIDENT_ONLY` | none assigned |
| `INC-2026-09-14-16` | Orion | `INCIDENT_ONLY` | none assigned |
| `INC-2026-09-14-17` | Orion | `INCIDENT_ONLY` | none assigned — `M-SCOPE-WIDEN` locator only |
| `INC-2026-09-14-18` | Orion | `INCIDENT_ONLY` | none assigned |

**Producers: Orion 14 · Kai 2 · DeepSeek 2. Total incidents 18.** The counts
carry no fairness, quality or producer-reliability inference; they are the
currently recorded population and nothing more.

**Mechanisms: `M-SCOPE-WIDEN` `PATTERN_CONFIRMED` · `M-PRODUCER-CURATION`
`PATTERN_CANDIDATE` · `P-ADJUDICATOR-PROPAGATION` `PATTERN_CANDIDATE` ·
`M-QUERY-OVERREACH` `PATTERN_CANDIDATE`.** Unchanged by this append. No
confirmed mechanism's recurrence count is altered. `INC-2026-09-14-17` cites
`M-SCOPE-WIDEN` as a LOCATOR and is NOT assigned to it, so that mechanism's
recurrence count and escalation state are untouched.

**Escalation state.** Doctrine 49.6 fires on the third independently
confirmed occurrence of ONE mechanism. Neither incident in this append is
assigned to a mechanism, so neither advances any escalation. The
independence / common-authority locator shape at `INC-2026-08-29-10` and
`INC-2026-09-12-14` still stands at two preserved occurrences, unchanged.

---

# APPEND 2026-09-15 — a policy the machine did not implement, and a
#                     candidate mechanism for the shape

One incident, and the first registration of a candidate mechanism drawn
from it and from `INC-2026-09-14-18`.


### `INC-2026-09-14-19` — an unwatched population member exited green

```
INCIDENT_ID             INC-2026-09-14-19
date                    2026-09-14
producer                Orion
subject/version         WF-3 shadow plan-aware floor consumer
                        scripts/security/uh_floor_gate.py and its calibration
                        scripts/test_uh_floor_gate.py
                        before: 53389986aa3f9d067be0bd4d429b6038e5dc2e1f
                        after:  18faee4a1d73306a31bc7e375e2d6a21479b4ab0

false_or_faulty_output  The consumer represented an unfloored population
                        member to the operator as:
                          "Absence of a floor is NOT zero and is NOT a pass"
                        while the consequential admission predicate was:
                          return 1 if fallen else 0
                        So a complete, valid aggregate with all 61 floors
                        satisfied and 17 admitted population members carrying
                        no floor exited 0. Reproduced before repair: rc=0,
                        adjudicated=true, fallen=0, unfloored=17.
                        THE REPORTING SURFACE WAS RED IN MEANING. THE MACHINE
                        ADMISSION SURFACE WAS GREEN. CI reads the second.

                        Scenario S checked that the 17 were reported, that
                        they carried no floor and that no zero was
                        synthesised. It did NOT assert the return code. The
                        test verified the explanation and left the
                        consequential outcome unverified.

corrected_output        UNFLOORED_TARGET is an admissible, adjudicated POLICY
                        FINDING. It is not a refusal -- the evidence is
                        perfectly adjudicable, the run completed and every
                        count is bound to its contracted result. It is not a
                        floor erosion -- no prior floor exists to fall from.
                        It is not zero and not an exemption.
                        Admission is now:
                          fully floored + all met  -> exit 0   (the only green)
                          fallen > 0               -> exit 1
                          unfloored > 0            -> exit 1
                          both                     -> exit 1
                        The terminal statement no longer claims a pass over F
                        while P is wider than F; a pass requires all 78
                        floored AND all met, and says so. A machine-readable
                        `findings: {fallen, unfloored}` was added so the
                        count is not inferred from array lengths.
                        No floor was assigned to the 17. No zero. No
                        exemption. Their values remain a separate decision
                        requiring evidence from a complete successful CI run,
                        which does not yet exist.

detection_method        Kai independent source review, triggered by Orion's
                        strategic challenge concerning the unresolved
                        17-target disposition. The challenge was strategic,
                        not evidential; the inspection it prompted found the
                        defect.

affected_scope          Shadow floor consumer only. No production caller. The
                        Makefile, workflow, legacy consumer and live floor
                        registry remained on the old production path.

downstream_impact       NONE realised. Had the shadow consumer been cut over,
                        CI could have returned green while 17 admitted
                        population members remained unwatched -- which is the
                        precise condition the assertion-floor gate exists to
                        make impossible.

mechanism_status        INCIDENT_ONLY
mechanism_id            none assigned directly
related_incidents       INC-2026-09-14-18 — cited as supporting evidence for
                        the candidate mechanism registered below, not as an
                        assertion of causal equivalence.
recurrence_count        1 measured occurrence of this incident

stop_signal             "I have written the policy in prose and the admission
                        in code -- prove they are the same proposition."

current_control         The corrected admission predicate, plus scenario S
                        asserting rc == 1 and that the run is admissible and
                        adjudicated rather than refused; S2 covering both
                        findings at once; S3 pinning the only green.
                        227 assertions, 0 failures, 39 scenarios.
control_type            STRUCTURAL -- SHADOW IMPLEMENTATION WITH AN EXECUTING
                        CALIBRATION SUITE, but NOT CONTROL_OPERATIONALISED:
                        no production caller invokes the consumer and the
                        suite is wired into no make target or workflow.
control_introduced_at   18faee4a1d73306a31bc7e375e2d6a21479b4ab0
recurred_after_control  NO measured recurrence

owner/stage             Orion (execution) · Kai (adjudication) · Dainius
                        (consequential authority)
status                  CLOSED AS A SHADOW IMPLEMENTATION INCIDENT. WF-3
                        remains OPEN.
```

---

## `M-POLICY-ADMISSION-DIVERGENCE` — PATTERN_CANDIDATE

**State: `PATTERN_CANDIDATE`. NOT `PATTERN_CONFIRMED`.**

**Candidate mechanism.** A gate carries a normative admission policy in one
surface — prose, documentation, reporting, or test intent — while the actual
machine predicate deciding PASS, FINDING, REFUSAL, admissibility,
adjudication or exit status implements a weaker or different proposition.
The producer's calibration then validates the descriptive surface without
asserting the consequential admission boundary, so policy and admission
diverge while the suite stays green.

**Supporting occurrence 1 — `INC-2026-09-14-18`.**
Normative policy: every population member must prove one exact target-bound
result. Actual admission before repair: the consumer did not bind the
canonical `result_label`, the ordinal, the completed target's zero exit, or
a sane `passed`/`failed` structure before counts reached floor adjudication.
Calibration defect: the positive fixture itself fabricated
`result_label = f"{target} label"` and passed anyway. The intended policy and
the consequential admission predicate were not the same proposition.

**Supporting occurrence 2 — `INC-2026-09-14-19`.**
Normative policy: unfloored population members are not a pass. Actual
admission before repair: the return code depended only on `fallen`, so
`unfloored > 0` with `fallen == 0` exited 0. Calibration defect: scenario S
checked the report and not the consequential return code. Again the policy
claim and the machine admission predicate differed.

**`INC-2026-09-14-18` is retrospectively included as supporting evidence for
candidate `M-POLICY-ADMISSION-DIVERGENCE`; its original ledger record remains
unchanged, because the candidate mechanism had not yet been adjudicated when
`INC-2026-09-14-18` was recorded.** This file is append-only and history is
not rewritten to make a pattern look tidier than its discovery.

**Why this is only a candidate.** The two occurrences share a producer, a
gate family, a tranche and a development period. That is recurrence evidence
and it is not independence. Causal equivalence is **not** established.

```
candidate occurrence count      2
confirmed recurrence count      NOT ESTABLISHED
doctrine 49.6 escalation        NOT TRIGGERED
machine-control escalation      NOT CLAIMED
cause                           NOT PROVEN
```

**Control hypothesis — NOT OPERATIONALISED, NOT IMPLEMENTED.**
The denominator is not "every sentence the gate prints"; that is too broad
to be a control. The load-bearing denominator is **normative admission
propositions**: any proposition capable of changing `admissible`,
`adjudicated`, PASS / FINDING / REFUSAL, process exit, finding code or
refusal code.

The proposed form is a decision-contract matrix whose expected outcomes are
derived from the **frozen architecture**, not read back out of the
implementation:

| condition | admissible | adjudicated | disposition | exit | buckets |
|---|---|---|---|---|---|
| status != 0 | false | false | `AGGREGATE_INCOMPLETE` | 2 | none adjudicated |
| complete + fallen | true | true | `FLOOR_FINDING` | 1 | present |
| complete + unfloored | true | true | `UNFLOORED_TARGET` | 1 | present |
| complete + no findings | true | true | `PASS` | 0 | present |
| green aggregate + label mismatch | false | false | `RESULT_CONTRACT_CONFLICT` | 2 | none |

This is a hypothesis recorded for later adjudication. Nothing implements it.

**Detection-method correction, recorded because the earlier claim was
wrong.** Orion stated in strategic advice that three defects —
`INC-2026-09-14-15`, `-18` and `-19` — were all found by adversarial source
reading. That is false and collapses two distinct detectors.
`INC-2026-09-14-15` was found by **dynamic calibration against real GNU Make
behaviour**; `-18` and `-19` were found by **independent adversarial contract
and source review**. The supported programme lesson is that hostile execution
and independent adversarial review are **complementary detectors, and neither
substitutes for the other**.

**Same-producer calibration, qualified.** A calibration suite written by the
producer of the implementation can inherit that producer's blind spots — both
supporting occurrences demonstrate it. That does **not** make same-producer
hostile calibration invalid: it is useful MACHINE evidence. It is simply not
independent corroboration. Same authority, different method, is cross-method
convergence. For consequential admission logic, future acceptance should
carry both producer hostile calibration **and** independent adversarial
contract review, with the independent reviewer contributing at least one
falsifier rather than only reading the existing suite.

---

## Ledger state after this append

| id | producer | mechanism_status | assigned mechanism |
|---|---|---|---|
| `INC-2026-08-29-01` … `-03` | Orion | `PATTERN_CONFIRMED` | `M-SCOPE-WIDEN` |
| `INC-2026-08-29-04` `-05` | Orion | `INCIDENT_ONLY` | none |
| `INC-2026-08-29-06` | Kai | `INCIDENT_ONLY` | none — locator only |
| `INC-2026-08-29-07` `-08` | DeepSeek | `INCIDENT_ONLY` | none |
| `INC-2026-08-29-09` `-10` | Orion | `INCIDENT_ONLY` | none |
| `INC-2026-08-29-11` | Kai | `INCIDENT_ONLY` | none |
| `INC-2026-08-30-12` | Orion | `PATTERN_CANDIDATE` | `M-QUERY-OVERREACH` |
| `INC-2026-09-12-13` | Orion | `PATTERN_CANDIDATE` | none assigned |
| `INC-2026-09-12-14` | Orion | `PATTERN_CANDIDATE` | none assigned |
| `INC-2026-09-14-15` | Orion | `INCIDENT_ONLY` | none assigned |
| `INC-2026-09-14-16` | Orion | `INCIDENT_ONLY` | none assigned |
| `INC-2026-09-14-17` | Orion | `INCIDENT_ONLY` | none assigned — `M-SCOPE-WIDEN` locator only |
| `INC-2026-09-14-18` | Orion | `INCIDENT_ONLY` | none assigned — supporting evidence for `M-POLICY-ADMISSION-DIVERGENCE` |
| `INC-2026-09-14-19` | Orion | `INCIDENT_ONLY` | none assigned — supporting evidence for `M-POLICY-ADMISSION-DIVERGENCE` |

**Producers: Orion 15 · Kai 2 · DeepSeek 2. Total incidents 19.** The counts
carry no fairness, quality or producer-reliability inference; they are the
currently recorded population and nothing more.

**Mechanisms: `M-SCOPE-WIDEN` `PATTERN_CONFIRMED` · `M-PRODUCER-CURATION`
`PATTERN_CANDIDATE` · `P-ADJUDICATOR-PROPAGATION` `PATTERN_CANDIDATE` ·
`M-QUERY-OVERREACH` `PATTERN_CANDIDATE` · `M-POLICY-ADMISSION-DIVERGENCE`
`PATTERN_CANDIDATE` (new).** No existing mechanism's recurrence count is
altered by this append. `INC-2026-09-14-18` and `-19` remain
`INCIDENT_ONLY`: they are supporting evidence for a candidate, which is not
an assignment.

**Escalation state.** Doctrine 49.6 fires on the third independently
confirmed occurrence of ONE mechanism. `M-POLICY-ADMISSION-DIVERGENCE` is a
candidate with two occurrences that are not independent — same producer,
same gate family, same tranche — so it has no confirmed occurrences and
advances no escalation. The independence / common-authority locator shape at
`INC-2026-08-29-10` and `INC-2026-09-12-14` still stands at two preserved
occurrences, unchanged.

---

# APPEND 2026-09-15 (second) — a machine control that fired, and a review
#                              loop that advanced nine commits without
#                              consuming its verdict

One incident. The detector worked. We did not listen to it.

This append also corrects a chronology that was wrong in the message that
authorised it, and records a second prohibited condition that the first one
had been hiding.


### `INC-2026-09-15-20` — a banked calibration suite left unwired, detected
###                       by the existing machine control, and advanced past

```
INCIDENT_ID             INC-2026-09-15-20
date                    2026-09-15
producer                Orion (implementation defect)
                        Programme review loop — Orion · Kai · Dainius
                        (acceptance defect). The acceptance side is NOT
                        attributed to a single producer and is not counted
                        against one.
subject/version         scripts/test_uh_floor_gate.py — the WF-3 shadow floor
                        consumer's calibration suite
                        introduced at: 009cfadd (2026-09-14)
                        still present at: 0c717ec3f8f49bb4135414a46fdf7068c9b5d4a2
                        Policy-as-Code control: scripts/security/check_test_wiring.py
                        wired as .github/workflows/policy-checks.yml step 26,
                        "Test wiring — no test defined and never called"

false_or_faulty_output  IMPLEMENTATION SIDE.
                        scripts/test_uh_floor_gate.py reports through a
                        check() helper and an `EXIT GATE` line, calling
                        sys.exit(1) only under `if __name__ == "__main__"`.
                        No Makefile recipe runs it as a script.

                        DO NOT READ THIS AS "the 227 assertions never ran."
                        They ran, repeatedly, under manual invocation during
                        WF-3 development, and their results were transmitted
                        and adjudicated. The supported statement is narrower
                        and is the one that matters:

                          once banked, the repository had NO Makefile
                          execution path that made this EXIT-GATE suite's
                          failures consequential through the normal governed
                          test wiring.

                        Under pytest collection every test_* function returns
                        normally whatever check() recorded, so collection
                        reports pass while the suite is failing. That is the
                        exact condition check_test_wiring.py exists to detect.

                        ACCEPTANCE SIDE — the load-bearing half.
                        The condition was detected, on the exact banked SHA,
                        by an existing machine control, and the programme
                        advanced anyway.

                        Verified at primary source (GitHub Actions API, this
                        session, not from recollection):

                          commit 18faee4a1d73306a31bc7e375e2d6a21479b4ab0
                          workflow "Policy-as-Code Checks" run #282
                            run id 34905078315 · job "policy" id 104179749481
                            conclusion: failure
                            steps 1–25 : success
                            step 26 "Test wiring — no test defined and never
                                     called" : FAILURE
                            steps 27–45 : SKIPPED (19 steps)

                        Step-26 log, quoted verbatim from the run:

                          FAIL: 1 EXIT GATE suite(s) that no recipe runs as a
                          script:
                            - test_uh_floor_gate.py: reports through `check()`
                              and `EXIT GATE`, but no Makefile recipe runs it
                              as a script. Under pytest its failures are
                              invisible — every test function returns normally
                              whatever check() recorded.

                        MACHINE DETECTOR: WORKED.
                        RED SIGNAL: EXISTED, ON THE EXACT SHA.
                        RESPONSE / ADJUDICATION: DID NOT HAPPEN.

                        Aggravating detail, recorded because it is worse than
                        an oversight. `INC-2026-09-14-19`, banked in this same
                        ledger at this same commit 18faee4, states in its own
                        `control_type` field:

                          "...the suite is wired into no make target or
                          workflow."

                        The prohibited condition was written into the
                        authoritative append-only ledger, by the producer, at
                        the very commit whose required CI job was failing on
                        that precise condition — and the tranche advanced.
                        The signal was not missed for want of visibility. It
                        was recorded and not treated as consequential.

correction_to_the       The authorising message stated the red began at
authorising_message     18faee4 and named run #282. Run #282 and its step are
                        confirmed exactly as described. THE START POINT IS
                        NOT.

                        Policy-as-Code history on this branch, enumerated
                        from the API (20 runs returned, branch-filtered):

                          #269  abea33cf  success   <- LAST GREEN
                          #270  859526c2  failure   <- FIRST RED
                          #271  859526c2  failure
                          #272  8fc5bc08  failure
                          #273  8fc5bc08  failure
                          #274  cb8cc270  failure
                          #275  009cfadd  failure
                          #276  009cfadd  failure
                          #277  79cd2dbc  failure
                          #278  53389986  failure
                          #279  53389986  failure
                          #280  efb9c63a  failure
                          #281  18faee4a  failure
                          #282  18faee4a  failure
                          #283  0c717ec3  failure   <- CURRENT HEAD

                        14 consecutive failing runs across 9 consecutive
                        commits, beginning at 859526c2 — five commits before
                        18faee4a, and before scripts/test_uh_floor_gate.py
                        existed at all (it enters at 009cfadd).

                        So the span of the acceptance defect is nine banked
                        commits, not one.

second_prohibited       At 859526c2 (run #270, id 34886024489, job id
condition_found         104116773548) the failing step was NOT step 26.
                        Step 26 "Test wiring" was SUCCESS there. The job
                        failed at:

                          step 45 "Instrumentation invariants (I-4 enforced)"
                          run: python scripts/security/check_gate_registry.py --gate

                        Reproduced locally at HEAD 0c717ec3 in this session:

                          exit 1
                            - uh_floor_gate: exists but is not in the registry
                            - uh_runner: exists but is not in the registry
                          GATE FAILED: 2 breach(es) of I-1, I-2, I-3, I-4,
                          I-5, I-6, I-7.

                        THERE ARE THEREFORE TWO DISTINCT PROHIBITED
                        CONDITIONS LIVE ON THIS BRANCH, NOT ONE:

                          (1) I-4 — uh_runner and uh_floor_gate on disk and
                              unregistered. Live since 859526c2. Step 45.
                          (2) test-wiring — test_uh_floor_gate.py unwired.
                              Live since 009cfadd. Step 26.

                        AND (2) MASKS (1). Step 26 runs before step 45, so
                        from 009cfadd onward the job short-circuits and step
                        45 is reported as `skipped`, not as the failure it
                        would still be. Anyone reading only the FIRST FAILED
                        STEP at 18faee4 concludes "one defect, wiring" and
                        silently loses an I-4 breach that has been red for
                        nine commits.

                        This is not a new mechanism claim. It is a measured
                        property of a short-circuiting job, recorded so the
                        process control below is written wide enough to
                        survive it.

corrected_output        No repository code, test, Makefile, workflow, registry
                        or floor value is changed by this append. The two
                        prohibited conditions remain OPEN and are scheduled:
                        DOC-1 first (authorised), then a bounded §24 tranche
                        that must clear BOTH step 26 and step 45, not only the
                        first-failing one.

detection_method        The unwired condition: EXISTING MACHINE CONTROL —
                        check_test_wiring.py via Policy-as-Code step 26, on
                        the exact banked SHA, at bank time.

                        Its re-surfacing on 2026-09-15: Orion source review
                        while scoping §24, independently of the CI record.

                        The nine-commit span, the 859526c2 start point and
                        the step-45 masking: Orion primary-source enumeration
                        of the Actions API in this session, prompted by Kai's
                        chronology and NOT by trusting it.

                        Recorded explicitly because an earlier draft of this
                        incident was going to say the inert suite "was found
                        by later source review rather than by any control we
                        have." THAT WOULD HAVE BEEN FALSE, and Kai caught it.
                        A finding that a control did not exist, when it
                        existed and fired, is the most expensive kind of
                        wrong entry this ledger can carry: it argues for
                        building a detector we already have, and it conceals
                        the actual defect, which is ours.

affected_scope          Branch claude/project-rework-plan-pgvp35 only.
                        Nine commits: 859526c2, 8fc5bc08, cb8cc270, 009cfadd,
                        79cd2dbc, 53389986, efb9c63a, 18faee4a, 0c717ec3.
                        No merge to main. No production cutover. No D-number.
                        No floor value altered. PR #122 remains DO NOT MERGE.

                        BOUNDED-SEARCH QUALIFIER. Three other workflows are
                        also non-success at these SHAs — "Core Tests",
                        "Python application" and "Unified Hunter Suites".
                        THEIR FAILING STEPS WERE NOT INSPECTED. Nothing here
                        classifies them as related, pre-existing, expected or
                        unrelated. They are named because they were returned
                        by the enumeration, not because their cause is known.
                        "PM Status Check" was success at every SHA inspected.

downstream_impact       No realised production impact. The cost is
                        programme-internal and real:

                        (a) WF-3 tranche progression, the real-78 shadow
                            traversal and its acceptance all proceeded on
                            commits whose required Policy-as-Code job already
                            carried an unexplained red;
                        (b) the floor-gate calibration was reasoned about as
                            though its repository wiring were settled, when
                            CI had already disproved that;
                        (c) the real-78 traversal could not have reached 78
                            under any outcome at plan member 27, because plan
                            member 32 (test-test-wiring) was already failing
                            for condition (2) — measured locally at HEAD:
                            `make test-test-wiring` -> exit 2,
                            "Test Wiring Tests: 18 passed, 1 failed".
                            Traversal effort was spent against a population
                            that could not complete.

mechanism_status        INCIDENT_ONLY
mechanism_id            none assigned. Explicitly NOT promoted into
                        M-POLICY-ADMISSION-DIVERGENCE, NOT into M-SCOPE-WIDEN,
                        NOT into any existing mechanism.
related_findings        KAI-GATE-018 / test-wiring is a strong LOCATOR: the
                        prohibited state is exactly what that machine control
                        exists to detect. A repository finding ID is not
                        automatically a ledger mechanism, and this one is not
                        being made into one.
related_incidents       INC-2026-09-14-19 — same commit, same suite; cited for
                        the self-documented control_type field quoted above,
                        not as causal equivalence.
recurrence_count        1 measured occurrence of this incident

stop_signal             "I am about to bank, advance a tranche, or accept an
                        evidence package. What is the exact-SHA CI status,
                        and have I read every required red down to its first
                        failing step AND accounted for the steps that were
                        skipped rather than passed?"

current_control         MANUAL PROCESS CONTROL. Before any tranche is
                        accepted or released after a banked commit:
                          1. enumerate every workflow run for that exact SHA;
                          2. inspect every required red;
                          3. identify the first failed step;
                          4. ALSO account for every step reported `skipped`
                             after that failure — a short-circuiting job hides
                             later breaches, as measured above, so the first
                             failed step is a LOCATOR, NOT THE POPULATION;
                          5. classify each red as expected known-negative /
                             pre-existing / tranche-caused / unrelated, with
                             evidence;
                          6. do not advance while any required red is
                             unexplained.
control_type            MANUAL. NOT MACHINE. NOT CONTROL_OPERATIONALISED.
                        No executing mechanism enforces any of the six steps.
                        It must not be described as machine-enforced until
                        one does.
control_introduced_at   this append

recurred_after_control  YES — the prohibited condition occurred while the
                        KAI-GATE-018 machine control already existed, and the
                        control DETECTED it.

                        THIS IS NOT EVIDENCE THAT THE MACHINE DETECTOR
                        FAILED. The detector contained the condition and
                        reported it correctly on every one of the nine
                        commits. The newly exposed weakness is the human
                        acceptance loop: a red exact-SHA CI result was not
                        reconciled before the next tranche advanced. Do not
                        read this field as a detector-reliability signal.

owner/stage             Orion (execution) · Kai (adjudication) · Dainius
                        (consequential authority)
status                  OPEN. The two prohibited conditions are unrepaired by
                        design — DOC-1 is sequenced ahead of them. This
                        incident closes only when §24 clears BOTH step 26 and
                        step 45 on an exact-SHA CI run that is read and
                        classified.
```

---

## Ledger state after this append

| id | producer | mechanism_status | assigned mechanism |
|---|---|---|---|
| `INC-2026-08-29-01` … `-03` | Orion | `PATTERN_CONFIRMED` | `M-SCOPE-WIDEN` |
| `INC-2026-08-29-04` `-05` | Orion | `INCIDENT_ONLY` | none |
| `INC-2026-08-29-06` | Kai | `INCIDENT_ONLY` | none — locator only |
| `INC-2026-08-29-07` `-08` | DeepSeek | `INCIDENT_ONLY` | none |
| `INC-2026-08-29-09` `-10` | Orion | `INCIDENT_ONLY` | none |
| `INC-2026-08-29-11` | Kai | `INCIDENT_ONLY` | none |
| `INC-2026-08-30-12` | Orion | `PATTERN_CANDIDATE` | `M-QUERY-OVERREACH` |
| `INC-2026-09-12-13` | Orion | `PATTERN_CANDIDATE` | none assigned |
| `INC-2026-09-12-14` | Orion | `PATTERN_CANDIDATE` | none assigned |
| `INC-2026-09-14-15` | Orion | `INCIDENT_ONLY` | none assigned |
| `INC-2026-09-14-16` | Orion | `INCIDENT_ONLY` | none assigned |
| `INC-2026-09-14-17` | Orion | `INCIDENT_ONLY` | none assigned — `M-SCOPE-WIDEN` locator only |
| `INC-2026-09-14-18` | Orion | `INCIDENT_ONLY` | none assigned — supporting evidence for `M-POLICY-ADMISSION-DIVERGENCE` |
| `INC-2026-09-14-19` | Orion | `INCIDENT_ONLY` | none assigned — supporting evidence for `M-POLICY-ADMISSION-DIVERGENCE` |
| `INC-2026-09-15-20` | Orion (implementation) · review loop (acceptance) | `INCIDENT_ONLY` | none assigned — `KAI-GATE-018` locator only |

**Producers: Orion 16 · Kai 2 · DeepSeek 2. Total incidents 20.** Derivation:
`grep -oE 'INC-2026-[0-9]{2}-[0-9]{2}-[0-9]+' kai-pm/FAILURE_PATTERN_LEDGER.md
| sort -u | wc -l` returned 19 before this append; this append adds
`INC-2026-09-15-20`. The counts carry no fairness, quality or
producer-reliability inference; they are the currently recorded population
and nothing more. `INC-2026-09-15-20`'s acceptance-side defect is attributed
to the review loop as a whole and is deliberately NOT counted against any
single producer — the implementation side is counted against Orion.

**Mechanisms: unchanged by this append.** `M-SCOPE-WIDEN`
`PATTERN_CONFIRMED` · `M-PRODUCER-CURATION` `PATTERN_CANDIDATE` ·
`P-ADJUDICATOR-PROPAGATION` `PATTERN_CANDIDATE` · `M-QUERY-OVERREACH`
`PATTERN_CANDIDATE` · `M-POLICY-ADMISSION-DIVERGENCE` `PATTERN_CANDIDATE`.
No mechanism's recurrence count is altered. `INC-2026-09-15-20` is
`INCIDENT_ONLY` with no mechanism assigned.

**Escalation state.** Doctrine 49.6 fires on the third *independently
confirmed* occurrence of ONE mechanism. This append confirms no mechanism
and therefore advances no escalation. The nine-commit span recorded above is
NINE OCCASIONS OF ONE UNCONSUMED SIGNAL, not nine independent occurrences of
a mechanism, and must not be counted as recurrence evidence.

**Doctrine 49.2 note.** The detector firing nine times is an incident record,
not a mechanism. What is established is that on this branch, between
`859526c2` and `0c717ec3`, the machine layer was correct and the acceptance
layer did not consume it. Whether that acceptance failure has a mechanism —
and whether it has occurred elsewhere in the programme — is UNMEASURED and
is not claimed here.

---

# APPEND 2026-09-15 (third) — the calibration proved a copy of the policy,
#                             and the mechanism is now confirmed

One incident, and the first promotion of a mechanism in this ledger from
`PATTERN_CANDIDATE` to `PATTERN_CONFIRMED`.

The incident is small. What it establishes is not: the third occurrence
crosses out of the gate family the first two shared, which is the evidence
that was missing when the candidate was registered.


### `INC-2026-09-15-21` — the registration's calibration verified a
###                       duplicate of the declaration, not the declaration

```
INCIDENT_ID             INC-2026-09-15-21
date                    2026-09-15
producer                Orion
subject/version         §24 WF-3 registration
                        scripts/security/gate_registry.py — the uh_runner and
                        uh_floor_gate entries
                        scripts/test_uh_runner.py
                        scripts/test_uh_floor_gate.py
                        at: a4296edf527f07c8c661fca686d92804bc35cfba

false_or_faulty_output  Both new registry rows declare a denominator, and
                        both carry `probe=False` with a skip reason naming
                        the calibration suite that verifies it instead. That
                        is the house form and it is the right shape.

                        THE SUITES DID NOT READ THE DECLARATION. Each typed
                        its own second copy:

                          gate_registry.py
                            uh_runner.denominator =
                              r"Unified Hunter — \d+ targets from "
                            uh_floor_gate.denominator =
                              r"Assertion floors — \d+ targets, \d+ floored,
                                \d+ assertions"

                          test_uh_runner.py
                            DECLARED_DENOMINATOR =
                              r"Unified Hunter — \d+ targets from "
                          test_uh_floor_gate.py
                            DECLARED_DENOMINATOR =
                              r"Assertion floors — \d+ targets, \d+ floored,
                                \d+ assertions"

                        The scenarios then proved the instrument's real
                        output matched the LOCAL COPY. The comment beside
                        each said "the registry's declared denominator is
                        what it prints". What was proven is "a second regex
                        typed into this file matches what it prints". Those
                        are different propositions, and only the first is
                        the one the machine admits on.

                        The consequential path does not close the gap.
                        `check_gate_registry.probe_denominator` returns
                        `"skipped"` when `probe` is False, and `cross_check`
                        raises a denominator finding only for `"missing"` or
                        `"absent"`. I-3 checks that `proven_by` NAMES A FILE
                        THAT EXISTS; it never executes it and never compares
                        it to the declaration. So nothing anywhere binds the
                        registry regex to the suite that claims to verify it.

                        REPRODUCED, not reasoned. One character of the
                        authoritative registry changed and nothing else:

                          uh_runner.denominator = r"BOGUS NEVER MATCHES"

                          make test-uh-runner            exit 0
                                                         163 passed, 0 failed
                          make test-test-wiring          exit 0
                          check_gate_registry.py --gate  exit 0
                            "GATE PASSED: I-1, I-2, I-3, I-4, I-5, I-6, I-7
                             hold."

                        Every surface green on a declaration that can never
                        match anything. The same hole exists independently
                        for uh_floor_gate.

corrected_output        See the repair commit recorded beside this entry.
                        One authoritative declaration per gate — the registry
                        row — imported by the calibration through
                        `gate_registry.BY_MODULE`, with the local copies
                        removed; the suites additionally assert that
                        `proven_by` names the suite that is executing, so the
                        binding is checked in both directions. Both
                        calibration suites are invoked by required
                        Policy-as-Code so the assertions are consequential
                        rather than locally green.

                        This repairs TWO INSTANCES. It does not implement the
                        generic control — see the escalation section.

detection_method        Kai independent source review, performed AFTER the
                        exact-SHA §24 CI came back green. Recorded plainly:
                        the producer's own evidence package, the mutation
                        testing in it, and a fully green required workflow
                        all failed to surface this. Green CI is what the
                        defect looks like.

                        Worth stating because the §24 package did carry
                        12-of-12 mutation detection — but every mutation was
                        injected into `uh_runner.py`, the implementation. Not
                        one was injected into the REGISTRY, the authority. A
                        mutation set aimed only at the implementation cannot
                        find a defect in what the implementation is measured
                        against.

affected_scope          Two shadow registry entries on branch
                        claude/project-rework-plan-pgvp35. Both gates are
                        `pending_wiring` and enforce nothing. No production
                        caller, no floor value, no cutover, no merge to main.

downstream_impact       NONE realised. The declarations happened to be
                        correct; the defect is that nothing was checking
                        whether they were. The realisable impact is a false
                        denominator coexisting with green calibration, green
                        gate-registry and green Policy-as-Code — I-2's "pass
                        that cannot be falsified" inside the file whose job
                        is to prevent exactly that.

mechanism_status        PATTERN_CONFIRMED
mechanism_id            M-POLICY-ADMISSION-DIVERGENCE
related_incidents       INC-2026-09-14-18, INC-2026-09-14-19 — the two
                        supporting occurrences, now confirmed as occurrences
                        of the same mechanism. Their historical records are
                        UNCHANGED; the adjudication is made here.
recurrence_count        3 confirmed occurrences

stop_signal             "I have written the authoritative value in one file
                        and the expected value in another. Which one does
                        the machine admit on, and is my test reading THAT
                        object or a copy of it?"

current_control         Repaired for these two instances only: the
                        calibration imports the registry object, the local
                        copies are gone, and a hostile mutation of the
                        registry row alone now turns the calibration red.
control_type            STRUCTURAL and OPERATIONALISED for the two named
                        gates — the suites execute in required
                        Policy-as-Code, so the binding is consequential.
                        NOT MECHANISED as a general rule: nothing prevents
                        the next registry entry from being calibrated
                        against a retyped copy.
control_introduced_at   the repair commit recorded beside this append

recurred_after_control  YES — third occurrence of a mechanism registered as
                        a candidate on 2026-09-15, one day after the second.

owner/stage             Orion (execution) · Kai (adjudication) · Dainius
                        (consequential authority)
status                  CLOSED as an incident. The mechanism is OPEN and
                        escalated.
```

---

## `M-POLICY-ADMISSION-DIVERGENCE` — PROMOTED TO `PATTERN_CONFIRMED`

**This section does not replace the `PATTERN_CANDIDATE` registration above.
That entry stands as written, with its explicit statement that the two
occurrences were not independent. This is the adjudication that followed.**

**State: `PATTERN_CONFIRMED`. Adjudicated by Kai, 2026-09-15.**

**The mechanism, unchanged in substance.** A gate carries a normative
admission policy in one surface — prose, documentation, reporting or test
intent — while the machine predicate that actually decides PASS, FINDING,
REFUSAL, admissibility, adjudication or exit implements a different
proposition. The producer's calibration then validates a SURROGATE of the
policy rather than the policy itself, so the two can diverge while every
surface stays green.

**The three confirmed occurrences, and the one proposition they share.**

| # | incident | the authoritative proposition | what the calibration actually checked |
|---|---|---|---|
| 1 | `INC-2026-09-14-18` | every member proves one exact target-bound result | a fixture that fabricated `result_label` — the field the contract turns on |
| 2 | `INC-2026-09-14-19` | an unfloored member is not a pass | the explanatory report, never the consequential exit code |
| 3 | `INC-2026-09-15-21` | the registry row's denominator | a second regex typed into the test file |

In all three the calibration proved a surrogate, and the consequential
authority remained free to diverge while green.

**Why the third occurrence confirms what two could not.** The candidate
registration recorded, correctly, that occurrences 1 and 2 were not
independent: same producer, same gate family, same tranche, same development
period. That reservation is still true of them. The third occurrence
supplies what they lacked — it crosses a component boundary, from the WF-3
floor consumer into the **instrumentation registry and its meta-gate**,
which is a different component with a different author history and its own
invariant framework (I-1 to I-7). The mechanism is therefore not a property
of one gate's construction. Same producer still; the common-authority
qualifier below is not withdrawn.

**Doctrine 49.6: TRIGGERED.** Third confirmed occurrence of ONE mechanism.
The producer is not the finding — the CONTROL is. Prose is no longer an
acceptable remedy, and this append does not reissue the reminder.

**Escalation, stated at the level actually reached.**

```
mechanism                          PATTERN_CONFIRMED
doctrine 49.6                      TRIGGERED
machine escalation                 BEGUN — PARTIAL
generic cross-component control    NOT IMPLEMENTED
mechanism controlled               NO
```

*Operationalised now, for the two named gates only:*

1. one authoritative denominator declaration per gate, in the registry;
2. the calibration imports that object rather than restating it;
3. both suites assert `proven_by` names the suite that is executing;
4. both suites are invoked by required Policy-as-Code, so their assertions
   are consequential rather than locally green;
5. a hostile mutation of the registry row **alone** makes the calibration
   red — proven in both directions before banking.

*Not implemented, and not to be reported as though it were:* the general
rule that **a calibration must consume the same authoritative object the
admission path consumes, never a retyped equivalent.** Nothing today stops
the next registry entry, the next gate or the next producer from calibrating
against a copy. `check_gate_registry` still admits `probe=False` without
ever comparing the declaration to anything, and I-3 still checks only that
`proven_by` names a file that exists.

**Machine-hook owner:** WF-3 / instrumentation-governance tooling. The
shape a generic control would take — an I-8 extension requiring that a
`probe=False` entry's `proven_by` suite demonstrably consumes
`BY_MODULE[module]` — is recorded as a direction, not a design, and nothing
implements it.

**What this append does NOT claim.** It does not claim the mechanism is
controlled. It does not claim the two repaired instances generalise. It does
not withdraw the same-producer qualifier: all three occurrences are Orion's,
so this is recurrence across components under one producer, not across
producers. And it does not alter `M-SCOPE-WIDEN`, `M-PRODUCER-CURATION`,
`P-ADJUDICATOR-PROPAGATION` or `M-QUERY-OVERREACH`, whose counts and states
are untouched.

---

## Ledger state after this append

| id | producer | mechanism_status | assigned mechanism |
|---|---|---|---|
| `INC-2026-08-29-01` … `-03` | Orion | `PATTERN_CONFIRMED` | `M-SCOPE-WIDEN` |
| `INC-2026-08-29-04` `-05` | Orion | `INCIDENT_ONLY` | none |
| `INC-2026-08-29-06` | Kai | `INCIDENT_ONLY` | none — locator only |
| `INC-2026-08-29-07` `-08` | DeepSeek | `INCIDENT_ONLY` | none |
| `INC-2026-08-29-09` `-10` | Orion | `INCIDENT_ONLY` | none |
| `INC-2026-08-29-11` | Kai | `INCIDENT_ONLY` | none |
| `INC-2026-08-30-12` | Orion | `PATTERN_CANDIDATE` | `M-QUERY-OVERREACH` |
| `INC-2026-09-12-13` | Orion | `PATTERN_CANDIDATE` | none assigned |
| `INC-2026-09-12-14` | Orion | `PATTERN_CANDIDATE` | none assigned |
| `INC-2026-09-14-15` | Orion | `INCIDENT_ONLY` | none assigned |
| `INC-2026-09-14-16` | Orion | `INCIDENT_ONLY` | none assigned |
| `INC-2026-09-14-17` | Orion | `INCIDENT_ONLY` | none assigned — `M-SCOPE-WIDEN` locator only |
| `INC-2026-09-14-18` | Orion | `PATTERN_CONFIRMED` | `M-POLICY-ADMISSION-DIVERGENCE` — confirmed occurrence 1, adjudicated here; the incident's own record is unchanged |
| `INC-2026-09-14-19` | Orion | `PATTERN_CONFIRMED` | `M-POLICY-ADMISSION-DIVERGENCE` — confirmed occurrence 2, adjudicated here; the incident's own record is unchanged |
| `INC-2026-09-15-20` | Orion (implementation) · review loop (acceptance) | `INCIDENT_ONLY` | none assigned — `KAI-GATE-018` locator only |
| `INC-2026-09-15-21` | Orion | `PATTERN_CONFIRMED` | `M-POLICY-ADMISSION-DIVERGENCE` — confirmed occurrence 3 |

**Producers: Orion 17 · Kai 2 · DeepSeek 2. Total incidents 21.** Derivation:
`grep -oE 'INC-2026-[0-9]{2}-[0-9]{2}-[0-9]+' kai-pm/FAILURE_PATTERN_LEDGER.md
| sort -u | wc -l` returned 20 before this append; this append adds
`INC-2026-09-15-21`. The counts carry no fairness, quality or
producer-reliability inference.

**Mechanisms.** `M-SCOPE-WIDEN` `PATTERN_CONFIRMED` · **`M-POLICY-ADMISSION-DIVERGENCE`
`PATTERN_CONFIRMED` (promoted by this append, 3 confirmed occurrences)** ·
`M-PRODUCER-CURATION` `PATTERN_CANDIDATE` · `P-ADJUDICATOR-PROPAGATION`
`PATTERN_CANDIDATE` · `M-QUERY-OVERREACH` `PATTERN_CANDIDATE`. No other
mechanism's state or recurrence count is altered.

**Escalation state.** Doctrine 49.6 has fired for
`M-POLICY-ADMISSION-DIVERGENCE` on its third confirmed occurrence. Machine
escalation is BEGUN and PARTIAL: operationalised for the two named gates,
with no generic cross-component control implemented. The mechanism is NOT
controlled. The independence / common-authority locator shape at
`INC-2026-08-29-10` and `INC-2026-09-12-14` still stands at two preserved
occurrences, unchanged.

---

# APPEND 2026-09-16 — a freeze read as permanent state, and a distribution
#                     claim that was never measured

Two incidents, both mine, both caught by someone else before they entered a
governed record — one of them by the adjudicator reopening a lineage I had
stopped reading, the other by an allocator check I ran for an unrelated
reason.

The first is the more serious. It would have put a defect INTO an
append-only correction written to remove a defect.


### `INC-2026-09-15-22` — a dated freeze decision reported as the current
###                       governance identity, six days after it moved

```
INCIDENT_ID             INC-2026-09-15-22
date                    2026-09-15
producer                Orion
subject/version         RC-1 lineage analysis and the draft D375 correction
                        kai-pm/DECISIONS.md D357 (freeze)
                        kai-pm/house_in_order_census_v11/MANIFEST.sha256
                        at: 5be90db24e054ffbc4d18b58ddc98273dafd911a

false_or_faulty_output  Transmitted to Kai, twice, and carried into a draft
                        append-only correction of D375:

                          "for frozen Census v1.1 the governance identity
                           is eb7aad7c...fa0e"

                        and, built on it, the claim that D375's Census value
                        29064d65...757a was "not the governance identity",
                        offered as one of four rows supporting a finding that
                        D375's aggregate column was wrong.

                        I also reported the aggregate derivation as
                        UNRESOLVED after failing to reproduce eb7aad7c from
                        the manifest by six candidate formulas.

what was actually true  D357 froze the PREDECESSOR Census identity at
                        eb7aad7c...fa0e. On 2026-09-09, commit bd1cbb4b
                        performed S1 O3 consumption-time source hardening
                        UNDER A NARROW FROZEN-PACKAGE EXCEPTION KAI
                        AUTHORISED, changing docgraph.py, opscan.py, the
                        package MANIFEST.sha256, and the consuming
                        h2_v13/passa.py. The manifest was regenerated and the
                        movement was surfaced deliberately; retaining the old
                        manifest to preserve eb7aad7c was explicitly rejected
                        as false attestation. S1 FINAL ACCEPTANCE banked the
                        hardened lineage and preserves BOTH digest lineages,
                        neither silently rewritten into the other.

                        So 29064d65...757a is the LATER AUTHORISED HARDENED
                        lineage, and D375 records it correctly.

                        The "UNRESOLVED" derivation resolved the moment the
                        commit boundary was measured instead of the current
                        tree:

                          sha256(MANIFEST.sha256) @ bd1cbb4b~1 = eb7aad7c...
                          sha256(MANIFEST.sha256) @ bd1cbb4b   = 29064d65...

                        Two aggregate conventions exist, both principled and
                        both reproduced exactly: house_in_order_h2 embeds its
                        own aggregate (sha256 of the entry lines, fa847726...,
                        written back as a comment, so the whole-file hash
                        necessarily differs); the later packages use
                        sha256 of the whole manifest file.

mechanism               A DATED ADJUDICATION READ AS CURRENT STATE.

                        This is not R16. I did open the authoritative source
                        and I read it correctly. D357 says what I said it
                        says. What I never did was ask whether anything had
                        happened to it SINCE, and a freeze is exactly the
                        kind of record that invites that omission: it
                        announces permanence in its own language.

                        Doctrine 0.0 states the principle -- nothing is true
                        because it was true last time -- and I have quoted it
                        in a staleness predicate I wrote into a document the
                        day before this incident.

                        The six failed derivation formulas are the tell I
                        missed. I was trying to reproduce a digest from the
                        CURRENT manifest that had been computed from a
                        DIFFERENT one. Six failures in a row is not a hard
                        problem; it is the wrong subject. R15 applied and I
                        recorded the failure as a limitation instead of
                        checking why.

detection               KAI, by independently reopening the S1 lineage and
                        supplying the commit. Not by me, and not by any
                        control. I then verified it at source with
                        `git show --numstat bd1cbb4b` and by hashing the
                        manifest on both sides of the commit boundary.

cost                    Kai was reasoning with a false premise across two
                        exchanges. Had he not reopened the lineage, the
                        correction of D375 would have entered an APPEND-ONLY
                        log asserting that D375 got Census wrong, when D375
                        had it right -- a correction carrying the defect it
                        was written to fix, in the one file where a mistake
                        cannot be edited out.

status                  INCIDENT_ONLY. No mechanism assigned.

                        One occurrence. Doctrine 37 and R18 bind: two
                        mistakes that look alike are a locator, not a cause,
                        and the mechanism has to be earned. Searched this
                        ledger for a prior staleness-shaped occurrence before
                        writing this entry: none found. The shape is recorded
                        so a second occurrence can be recognised, and it is
                        NOT registered as a mechanism on a population of one.

control state           NONE. No control exists that asks whether a cited
                        decision has been acted on since. Deliberately not
                        proposed here: an automatic supersession detector is
                        exactly the kind of generic mechanism this programme
                        has just spent a tranche refusing to build on one
                        occurrence.
```


### `INC-2026-09-15-23` — a count measured, a distribution asserted

```
INCIDENT_ID             INC-2026-09-15-23
date                    2026-09-15
producer                Orion
subject/version         KAI_KINGSMAN_COLD_START_MASTER.md §5.2,
                        banked at 5be90db24e054ffbc4d18b58ddc98273dafd911a

false_or_faulty_output  "D344 through D353. Nothing else in the entire range
                        is absent." and "the only hole in a 358-entry
                        ledger". Transmitted to Dainius, transmitted to Kai,
                        and COMMITTED to the repository.

what was actually true  Three gaps, not one. Absent in 1..375:
                        D103, D104, D105, D106, D107, D108, D260, and
                        D344..D353. Seventeen absent, not ten.
                        `grep -c "\bD103\b"` etc. return 0: they are absent
                        in any form, not merely absent as headings.

                        The entry count was right. 375 - 17 = 358, which is
                        the figure I reported. I had the correct total and
                        described its distribution without ever computing
                        one.

mechanism               `M-SCOPE-WIDEN` -- measured subject is not the
                        transmitted subject. I measured CARDINALITY and
                        transmitted a claim about CONTIGUITY. Those are
                        different predicates over the same data, and the
                        second was never run.

                        R17 in its exact form: the sentence covered more
                        ground than the check. The universal quantifier
                        "nothing else in the entire range" is the tell; I
                        wrote it without a predicate that could have
                        falsified it.

detection               By me, from an allocator check run for an unrelated
                        purpose -- establishing the next free D-number before
                        banking. The absent list printed itself. Nothing was
                        looking for this.

cost                    Low so far and bounded: no decision rests on it. But
                        it is committed in a navigation document written to
                        stop a cold thread re-deriving programme state, so
                        its cost is deferred rather than absent. Recorded
                        here for the next controlled correction of that
                        document; NOT repaired in this append, because
                        bundling an unrelated documentation repair into a
                        ledger commit is the defect shape this programme
                        keeps removing.

status                  `M-SCOPE-WIDEN`, confirmed mechanism, one further
                        occurrence. The mechanism's state is unchanged --
                        it was already PATTERN_CONFIRMED and remains so.

control state           UNCHANGED. R17 is banked doctrine, quoted in the R0
                        stop-signal table, and it did not stop me. This is a
                        producer-side recurrence against an existing rule,
                        not evidence that the rule is wrong. No escalation is
                        claimed on one further occurrence; doctrine 49.6's
                        third-occurrence test applies to a mechanism's
                        CONTROL, and M-SCOPE-WIDEN's control is the doctrine
                        rule itself, which is already banked and already
                        cited.
```


### Roster delta

| incident | producer | status | mechanism |
|---|---|---|---|
| `INC-2026-09-15-22` | Orion | `INCIDENT_ONLY` | none assigned — dated-adjudication-as-current-state, locator only |
| `INC-2026-09-15-23` | Orion | recurrence | `M-SCOPE-WIDEN` — one further occurrence |

**Producers: Orion 19 · Kai 2 · DeepSeek 2. Total incidents 23.** Derivation:
`grep -oE 'INC-2026-[0-9]{2}-[0-9]{2}-[0-9]+' kai-pm/FAILURE_PATTERN_LEDGER.md
| sort -u | wc -l` returned 21 before this append; this append adds
`INC-2026-09-15-22` and `INC-2026-09-15-23`. The counts carry no fairness,
quality or producer-reliability inference.

**Mechanisms.** `M-SCOPE-WIDEN` `PATTERN_CONFIRMED` (one further occurrence,
state unchanged) · `M-POLICY-ADMISSION-DIVERGENCE` `PATTERN_CONFIRMED` ·
`M-PRODUCER-CURATION` `PATTERN_CANDIDATE` · `P-ADJUDICATOR-PROPAGATION`
`PATTERN_CANDIDATE` · `M-QUERY-OVERREACH` `PATTERN_CANDIDATE`. **No new
mechanism is registered by this append.** No other mechanism's state or
recurrence count is altered.

**Escalation state.** Unchanged. Doctrine 49.6 remains fired for
`M-POLICY-ADMISSION-DIVERGENCE`, machine escalation BEGUN and PARTIAL, the
mechanism NOT controlled. Nothing in this append advances or retires it.

**What both incidents share, recorded as an observation and not as a
mechanism.** In each, the measurement was correct and the sentence built on
it was not: a freeze correctly read but not followed forward, a cardinality
correctly counted but described as a distribution. That is a locator across
two entries by the same producer on the same day. It is not promoted here,
and it must not be cited as a cause.

---

# APPEND 2026-09-17 — an authority record that declared a complete mutation
#                     surface it had not measured

One incident. It reached an APPEND-ONLY authority record and was caught by
independent contract review **before any implementation began**, so no
out-of-authority mutation occurred. The stop worked.


### `INC-2026-09-17-24` — D376's "exact mutation surface" omitted the paths
###                       a governed generator necessarily derives

```
INCIDENT_ID             INC-2026-09-17-24
date                    2026-09-17
producer                Orion
subject/version         kai-pm/DECISIONS.md — D376 §4 EXACT MUTATION SURFACE
                        banked at e0507701b31fe9444c6fe3e6156512cd861993df

false_or_faulty_output  D376 §4 declares:

                          "EXACT MUTATION SURFACE — 15 PATHS"

                        and closes the boundary explicitly:

                          "Any path not in this list is outside authority."

                        README.md and docs/PROJECT_BACKLOG.md appear
                        nowhere in that list.

what was actually true  scripts/sync_docs.py is an existing governed
                        generator. Measured at this subject, not assumed:

                          count_test_files()     globs scripts/test_*.py
                          count_test_functions() counts ^\s*def test_ in
                                                 those same files
                          count_python_loc()     rglobs every *.py except
                                                 .git, __pycache__, _archive

                        and it writes exactly two files, at
                        sync_docs.py:200 and :287 —
                          README.md
                          docs/PROJECT_BACKLOG.md

                        The tranche authorised by D376 adds 8 new .py files,
                        4 of them scripts/test_*.py, and the house
                        calibration style does use `def test_` — measured,
                        not assumed: 22 in test_uh_runner.py, 40 in
                        test_uh_floor_gate.py, 39 in test_gate_registry.py.
                        It also retires one `def test_` from
                        test_p1_p4_enhancements.py.

                        So the authorised work NECESSARILY moves the
                        individual-test count, the test-file count and the
                        Python LOC figure, and therefore both generated
                        files.

                        Sharper than the review's statement, and measured:
                        the TARGETS metric does NOT move. count_test_targets()
                        parses the `test-core:` dependency list, and the
                        tranche deliberately adds no prerequisite to it.

                        The contract as banked is therefore not executable:
                        syncing the documentation violates D376's own
                        boundary clause, and not syncing it knowingly leaves
                        generated documentation stale and the documentation
                        gate red. Neither branch is acceptable.

mechanism               `M-SCOPE-WIDEN`.

                        Assigned against the roster's definition rather than
                        by resemblance, as the review required. The roster
                        at the time of writing: M-SCOPE-WIDEN and
                        M-POLICY-ADMISSION-DIVERGENCE PATTERN_CONFIRMED;
                        M-PRODUCER-CURATION, P-ADJUDICATOR-PROPAGATION and
                        M-QUERY-OVERREACH PATTERN_CANDIDATE. The defining
                        test for M-SCOPE-WIDEN is MEASURED SUBJECT IS NOT
                        TRANSMITTED SUBJECT, and that test is met exactly: I
                        enumerated THE PATHS I WOULD AUTHOR and transmitted
                        it as THE COMPLETE SET OF PATHS THAT WOULD CHANGE.
                        Those are two different sets and only the first was
                        measured.

                        A distinguishing sub-shape is recorded WITHOUT
                        minting a new identifier, on one occurrence, per
                        doctrine 37: the omitted paths are produced by an
                        EXISTING GOVERNED GENERATOR, not by the author. No
                        amount of enumerating my own edits would have
                        surfaced them; only asking what the repository
                        derives FROM those edits would. First-order surface
                        declared as total surface.

                        NO ESCALATION IS ADVANCED BY THIS ENTRY. The review
                        that found this explicitly warned against advancing
                        escalation mechanically from its wording, and
                        doctrine 49.6's third-occurrence test applies to a
                        mechanism's CONTROL. M-SCOPE-WIDEN's control is the
                        banked doctrine rule R17/48, already in the R0
                        stop-signal table. What this occurrence establishes
                        is a RECURRENCE RATE worth an adjudicator's
                        attention — INC-2026-09-15-23 was two days ago — and
                        that judgement is Kai's, not mine.

detection               KAI, by independent contract review of the banked
                        D376 text before implementation started. Not by me,
                        and not by any control. Verified at source here:
                        sync_docs.py metric scanners and both write sites,
                        and the `def test_` population of three existing
                        house suites.

cost                    NONE REALISED. No implementation had begun; no path
                        outside authority was touched; the working tree was
                        clean at e050770. The counterfactual is the finding:
                        had the build started, the FIRST CORRECT ACTION —
                        running the documentation sync the repository already
                        requires — would itself have exceeded the recorded
                        authority, and the alternative would have been to
                        suppress a sync and call a knowingly stale tree
                        verified.

status                  `M-SCOPE-WIDEN`, confirmed mechanism, one further
                        occurrence. The mechanism's state is UNCHANGED — it
                        was already PATTERN_CONFIRMED and remains so.

control state           UNCHANGED, and the gap is worth naming precisely: no
                        control asks "what does this repository DERIVE from
                        the paths I am about to change?" before a mutation
                        surface is declared. None is proposed here. Proposing
                        a generic derived-surface control on one occurrence is
                        the shape this programme has spent the whole RC-1
                        tranche declining to build, and the correction that
                        matters is the authority record, not a new detector.
```


### Roster delta

| incident | producer | status | mechanism |
|---|---|---|---|
| `INC-2026-09-17-24` | Orion | recurrence | `M-SCOPE-WIDEN` — one further occurrence, state unchanged |

**Producers: Orion 20 · Kai 2 · DeepSeek 2. Total incidents 24.** Derivation:
`grep -oE 'INC-2026-[0-9]{2}-[0-9]{2}-[0-9]+' kai-pm/FAILURE_PATTERN_LEDGER.md
| sort -u | wc -l` returned 23 before this append; this append adds
`INC-2026-09-17-24`. The counts carry no fairness, quality or
producer-reliability inference.

**Mechanisms.** `M-SCOPE-WIDEN` `PATTERN_CONFIRMED` (one further occurrence,
state unchanged) · `M-POLICY-ADMISSION-DIVERGENCE` `PATTERN_CONFIRMED` ·
`M-PRODUCER-CURATION` `PATTERN_CANDIDATE` · `P-ADJUDICATOR-PROPAGATION`
`PATTERN_CANDIDATE` · `M-QUERY-OVERREACH` `PATTERN_CANDIDATE`. **No new
mechanism is registered. No mechanism state is altered by this append.**

**Escalation state.** UNCHANGED, and deliberately so. Doctrine 49.6 remains
fired for `M-POLICY-ADMISSION-DIVERGENCE` only, machine escalation BEGUN and
PARTIAL, that mechanism NOT controlled. **Nothing in this append advances,
retires or re-triggers any escalation.** The `M-SCOPE-WIDEN` recurrence rate
is surfaced for adjudication and is not acted on by the producer who caused it.

---

# ADJUDICATION 2026-09-17 — `M-SCOPE-WIDEN` advances to
#                           `RECURRED_AFTER_CONTROL`

**This is a MECHANISM-STATE ADJUDICATION, not an incident.** No `INC-` is
allocated. `INC-2026-09-17-24` is the occurrence; this records its effect on
the existing mechanism's state. The original `M-SCOPE-WIDEN` mechanism
section, `INC-2026-09-17-24`, and every prior entry are **unedited**.

**Adjudicator: Kai**, on independent review of the D376/D377 authority
surface, 2026-09-17.

```
mechanism                       M-SCOPE-WIDEN
previous confirmed occurrences  3
                                INC-2026-08-29-01, -02, -03
                                all adjudicated by Kai
new supporting occurrence       INC-2026-09-17-24
confirmed occurrence count      4

control existing before
  the occurrence                doctrine 48 / CLAUDE.md R17 /
                                R0 stop-signal table
control type at recurrence      MANUAL
recurred_after_control          YES
mechanism state                 RECURRED_AFTER_CONTROL
machine control                 NOT IMPLEMENTED
controlled                      NO
mechanised                      NO
```

### What this establishes

**The fourth occurrence does not re-run the third-occurrence rule.** Doctrine
49.6 triggered at occurrence 3, on 2026-08-29, and the control selected in
response was the manual one: doctrine 48, CLAUDE.md R17, and the R0
stop-signal rows. That control existed, was banked, was cited, and appears in
the stop-signal table a producer is meant to consult in flight.

`INC-2026-09-17-24` occurred on 2026-09-17, **after** that control existed.

**Therefore what the fourth occurrence proves is not that the mechanism
recurs — that was already established — but that THE CONTROL CHOSEN AFTER
ESCALATION HAS NOT PREVENTED RECURRENCE.** That is stronger evidence than
the original escalation, and it is the reason the state changes rather than
staying at `PATTERN_CONFIRMED`.

### What this explicitly does NOT establish

* **Doctrine 49.6 is NOT triggered "for the first time."** It fired at
  occurrence 3 and that firing stands unaltered.
* **No new mechanism is registered.** The generated-path sub-shape recorded
  in `INC-2026-09-17-24` — omitted paths produced by an existing governed
  generator rather than by the author — is a sub-shape of `M-SCOPE-WIDEN`
  and is **NOT** counted, promoted or named as a separate mechanism.
* **No causal independence from `M-SCOPE-WIDEN` is claimed.**
* **No generic scope-control system is authorised, designed or begun.**

### Machine-hook obligation — governance status changed, not implemented

The mechanism record already specifies the shape of the missing hook:
*reconcile measured/extracted scope against the transmitted consequential
claim scope, and refuse a claim whose declared universe exceeds what was
measured.*

```
BEFORE INC-2026-09-17-24   machine hook SPECIFIED but UNIMPLEMENTED
AFTER  INC-2026-09-17-24   machine hook is a REQUIRED OPEN CONTROL
                           OBLIGATION. M-SCOPE-WIDEN may NOT be described
                           as CONTROLLED or MECHANISED until it exists.
```

**THIS IS NOT IMPLEMENTATION AUTHORITY.** It is an outstanding control
requirement. A later design tranche must decide its architecture,
denominator and enforcement point. **It must NOT be bolted onto the D376
tranche because the recurrence happened there** — that is the opportunistic
shape this programme has spent the whole RC-1 sequence declining.

### Arithmetic made legible — one attribution this count does NOT include

`INC-2026-09-15-23` carries a **producer-assigned** `M-SCOPE-WIDEN`
attribution, banked by Orion on 2026-09-15 and **not adjudicated**.
**It is not included in the confirmed occurrence count of 4**, because
producer attribution is not adjudication and the three counted predecessors
are each recorded as `adjudicator (Kai)`.

**Recorded so the ledger's arithmetic is checkable rather than apparently
inconsistent:** a reader will find five `M-SCOPE-WIDEN` attributions in this
file and a confirmed count of four. The difference is `INC-2026-09-15-23`.
**If Kai adjudicates it as a confirmed occurrence, the count becomes 5 and
this append is superseded by that adjudication — which is Kai's to make and
is not made here.** Neither the count nor `INC-2026-09-15-23` is edited by
this entry.

### Unchanged by this adjudication

`M-POLICY-ADMISSION-DIVERGENCE` `PATTERN_CONFIRMED`, doctrine 49.6 fired,
machine escalation BEGUN and PARTIAL, mechanism NOT controlled — **no change
of any kind.** `M-PRODUCER-CURATION`, `P-ADJUDICATOR-PROPAGATION` and
`M-QUERY-OVERREACH` remain `PATTERN_CANDIDATE`. **No new mechanism. No new
incident. Highest incident remains `INC-2026-09-17-24`. Total incidents 24.**
