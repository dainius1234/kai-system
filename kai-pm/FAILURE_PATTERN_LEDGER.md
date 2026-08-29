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
