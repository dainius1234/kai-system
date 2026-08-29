# Engineering doctrine

Standing operating rules. **Not** temporary rules for KAI-GATE-048.

Every rule here was earned by a specific failure during that
investigation, and each is stated so it can be applied in flight rather
than admired afterwards. `CLAUDE.md` holds the repository's operating
rules and remains binding; this file holds the doctrine those rules serve
and applies equally to work delegated to subagents.

The operator's framing, which is the point of the whole document:

> Where a new failure teaches a defensible general rule, flag it,
> preserve the evidence that earned it, and propose whether it should
> join this doctrine. Do not quietly turn every incident into a law, but
> do not let earned lessons disappear either.

---

## 0.0 Nothing is true because it was true last time

**Every claim is re-earned at the moment it is relied upon.** A fact
established in a previous run, a previous commit or a previous
conversation is a **record of a past measurement**, never a licence for a
present one.

The operator's framing, which is the one to keep:

> Thor's hammer was never inherited. Odin did not hand it over — Thor had
> to be worthy of it, **every single time he picked it up.**

This is the spine the rest of this document hangs from. R1 is this rule
applied to assertions; R2 to contingencies; R5 to denominators; R11 to
prerequisites. It had never been stated on its own, which is why it kept
being rediscovered one venue at a time.

**Earned by, in one week:**

* `docker-compose.full.yml` declared
  `memu-graph → ollama-pull: service_completed_successfully`. That was
  true in the file and **not in force at runtime**, because the replay's
  `--no-deps` told Compose to ignore it. Ten replays hit a model that was
  not there and the run went green. A declaration that reads as
  protection while not being enforced is worse than a missing one,
  because everyone who greps for it stops looking. (D265, D266)
* The gate registry's `in_workflows` was accurate for every gate written
  before mine and false for the one I added the same hour.
  Correct-last-time is not correct-now. (D266)
* Attempt 1 froze the request correctly. That licensed nothing about
  Attempt 3 — which is why `--verify-request-hash` exists and why S1
  re-selection runs live on every attempt instead of trusting the
  frozen record. (D264, D267)

**The tell, in flight:** *"we already established that."* The moment a
claim is load-bearing because of something earlier rather than something
now — run it again.

**And the corollary that makes it more than a slogan:** a principle that
lives only in a document is itself a declaration that was true when
written. This one is enforced by
`scripts/security/check_declared_prerequisites.py`, which requires every
site that bypasses a declared condition to say which condition it skips
and what compensates for it. Without that, the rule would violate itself
on the day it was written. (D268)

---

## 0. Proactive engineering duty

**If you see a materially safer, stronger, more correct, more
maintainable or more evidentially defensible route, you must flag it —
even when nobody asked.**

Silence is not permission to take the easiest path. **The operator not
knowing that a technical question exists is not permission to ignore
it.**

When such a condition appears:

1. state what you observed;
2. explain why it matters in plain language;
3. distinguish **FACT / EVIDENCE / INFERENCE**;
4. present the realistic options;
5. recommend the strongest justified route;
6. identify cost, scope and risk;
7. **do not implement scope expansion without authorisation.**

Proactive engineering is not autonomous scope expansion.
**Flag → explain → recommend → request authority.**

---

## The rules

### Truth and promotion

1. **Truth outranks progress.** If something cannot be proved, do not
   promote it.
2. **Present ≠ executed ≠ enforced.** Configuration or source presence
   does not prove runtime behaviour.
24. **No finding closes because source looks better or a related test
    passed.** Closure requires evidence appropriate to the claim.
27. **UNKNOWN remains UNKNOWN until evidence moves it.**
34. **`UNKNOWN` and `UNRESOLVED` are different states at different
    levels.** `UNKNOWN` means the instrument has **insufficient
    qualified evidence** to earn a positive. `UNRESOLVED` means the
    **source truth itself** cannot be adjudicated because the qualified
    evidence is genuinely ambiguous, conflicting or indeterminate.
    **Lack of proof is not ambiguity**, and contract silence about a
    particular label never creates `UNRESOLVED` — the burden sits on the
    positive, so failing it yields abstention.

    > *ENFORCEMENT: manual now.*
    > *MACHINE HOOK: adjudication schema that will not accept
    > `UNRESOLVED` without a recorded ambiguity witness in the source.*
    > *OWNER/STAGE: holdout adjudication tooling.*

35. **Evidence conservation — no unproven promotion.** A downstream
    claim may not silently broaden the **subject**, **predicate or
    evidence kind**, **applicability scope**, **polarity**, **temporal
    applicability**, **certainty** or **authority status** of the
    evidence supporting it. Any such widening must itself carry
    qualified evidence; absent that, abstain. *Reviewed-at* is not
    *valid-as-of*; a commit cited for one sentence is not a whole-file
    binding; a pronoun is not a subject.

    > *ENFORCEMENT: partially mechanised — an ordered envelope lattice
    > refusing undeclared promotions exists in HOUSE_H2 v1.2.*
    > *MACHINE HOOK: the same conservation check applied at the CAPTURE
    > stage, since a lattice cannot detect a widening it receives
    > already widened.*
    > *OWNER/STAGE: instrument capture layers.*

40. **An anomaly signal is an instruction to check, not a caveat to
    record.** When a result registers as wrong — including when the
    reason cannot yet be articulated — investigate before continuing.
    **A doubt written down beside the deliverable it doubts is not
    action; it is insurance against being wrong later.**

    The signal is a **pointer, not a claim**. It licenses no assertion
    whatever — rule 1 and R1 bind absolutely — it decides only **where
    the next check is spent**.

    The asymmetry is never close, and the characteristic failure is not
    weighing it at all:

    ```
    check, clean      -> minutes spent, confidence gained
    check, not clean  -> caught before it reaches anyone else
    do not check      -> the cost lands downstream, later and larger
    ```

    This generalises rule 14 from detector output to any anomalous
    observation, and it binds every producer — human, model or
    instrument. **"The work is finished and the tests are green" is sunk
    cost, not judgement.**

    > *ENFORCEMENT: manual now.*
    > *MACHINE HOOK: hedging language in a deliverable ("deserves
    > suspicion", "worth noting", "I would flag") detected and required
    > to resolve to either a completed check or an explicit accepted-risk
    > record before the artefact ships.*
    > *OWNER/STAGE: report/deliverable tooling.*

36. **A sound negative indicator does not authorise its converse.**
    That *X* is evidence **against** *Y* does not make *not-X* evidence
    **for** *Y*. Repeated per-record date fields may be evidence against
    whole-file applicability; a unique date field does not thereby prove
    it.

    > *ENFORCEMENT: manual now.*
    > *MACHINE HOOK: review checklist requiring each positive rule to
    > name the evidence that earns it, not merely the absence of a
    > disqualifier.*
    > *OWNER/STAGE: rule authoring.*

### Derived claims

33. **A derived claim travels with the derivation that earns it.** Any
    statement that could materially influence adjudication, admission,
    repair scope, closure or programme state must carry its computation
    **at the moment it is transmitted**, not on later request. The
    obligation is the **producer's**; a consumer is never required to
    discover afterwards that a computation was missing.

    Mandatory classes: counts, ratios and aggregates · absence and
    negative claims · structural-impossibility claims · causal or
    mechanism claims · any *clean / green / sound / proven*
    self-assessment · any *nothing changed / boundary only /
    byte-identical* claim · independence or corroboration claims · and
    anything else derived that could change a consequential decision.

    ```
    DERIVED_CLAIM
      claim:
      claim_class:
      producer:
      subject_commit:
      subject_tree:
      source_or_artefact:
      query_or_command:
      instrument_and_version_or_hash:
      denominator_or_search_universe:
      unit:                 rows | cells | literal matches | mechanisms | runs
      raw_result:
      derived_result:
      interpretation:
      limitations:
      independence_status:
      rerun:
    ```

    A consequential derived claim transmitted without sufficient
    derivation is **`UNVERIFIED_DERIVED_CLAIM`**. It may be logged or
    discussed; it **must not alter adjudication, admission or repair
    scope** until the derivation is supplied and — where consequential —
    reproduced through the required independent leg.

    **Single-producer reproducible evidence is still evidence.** Exact
    subject plus exact query plus reproducible raw output is a valid
    engineering measurement. What it is not is *independent
    corroboration*, and the card's `independence_status` field exists to
    keep those apart.

    **The card is universal; independent reproduction is graded.** Every
    consequential derived claim carries a card. Independent reproduction
    is **mandatory only** where the claim can itself determine
    admission, closure, authority or irreversible repair scope, or where
    the producer would otherwise be self-certifying (rule 26). Ordinary
    reproducible machine measurements may remain single-producer
    evidence **provided they are labelled exactly that.** Requiring an
    independent leg for every figure makes the adjudicator the
    bottleneck and buys nothing where the claim decides nothing.

    **A complete card is not a true card.** `interpretation` and
    `limitations` are free text, and unearned promotion lives precisely
    there. A schema check proves non-emptiness, not correctness. Prefer
    constraints that bind CONTENT — a mechanism claim must carry a
    mechanism definition, an absence claim must carry its searched
    universe, an independence claim must expose shared dependencies —
    over additional fields.

    > *ENFORCEMENT: manual operational control now.*
    > *MACHINE HOOK: a transmission-time schema check that refuses to
    > emit a consequential claim lacking required fields, and a linter
    > over decision entries flagging bare figures with no adjacent card.*
    > *OWNER/STAGE: evidence-plane tooling, after the H2 repair cycle.*

48. **EVIDENCE HANDOFF IS LOSSLESS AND SCOPE-PRESERVING.** A transmitted
    claim may never have a broader **universe**, **subject** or
    **certainty** than the measurement that earned it, and a producer may
    never **silently curate** the population a declared extraction
    returned.

    ```
    CLAIM_SCOPE ⊆ MEASURED_SCOPE
    TRANSMITTED_RESULT = PREDICATE_RESULT     (before any DECLARED transformation)
    ```

    1) **A bounded search earns only a bounded statement.** Searching
       `/tmp/a` and `/tmp/b` earns `NOT_FOUND_IN_SEARCHED_PATHS` — never
       `NOT_FOUND_ANYWHERE`.
    2) **A field-level measurement earns only a field-level statement.**
       *0 witnesses in `classification.json`* must not become *0
       witnesses in the package*.
    3) **A snippet is not its source.** An excerpt, a `local_context`
       field, a log tail, an API window or any truncated diagnostic must
       never be described as though it were the underlying artefact.
    4) **The consumer defines the universe and the predicate; the
       producer executes them literally.** If the predicate is poor,
       return its actual result **first**, then propose a corrected
       predicate separately.
    5) **Everything returned travels.** Unexpected rows, nulls,
       duplicates, malformed records, counterexamples and inconvenient
       results are not silently pruned, ranked, deduplicated or cleaned.
    6) **Every exclusion arises from the declared predicate or a declared
       transformation.** Producer judgement is never an invisible
       exclusion criterion.
    7) **Post-extraction transformations are named.** Filtering,
       normalisation, deduplication, aggregation and ranking are
       separately identified, reproducible, and reconciled against the
       raw population.
    8) **Transport limits are not filtering authority.** Large results
       are chunked deterministically, carrying total count, chunk range
       and a final coverage reconciliation.
    9) **Negative, completeness and universality claims state the
       universe inspected.** *none · nothing · no · all · only · never*
       each carry that burden.
    10) `claim_scope > measured_scope` ⇒ **`SCOPE_OVERCLAIM`**: zero
        weight for admission, closure, repair scope or programme state
        until re-earned at the claimed scope.
    11) `transmitted_result ≠ predicate_result` without a declared
        transformation ⇒ **`SILENTLY_CURATED`**: zero evidential weight
        until reproduced.

    This binds **humans, AI producers, scripts, CI, audit tooling, search
    systems, dashboards, metrics, model evaluations and evidence
    pipelines** alike. It is the transmission-side enforcement of
    evidence conservation and **supplements rules 33, 35, 46 and 47
    rather than replacing them**: 33 governs whether the derivation
    travels, 35 whether the meaning was promoted, 46 whether the source
    was opened, 47 whether it was read far enough — and 48 whether the
    sentence finally sent is no wider than what was actually measured.

    Every consequential extraction transmits: subject/version ·
    measurement universe · predicate/method · **raw returned count** ·
    **transmitted count** · transformations · claim scope · limitations.

    **A recipient who detects a scope-overclaim stops using the claim**
    and requests or reproduces the bounded version rather than reasoning
    through it.

    > *ENFORCEMENT: `RULE_BANKED` / manual operational control, effective
    > immediately. Directed by Dainius, 2026-08-29.*
    > *MACHINE HOOK: reconcile extraction count against transmission
    > count, and refuse a consequential claim whose declared scope
    > exceeds its measurement universe.*
    > *OWNER/STAGE: evidence-plane / professionalisation machinery. NOT
    > to be implemented inside the frozen H2 candidate under this
    > authority.*

### Evidence identity

3. **Evidence identity is immutable.** Bind runtime evidence to the exact
   run, tree and artifact that produced it. Later applicability must be
   independently established.
4. **LOOKUP → VERIFY SUBJECT → USE IDENTIFIER.** Never use a remembered
   run id, SHA, artifact or subject where an authoritative lookup exists.
46. **MEMORY IS A LOCATOR, NEVER EVIDENCE.** A remembered fact may tell
    you **where to look**. It may never establish **what is true**.

    Any source-dependent claim — engineering, repository, governance,
    programme-state, runtime, evidence, decision, sequence, existence,
    absence, identity, count, status or historical content — must be
    **re-earned from the authoritative source at the moment it is
    relied upon**.

    **PRIMARY SOURCE BEFORE SYNTHESIS.** No remedy, decision, scope
    change, repository change, challenge to another producer, or
    consequential interpretation may be designed from a **remembered
    description of what an artefact contains or lacks**. Open the
    artefact first.

    **Positive and negative claims are equally bound.** *"It contains X"*
    requires inspection. *"It does not contain X"* requires inspection
    **plus a bounded search sufficient to earn the negative** (rule 33,
    P13/P14).

    Prior chat, summaries, handovers, field notes, another model's
    statement and a producer's own recollection are **navigation aids**
    when a primary source is accessible — never substitutes for it.

    **If the authoritative source cannot be retrieved, or the exact
    subject cannot be established, return `UNVERIFIED`,
    `STATE RECOVERY INCOMPLETE` or the governing abstention state.**
    Never complete a missing fact from memory.

    **NO CASCADED MEMORY AUTHORITY.** One producer's unverified
    recollection cannot become another producer's verified premise.
    Orion saying *"D359 lacks X"* does not entitle Kai to design against
    that premise; Kai must open D359. This binds in every direction —
    Orion, Kai, DeepSeek, human operators and scripts alike, **including
    the adjudicating authority.**

    **Current claims require current evidence.** Historical evidence
    proves the historical subject only, until present applicability is
    separately re-earned.

    This **generalises** existing controls rather than duplicating them:
    doctrine §0.0 (*every claim is re-earned when relied upon*), rule 4
    (*lookup, verify subject, use identifier* — for ids and artefacts),
    rule 33 (*derived and negative claims travel with their
    derivation*), and CLAUDE.md R1 (*do not assert what you have not
    run*). What was implicit across those four is now an explicit
    **no-memory-authority boundary**.

    > *ENFORCEMENT: manual operational control, effective immediately.*
    > *MACHINE HOOK: where technically feasible, a consequential
    > repository or programme claim carries source identity and current
    > subject, and emission is refused when the required source lookup
    > has not occurred.*
    > *OWNER/STAGE: evidence-plane tooling. NOT to be implemented during
    > the H2 hold.*

47. **SOURCE OPENED ≠ SOURCE READ. INSPECTION MUST BE
    CLAIM-SUFFICIENT.** Retrieving or opening an authoritative source
    earns **zero substantive claim by itself**. Before emitting a
    source-dependent conclusion, the inspection performed must be
    sufficient for the **exact predicate and scope** of the claim.

    ```
    identify the claim
      -> identify the authoritative source
      -> establish exact subject / version
      -> determine the evidence scope the CLAIM requires
      -> inspect that scope COMPLETELY
      -> follow governing internal references or delegations where material
      -> only then emit
    ```

    **READ TO THE CLAIM BOUNDARY.** A paragraph glance cannot support a
    document-wide conclusion. A heading scan cannot establish that a
    phase is undefined. A single search hit cannot establish programme
    meaning where another section qualifies it. A retrieved snippet or
    line window cannot establish absence outside that window. A summary
    cannot substitute for available primary material.

    Whole-artefact claim ⇒ the whole artefact inspected or completely
    searched. Bounded claim ⇒ the conclusion stays bounded to that
    section. Where a source **delegates or promotes** substantive detail
    into another identified artefact, that dependency is followed before
    a conclusion whose truth depends on it.

    **PARTIAL READING REQUIRES PARTIAL CLAIMS.** Having inspected only
    lines 1-100, `NOT_FOUND_IN_INSPECTED_RANGE` may be earned;
    `NOT_FOUND_IN_DOCUMENT` may not. Having inspected only one section,
    report what that section says — never that the document contains
    nothing else.

    **NEGATIVES CARRY A COVERAGE BURDEN.** For existence, absence,
    completeness, definition, authority, sequence, enumeration and
    *all/none* claims, record the inspection universe that supports the
    claim. **"I opened it" is not a proxy for "I checked it."**

    **RULES 46 AND 47 ARE A COUPLED CONTROL.**

    | | asks |
    |---|---|
    | rule 46 | did you go to the authoritative source rather than memory? |
    | rule 47 | did you inspect **enough of it** to earn the exact claim? |

    Either alone is insufficient: memory yields a wrong premise; the
    correct source read shallowly yields a wrong *source-derived*
    premise, which is more dangerous because it arrives wearing a
    citation.

    **This supplements rule 46. It does not rewrite it, and no
    overlapping doctrine is to be created elsewhere.**

    > *ENFORCEMENT: `RULE_BANKED` / manual, effective immediately. A
    > consequential source-derived answer must state enough of its own
    > reading or search scope to justify its conclusion. Whole-source
    > negatives require whole-source or demonstrably complete coverage.*
    > *MACHINE HOOK: claim metadata carrying `source_identity`,
    > `subject/version`, `claim_predicate`, `claim_scope`,
    > `inspection_scope`, `coverage_mode` =
    > `FULL_ARTEFACT | COMPLETE_BOUNDED_SEARCH | SECTION_BOUND`, and
    > `followed_dependencies` — refusing promotion where
    > `inspection_scope < claim_scope`.*
    > *OWNER/STAGE: assurance/evidence tooling when authorised. NO
    > implementation during the H2 hold.*
23. **Historical corrections are append-only.** Do not erase mistakes
    that taught us something.

### Measurement vs subject

5. **Measurement state and subject verdict are separate.** Crash,
   observer failure, instrument failure and UNMEASURED are **not** adverse
   results about the subject.
6. **A refusal must return a verdict.** Crashing while attempting to
   refuse is instrument failure, not fail-closed proof.
7. **Observer liveness ≠ subject observation.**
8. **Traversal ≠ transparency.** A hook being installed or traversed does
   not prove it preserved the subject's behaviour.

### Gates and triggers

9. **Gate trigger conditions are part of the gate.**
10. **Evidence-admission rules do not authorise evidence production.**
    "We will not use this result" is not permission to perform the action
    that generates it.
11. **LIVE-MODEL and CAPTURE-WRITING are different properties.** Assess
    every relevant change against both.
12. **Trigger analysis must operate on executable behaviour, not textual
    resemblance.** Docstrings and comments containing commands are not
    execution.

### Populations and detectors

13. **Population and denominator must be explicit and reproducible.**
    Never silently discard inconvenient rows or runs.
14. **Detector surprise means inspect the detector first.** If the
    expected population is 1 and the detector reports 50, do not begin
    fixing 50 defects.
37. **A query proves what it actually matches.** A search over a
    literal or token establishes *N instances of that literal*. It does
    **not** establish *N instances of one causal mechanism* — mechanism
    equivalence requires evidence of the mechanism. When reporting a
    population, the **unit must be stated**: literal matches, rows,
    cells, mechanisms or runs are different denominators and are not
    interchangeable.

    **A ROUTING SIGNATURE IS NOT A MECHANISM.** The same applies one
    level up from literals: a witness key, a classifier route or an
    emitted-value tuple is an **observation class**, not a causal
    equivalence class. `witness_type / binding label / emitted value`
    groups rows that TRAVELLED THE SAME PATH; it does not establish that
    one cause produced them. Call such a grouping a **routing
    signature** until mechanism equivalence is separately earned.

    Earned twice in one fortnight: first by filtering a literal date and
    reporting a mechanism population, then -- while correcting that very
    error -- by reporting six machine-constructed routing keys as "six
    distinct witness mechanisms" in a Fact Card whose own `limitations`
    field admitted the key was the producer's construction.

    > *ENFORCEMENT: manual now, via rule 33's mandatory `unit` field.*
    > *MACHINE HOOK: population reports that emit their unit and their
    > exact query alongside the count, so a literal filter cannot be
    > presented as a mechanism population; and a `mechanism_count` claim
    > class that refuses to validate without a `mechanism_definition`.*
    > *OWNER/STAGE: audit instruments.*

41. **Unadmitted-candidate quarantine.** The output of a failed or
    unadmitted candidate may be used as a **diagnostic signal** -- it can
    say where to look -- but it may **not** become the baseline for an
    architecture, utility or scope conclusion until the frozen
    truthfulness adjudication that governs it is complete. A candidate
    whose positives are known to be partly wrong cannot measure how
    useful the corrected instrument will be.

    > *ENFORCEMENT: manual now.*
    > *MACHINE HOOK: candidate artefacts carry an admission status, and
    > a claim citing an unadmitted candidate as a baseline is refused.*
    > *OWNER/STAGE: evidence-plane tooling.*

42. **Consumer requirements must not be reverse-engineered from
    producer output.** A downstream stage's requirement comes from the
    governing need and the problem that predates the producer -- never
    from what the current producer conveniently emits. **Where the
    builder has already seen the producer's output, the builder cannot
    be the sole author of the consumer contract.** The requirement must
    include **negative/rejection cases** and at least one need the
    current producer may fail to satisfy, drawn from a pre-existing
    problem statement.

    Without this, a requirement written late is shaped by what already
    works, the causality is reversed, and the contract certifies the
    instrument instead of governing it.

    > *ENFORCEMENT: manual now -- authorship separation and blind drafting.*
    > *MACHINE HOOK (NARROWED, D370 CORRECTION): every consequential
    > consumer requirement traces to a governing need, problem or
    > evidence source that PREDATES EXPOSURE TO THE PRODUCER OUTPUT. A
    > contract authored later declares its authorship and contamination
    > status and may not be shaped solely by the exposed builder. The
    > original wording -- "hash-bound to a commit predating the producer
    > artefact" -- was withdrawn: it would make a legitimate H3 contract
    > impossible by chronology, since H2 already exists. What must
    > predate the producer output is THE NEED, not the document that
    > states it.*
    > *OWNER/STAGE: contract format.*

43. **Coverage is reported by DETECTION CAPABILITY, not by sample
    size.** An assurance sample must state, per axis, which failure
    classes it can actually expose: false-positive opportunity,
    over-abstention opportunity, evidence-fact opportunity, boundary and
    common-mode cases, plus family and routing concentration and the
    classes it does **not** cover. **Forty rows is not forty independent
    questions.**

    Do **not** compute an "effective sample size" without a defensible
    statistical model; report the dimensions separately instead.

    Earned by the D368 holdout: 40 rows, but 38 of 42 non-abstention
    cells shared one routing signature, and three of six axes contained
    **zero** non-abstention cells -- so the sample had no false-positive
    detection opportunity on those axes at all. Nobody noticed until
    somebody counted, after the sample was frozen.

    > *ENFORCEMENT: manual now.*
    > *MACHINE HOOK: a coverage matrix emitted and reviewed BEFORE an
    > assurance sample is frozen.*
    > *OWNER/STAGE: assurance tooling.*

### Calibration

15. **Calibration must prove the instrument can fail.** Positive-only
    tests are comfort, not control.
16. **Calibration fixtures must reproduce the hostile production property
    they claim to test.**
17. **The shipped entry point must be directly exercised.** Testing
    internal functions does not prove the CLI or workflow invocation can
    start.
18. **Programmatic edits must assert mutation cardinality.** An edit
    expected to change one target must prove exactly one intended target
    changed. A zero-match silent edit is a failure.
32. **A shared-fate control cannot prove the shared property.**
    Agreement between two observations is not independent corroboration
    when both depend on the same decisive mutable source, parser, state,
    authority or failure path. For a control to detect failure of
    dependency X, **at least one decisive verification leg must not
    share dependency X**.

    Before trusting a gate, ask what failure dependencies its evidence
    legs share. Where they share the decisive one, the gate reports on
    itself and calls it the world.

    This does **not** demand universal independence, which is
    unaffordable and usually impossible. Independence remains
    claim-specific. The rule governs only the decisive dependency whose
    failure the control claims it can detect.

    Rule 21 is the special case for two counters inside one instrument;
    this is the general form.

### Repair

29. **A repair is not complete until a fixture can detect its absence.**
    Before a repair may be banked as proven, there must exist an
    independent fixture that **fails on the defective behaviour and
    passes on the repaired one**. Test-first is not the requirement —
    some defects are only visible at runtime, and turning method into
    ceremony helps nobody. The requirement is about **proof order**: the
    deletion-sensitivity of a repair is established *before* the repair
    is called done, not discovered later by a reviewer.

    Earned three times in two patches, always identically: the repair
    was written, the suite stayed green, and only a destructive
    reinjection revealed that **nothing specifically proved the repair**
    — `--no-cache` reconciliation, the invocation commit contract, and
    the unmeasurable-corroborator path each survived their own deletion
    at 486/0, 486/0 and 515/0. A mechanism can become untestable by
    deletion while still accumulating reassuring green tests, and green
    tests are exactly what stops anyone looking.

    For repairs made before this rule, destructive reinjection is
    acceptable as retrospective evidence.

30. **Qualification and mutation may not share an uncontrolled subject
    state.** A test or collector may not claim evidence over files or
    processes that it — or anything running beside it — is actively
    mutating, unless that interaction is itself part of the declared
    experiment.

    Earned by running a suite in the foreground while a reinjection
    sweep held the same files patched, and reading three failures as
    real for long enough to start diagnosing them. R9 is the special
    case where the instrument observes *itself*; this is the general
    one, where the instrument observes a subject somebody else is
    changing underneath it. The tell is the same: **the measurement was
    true of something, just not of the world it claimed.**

31. **Failure identity before causal attribution.** The failed CI step
    must be established from authoritative **step-level execution
    state**. Root-cause attribution must then come from evidence
    generated **by that failing step**, or from independently
    corroborated evidence. `if: always()` diagnostics, post-mortems,
    downstream skips and log-tail excerpts are **consequences and
    context** unless separately proven causal.

    Earned 2026-08-22, in the census. Two red workflows were recorded
    with the wrong failing step: a **skipped** ratchet was named as the
    failure while the JSON quoted from it came from a later step that
    **passed**, and "the image builds produced no output" described
    steps that never ran because an earlier step had already failed.

    The mechanism generalises, and it is the reason this is a rule
    rather than a note. A diagnostic that runs `if: always()` is
    deliberately placed **last so it survives log truncation** — this
    repository does that on purpose and even names the step for it. So
    **the tail of a failed job's log is systematically the output of a
    step that succeeded.** Reading the tail and taking the most
    failure-shaped text in it inverts cause and consequence.

    That is R10 seen from the other side: R10 says the full output must
    survive and excerpts must declare themselves; this adds that **an
    excerpt at the end of a log is not a neutral sample** — it is the
    part most likely to be a post-mortem, and therefore the part least
    likely to be the failure.

    It also invalidated a *classification*, not just a label: an origin
    was called PRE-EXISTING by comparing post-mortems between two runs,
    evidence that cannot distinguish **which step** failed, because any
    failure before the same phase produces an identical empty
    post-mortem. The conclusion happened to survive re-derivation at
    step level. The method did not, and nothing but redoing it would
    have revealed that.

### Records and streams

19. **Machine evidence and human prose stay separate.**
20. **ABSENT / NULL / VALUE remain distinct** wherever invocation identity
    depends on them.
21. **Internal reconciliation is not independent proof.** Two counters
    inside one instrument can detect internal loss and both be blind to a
    path that never reached the instrument.
22. **Evidence outside the retrievable observation window is not
    available evidence.** Prefer concise identity and verdict in logs,
    authoritative detail in artifacts.

44. **`RULE_BANKED` and `CONTROL_OPERATIONALISED` are different
    states.** A doctrine entry with provenance, an earned example and a
    named enforcement hook is **banked governance** -- authoritative, and
    citable. It becomes **operationally closed** only when the hook
    exists, carries hostile calibration and is actually enforced.

    Track both. Do not retrospectively erase an entry's banked status
    because its hooks are future work, and do not claim a recurrence has
    been prevented when only the prose exists. **The programme's own
    record shows why: 14 of 17 defects were already covered by written
    doctrine at the moment they occurred.**

    > *ENFORCEMENT: a status field on every doctrine rule.*
    > *MACHINE HOOK: a report listing rules whose hook is unbuilt, so the
    > gap between banked and operationalised stays visible.*
    > *OWNER/STAGE: doctrine tooling.*

45. **Assurance-tool trust is purpose-limited.** An audit, classifier,
    linter or gate does **not** acquire admission weight for a new
    purpose merely because it once produced output that was believed.
    Before its result may gate a consequential decision it needs
    known-positive, known-negative and boundary/adverse fixtures, and
    exact tool and version identity bound to the result.

    **Historical output remains historical evidence** -- this does not
    retrospectively void what a tool has already produced. It governs
    what that tool may be trusted to decide NEXT. A repeatable tool can
    be repeatably wrong.

    > *ENFORCEMENT: manual now.*
    > *MACHINE HOOK: seeded fixture corpus per governance instrument,
    > with the tool hash recorded in every claim card it produces.*
    > *OWNER/STAGE: assurance tooling.*

49. **FAILURE PATTERNS ARE FIRST-CLASS ENGINEERING EVIDENCE.** A material
    mistake does not end when its immediate output is corrected. Every
    verified material failure enters a durable learning cycle:

    ```
    FAILURE -> INCIDENT RECORD -> MECHANISM -> RECURRENCE SIGNATURE
            -> STOP-SIGNAL / CONTROL -> CALIBRATION -> FUTURE RETRIEVAL
    ```

    The purpose is not blame. It is that the system, its operators, its
    models and its instruments stop paying repeatedly for one underlying
    mistake.

    1) **Material errors are logged, not merely corrected.** A correction
       that erases how the error happened destroys the future defence.
    2) **Incident and mechanism are separate.** Similar outputs do not
       prove a shared cause. Wording similarity, common file paths and
       routing signatures are **locators only** (rule 37). A pattern is
       promoted only once the causal mechanism is supported by evidence.
    3) **Recurrence is actively searched for**, not waited for. Once a
       mechanism is confirmed, inspect current and prior work for the
       same mechanism wherever the denominator is material (rule 13).
    4) **Known patterns become pre-action checks.** Before consequential
       work, the relevant confirmed patterns and their stop-signals are
       consulted — not recalled (rule 46).
    5) **A repeated mistake is evidence against its current control.**
       When a confirmed pattern recurs after a rule exists, do not
       reissue the reminder; reassess the control.
    6) **THREE CONFIRMED OCCURRENCES FORCE ESCALATION.** On the third
       independently confirmed occurrence of one mechanism, a
       prose-only/manual control is **presumed insufficient** unless
       evidence shows otherwise, and must receive a machine-enforcement
       design, a stronger structural constraint, or an explicit
       accepted-risk decision carrying its justification.
    7) **Corrections are append-only.** The false claim stays visible
       beside its correction. History is never rewritten to make a
       producer look consistently right (rule 23).
    8) **Everyone is inside the denominator** — operator, Orion, Kai,
       DeepSeek, subagents, scripts, classifiers, gates and any future
       autonomous component. **No role is exempt because it adjudicates
       others.**
    9) **Patterns must travel into future reasoning.** A ledger that is
       never consulted is archival documentation, not a control.
    10) **Pattern effectiveness is measured.** Track recurrence after a
        control is introduced. *Rule written* is not *failure
        prevented* — rule 44 is the reason that distinction exists.

    **The authoritative record is `kai-pm/FAILURE_PATTERN_LEDGER.md`,
    append-only.** It is distinct from `ORION_FIELD_NOTES.md`, which
    remains non-authoritative working memory and creates no programme
    state. Its schema, pattern states and the initial banked patterns
    live in that file. A producer may **propose** a pattern; it may never
    silently self-certify causal equivalence.

    **ARCHITECTURAL DESTINATION — recorded as a requirement, carrying no
    implementation authority.** A governed *Failure Pattern Memory /
    Reasoning Guard* that ingests only verified incidents, derives
    candidate mechanism similarity without automatic causal promotion,
    retrieves confirmed patterns **before** consequential reasoning,
    injects their stop-signals into the reasoning context, checks
    proposed outputs and actions against known mechanisms, logs
    recurrences, measures whether controls reduce them, escalates
    repeated manual-control failures toward machine enforcement, and
    **never autonomously rewrites doctrine or its own governing rules.**
    House Doctor is one *consumer* of this intelligence, not its owner:
    the pattern memory must reach the reasoning and orchestration layer,
    so a known failure is avoided before there is anything to diagnose.
    The target is not a producer that remembers being wrong — it is one
    that has become **progressively harder to fool in ways that have
    already fooled it.**

    > *ENFORCEMENT: `RULE_BANKED` / manual operational control, effective
    > immediately. Directed by Dainius, 2026-08-29.*
    > *MACHINE HOOK: ledger schema validation; a recurrence report per
    > confirmed mechanism; an escalation trigger at the third confirmed
    > occurrence; and retrieval-before-reasoning in the evidence plane.*
    > *OWNER/STAGE: evidence-plane / reasoning-architecture programme
    > stage. NO implementation during the H2 hold.*

### Delegation and authority

38. **Reconcile an instruction against its governing contract before
    executing it.** When an instruction changes, narrows, shortcuts or
    expands work governed by a frozen or active contract, re-read the
    exact governing artefact first:

    ```
    instruction received
      → identify the governing artefact
      → exact-subject read
      → reconcile instruction against contract
      → execute only if compatible
      → otherwise HOLD and escalate
    ```

    **A verified repository state under an unverified instruction is not
    a verified programme position.** This binds regardless of who issued
    the instruction, including the adjudicating authority.

    > *ENFORCEMENT: manual now.*
    > *MACHINE HOOK: contract artefacts carrying a machine-readable
    > obligations block that an executor can diff an instruction against.*
    > *OWNER/STAGE: contract format, next contract cycle.*

39. **Name the authority behind every independence claim.** Before
    writing *independent*, *corroborated* or *equivalent*, state for each
    leg its **producer/authority** and its **decisive dependency**. Same
    authority with a different method is **cross-method convergence under
    the same authority** — real evidence, and not authority-independent
    corroboration. Rule 32's shared-fate semantics continue to apply.

    > *ENFORCEMENT: manual now, via rule 33's `independence_status` field.*
    > *MACHINE HOOK: independence assertions required to enumerate legs
    > with producer and dependency before the word is accepted.*
    > *OWNER/STAGE: evidence-plane tooling.*

25. **No agent may silently expand its remit.** Subagents inherit these
    standards and return **evidence, not confidence**.
26. **No consequential mechanism self-approves or self-verifies.**

### Governing material

28. **Governing material must be checkable from the work it governs.** A
    rule, priority order or constraint that binds this repository must
    exist as an artefact **inside** it. A copy held elsewhere is a
    convenience, never the canon, and every copy must be reconcilable by
    **mechanical comparison** rather than by reading. Where two records
    disagree, the one both parties can open wins.

    **Working notes are non-authoritative memory.** Field notes,
    retrospectives and lesson logs may preserve observations and
    provenance, but may **never** create programme state, authority,
    sequence, acceptance criteria, finding closure, implementation
    permission or admission. A later governing decision may cite such a
    file as a **provenance pointer**, but must re-earn and cite the
    original evidence or governing artefact. Authority is never acquired
    by repetition, and no second status ledger may be created.

    > *ENFORCEMENT: a prominent non-authoritative status block at the
    > head of each working-notes file.*
    > *MACHINE HOOK: a check that decision entries citing a working-notes
    > file also cite a primary source or governing artefact.*
    > *OWNER/STAGE: ledger tooling.*

---

## Where each rule was earned

| rule | the failure that earned it |
|---|---|
| 48 | **three incidents in one H2 adjudication session, one mechanism: *bounded measurement → correct local result → unbounded transmitted claim*.** (1) A Pass A `local_context` cut at the 6000-byte window was described as *the source document carries a truncated SHA with no closing backtick* — the source was intact; the instrument's own output had been read as the artefact. (2) `find` over `/tmp/tmp.6xNl2hBs2V` and `/tmp/claude-0` became *"passA.json was not preserved anywhere on disk. I looked."* — it was tracked at `f196366`, 359,173 bytes, and `sha256sum -c PACKAGE.sha256` verified all 14 entries. (3) *0 positive evidence facts carry a witness* was measured over `h2v12-classification.json` and transmitted as a claim about the candidate package — **235 of 316 carry the full nine-field §5 trace** in the package-bound sidecar. Each measurement was locally correct; each sentence was wider than what was measured. Adjudicated by Kai across three exchanges, directed into doctrine by Dainius (D373) |
| 49 | **the same three incidents, seen from the control side.** Rules 33, 35, 46 and 47 were all banked and all cited — one of them quoted in the very message that broke it — and the mechanism still recurred three times inside a single session. The programme's own record already showed 14 of 17 defects were covered by written doctrine when they occurred (rule 44). What was missing was not another rule but the closed loop: preserve the incident, name the mechanism, search for recurrence, and escalate the control when a confirmed pattern repeats. Directed by Dainius (D374) |
| 1, 24, 27 | findings "closed" on argument rather than evidence; counts that changed because a fix landed |
| 29 | three repairs in two patches survived their own deletion at 486/0, 486/0 and 515/0 — repaired, green, and proved by nothing |
| 30 | a suite read in the foreground while a reinjection sweep held the same files patched; three failures diagnosed as real |
| 31 | a **skipped** ratchet named as a failing step, quoting JSON emitted by a later step that **passed**; and an origin classified PRE-EXISTING by comparing post-mortems, which cannot see which step failed |
| 2 | a hook installed and never traversed, reported as a measurement |
| 3 | a re-analysis whose evidence and analyser came from different trees |
| 4 | five 404s from guessed run ids |
| 5, 6 | P1 run 5: the census **crashed** while attempting to refuse, and a crash is not a refusal |
| 7 | a watcher that proved its own timer was alive and called it observation |
| 8 | run 16: traversal proven, transparency not |
| 9 | a repaired collector that fired nothing, absent from its own workflow's filter |
| 10 | D251: admissibility pre-registered, authorisation never asked for |
| 11, 12 | `core-tests.yml` starts a model on every push; the first detector counted docstrings as execution |
| 13, 14 | 100+ findings for 1 real defect; 69 findings against a correct tree |
| 15, 16 | a stub that could not reproduce the hostile property it was testing |
| 47 | **the same D359 artefact, the second half of the failure.** Rule 46 worked — D359 was opened rather than answered from memory. But only the §2 sequence block was inspected: **1,400 characters of a 12,939-character, ten-section entry — 10.8% coverage** — and a WHOLE-DOCUMENT negative, `NOT_FOUND_IN_D359`, was emitted from it. §5 defines `HOUSE_H3` (active claim qualification), `H4` (repair the control sources), `H5` (document authority / drift enforcement) and `H6` (bank the baseline; prove the drift checks detect their own absence). The same artefact therefore demonstrates that **source retrieval and source qualification are separate operations** (D372) |
| 46 | **the D359 incident.** Six specific absence claims were produced about D359 without opening it — *Item 8 inside the 048 path · `A-4_PROVENANCE` distinct from `A4_SELF_DIAGNOSIS` · Assurance Integration · repository consolidation · Evidence Plane last · the explicit Dainius House Exit Ruling*. **All six were false; every one was already in D359.** A second producer accepted the summary and designed a governance remedy on the false premise. Only operator scrutiny forced primary-source inspection, which disproved it before any repository mutation. The enumerated negative was written in a message about durability discipline, one paragraph after quoting the rule it broke (D371) |
| 15, 44 (and R1) | **PRODUCED ≠ USABLE ≠ VERIFIED.** A sidecar's `.sha256` file carried two CORRECT digests and was reported as "hash-bound" — but it had been hand-formatted with three spaces where `sha256sum` requires two, so `sha256sum -c` could not parse it at all. The digests were right; the verification artefact did not work, and the consumer command had never been run before the property was reported. Existing controls, not a new law: R1 forbids asserting what has not been run; rule 15 asks a control to demonstrate it can fail; rule 44 separates a written control from an enforced one (D370 errata C5) |
| 17 | P1 run 5 again: 67 assertions on the parts, none on the shipped entry point |
| 18 | a string replacement whose anchor no longer matched, applied without an assertion, silently doing nothing |
| 19 | the probe's own denominator line inside `capture.jsonl`, which made the file uncertifiable |
| 20 | `temperature: None` unable to say *absent* from *explicitly null* |
| 21 | a manifest counter that re-read the rows it was meant to reconcile |
| 22 | the ~15.8KB Actions log window, three times |
| 23 | disproven claims struck through rather than deleted, so the pattern stays visible |
| 25, 26 | a meta-check that wanted to probe a key generator to read its own denominator |
| 28 | the programme's binding order of work existed nowhere in the tree (D270 §2); and the external doctrine copy was silently missing rule 4 — the anti-drift rule itself — while reaching 27 by splitting rule 26, so "Rule 17" named different rules in each record (D272) |
| 33 | "18 of 40 share the mechanism" — computed by filtering the literal `Reviewed: 27 July 2026` while describing a mechanism, so the 26 July member was missed. The figure reached the adjudicator with its derivation invisible, and was in front of him while he reasoned (D369) |
| 34 | a plain `Reviewed: <date>` witness parked as `UNRESOLVED PENDING SEMANTIC AUTHORITY` when the source was perfectly intelligible and it was the *positive* that had failed its burden. Searching for authority forbidding a label, when abstention was already the default (D369) |
| 35 | seventeen HOUSE_H2 defect classes, all one shape: an observation recorded at one scope and reported at a wider one. `reviewed-at` promoted to `time-bound-validity`; a commit cited for one sentence promoted to a whole-file snapshot binding; a bare pronoun promoted to a subject (D364–D368) |
| 36 | a uniqueness rule shipped on a denominator of one in 162: repetition of a binding predicate is evidence *against* whole-document scope, and its converse was implemented as evidence *for* it (D368) |
| 41 | a strategic reframe argued from D368's cell counts while D368 was an unfrozen failed-admission candidate with an incomplete holdout and known-wrong positives — turning a failure into a baseline while the failure was still open (D370) |
| 42 | three weeks of H2 work with no statement anywhere of what HOUSE_H3 needs from it; a requirement written afterwards, by the builder, would have been shaped by what H2 already emitted (D370) |
| 43 | the D368 holdout: 40 rows, 38 of 42 non-abstention cells sharing one routing signature, and three of six axes with zero non-abstention cells — no false-positive detection opportunity on those axes, discovered only by counting after the freeze (D370) |
| 44 | D369 banked eight rules with enforcement hooks and built none of them; without two states the entry is either overclaimed as prevention or wrongly dismissed as unbanked (D370) |
| 45 | `validity_binding_audit.py` and `cross_axis_semantic_audit.py` produced evidence banked in D364 and D365 with no seeded fixtures and no adversarial calibration of their own (D370) |
| 40 | HOUSE_H2 v1.2 W2: `VALIDITY` rose 56 → 161 while every other axis fell. The anomaly was registered, written into the report as a worry, and the candidate shipped anyway on an explanation that was liked but unproven. The skipped check was minutes of source reading; the cost landed on the adjudicator, who reasoned with the wrong figure in front of him. Directed into doctrine by Dainius (D368/D369) |
| 37 | the same 18-vs-19 count as rule 33, seen from the population side: a literal filter reported as a mechanism population, where the corpus split 120/24 across two dates (D369) |
| 38 | an instruction to stop the 40-row holdout at the first blocker, when the frozen contract required all forty. Repository state had been verified; the instruction against the contract had not (D369) |
| 39 | six `EXACT_SNAPSHOT` documents called "independent corroboration" when both the machine rule and the prior source adjudication traced to the same authority (D369) |
| 32 | the same structural failure in three different mechanisms: R9's watcher whose `pgrep` pattern matched its own command line; a detector whose population included its own docstring (I-8); and Census v1.1's subject reconciliation, whose two supposedly independent sides both dereferenced the same moving symbolic ref, so they agreed perfectly and reported `reconciles: True` while the result was stamped with a different commit than the one measured (D356) |
