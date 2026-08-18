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

### Evidence identity

3. **Evidence identity is immutable.** Bind runtime evidence to the exact
   run, tree and artifact that produced it. Later applicability must be
   independently established.
4. **LOOKUP → VERIFY SUBJECT → USE IDENTIFIER.** Never use a remembered
   run id, SHA, artifact or subject where an authoritative lookup exists.
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

### Delegation and authority

25. **No agent may silently expand its remit.** Subagents inherit these
    standards and return **evidence, not confidence**.
26. **No consequential mechanism self-approves or self-verifies.**

---

## Where each rule was earned

| rule | the failure that earned it |
|---|---|
| 1, 24, 27 | findings "closed" on argument rather than evidence; counts that changed because a fix landed |
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
| 17 | P1 run 5 again: 67 assertions on the parts, none on the shipped entry point |
| 18 | a string replacement whose anchor no longer matched, applied without an assertion, silently doing nothing |
| 19 | the probe's own denominator line inside `capture.jsonl`, which made the file uncertifiable |
| 20 | `temperature: None` unable to say *absent* from *explicitly null* |
| 21 | a manifest counter that re-read the rows it was meant to reconcile |
| 22 | the ~15.8KB Actions log window, three times |
| 23 | disproven claims struck through rather than deleted, so the pattern stays visible |
| 25, 26 | a meta-check that wanted to probe a key generator to read its own denominator |
