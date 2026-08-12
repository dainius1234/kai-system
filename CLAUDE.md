# Operating rules — read this first

Standing rules for anyone (including me) working in this repository.
Every rule here exists because it was broken, and the incident is named
so the rule is not mistaken for taste. The long form is in
`kai-pm/ORION_FIELD_NOTES.md`; this file is the part that binds.

---

## R0. Stop signals — the tells, in one place

The rules below explain *why*. This is the part that has to work **in
flight**, so it is first and it is short. If a sentence I am about to
write or a command I am about to run matches one of these, stop and
check before continuing.

| the tell | what it usually means |
|---|---|
| "would have", "cannot", "is simply", "should work", "it's just a" | R1 — I am asserting something I have not run |
| "saved to…", "added to…", "wired into…" | R1 — a claim about *my own action*. `ls` it, or do not say it |
| "recorded", "banked", "documented", "carried forward" | R1 — verifiable state claims. `grep` the record before saying them |
| I am describing a safety net in the present tense | R2 — I have written a contingency, not run one |
| I am about to describe the *shape* of a failure ("it's X, not Y") | §3.1 — my scope may be narrower than my claim. What is the full set? |
| I am writing a name I recognise rather than one I looked up | R5 — a list beside the thing. Derive it |
| I am about to `cut`, `head` or `tail` a diagnostic | R10 — keep the full output somewhere, and say how big the excerpt is |
| A prerequisite failed and I am still collecting rows | R11 — no subject, no observation. Abort at the boundary |
| I am fixing the second instance of something | R6 — count the population first, then fix all of it |
| "we'll tune this once we measure it" | ship the instrument that produces the number in the *same* commit |
| I am typing `;` between commands | R3 — use `&&` |
| I am writing a loop that waits for a process by name | R9 — the loop's own command line contains that name. It will match itself and wait forever |
| My evidence that a check works comes from the same place as the thing it checks | I-8 — calibrate against something independent, with a known-positive and a known-negative |

**The trigger is speed, not ignorance.** Every R1 breach on 2026-08-07
happened while moving fast — a hard-coded image name, a guessed job id,
two files claimed as saved. None happened because I did not know better;
each happened because checking felt slower than continuing. When the
work is flowing is exactly when these tells matter, which is the
opposite of when they feel necessary.

## R1. Do not assert what you have not run

If a sentence contains **"would have"**, **"cannot"**, **"is simply"**,
**"should work"**, or **"it's just a"** — stop. Run the thing. Read the
field. Compute the date.

Earned by, in one week:

* Claimed a CI re-run "bought nothing" — which assumed it would
  complete. It was cancelled.
* Claimed run 695 had "survived past the 15-minute mark". It had not; I
  judged elapsed time instead of computing it.
* Wrote in a post-mortem that `workflow_dispatch` "cannot help" — it was
  the thing that unblocked the incident.
* Argued a failure could not be mine because the change was downstream.
  The argument was sound. It was still only an argument. The re-run
  proved the failure deterministic when I had predicted a flake.

The cost of checking is seconds. The cost of being wrong in writing is
that someone believes it.

**Corollary — say which half you verified.** When part of a change can
only be tested elsewhere (CI, a runner, a network you cannot reach), say
so explicitly in the commit message: what was verified here, what was
not, and what will verify it.

**Corollary — this applies to your own actions, not only to the code.**
"Saved to `<path>`", "added to the register", "wired into the Makefile"
are all assertions, and they are the easiest kind to get wrong because
they feel like memory rather than inference. On 2026-08-07 I told the
operator twice that a document had been saved to `kai-pm/`. Neither file
existed; the text had only ever been in a chat message. Both claims were
made in the same conversation where this rule was written down.

`ls` it, or do not say it.

**Earned a second time, 2026-08-12.** I ended a session by listing what
to "carry forward" — a run's result, a three-layer mechanism, a
correction, an unmeasured question — as though naming them preserved
them. `grep` over `kai-pm/` found none of it: the run id appeared
nowhere, the differential nowhere. Everything since the previous decision
entry existed only in chat, and the session was one message from ending.
The check took seconds; without it a full day's evidence would have been
lost while I reported a clean handoff.

The operator's formulation, which is the one to keep:

> **"Recorded", "saved", "banked" and "documented" are verifiable state
> claims. Verify the artefact exists before asserting them.**

This applies to our own work at least as aggressively as to evidence from
the system — more so, because nobody else is auditing it.

## R2. Always *run* the contingency, never merely write one

A rollback that has not been executed is a hypothesis with good
presentation. Before relying on one:

```bash
git revert --no-commit <sha>     # does it even apply?
```

and check the resulting tree actually has the property you are claiming
for it. Then write down what you ran, not what you intended.

Earned 2026-08-07: asked whether a contingency existed, checking rather
than answering found that the "fail closed" model download I had shipped
an hour earlier had **no retry** — so a single dropped packet failed
every build. Fail closed means *never ship a broken image*; it does not
mean *die on the first blip*. Brittle and strict are different
properties wearing the same word.

**A contingency must also survive the failure it is for.** The rollback
target for a build that depends on huggingface.co must be a commit that
builds *without* huggingface.co. Verify that specifically.

## R3. `&&`, never `;`

    make policy-check && git commit && git push

Twice in one day I ran `make policy-check ; git commit ; git push` and
pushed past a failing gate — the second time hours after writing the
first one down. If a chain contains a gate, the gate must be able to
stop it.

## R4. Measure the population before fixing it

Never apply a new rule to a large denominator before counting what it
hits. The sequence, in order:

1. One **confirmed** instance — observed, not suspected.
2. Derive the rule from why that instance is wrong.
3. **Count the hits. Fix nothing yet.**
4. Calibrate against a known answer — run it against the tree one commit
   before a known fix; it must report exactly the known defects.
5. Fix the whole population.
6. Gate it, printing the denominator, with proof it can fail.

Step 3 before step 5 is load-bearing. Uncalibrated rules produced 100+
findings for 1 real defect, and 69 findings against a tree that was
already correct. A gate with false positives sends people to break
working code and buries the true finding.

Counting has twice changed the design and twice stopped me "fixing"
things that were right. It also caught me over-claiming a scope of three
when the real population was one.

## R5. State the denominator, and derive it from the tree

A check's scope is defined by the data it traverses, not by a list kept
beside it. Seventeen defects in one stint were all the same shape: *a
check whose scope was smaller than its name implied*. Any hand-written
tuple of names in a gate is a defect waiting to be found.

The inverted form is worse: a scope **larger** than reality reports
failure over things that are right.

## R6. Fix the class, not the instance

If a remedy has a denominator, apply it to the denominator. Fixing one
file and declaring the class closed is how 16 `depends_on` declarations
got fixed in one compose file while 11 survived in two others, on the
same day.

## R7. Findings stay open until a formal closure review

Counts do not change because a fix landed. Closure is a separate,
evidence-backed register action.

## R8. Never-executed code is where the defects are

Every defect of the 2026-08-07 stint lived in code or configuration that
had never run. Not one was code that used to work and broke. When
choosing where to look, ask what has no execution path — not what looks
suspicious.

## R9. A watcher must not be able to observe itself

    until ! pgrep -f "make prepush"; do sleep 15; done

This never exits. `pgrep -f` matches the full command line, and the
command line of the waiting shell *contains the string it is searching
for*. The loop finds itself, concludes the job is still running, and
waits forever.

Earned 2026-08-07. Eight of these accumulated in one stint. The gate
chain they were guarding never ran, so `prepush.log` stayed empty, the
tree stayed dirty, and the commit never happened — while I reported,
repeatedly and in good faith, that the gates were "still running". The
observation was true of the watcher and false of the world.

Two fixes, in order of preference:

1. **Do not wait.** Run the chain in the foreground with a long timeout.
   `make prepush && git commit && git push` needs no supervision, and
   the `&&` still stops it at a failing gate.
2. If a wait is genuinely needed, watch something the watcher cannot
   be — a pid captured before the loop, a sentinel file, an exit-code
   file — never a name that appears in the watcher's own text.

The general shape is worth more than the instance: **an instrument whose
own presence changes what it measures reports on itself and calls it the
world.** That is the same defect as a check whose scope is wrong, seen
from the other side, and §3.5 predicts it — diagnostics are structurally
the least-executed code, so they are where this lands.

## R10. Full diagnostic output survives; excerpts say they are excerpts

**The full output of a diagnostic is authoritative and must survive.**
Any human-facing excerpt must visibly declare that it is partial, and
must preserve the terminal failure context by default.

Earned twice, the same defect wearing different clothes:

* 2026-08-11, #47 run 2: a resolver sent stdout, stderr and the exit
  status of every attempt to `/dev/null` and returned one string. When it
  came back empty the chain collapsed to `UNRESOLVED` and the failing
  stage was unknowable.
* 2026-08-12, #41-B deployed run 1: a collector recorded
  `cut -c1-1500` of a 197KB build log. The first 1500 characters are
  Dockerfile parsing; the failure is at the end. The record stopped at
  `#13 … load build definition`, and the run could not say why the stack
  had not started.

**A truncation is a `/dev/null` with better manners.** Tail-by-default is
right for build and process failures, because the causal error is
normally terminal — but the tail is a convenience, and the full artefact
is what stops one blind spot being swapped for another. Record the byte
count too: an excerpt that does not announce its own size reads like the
whole thing.

## R11. No subject → no observation

**A dependent measurement may not execute after its prerequisite state is
unproven.**

Earned 2026-08-12. `compose up` failed, so the system under test did not
exist — and the collector went on to run fifty probes against it,
recording fifty results. Every one was correct. `service "dashboard" is
not running`, fifty times. Not fifty measurements: **one failed
prerequisite repeated fifty times**, in a table whose shape said
otherwise.

That shape is the danger. An empty table invites a question; a full table
of correct-looking rows invites a conclusion.

Abort at the prerequisite boundary, say which prerequisite failed, and
say what was therefore not measured.

**And classify the abort honestly.** An instrument that detects an unmet
prerequisite and refuses to measure has *worked*. Calling that an
"instrument failure" blames the thermometer for the fever — the correct
label names the unmet prerequisite and leaves its root cause UNKNOWN
until something measures it.

## I-8. Calibrate against independent evidence

**Every instrument must be calibrated against evidence independent of
the result it is trying to prove, with a known-positive and a
known-negative case wherever that is feasible.**

Not a second measurement system for everything — that is unaffordable
and usually impossible. The requirement is that the evidence and the
claim come from *different places*, so neither can excuse the other.

Six failures in one stint were this one rule, unnamed:

* a detector's population included **its own docstring**, and the count
  was checked against itself;
* `prepush` reported "gates green" while skipping the suite that tests
  the architecture — the pass was evidence only about what it ran;
* a watcher used `pgrep` on a string its own command line contained, so
  it measured itself and reported the world;
* a denominator **shrank when a defect was repaired**, so progress and
  absence became indistinguishable;
* a meta-check wanted to *probe* a key generator to read its
  denominator, which would have written secrets as a side effect of
  measuring;
* a `KEY_ID` was classified by the word in its name rather than by what
  it is, and an exception for it would have blinded the detector to real
  key material.

The practical form, in order:

1. a **known-positive** — inject the defect, prove the check fires;
2. a **known-negative** — the correct case, prove it does not;
3. the *source of the expected answer* must not be the thing under test.

A test list derived from the module it tests is fine; a test list
**maintained beside** it is not, and neither is a count the instrument
computes about itself.

---

## Hard constraints

* **`BINANCE_API_KEY` and `BINANCE_API_SECRET` never leave the
  broker-bridge service.** They must not reach the dashboard layer under
  any bring-up, profile, or debug path.
* **No push to `main` without explicit authorisation.** Development
  happens on the designated branch.
* **`kai-pm/DECISIONS.md` is append-only.** A correction is a new entry,
  never an edit. Disproven claims stay, struck through, with what
  disproved them — deleting them hides the pattern that made the
  mistakes visible.
* **No destructive git operations without explicit permission.**
* **Do not open a pull request unless asked.**

## Where things are

| file | what it holds |
|---|---|
| `kai-pm/ORION_FIELD_NOTES.md` §0 | **start here** — where the last stint stopped, what to look at first, and what is on `main` versus the branch |
| `kai-pm/WAYPOINTS.md` | known-good commits with evidence, and standing contingencies |
| `kai-pm/DECISIONS.md` | append-only decision log |
| `kai-pm/ORION_FIELD_NOTES.md` | defect shapes, my failure modes and the tell for each |
| `kai-pm/NEXT_STINT_PLAN.md` | current plan of work and its ordering |
| `scripts/security/` | the gates; `check_gate_registry.py --gate` audits them |

## Facts about this system worth not re-deriving

* Images are `python:3.11-slim`. **No Dockerfile installs `wget` or
  `curl`** — healthchecks must use `python -c "import urllib.request…"`.
* `docker compose config` runs client-side, no daemon needed.
* Docker seeds a named volume from the image directory's contents *and*
  ownership — **and only when the volume is new.** Image content under a
  mount path is shadowed on any pre-existing volume.
* A bare `depends_on` list waits for container **creation**, not
  readiness.
* Compose passes a variable into a container **only if the service names
  it**.
* A workflow step with neither `run` nor `uses` is schema-invalid:
  GitHub rejects the file, schedules **zero jobs**, and the run reads
  like an ordinary red build.
* The Actions log API serves a **fixed byte window** from the end —
  measured identical at 15,780 characters for two different
  `tail_lines`. Diagnostics outside it may as well not exist, which is
  why live steps tee to a file the post-mortem reprints last.
* Several services are attached only to networks declared
  `internal: true` and therefore have **no egress at runtime**, by
  design. Anything they need from the network must be in the image.
