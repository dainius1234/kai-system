# Field notes — Orion

Written 2026-08-07, after the stint that took `core-tests.yml` from
dying at step 7 to 67 of 67 green across four compose profiles.

Not a summary of what was built. This is for **me, next time**: the
defect shapes worth looking for first, my own failure modes and the tell
for each, and which habits actually paid.

---

## 1. The one finding

Every defect in this stint — system and instrument alike, seventeen
venues — was the same shape:

> **A check whose scope was smaller than its name implied.**

The operator's sharper form: *a check's scope is defined by the data it
traverses, not by a static list.*

The remedy never varied: **state the denominator, and derive it from the
tree rather than from a list kept beside it.**

### The corollary that costs more when ignored

The finding **inverts**, and the inverted form is worse:

> A scope **larger** than reality reports failure over things that are
> right. False positives send people to break working code and bury the
> one true finding.

I hit this three times in two days. Each time the broad rule "worked" in
the sense of catching the real defect, and would have done net harm:

| broad rule | findings | real |
|---|---|---|
| every third-party import in a shipped package | 100+ | 1 |
| every command in a healthcheck (regex-split) | 69 | 0 (tree was correct) |
| every import anywhere in `common/` | flagged `torch` at weather-service | none |

**Rule for next time:** before widening a scope, run the widened rule
and count. If it produces more than a handful, the rule is wrong, not
the tree. Narrow to the decidable part and say out loud what is left
uncovered.

---

## 2. Defect shapes, in the order they are worth looking for

Ranked by how often they appeared and how cheaply they are found.

1. **The list beside the thing.** A hand-written list of names next to
   the thing it describes. Fourteen-plus venues: `CORE_SERVICES` (7 of
   49), `COMPOSE_FILES` written out three times, dump steps naming three
   services, a workflow's dependency list, a registry scoped to one
   directory. *Look for:* any literal tuple/list of names in a gate.
   *Fix:* derive it, then assert the derived count is plausible.

2. **Never-executed code.** Every single defect lived in code that had
   never run. Not one was code that used to work and broke. *Look for:*
   what has no execution path in CI. `report_execution_coverage.py`
   exists for exactly this and its number is the map of remaining risk.

3. **Configuration that was never loaded.** The subject-changed variant.
   `security/policy.yml` calls itself "the single source of truth"; 35
   images shipped `common/` without pyyaml, so no container had ever
   read it. *Look for:* a file described as authoritative, and evidence
   that something actually parsed it.

4. **Two copies of one fact, disagreeing.** Same service, three compose
   profiles, one different: `memu_db` vs `sovereign`; `wget` vs
   `python -c`; bare `depends_on` vs `condition: service_healthy`. The
   different copy is always the one that never ran. *Look for:* diff the
   same service across profiles.

5. **A guard that cannot fire.** `if not X.exists(): continue` — absence
   reading as correctness (I-1). Also: an advisory `print()` where a
   failure belongs (I-5), and a gate whose scope is a *directory* rather
   than the property that matters.

6. **A message that names the wrong cause.** `POLICY FILE CORRUPT` when
   the file was fine and pyyaml was missing. This is the one that makes
   defects survive for months: it sends every reader to the wrong place.
   *Fix:* the error must name the thing to change.

---

## 3. My failure modes, and the tell for each

Written plainly because the correction only happens if the pattern is
recognisable in flight.

### 3.1 Concluding from a scope narrower than the claim

The single most common one, and the same shape as the systemic finding
— which is uncomfortable and worth remembering. Instances:

* Said "platform-wide outage" without checking whether *other*
  workflows were getting runners.
* Over-corrected to "only core-tests affected" without checking whether
  the other workflow's *failures* were the same signature.
* Said "push events are broken" by comparing dispatch-at-06:15 against
  push-eleven-hours-earlier — two variables, one conclusion.
* Said "CI" all day while watching one workflow out of nine. That is how
  `policy-checks.yml` ran nothing on every push for a day unnoticed.

**Tell:** I am about to describe the *shape* of a failure ("it's X, not
Y"). **Stop and ask:** what is the full set of things that could exhibit
this, and have I looked at all of them?

### 3.2 Asserting something I have not run

* Claimed a re-run "bought nothing" — which assumed it would complete.
  It was cancelled.
* Claimed run 695 had "survived past the 15-minute mark". It hadn't; I
  misjudged elapsed time instead of computing it.
* Wrote in a post-mortem that `workflow_dispatch` "cannot help" — then
  it was the thing that unblocked the incident.

* Argued a CI failure could not be mine because my change was
  *downstream* of the failing service. The argument was sound. It was
  still only an argument — and the re-run showed the failure was
  deterministic when I had predicted a flake.

**Tell:** a sentence containing "would have", "cannot", "is simply",
"should work", or "it's just a". **Fix:** `date -u`, run the command,
read the field. Cheap every time.

**Promoted to `CLAUDE.md` R1 on 2026-08-07**, at the operator's
instruction, because a rule recorded only in my own notes is a rule I
have to remember to re-read.

### 3.2b Writing a contingency instead of running one

Asked whether a contingency existed, I could have answered from the
files — `main` untouched, a waypoint recorded, one isolated commit — and
every part of that answer would have been true. Checking instead found
that the "fail closed" model download I had shipped an hour earlier had
**no retry**, so one dropped packet would fail every build. Fail closed
means *never ship a broken image*. It does not mean *die on the first
blip*. Brittle and strict are different properties wearing the same
word.

I then ran `git revert --no-commit` rather than asserting it would
apply, and confirmed the rollback target builds *without* the dependency
whose outage it exists to survive. Both took under a minute.

**Tell:** I am about to describe a safety net in the present tense.
**Fix:** execute it. A rollback that has not been run is a hypothesis
with good presentation. `CLAUDE.md` R2.

### 3.3 The `;` versus `&&` push

Twice in one day I ran `make policy-check ; git commit ; git push` and
pushed past a failing gate. The second time was hours after recording
the first.

**Fix, permanently:** `make policy-check && git commit && git push`.
Never `;`. If a chain has a gate in it, the gate must be able to stop it.

### 3.4 Shipping a detector before calibrating it

Every gate I wrote was wrong before it was right, and each time the
error was found by *pointing it at input whose answer I already knew* —
never by re-reading it.

**Rule:** calibrate before trusting. Run the new detector against the
tree as it was one commit ago and count. Expect the exact number of
known defects; anything else means the detector is wrong.

---

## 3.5 The recursion — my instruments keep inheriting the defect they hunt

Written 2026-08-07 evening, because this is the strongest pattern of the
week and I did not see it until the fourth instance.

Every tool I built to *see* failures has carried the same defect as the
system it was watching:

| instrument | its own defect |
|---|---|
| the gates | scope smaller than the name implied — the systemic finding, in the thing looking for it |
| `check_gate_registry` | resolved 3 of 8 modules into a directory where the file did not exist, so every AST check returned "no findings" |
| the post-mortem | printed twelve empty sections and evicted the one with content from the log window |
| D167's "tune from a measured number" | promised a measurement and built no instrument to read it |

Four venues. Not coincidence.

**The mechanism, which I think is structural rather than personal.**
Diagnostics are, by construction, the *least-executed code in any
system*. A post-mortem runs only on failure. A gate's failing branch
runs only when something is wrong. An error message's text is reached
only in the case nobody tested. So the observability layer is made
almost entirely of the exact category that §2.2 identifies as where
every defect lives — **never-executed code** — and it is the category
you are most dependent on precisely when you can least afford it to be
wrong.

That reframes the remedy. "Write better diagnostics" is not a plan;
diagnostics fail for the same reason all never-executed code fails, and
willpower does not execute code. The plan has to be **exercise the
failure path on purpose, routinely**, so it stops being never-executed:

* inject a failure on a schedule and assert the post-mortem produced a
  *readable* answer, not merely that it ran — chaos engineering aimed at
  the observability layer rather than at the system
* the assertion should be a **property of the output**, e.g. "under 20
  lines when only the build failed", not "did not crash".
  `scripts/test_post_mortem.py` does exactly this and is the first
  instrument here whose test asserts legibility rather than function
* every "we will tune this from a measured number" must ship the
  instrument that produces the number **in the same commit**, or the
  sentence is a promise to nobody

**The uncomfortable corollary.** If diagnostics inherit the defect of
the observed, then the *rules* in `CLAUDE.md` are also a diagnostic —
they are an instrument pointed at me — and they will inherit it too. A
rule that is never exercised is a rule I will discover was wrong at the
worst moment. Which is an argument for turning rules into gates wherever
they can be turned, and for being honest about the floor of ones that
cannot.

## 3.6 A caution about the rules themselves

Eight rules exist as of today. Four of them caught me within an hour of
being written, which reads as success and might not be.

The failure mode to watch: **rules accumulate, and a wall of rules has
the same defect as a warning printed on every run** — I wrote in §5 that
printing forever is what teaches everyone to ignore it. Twenty rules is
a document nobody reads, including me. So the ratchet on this file is
not "add a rule per incident"; it is:

1. Can this be a **gate**? Then it belongs in `scripts/security/`, not
   in prose. I-2 already enforces R5. The register already enforces R7.
2. If it cannot be a gate, is it a *tell* — a recognisable signal in
   flight — or merely good advice? Only tells earn a place. R1's value
   is not "be careful"; it is the five specific words that mean stop.
3. If a rule has not fired in a month, it is either working or dead, and
   those look identical. That is I-7 aimed at myself, and I do not yet
   have an answer for it.

## 4. What actually worked

* **Calibration against a known-answer input.** Caught every bad
  detector. `git show HEAD:file > file`, run, count, restore.

* **Deriving the denominator.** Every real fix was "read it from the
  tree" rather than "update the list".

* **Fixing at the root, not the instance.** When a remedy was applied to
  one dump step and three others lacked it, applying it to all four was
  right — and I had to be told that four times before it stuck. *If a
  fix has a denominator, apply it to the denominator.*

* **Teeing every live step to a file the post-mortem reprints.** The
  Actions log API serves a **fixed byte window** — measured: identical
  15,780 characters for `tail_lines=130` and `255`. Diagnostics outside
  it may as well not exist. Every sovereign and full-profile cause was
  found because of this.

* **Narrowing diagnostics to what is wrong.** `dump_unhealthy.py` prints
  only containers Docker says are unhealthy, derived from `compose ps`.
  40 lines × 20 services does not fit the window; burying the answer is
  the same failure as truncating it.

* **Asking for a second opinion.** DeepSeek's reframe — *a gate cannot
  decide whether a guarded fallback works, but the service can decide
  its own output is nonsense* — turned an undecidable static question
  into a decidable runtime one, and produced the policy-loader refusal.
  Their Q2 answer was wrong on mechanism, which was also useful: it made
  me check rather than accept.

* **Writing the record honestly, including strike-throughs.** The
  post-mortem keeps disproven claims struck through with what disproved
  them. Deleting them would have hidden that I made the same error six
  times, which is the only reason the pattern became visible.

---

## 5. Things that are true about this system

Worth not re-deriving:

* `docker compose config` runs **client-side, no daemon**. Validates
  compose files, `depends_on` targets and condition values without a
  runner. Usable when CI is unavailable.
* Docker seeds a fresh named volume from the image directory's
  **contents *and* ownership**. A mountpoint must exist and be owned
  correctly *before* the volume is created.
* Compose interpolates `$VAR` at parse time, before any container
  exists. `$$` passes a literal `$` to the shell.
* Compose passes a variable into a container **only if the service names
  it**. Setting it at step scope does nothing otherwise.
* `docker compose ps` reports empty `Health` for a container that is not
  *running* — so "no healthcheck" and "dead" look identical unless
  `State` is read too.
* A bare `depends_on` list waits for **creation**, not readiness.
* A workflow step with neither `run` nor `uses` is schema-invalid:
  GitHub rejects the file, schedules **zero jobs**, and the run reads
  like an ordinary red build.
* The images are `python:3.11-slim`. **No Dockerfile installs `wget` or
  `curl`.** Healthchecks must use `python -c "import urllib.request…"`.

---

## 6. Open, for the next stint

* **26 of 49 services have never been started by anything** — all behind
  `profiles:` gates (executor, verifier, supervisor, fusion-engine among
  them). Given that every defect this stint lived in never-executed
  code, this is where the next ones are.
* The stack is proven to **boot and answer smoke probes** with fake
  embeddings and throwaway secrets. Sustained operation under real
  embeddings and real credentials is not proven.
* Lint waves: F401 321, E501 246, E127/E221 272, F841 24, F541 20.
* A-03 sentinel mutation audit; KAI-GATE-026 container CVEs.
* KAI-GATE-034 tolerated, owner Orion, review 2026-08-12.
* 60 commits ahead of `main`, unmerged — awaiting explicit authorisation.
