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

**Tell:** a sentence containing "would have", "cannot", or "is simply".
**Fix:** `date -u`, run the command, read the field. Cheap every time.

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
