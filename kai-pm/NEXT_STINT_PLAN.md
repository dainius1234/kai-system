# Next stint — plan of work

Written 2026-08-07, after `core-tests.yml` reached 67 of 67 green across
four compose profiles (waypoint `097c91d`, see `WAYPOINTS.md`).

Agreed with the operator on 2026-08-07. The ordering below is
deliberate and the reasoning for it is in §1 — **do not reorder without
reading it**, because the ordering is the whole point.

---

## 1. The strategy: fix the class, not the instance

The operator's framing, kept in their words because it is the clearer
statement:

> "If we have now just one set of wheels and 50 cars and try to move
> them, and each time we need to take the wheels off to put them on
> another car — same again 50 times — it's painful. So why not put all
> cars on wheels, or on a trailer, and make the issue go away."

**They are right, and it is the same finding this programme has been
chasing all week from the other side.** The seventeen defects of the
last stint were each *a check whose scope was smaller than its name
implied*. Fixing one instance of a class and leaving the rest is the
identical error wearing a different hat: a **fix** whose scope is
smaller than the class it belongs to.

I have already made this mistake, repeatedly and in writing —
`ORION_FIELD_NOTES.md` §4 records that I had to be told four times that
a remedy applied to one dump step belonged on all four. So this is not
an improvement being adopted on advice; it is a correction.

### 1.1 A live example, found while writing this file

The subplan below listed "full.yml: 16 services declare `depends_on`
with no readiness condition" as done. It is done — for `full.yml`.
Measured just now:

| file | bare `depends_on` lists |
|---|---|
| `docker-compose.full.yml` | 0 |
| `docker-compose.minimal.yml` | 1 (`skill-hunter`) |
| `docker-compose.sovereign.yml` | 10 |

A bare list waits for container **creation**, not readiness. Eleven of
them survived a fix that was declared complete, in the same tree, on the
same day, and **no gate exists for this class** — 30 gate scripts, none
mentions `depends_on` readiness. The class was fixed in one file and
called finished.

That is the operator's fifty cars exactly, and it was found in about
ninety seconds by counting the class across the tree instead of trusting
the completion note. It becomes **item 0** below.

### 1.2 What is already on the trailer

The good news, verified today rather than assumed:

```
built services (denominator derived from compose `build:` stanzas) .. 49
  started by CI at least once ....................................... 23
  never started by anything ......................................... 26
```

The static gates scan **all 49**, not the 23 that CI boots:

| gate | inspected |
|---|---|
| `check_image_modules` | 49 services |
| `check_shipped_package_deps` | 38 first-party packages |
| `check_healthcheck_runnable` | 69 healthchecks across 49 services |
| `check_dockerfile_context` | 267 `COPY` sources |

So for every class we have encoded statically — missing modules, missing
package dependencies, unrunnable healthchecks, broken `COPY` contexts —
**the 26 never-started services are already scanned and already clean.**
The trailer exists. The last stint built it without naming it.

### 1.3 Where the trailer does not reach — the real gap

Static scanning cannot answer questions whose answer only exists at
runtime. Every one of these bit us last stint and none is currently
pre-scannable:

* **Does the process actually start?** `uvicorn` reaching "Application
  startup complete" is not implied by imports resolving.
* **Is the value right, or merely present?** `memu_db` vs `sovereign`
  were both syntactically valid database names. A gate sees a string.
* **Does the volume mount with the right ownership?** Docker seeds a
  fresh named volume from the image directory's contents *and*
  ownership. Only visible on first boot.
* **Does the readiness signal ever arrive?** The healthcheck gate proves
  the binary exists. It does not prove the endpoint answers.
* **Does the service degrade correctly when a dependency is absent?**
  DeepSeek's reframe applies here: a gate cannot decide whether a
  guarded fallback works, but the service can decide its own output is
  nonsense.

**Therefore the 26 never-started services remain the highest-value work
— but for runtime reasons, not static ones.** Booting them is the only
instrument that answers the list above. That is item 2, and it is the
main body of the stint.

### 1.4 The rule that makes batch-fixing safe

Batch-fixing is how this programme produced its three worst incidents. A
rule applied to a large denominator before it is calibrated does not
find fifty defects; it manufactures fifty false findings that send
people to break working code and bury the one true one:

| broad rule, uncalibrated | findings | real |
|---|---|---|
| every third-party import in a shipped package | 100+ | 1 |
| every command in a healthcheck (regex-split) | 69 | 0 |
| every import anywhere in `common/` | flagged `torch` at weather-service | 0 |

So the sequence is fixed, and every batch fix in this plan follows it:

1. **One confirmed instance.** A defect actually observed, not suspected.
2. **Derive the rule** from why that instance is wrong.
3. **Measure the population** — run the rule, print the count, *do not
   fix anything yet*.
4. **Calibrate against a known answer.** Run it against the tree one
   commit before the known fix. It must report exactly the known
   defects. `git show HEAD~1:file > file`, run, count, restore.
   *If step 3 produced more than a handful, the rule is wrong, not the
   tree.*
5. **Fix the whole population.**
6. **Gate it**, so the class cannot come back — with the denominator
   printed in the gate's own output (I-2), and proof it can fail (I-3).

Step 3 before step 5 is the load-bearing part. Counting is cheap;
un-breaking fifty files that were right is not.

---

## 2. The work, in order

### Item 0 — bare `depends_on`, all profiles — **DONE, same commit**

Kept here in full because the point of it was to run §1.4 end to end on
real input before the sequence is trusted on bigger work. What each step
actually produced:

| step | result |
|---|---|
| 1. confirmed instance | `full.yml`'s 16, fixed in `e47622b` |
| 2. rule | every `depends_on` is a mapping with a valid explicit condition |
| 3. **measure first** | 15 findings — 11 bare lists, 4 clause-2 |
| 4. **calibrate** | against `e47622b~1:docker-compose.full.yml`: **16 findings, 0 advisories** — exactly the known answer |
| 5. fix | 11 fixed; `sovereign` redis given the healthcheck the other two profiles already had |
| 6. gate | `scripts/security/check_depends_on_readiness.py`, 90 edges, in `policy-check` + `policy-checks.yml` |

**Step 3 changed the design, which is the whole argument for doing it.**
The first draft enforced a second clause — *if the target has a
healthcheck, the condition must be `service_healthy`* — and measuring
before fixing showed 4 hits, all on `dashboard` in `minimal.yml`, in the
profile CI proves green:

    tts-service, notify-service, document-parser, agentic

Blocking a UI's start on a slow optional dependency is a design
decision, and `service_healthy` against a service that never reaches
healthy blocks the whole bring-up. Enforcing that clause would have
failed on something that is right and sent someone to break a green
profile. It is now **reported and counted, not enforced**, with the
reason written into the gate.

Had I fixed before measuring, I would have "fixed" four correct
declarations and called it a clean sweep.

Register: this does **not** close the class per Programme Rule 7.
Closure is a separate evidence-backed action, and the evidence here is
static only — no profile has been booted with the new conditions yet.
CI on this commit is the first runtime evidence.

Still open on this item: `sovereign`'s `vault-rotator` waits with
`service_started` because `vault` has no healthcheck, and `vault` is in
the `dev` profile which nothing has ever started. Adding a probe there
would be an unverifiable healthcheck on a never-run service — precisely
the `wget` defect. It goes into item 2 with the rest.

### Item 1 — delete the stub eraser — **DONE**

`scripts/kai_supervisor.py:111-118`:

```python
if "TODO" in src or "pass  # stub" in src or "NotImplementedError" in src:
    new_src = (src.replace("TODO", "")
                  .replace("pass  # stub", "")
                  .replace("NotImplementedError", ""))
    pyfile.write_text(new_src, encoding="utf-8")
    log_supervisor_action("auto_stub_removed", ...,
                          rationale="Stub/TODO removed for production readiness.")
```


**Done 2026-08-07.** The whole file removed, not just the eraser —
183 lines, wired to nothing, containing three paths that rewrite the
repository and one that posts to a hard-coded URL swallowing every error.

Measured before deleting (§1.4 step 3), against its own glob:

| what it would rewrite | files |
|---|---|
| denominator — `scripts/*.py` | 251 |
| files containing `TODO` | 3 |
| files containing `pass  # stub` | 2 |
| files containing `NotImplementedError` | 2 |
| files it would inject a generic docstring into | 26 |
| files with a bare `raise NotImplementedError` (→ `raise `, a SyntaxError) | **0** |

That last zero is luck, not safety. Any future bare `raise
NotImplementedError` in `scripts/` would have become a syntax error, and
`safe_experimentation()` separately writes `scripts/sandbox_experiment.py`
into the tree on every call.

**And measuring stopped a gate that would have been wrong.** The obvious
R6 move was "no script may rewrite `.py` files it did not create". Counted
first: 469 python files inspected, **5** glob `*.py` and write files —
and four are correct.

| script | writes | verdict |
|---|---|---|
| `sync_docs.py` | README / BACKLOG markdown | legitimate |
| `hygiene_survey.py` | its own baseline | legitimate |
| `test_image_modules.py`, `test_test_wiring.py` | test fixtures | legitimate |
| `kai_supervisor.py` | **source files it did not create** | the defect |

Population one, about to be zero. **No gate written** — a gate for a
class with no members is an inert rule (I-5), and this one would have
reported failure over four things that are right. Third time this week
that counting first changed the answer.

This does not remove stubs. It removes the *words* that name them, over
every `*.py` in `scripts/`, and logs the result as production readiness.
Deleting the token `NotImplementedError` from a `raise` leaves a
`NameError` at runtime, and every marker a future audit would search for
is gone.

Verified wired to nothing — absent from the Makefile and from all nine
workflows. It is dormant, not running. Delete it; keep the docstring
pass if it is wanted, but the eraser goes.

**First, because it is a tool that damages the tree, and everything
after this touches the tree.**

### Item 2 — the 26 never-started services

**Measured 2026-08-07 before starting.** They are not a flat list of 26;
they are **8 profile groups**, and which compose file defines them
matters more than I had assumed — most live only in `minimal.yml`, not
`full.yml`:

| profile | services | defined in |
|---|---|---|
| `sensors` | audio, camera, clipboard, files, screen-capture, screen-watcher, vision, wake | 8 — split across minimal and full |
| `watchers` | docker-watcher, git-watcher, monitor-service, sysmetrics | 4 — minimal only |
| `external-egress` | browser-agent, email-reader, news-feed, telegram-bot | 4 — minimal and full |
| `introspection` | agentic-introspect, cortex, letta-agent | 3 |
| `recovery` | fusion-engine, supervisor, verifier | 3 |
| `finance` | broker-bridge, financial-awareness | 2 |
| `execution` | executor | 1 |
| `vault` | vault-sync | 1 |

Verified at the same time, because it was an assumption I had written
into `WAYPOINTS.md` without checking: **26 of 26 are behind a
`profiles:` gate in every file that defines them. None would start on a
bare `up`.** The claim held.

**Boot order — cheapest and least dangerous first**, because each batch
is a chance to learn a defect class and ask whether it is scannable
across the rest (§1.4 step 1):

1. **`watchers`** (4). Zero `depends_on`, no secrets, no host devices,
   minimal.yml only. The safest possible first contact with a
   never-executed service.
2. **`introspection`** (3) and **`recovery`** (3). 1–3 dependencies, no
   secrets. `supervisor` and `verifier` are the two the register has
   been carrying longest.
3. **`sensors`** (8). The largest group. `camera-service` declares a
   secret and several want host devices — expect the volume-ownership
   and device-node classes here, and expect some to be un-bootable in
   CI by nature. Say which, out loud, rather than quietly skipping.
4. **`external-egress`** (4). By definition these want the network. On a
   runner they will exercise the *degradation* path, which is worth
   knowing and is not the same as exercising the service.
5. **`execution`** (1) and **`vault`** (1).
6. **`finance`** (2) **last, and carefully.** `broker-bridge` carries the
   standing constraint: `BINANCE_API_KEY` and `BINANCE_API_SECRET` never
   leave that service and must not reach the dashboard layer under any
   bring-up. Booting it is the first time that constraint is tested by
   something other than reading.

**What each batch must produce**, or it does not count as done: the
services boot and answer a probe; every failure is classified; and for
each class, an answer to *is this scannable across the other groups?* A
batch that fixes its own failures and encodes nothing is the outcome §4
names as failure.



The main body. All 26 sit behind `profiles:` gates and have never been
started by anything:

```
agentic-introspect  audio-service     broker-bridge     browser-agent
camera-service      clipboard-service cortex            docker-watcher
email-reader        executor          files-service     financial-awareness
fusion-engine       git-watcher       letta-agent       monitor-service
news-feed           screen-capture    screen-watcher    supervisor
sysmetrics          telegram-bot      vault-sync        verifier
vision-service      wake-service
```

Every defect of the last stint lived in code that had never executed.
Not one was code that used to work and broke. So this list is not a
comfortable remainder — it is the map of where the next findings are.

Approach: **boot them in batches, smallest dependency footprint first**,
and after each batch ask what class the failures belonged to and whether
that class is scannable across the other 25. The batches feed §1.4 step
1 with confirmed instances; that is how the runtime gap in §1.3 gets
converted into static gates.

`broker-bridge` carries the standing constraint: `BINANCE_API_KEY` and
`BINANCE_API_SECRET` never leave the broker-bridge service. They must
not reach the dashboard layer under any bring-up.

### Item 3 — no-op fallback audit

77 functions repo-wide whose entire body is `pass` or a bare
`return`/`return None`/`return 0`/`return ""` (measured today by AST,
excluding `.venv` and `_archive`). Includes a repeated `ErrorBudget`
null-object pattern duplicated across service `app.py` files
(`sysmetrics`, `weather-service`, `browser-agent`, `email-reader`, … —
`__init__` and `record`, both empty, at nearly identical line numbers).

**Not all 77 are defects.** An abstract base method, a protocol stub and
a deliberate null object are all legitimately empty. This is the class
most likely to produce a 100-findings-1-real repeat, so §1.4 step 3 is
mandatory and the split must be by *decidable* property — e.g. "empty
body on a function whose caller checks its return value" — not by
"looks like a stub". Expect to narrow the rule hard and to state out
loud what is left uncovered.

### Item 4 — lint waves

Measured today with the exact CI-equivalent invocation
(`--max-line-length=127 --exclude=.venv,_archive,__pycache__`), total
**1,028**:

| code | count | note |
|---|---|---|
| F401 imported but unused | 322 | mostly `common.degraded.degradation_report` |
| E402 module-level import not at top | 321 | |
| E501 line too long (>127) | 171 | |
| C901 too complex | 108 | advisory |
| E701 multiple statements (colon) | 27 | |
| F841 assigned but never used | 24 | |
| F541 f-string without placeholders | 21 | |
| E741 ambiguous name `l` | 8 | |
| E401 multiple imports on one line | 17 | |
| E702 multiple statements (semicolon) | 3 | |
| F811 redefinition of unused | 2 | |
| F601 dict key repeated with different values | 2 | |

These are **advisory in CI** (`--exit-zero`); the blocking selection is
`E9,F63,F7,F82` only, and that is at zero.

Two of these are not cosmetic and should be read before any bulk
autofix: **F811** (a redefinition means one of the two definitions never
takes effect — the never-executed-code shape) and **F601** (a repeated
dict key with *different values* means one value is silently discarded).
Triage those twelve by hand first; batch the rest.

F401 at 322 with one import dominating suggests a single template
copied across services rather than 322 independent decisions — check
that before removing, because if the import has an import-time side
effect, removing it changes behaviour.

### Item 5 — carried debt

* **A-03** sentinel mutation audit — register entry says unscheduled,
  audit tool is not a gate.
* **KAI-GATE-026** container CVEs.
* **KAI-GATE-034** tolerated, owner Orion, **reviewed 2026-08-12 (D176),
  extended to 2026-08-17**. `weekly-report-card` half SATISFIED — run 443,
  `194db0a`, green, and that tree verified to contain both fix halves.
  `friday-cleanup` half has **never run from a fixed tree**: its last
  firing (2026-08-07, `a0298c6`) predates the merge and failed with the
  tracked defect. Next admissible evidence 2026-08-14 09:00 UTC.
* Tag `v0.1-all-profiles-green` exists locally only; five push attempts
  died on `send-pack: unexpected disconnect`. Retry when the network
  allows. `WAYPOINTS.md` is the source of truth regardless.
* From the post-mortem, still open: githubstatus.com could not be
  reached from this environment (proxy-blocked) during the runner
  starvation incident, so the platform-status question was never
  answered from a primary source. Billing was ruled out — the operator
  holds a paid subscription.

---

## 3. Programme rules that apply to this stint

Restated because each one exists because it was broken.

* **Rule 7.** Finding counts do not change until a formal closure
  review. Closure is a separate, evidence-backed register action — not a
  side effect of a fix landing.
* **`&&`, never `;`.** `make policy-check && git commit && git push`.
  Twice in one day a `;` chain pushed past a failing gate, the second
  time hours after the first was written down.
* **Calibrate before trusting.** Every gate written last stint was wrong
  before it was right, and every error was found by pointing it at input
  whose answer was already known — never by re-reading it.
* **`DECISIONS.md` is append-only.** A correction is a new entry.
* **No push to `main` without explicit authorisation.**

## 4. What would make this stint a failure

Named in advance so it is recognisable in flight:

* Finishing item 2 having booted the 26 services and fixed what broke,
  but having added **no new gate** — that would mean the runtime
  findings were converted into fixes and not into instruments, and the
  next tree gets bitten by the same class.
* Any batch fix landing without its step-3 count in the commit message.
* Declaring a class closed on the evidence of one file, which is item 0's
  entire subject.
