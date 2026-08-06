# Post-mortem — CI runner starvation, 2026-08-06

**Status:** ongoing at time of writing. Cause not fully determined; this
records what is *known* versus *inferred*, and counts the reasoning
errors made along the way.

---

## 1. Timeline

All times UTC. `runner_id: 0` means the job was never assigned a
machine.

| time | run | outcome |
|---|---|---|
| 15:23:09 | core-tests 694 attempt 1 | `Failed to resolve action download info. Error: Service Unavailable` — 2 retries, died before `actions/checkout` |
| 15:23:16 | unified-hunter 66, python-app 617 | **success** |
| 15:52:12 | core-tests 694 attempt 2 | cancelled, `runner_id: 0`, **15m02s** |
| 16:00:13 | core-tests 695 attempt 1 | cancelled, `runner_id: 0`, **15m02s** |
| 16:00:13 | **python-app 618** | cancelled, `runner_id: 0`, **15m02s** |
| 16:20:05 | core-tests 695 attempt 2 | cancelled, `runner_id: 0`, **15m02s** |
| 16:31:28 | core-tests 696 | cancelled, `runner_id: 0`, **15m02s** |
| 16:31:27 | python-app 619 | **success** |
| 16:44:42 | core-tests 697 | cancelled, `runner_id: 0`, **15m02s** |

Six consecutive `core-tests` starvations. Every cancellation is exactly
15m02s from its own queue entry.

## 2. Signature

Deterministic 15m02s queue TTL with no runner assignment. Predicted
697's cancellation time (16:44:42 + 15m02s = 16:59:44) before it
happened; it landed to the second. The mechanism is confirmed even
though the cause is not.

## 3. What is known

* Runner allocation for this repository became intermittent from
  ~15:52 and had not recovered at time of writing.
* **It is not workflow-specific.** `python-app` starved with the
  identical signature at 16:00. `unified-hunter` failed in the same
  window.
* **It is intermittent, not absolute.** `python-app` succeeded at 16:31
  while `core-tests`, queued one second earlier, starved.
* No `concurrency:` block exists in any of the nine workflows, and runs
  686/687 were observed running simultaneously on the same branch
  earlier the same day.

## 4. Hypotheses raised and killed

| hypothesis | source | killed by |
|---|---|---|
| A new push cancels queued runs via branch concurrency | operator | No `concurrency:` block anywhere; 686/687 ran simultaneously; and **nothing was pushed after `e47622b`**, yet 695 died identically. Cancellations occur 15m02s after each run's *own* queue entry, not on the push. |
| Platform-wide outage, all Actions down | Orion, endorsed by DeepSeek | `python-app` 619 and `unified-hunter` 66 both **succeeded** inside the window. |
| Not an outage — only `core-tests` affected | Orion (over-correction) | `python-app` 618 starved with the identical `runner_id: 0` / 15m02s signature. |
| `models: read` permission blocks scheduling — the only structural difference between `core-tests` and the workflows that get runners | Orion | `python-app` requests no `models` scope and starves identically. **Experiment cancelled before it was run.** |

Remaining, unproven: intermittent GitHub-side capacity for this
repository or region. GitHub's status page could not be reached from
this environment (`CONNECT tunnel failed, 403` via the agent proxy);
verification needs a browser.

## 5. Reasoning errors, counted

Four, all mine, all corrected — but they should be counted rather than
narrated away.

1. **Claimed the isolated re-run "bought nothing"** because it would
   test `09313ce` regardless. That assumed the re-run would complete.
   It was cancelled, and the isolated data point was lost.
2. **Claimed 695 had "survived past the 15-minute mark"** as a sign of
   recovery. It had not; I misjudged elapsed time. It died at exactly
   15m02s like the others.
3. **Endorsed "platform-wide outage"** without checking the one thing
   that could falsify it — whether other workflows were getting
   runners. Three of them were, some of the time.
4. **Over-corrected to "not a blanket outage, `core-tests` specific"**
   on the strength of `python-app` succeeding once, without checking
   whether its *failures* were starvation. They were.

Errors 3 and 4 are the same error in opposite directions, and both are
the programme's own systemic finding applied to reasoning rather than
code: **a conclusion drawn from a scope narrower than the claim.** The
fix is identical too — before asserting the shape of a failure, check
the whole denominator of things that could exhibit it.

## 6. Impact

**Verification blocked.** Four changesets remain unexecuted in CI:

* the two sovereign fixes (`memu_db` → `sovereign`; pyyaml in 35 images)
* the `depends_on` readiness conversion (16 services in `full.yml`)
* the policy-loader refusal
* the `policy-checks.yml` schema repair

None of these is being described as fixed.

**Verification that did survive.** `unified-hunter` succeeded on
`09313ce`, so all 52 gate suites — 2,814 assertions — passed *in CI* on
the commit carrying the sovereign fixes. `python-app` covers lint and
`py_compile`. The Docker profiles are the uncovered part.

## 7. What the outage bought

Being unable to run CI is what caused the sideways look that found the
day's largest instrumentation defect:

* **`policy-checks.yml` had been executing nothing on every push**, for
  at least a day — a step with a `- name:` and no body, which GitHub
  rejects, scheduling zero jobs. Found only while checking whether
  other workflows were also starved.
* The blast radius was then *measured* rather than feared: the 26 gate
  scripts it invokes are exactly the 26 in `make policy-check`, which
  runs before every push. Nothing was unenforced; the CI-side
  redundancy was dead.
* `check_ci_tolerations.unparseable()` was extended from "does Python
  parse it" to "will GitHub run it" — the same scope error it was
  originally written to prevent, one level up. Calibrated: exactly 1
  finding on the real pre-fix file, 0 after.

## 8. Actions

* **Open — needs a browser:** check githubstatus.com for an Actions
  incident on 2026-08-06 15:50–17:00 UTC, and check org billing for a
  spending limit. Neither is reachable from this environment.
* **Done:** stop re-queueing continuously. Capacity is intermittent, so
  periodic retries are the correct cadence; each failed attempt costs 15
  minutes of wall-clock and produces no information.
* **Done:** treat a cancellation as *no result*, never as a code result.
  Run 694 attempt 1's `conclusion: failure` had nothing to do with the
  tree, and reading it as a verdict would have been the exact
  "diagnostic that reports something other than what happened" defect
  this programme exists to remove.
* **Recorded, not done:** if future isolation of a single commit is
  needed, push it to a throwaway branch rather than re-running on the
  working branch. Not because of concurrency — that was disproven — but
  because a re-run competes for the same scarce capacity as everything
  else on the branch.
