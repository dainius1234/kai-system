# Waypoints — known-good commits, with the evidence

Survey benchmarks for this repository. Each entry is an **exact commit
sha**, what was *proven* about it, and the CI run that proved it.

## Why this file exists rather than a git tag

A tag is a name pointing at a sha, stored in git's ref namespace and
pushed separately from your commits. On 2026-08-07 the tag
`v0.1-all-profiles-green` was created locally and **failed to push five
times**:

    send-pack: unexpected disconnect while reading sideband packet
    fatal: the remote end hung up unexpectedly
    Everything up-to-date

(That last line is `git push` reporting two contradictory things in one
breath — it refers to branch refs, not the tag that just failed. The
remote was checked with `git ls-remote` rather than trusting it.)

So the benchmark existed on one machine and nowhere else. A rollback
point that only one party can see is not a rollback point.

**This file is a normal file on `main`.** It travels with every clone,
survives any tag going missing, is readable without git commands, and
records *why* a commit was good rather than only that it was. The tag is
still worth having when it pushes — but the file is the source of
truth, and it is the thing to update first.

## How to use it

```bash
git log --oneline -1 <sha>          # what is this commit
git checkout <sha>                  # look at that state
git diff <sha>..HEAD --stat         # what changed since
git checkout -b recover <sha>       # start again from there
```

---

## Standing contingency — the build now depends on huggingface.co

From `bc70c7d` onward, `memu-core/Dockerfile` downloads the embedding
model at build time and **fails the build** if it cannot. That is
deliberate (I-1: an image without the model cannot do the job, and the
runtime has no egress to fetch one later), but it means this repository
has acquired an external build-time dependency it did not have before.
So the contingency is written down rather than assumed.

**If a build fails on the model download:**

1. **First check whether it is transient.** The step retries 5 times
   over ~100s with increasing backoff. Five consecutive failures is an
   outage, not a blip.
2. **Roll back to `b5deaaa`.** Verified 2026-08-07: its Dockerfile has
   **zero** network dependencies, so it builds when huggingface.co is
   unreachable. It is green at 67 of 67 steps. It runs on hash-based
   fake embeddings, which is what the tree did anyway.
3. **Or revert just this commit.** Verified to apply cleanly:

       git revert --no-commit bc70c7d

   This restores a Dockerfile with no download and leaves every other
   fix from today in place. Preferred over a full rollback.

**What is NOT at risk:** `main` is at `194db0a` and has received none of
this work. No deployment from `main` can be affected by any of it.

**Rebuild caution:** the running container sets `HF_HUB_OFFLINE=1`, so a
memu-core image built *before* `bc70c7d` and run *after* this change is
not a combination that exists — the variable ships in the image. But an
image built after `bc70c7d` and run with an old `memu_data` volume is
fine by construction: the cache lives at `/opt`, outside every mount.

---

## `b5deaaa3b21b2c6a9ba46f17e9a5e5b1b9057797` — the boot race removed

**Date:** 2026-08-07 · **Branch:** `claude/project-rework-plan-pgvp35`
· **Merged to `main`:** no — awaiting authorisation.

The first state where the minimal bring-up is not a coin toss.

**Proven:** run 709, `conclusion: success`, 24.8 minutes, all 67 steps —
minimal, memu-graph, full and sovereign.

**What it proves that `097c91d` did not.** `097c91d` was green six runs
running, and every one of them had won a race:

| run | commit | memu-core offline guard | result |
|---|---|---|---|
| 708 attempt 1 | `e0e9849` | no | fail, step 49, 109s |
| 708 attempt 2 | `e0e9849` | no | fail, step 49, 108s |
| 709 | `b5deaaa` | yes | success |

`memu-core` sits only on networks declared `internal: true`, so it has
no egress, and its image ships `sentence-transformers` without a model.
Every boot therefore attempted a download that could not succeed and
spent 70–100 seconds in DNS backoff, against a healthcheck that gives up
at ~100 seconds. Attempt 2 was run on the *unchanged* commit
specifically to establish that this was deterministic rather than a
flake — it was.

Also carried here, verified at runtime for the first time: 11
`depends_on` declarations converted from bare lists to explicit
conditions across `minimal` and `sovereign`, and `sovereign`'s redis
given the healthcheck the other two profiles already had.

**NOT proven — and one item is worse than at `097c91d`, not better,
because it is now known.** `MEMU_ALLOW_FAKE_EMBEDDINGS=false` is the
documented production default, and in these profiles it makes memu-core
**raise at import and die**. No profile can run real embeddings: the
service has nowhere to fetch the model from. CI overrides the flag to
`true`, which is why nothing has ever executed the default. Green here
still means *boots and answers smoke probes with hash-based fake
embeddings*.

Unchanged from `097c91d`: 26 of 49 services have still never started.

**Rollback:** safe to return to, and strictly better than `097c91d` for
running CI.

---

## `097c91d514781ae110d3346a5395c83fd8da6b49` — all four profiles green

**Date:** 2026-08-07 · **Branch at the time:**
`claude/project-rework-plan-pgvp35` · **Merged to `main`:** yes, this
is `main`.

The first state in the project's history where `core-tests.yml` passes
end to end.

**Proven — verified twice, on two different commits:**

| run | commit | result |
|---|---|---|
| 702 | `17c321a` | 67 of 67 steps, `conclusion: success` |
| 704 | `097c91d` | 67 of 67 steps, `conclusion: success` |

| profile | what ran |
|---|---|
| minimal | bring-up, live smoke (12 services healthy, 4 endpoints exercised), kill-isolation, restart-persistence |
| memu-graph | Cognee/Kuzu ingest → query → forget |
| full | 21 services by default, bring-up + live smoke |
| sovereign | boots in 11 seconds |

Also green on this commit: `python-app` (lint, coverage floors, repo
suite floor, cross-file isolation), `unified-hunter` (54 suites, 2,852
assertions), `make policy-check` (26 gates), docs current.

**NOT proven.** Green here means *boots and answers smoke probes* in an
ephemeral runner, with `MEMU_ALLOW_FAKE_EMBEDDINGS=true` and throwaway
secrets. It does not mean:

* sustained operation under real embeddings or real credentials
* any load, soak or failure-injection beyond kill-isolation
* **26 of 49 services** — all behind `profiles:` gates, including
  executor, verifier, supervisor and fusion-engine — which have still
  never been started by anything

Every defect fixed on the way to this commit lived in code that had
never executed. Not one was code that used to work and broke. That makes
those 26 services the highest-probability location of the next
findings, not a comfortable remainder.

**Rollback:** safe to return to. `main` is here.

---

## `a0298c6` — previous `main`

The state `main` sat at before the 2026-08-07 merge. Recorded only so
the merge has a visible before-and-after; **not** a good rollback
target — it predates every fix in the list above, including the
sovereign database-name error, pyyaml missing from 35 images, and ten
sovereign healthchecks that could never pass.

---

## Rules for this file

1. **A waypoint needs evidence, not a feeling.** Name the CI run and the
   step count. "It looked fine" is not a waypoint.
2. **Record what is NOT proven**, in the same entry. A rollback point
   that overstates itself is worse than none — somebody will return to
   it expecting guarantees it never had.
3. **Update this file before creating the tag**, not after. The tag can
   fail to push; a committed file cannot go missing without the commit
   going missing.
