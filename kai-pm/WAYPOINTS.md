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
