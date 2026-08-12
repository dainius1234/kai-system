# Embedding backends — measured state

**Last run:** 31568526480, commit `189500b`, tree `f4bd412`, 2026-08-12.
Run 2 was 31531007344 / `757d955`; run 1 was 31528178414 / `da566df`.

> **RETRACTION, run 3.** The row `fusion-engine claim-A: REAL` that run 3
> printed is **withdrawn**. It is a verdict about **memu-core's image**,
> mislabelled. See §"Run 3" below. fusion-engine's runtime backend
> remains **UNKNOWN**.

---

## What is proven

### memu-core Claim B — PROVEN, in CI, at this commit

```
MEMU_ALLOW_FAKE_EMBEDDINGS absent at the environment boundary (asserted,
  the step refuses if it is set)
  -> compose resolved ${MEMU_ALLOW_FAKE_EMBEDDINGS:-false}
  -> normal memu-core application startup
  -> "sentence-transformers loaded — model=all-MiniLM-L6-v2  dim=384
      embedding backend ready in 8.8s"
  -> the application produced a vector of width 384
```

The documented production default that had never executed anywhere now
has runtime evidence. **Scoped to this CI evidence boundary** — it is not
a claim about a production host.

### memu-core image capability — PROVEN, entailed by Claim B

We watched that image load the model and produce a real vector. Claim B
is the stronger claim and contains the weaker one. Recording the image
capability as UNKNOWN because a *separate probe* did not run would be a
scope error of the kind this programme exists to find.

### memu-core Claim A — PROVEN REAL, run 2

```
image  sha256:b5e68a39567a26e2b5cad14ebf62e136701d5a1feeb75736a8a4555d53d88056
       (the immutable id, resolved from the container Compose created)
probe  3 of 3 stages reached — library import, model load, semantic operation
       sentence-transformers 5.7.0, width 384
       run with --network none, so nothing could have been fetched
```

## What is not proven

| row | status | why |
|---|---|---|
| agentic runtime backend | **UNKNOWN** | run 3 measured the failing stages: `create --no-deps` is an invalid flag, and the name fallback resolved `ollama/ollama:0.6.8`, which is not built locally |
| fusion-engine runtime backend | **UNKNOWN** | run 3's `REAL` row is **retracted** — it probed `kai-system-memu-core`, not fusion-engine's image |

**"Behind a profile" is disproven.** Run 3 measured it: with
`COMPOSE_PROFILES='*'`, `config --services` lists `fusion-engine`
(`service_known_with_all_profiles=yes`) and `config --images` names
`kai-system-fusion-engine`. The profile was never the obstacle. Recording
this because it was carried as a hypothesis for two runs and the
temptation was to write it down as a cause.

`agentic` and `fusion-engine` remain **DECLARATION DEFECT** by static
measurement — the library is absent from their requirements, their
Dockerfiles install only that file from a bare base, and full transitive
resolution (46 and 18 packages) contains no `sentence-transformers`,
`torch` or `transformers`. That proves the declaration is missing. It
does not prove the built container lacks it, and dependency arithmetic
is not a built container.

## Run 1's instrumentation defect, and why it matters

**Workflow execution: SUCCESS. Claim-A evidence: INCOMPLETE.**

The job completed green while three intended measurements never ran. The
green tick said nothing about that, which is exactly the confusion this
work exists to prevent.

Cause, read from the command's own help text rather than inferred:

    docker compose images — "List images used by the created containers"

At Claim-A time the images were built and **no container existed**, so it
correctly returned nothing. A container-scoped listing was used to
resolve a build-scoped artefact: an instrument whose scope was narrower
than its question. Claim B ran later, created a container, and worked —
which is why B succeeded where A never started.

Repaired by resolving identity from the container **Compose itself
creates**, so the evidence is tied to the image Compose would run.
Calibrated in the run: a nonexistent service must resolve to nothing, or
"resolved" would mean nothing. And an unresolved image is now recorded
as an UNKNOWN verdict rather than skipped, because a skip is
indistinguishable from a pass in a summary.

## Run 2's instrumentation defects, and the repair for run 3

Run 2 proved memu-core and produced **no evidence at all** for the other
two, in two distinct ways. Both were the instrument, not the subject.

**One verdict suppressed an independent one.** Claim A was a single step
looping over three services, and it failed itself when its own evidence
was incomplete. GitHub then **skipped Claim B entirely** — a measurement
about image capability deleted a measurement about the service path.

> A measurement verdict may affect its own claim.
> It may never suppress an independent measurement.

Repaired by making each service its own step, each running a collector
that exits 0 for every **defined** probe verdict and non-zero only when
the instrument itself malfunctions. A probe proving the backend absent is
a *successful measurement of a failed capability*. Completeness is now
judged **once, last**, by `Evidence summary`, which is positioned after
every measurement and before an `if: always()` upload, so failing it
suppresses nothing.

**The resolver destroyed the evidence it existed to gather.** It sent
stdout, stderr and the exit status of every attempt to `/dev/null` and
returned one string; an empty string collapsed the whole chain to
`UNRESOLVED`. So run 2 cannot say whether agentic failed at container
creation, at the id lookup or at inspect. Repaired by recording every
stage's command, stdout, stderr and exit status, so an empty result is a
classified stage outcome rather than silence.

**Two axes, no longer conflated.** `measurement` (COMPLETE / INCOMPLETE /
INSTRUMENT_ERROR) answers *did we look?*; `claim_verdict` (REAL / FAKE /
WRONG_DIMENSION / NO_OBSERVATION / TIMEOUT_UNKNOWN / UNKNOWN) answers
*what was there?* Collapsing them makes "we could not look" and "we
looked and it was absent" produce the same number.

### Found while repairing the aggregator

`grep -c` **exits 1 on zero matches and still prints `0`**, so
`$(grep -c … || echo 0)` yields the two-line string `0\n0` and the
comparison after it dies with a syntax error. A completeness verdict was
one empty file away from being decided by an arithmetic accident. This is
the same mechanism that made a background wrapper report FAIL over a
green chain on 2026-08-10.

### What run 3 produced

Four rows from one commit, as required — and the structure held while
the resolver produced a **confidently wrong answer**.

```
memu-core     claim-A  measurement=INCOMPLETE  verdict=UNKNOWN  image=unresolved
agentic       claim-A  measurement=INCOMPLETE  verdict=UNKNOWN  image=unresolved
fusion-engine claim-A  measurement=COMPLETE    verdict=REAL     image=sha256:121b8200…
memu-core     claim-B  REAL (384)              dim=384, 8.4s, application path
Evidence summary: EVIDENCE SET INCOMPLETE -> exit 1
```

**What worked.** Every measurement step ran to completion; no verdict
suppressed another; the only failing step was the completeness judgement,
last, after the artefact upload had its `if: always()`. Claim B is
reproduced independently at a second commit. The per-stage evidence made
the defect below readable in a single pass — the previous collector
would have printed `UNRESOLVED` and nothing else.

**What did not.** Two measured causes.

1. `docker compose create --no-deps <service>` → **`unknown flag:
   --no-deps`**, for all three services. `create` has no such flag. That
   killed the container-scoped path — the one that worked in run 2 — and
   forced every row onto the name-based fallback.

2. `config --images <service>` **returns the service's whole dependency
   graph**, in an order that is not the service's own:

   ```
   memu-core      -> redis:7-alpine | kai-system-memu-core | pgvector…
   agentic        -> ollama/ollama:0.6.8 | kai-system-agentic | …
   fusion-engine  -> kai-system-memu-core | … | kai-system-fusion-engine
   ```

   The collector took `head -1`. For memu-core and agentic the first
   entry was not built locally, `docker image inspect` failed, and the
   rows honestly read UNKNOWN. **For fusion-engine the first entry WAS
   built** — so the probe ran against `kai-system-memu-core` and the
   collector recorded `fusion-engine claim-A: REAL`.

### The shape, named

**A confident verdict about the wrong artefact.** Worse than the UNKNOWN
it replaced, and it passed every verdict-integrity control in the job —
because those protect a verdict's **transport**, not its **subject**. The
exit code survived intact from producer to record; it was simply an
answer about a different image.

memu-core and agentic were saved only by their first-listed dependency
not being present locally. That is luck, not a control.

Repaired for run 4: the image name is read from the service's own
resolved definition (`config --format json` → `services.<name>.image`),
which is single-valued and cannot name a neighbour; the dependency
listing is kept as an independent corroborating channel; a **binding
check** refuses when a fallback name is also another service's image, as
`INSTRUMENT_ERROR` rather than as a claim; and `--no-deps` is gone.

Nothing about remediation until the runtime denominator is complete for
all three services. It currently stands at **1 of 3** — memu-core via
Claim B. fusion-engine and agentic are UNKNOWN.

## The rules this job holds to

* Claim A cannot satisfy Claim B.
* B may entail A where B exercises the same image capability.
* Probe execution, not loop iteration, creates a Claim-A verdict.
* Missing observation = UNKNOWN, never PASS.
* Infrastructure failure is not embedding failure.
* Workflow status is not evidence status.
* Vector width decides; logs corroborate.
