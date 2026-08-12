# Embedding backends — measured state

**Last run:** 31531007344, commit `757d955`, tree `0d5fe1a`, 2026-08-11.
Run 1 was 31528178414, commit `da566df`, tree `acb60b4`.

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
| agentic runtime backend | **UNKNOWN** | image never resolved in run 2; the failing stage was not recorded |
| fusion-engine runtime backend | **UNKNOWN** | `config --images` exposed no image name for it; **the cause is not measured** |

For fusion-engine, "behind a profile" is a **hypothesis and nothing
more**. It built successfully in the same run, so the failure is
somewhere in the resolution chain after the build. Run 3's collector
records each stage's command, stdout, stderr and exit status so the next
entry here can name the cause from evidence.

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

### What run 3 must produce

Four rows from one commit: Claim A for memu-core, agentic and
fusion-engine, and Claim B for memu-core. Then the completeness check,
last. Nothing about remediation until the runtime denominator is complete
for all three services.

## The rules this job holds to

* Claim A cannot satisfy Claim B.
* B may entail A where B exercises the same image capability.
* Probe execution, not loop iteration, creates a Claim-A verdict.
* Missing observation = UNKNOWN, never PASS.
* Infrastructure failure is not embedding failure.
* Workflow status is not evidence status.
* Vector width decides; logs corroborate.
