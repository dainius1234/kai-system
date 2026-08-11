# Embedding backends — measured state

**Last run:** 31528178414, commit `da566df`, tree `acb60b4`, 2026-08-11.

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

## What is not proven

| row | status | why |
|---|---|---|
| memu-core standalone Claim-A probe | **NOT EXECUTED / UNKNOWN** | image unresolved in run 1 |
| agentic runtime backend | **UNKNOWN** | nothing executed inside its image |
| fusion-engine runtime backend | **UNKNOWN** | nothing executed inside its image |

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

The step now **fails when fewer than three probes produce a verdict**.
Probe execution, not loop iteration, creates a Claim-A verdict.

## The rules this job holds to

* Claim A cannot satisfy Claim B.
* B may entail A where B exercises the same image capability.
* Probe execution, not loop iteration, creates a Claim-A verdict.
* Missing observation = UNKNOWN, never PASS.
* Infrastructure failure is not embedding failure.
* Workflow status is not evidence status.
* Vector width decides; logs corroborate.
