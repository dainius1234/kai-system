# Embedding backends — measured state

**#47 MEASUREMENT: COMPLETE.** Closed 2026-08-12 by operator decision.
**Last run:** 31570714150, commit `b747388`, tree `2c344ac`, 2026-08-12.
Runs 1–3: 31528178414 / `da566df`, 31531007344 / `757d955`,
31568526480 / `189500b`.

> **RETRACTION, run 3.** The row `fusion-engine claim-A: REAL` that run 3
> printed is **withdrawn**. It measured **memu-core's image**. See
> §"Run 3" below.

---

## THE TWO DENOMINATORS

The single most important correction in this work. They were being mixed,
and mixing them is what nearly spent a fifth CI run on a dormant artefact.

### Production runtime denominator — **2 of 2 MEASURED**

Services that a **repo-defined execution path actually starts**, and
whose embedding backend therefore matters in production.

| service | started by | verdict | evidence |
|---|---|---|---|
| `memu-core` | minimal, full, sovereign bring-ups | **REAL** | Claim A run 4, immutable image `sha256:ee94f1cb…`, 3/3 probe stages, `--network none`; **and** Claim B runs 1/3/4 through the normal application path under the production default, width 384 |
| `agentic` | minimal, full bring-ups | **BACKEND_UNAVAILABLE** | Claim A run 4, image `sha256:47d6a1b6…`, probe exit 5 = NO_OBSERVATION, executed inside agentic's own image with `--network none` |

Both rows carry `measurement=COMPLETE`. agentic's negative result is a
**successful measurement of an absent capability**, not a failed
measurement.

### Source-reachability denominator — **3 modules**

Every non-test module importing `sentence_transformers`, which is what
`report_embedding_backends.py` derives. Correct for *"which modules could
degrade"*; it is **not** the set of *"which running services do
degrade"*, and nothing had ever said so. The third module is
`fusion-engine/app.py`.

### fusion-engine — excluded from the runtime denominator

```
runtime classification        BUILD-ONLY / NOT CURRENTLY REPO-RUNTIME-REACHABLE
semantic-helper verdict       NOT APPLICABLE TO THE CURRENT RUNTIME DENOMINATOR
future runtime status         contingent on task #41
```

Measured topology, 2026-08-12:

* declared in `docker-compose.full.yml` **only**, under
  `profiles: ["recovery"]`; absent from `minimal.yml` and
  `sovereign.yml`;
* **no repo-defined path enables `recovery`** — zero occurrences of
  `COMPOSE_PROFILES=` or `--profile` in the Makefile, any workflow or any
  script. `core-tests.yml` runs `up -d` on `full.yml` with no profile, so
  fusion-engine is excluded by definition;
* `scripts/test_restart_persistence.py:178` states it outright:
  *"`profiles: ["recovery"]` — deliberately not in this bring-up"*;
* the **only** build of `kai-system-fusion-engine` in the repository is
  the #47 evidence job itself, which this work introduced.

**Its capability is also the wrong shape for this probe.** The
`sentence_transformers` call lives in `_semantic_agreement()` — a plain
helper, not an endpoint — wrapped in
`except ImportError: return _jaccard_agreement(texts)` with a second
`except Exception:` around the model load. Both paths return an
**agreement float**. There is no vector to measure, so REAL / FAKE /
384-vs-8 is not the right instrument for it. If #41 proves fusion-engine
is intended to run, it re-enters the runtime denominator and gets a
**purpose-built semantic-agreement probe**, not the embedding-vector one.

### Moved to task #41

**Four components expect `http://fusion-engine:8053` while nothing
starts that service** — the dashboard health map, the supervisor service
list, the metrics-gateway scrape list, and a prometheus job. Whether that
is missing runtime wiring, a dormant/superseded service, stale
health/metrics configuration, or another topology defect is **not decided
here**; it is re-derived from #41 evidence.

### Correction to this document's earlier wording

An earlier revision said *"'behind a profile' is disproven."* That was
too broad, and it is corrected rather than deleted. fusion-engine **is**
behind a profile — `profiles: ["recovery"]`. What run 3 disproved was
narrower: the profile was not what blocked the **resolver**; with
`COMPOSE_PROFILES='*'` it resolves fine. The topology claim was right all
along; the phrasing overreached the measurement.

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
| fusion-engine runtime backend | **NOT APPLICABLE** | build-only; no repo-defined path starts it. See §"The two denominators" |

`agentic` moved from UNKNOWN to **BACKEND_UNAVAILABLE** in run 4 — its
image was resolved from the container Compose created and the probe ran
inside it.

The profile was not what blocked the **resolver** — with
`COMPOSE_PROFILES='*'` run 3 showed `config --services` listing
`fusion-engine` and `config --images` naming `kai-system-fusion-engine`.
That is a statement about resolution only; see §"Correction to this
document's earlier wording" above.

### The declaration defect, now settled for agentic

Static measurement said the library was absent from `agentic`'s and
`fusion-engine`'s requirements — their Dockerfiles install only that file
from a bare base, and full transitive resolution (46 and 18 packages)
contains no `sentence-transformers`, `torch` or `transformers`.

That proved the **declaration** was missing and said nothing about the
built container. **Run 4 closed the gap for agentic through an
independent channel:** the probe ran inside agentic's own image, offline,
and reached no semantic backend. Two different methods, same answer —
which is the point of measuring rather than arguing.

For `fusion-engine` the static finding stands and is **untested at
runtime by design**: nothing starts it, so there is no runtime to test.

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
