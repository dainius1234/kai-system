# #41 runtime-topology census — measured 2026-08-12

> **#41-B CALLER-LOGIC MEASUREMENT = COMPLETE.** 41/41 edges: 31
> BOUNDED_DEGRADATION, 10 SILENT_FALLBACK (all in `dashboard/app.py`), 0
> MISLEADING_HEALTHY, 0 BLOCKED, 0 RETRY_STORM, 0 CRASH, 0 UNKNOWN.
>
> **DEPLOYED graceful-degradation behaviour = PARTIALLY UNKNOWN.** Kept
> as a separate row on purpose. Caller-logic evidence is not deployment
> proof, and no reading of this document may merge the two.
>
> `#53` / `KAI-GATE-047` open and deliberately unremediated.

Read-only. Nothing was changed, started, or wired. Every number here was
derived from this tree today; **no count was carried forward**, including
task #41's own previous figure of 26.

---

## 0. The headline, and it is not what the task said

```
services the repository DEFINES                      61
services a repo-defined path STARTS                  27
NEVER-STARTED                                        34   (task #41 said 26)
  ...of which EXPECTED by a component that IS live   25
```

**The dominant cause is one thing, not thirty-four.** Ten profiles are
declared, gating 32 service-definitions, and:

```
profiles ENABLED by any repo-defined invocation:  NONE
```

Zero `COMPOSE_PROFILES=` and zero `--profile` in the Makefile, in any
workflow, or in any script.

**Precise claim, and the imprecise one it replaces.** Do *not* say "the
profile mechanism has never been exercised" — that is broader than the
evidence and `memu-graph` is the counter-example: it is gated
(`introspection`) and `core-tests.yml` names it explicitly, and Compose
enables a service's own profiles when the service is targeted by name, so
it starts. What is source-confirmed is narrower:

> **No repo-defined path has exercised the intended explicit profile
> activation mechanism as a profile set.**

Individual gated services can be and are started by naming them. What has
never happened is a profile being *selected*, as a set, through
`--profile` or `COMPOSE_PROFILES`.

## 1. And the gating is deliberate — this is the finding that reframes #41

`scripts/security/check_default_profiles.py` is a **P0 security gate**
(P0-PR-03):

> Services with consequential capabilities must be behind an explicit
> profile. Only the contained core may start with a bare
> `docker compose up`.

Its `DANGEROUS_SERVICES` set names 29 services — including
`fusion-engine`, `supervisor`, `verifier`, `executor`, `broker-bridge`,
`browser-agent`, `vault-sync`, and the whole sensor family.

**So "boot the 26 never-started services" cannot mean "move them into the
default profile."** That would fail a P0 gate by construction. The task's
title encodes an approach the security model forbids, and re-deriving the
denominator is what surfaced it.

The defect is therefore **not** that these services are gated. It is:

1. **no profile has ever been activated as a profile set**, so the
   profile mechanism itself and 31 service runtimes carry no
   profile-level runtime evidence; and
2. **live components call gated services unconditionally** — whether they
   degrade gracefully when those endpoints are absent is **unmeasured**.

(2) is plausibly the more important of the two and is a different defect
class from (1).

## 2. Method, and its calibration

Derived, not listed:

* **defined** — every `services:` key across all three compose files.
* **started** — every `docker compose … up` in the Makefile, the
  workflows and `scripts/**/*.sh`, parsed and resolved. Seven
  invocations found; comment lines excluded. Compose semantics encoded
  deliberately: a bare `up -d` starts every **default-profile** service;
  `up -d <name>` **does** start a profiled service, because Compose
  enables a service's own profiles when it is named explicitly; and
  `depends_on` closure is followed.
* **expected** — every `http://<service>:<port>` and prometheus target in
  any non-`__pycache__` `.py` plus `prometheus.yml`, attributed to the
  service that owns the file, and self-references dropped.

**Calibrated before use, against answers derived independently:**

| case | expected | got |
|---|---|---|
| `memu-core` | ACTIVE | ACTIVE ✓ |
| `agentic` | ACTIVE | ACTIVE ✓ |
| `fusion-engine` | PROFILE-GATED / NEVER-STARTED | ✓ |

and all seven parsed invocations match the seven found by hand.

**One false positive caught and discarded before it was reported.** A
first pass said `memu-graph` was on the dangerous list but not gated. It
*is* gated (`introspection`, `full.yml`); it appeared ACTIVE only because
`core-tests.yml` names it explicitly. The comparison had conflated
"gated" with "never started". Corrected before publication.

| service | live expecters | startup definition | classification |
|---|---|---|---|
| `verifier` | 5 | full:recovery; minimal:recovery | PROFILE-GATED / NEVER-STARTED |
| `broker-bridge` | 2 | minimal:finance | PROFILE-GATED / NEVER-STARTED |
| `clipboard-service` | 2 | minimal:sensors | PROFILE-GATED / NEVER-STARTED |
| `docker-watcher` | 2 | minimal:watchers | PROFILE-GATED / NEVER-STARTED |
| `email-reader` | 2 | minimal:external-egress | PROFILE-GATED / NEVER-STARTED |
| `financial-awareness` | 2 | full:finance | PROFILE-GATED / NEVER-STARTED |
| `fusion-engine` | 2 | full:recovery | PROFILE-GATED / NEVER-STARTED |
| `git-watcher` | 2 | minimal:watchers | PROFILE-GATED / NEVER-STARTED |
| `news-feed` | 2 | minimal:external-egress | PROFILE-GATED / NEVER-STARTED |
| `screen-watcher` | 2 | minimal:sensors | PROFILE-GATED / NEVER-STARTED |
| `supervisor` | 2 | full:recovery; minimal:recovery | PROFILE-GATED / NEVER-STARTED |
| `sysmetrics` | 2 | minimal:watchers | PROFILE-GATED / NEVER-STARTED |
| `wake-service` | 2 | full:sensors; minimal:sensors | PROFILE-GATED / NEVER-STARTED |
| `agentic-introspect` | 1 | full:introspection; sovereign:introspection | PROFILE-GATED / NEVER-STARTED |
| `audio-service` | 1 | full:sensors; minimal:sensors; sovereign:sensors | PROFILE-GATED / NEVER-STARTED |
| `browser-agent` | 1 | minimal:external-egress | PROFILE-GATED / NEVER-STARTED |
| `cortex` | 1 | minimal:introspection | PROFILE-GATED / NEVER-STARTED |
| `executor` | 1 | full:execution; sovereign:execution | PROFILE-GATED / NEVER-STARTED |
| `files-service` | 1 | minimal:sensors | PROFILE-GATED / NEVER-STARTED |
| `letta-agent` | 1 | full:introspection | PROFILE-GATED / NEVER-STARTED |
| `monitor-service` | 1 | minimal:watchers | PROFILE-GATED / NEVER-STARTED |
| `perception-telegram` | 1 | sovereign:external-egress | PROFILE-GATED / NEVER-STARTED |
| `screen-capture` | 1 | full:sensors | PROFILE-GATED / NEVER-STARTED |
| `vault-sync` | 1 | minimal:vault | PROFILE-GATED / NEVER-STARTED |
| `vision-service` | 1 | minimal:sensors | PROFILE-GATED / NEVER-STARTED |
| `alertmanager` | 0 | sovereign:(default) | NEVER-STARTED |
| `camera-service` | 0 | full:sensors; sovereign:sensors | PROFILE-GATED / NEVER-STARTED |
| `grafana` | 0 | sovereign:(default) | NEVER-STARTED |
| `parakeet-server` | 0 | full:parakeet | PROFILE-GATED / NEVER-STARTED |
| `prometheus` | 0 | sovereign:(default) | NEVER-STARTED |
| `tailscale` | 0 | sovereign:external-egress | PROFILE-GATED / NEVER-STARTED |
| `telegram-bot` | 0 | full:external-egress | PROFILE-GATED / NEVER-STARTED |
| `vault` | 0 | sovereign:dev | PROFILE-GATED / NEVER-STARTED |
| `vault-rotator` | 0 | sovereign:dev | PROFILE-GATED / NEVER-STARTED |

## 3. fusion-engine — the first topology inconsistency, classified

| field | value |
|---|---|
| expected by | `dashboard/app.py`, `metrics-gateway/app.py`, `supervisor/app.py`, `prometheus.yml` |
| of which **live** | **2** — dashboard and metrics-gateway are both ACTIVE |
| of which not live | `supervisor` (itself PROFILE-GATED / never started), `prometheus` (NEVER-STARTED) |
| startup definition | `full.yml`, `profiles: ["recovery"]`; coherent, and the image builds |
| ever runtime-proven | **no** — nothing has ever started it |
| actual blocker | **the `recovery` profile is never enabled by any repo-defined path** |

**Classification: INTENTIONALLY PROFILE-GATED, NEVER EXERCISED — with
live callers.**

Ruling out the alternatives on evidence rather than by preference:

* **not missing wiring** — the compose definition, build context,
  healthcheck and entrypoint are all present and coherent;
* **not stale callers only** — two of the four expecters are live
  services;
* **not a profile/config mistake** — `check_default_profiles.py`
  *requires* fusion-engine to be gated; ungating it fails a P0 gate;
* **supersession: not established either way.** No replacement was
  searched for. Recorded as unknown rather than assumed absent.

Half of its expecters are themselves never started — `supervisor` and
`prometheus`. So part of what looked like "live components expect a dead
service" is really **a dead component expecting a dead service**, which
is a different and cheaper problem.

## 4. A scope finding, recorded not fixed

`DANGEROUS_SERVICES` is a **hand-written tuple of names beside the
thing** — R5's canonical shape. Measured against the tree:

```
hand-written list                                29
gated in EVERY file that defines them            32
on the list but default-profile somewhere         0     (no false negatives today)
gated everywhere but ABSENT from the list         3     parakeet-server, vault, vault-rotator
```

No live violation: nothing dangerous is currently in a default profile.
But the gate cannot notice if those three are ever ungated, because it is
not looking at them. Whether they *should* be considered consequential is
a judgement — `vault` and `vault-rotator` hold secrets, `parakeet-server`
is speech — so this is reported, not decided.

## 5. Recommended first #41 batch

Ordered by **evidenceability**, not importance.

**Work item 1 — DONE.** `scripts/security/report_runtime_topology.py`,
kind `REPORT`, proven and calibrated by
`scripts/test_runtime_topology.py` (33 assertions, 12 scenarios).
Registry: 51 declared, 51 found, I-1…I-7 hold. Every figure above is now
re-derivable with one command.

Its calibration immediately earned itself: the first parser anchored `up`
directly after `-f FILE`, so
`docker compose -f X --profile recovery up -d` **did not match at all** —
the one command form that could have disproved the headline finding was
the one form the instrument could not see. The real tree would never have
revealed it, because the finding and the blindness produce identical
output. Found by a known-positive written in the test. After the repair
the real tree reads identically, so the finding survived becoming
falsifiable.

**Work item 2 (operator-reordered: this is now THIRD) — exercise ONE
profile in CI, and observe.**
`recovery` is the natural first: 5 members, it contains `verifier`
(5 live expecters, the highest in the tree), `supervisor` and
`fusion-engine`, and it needs **no change to any default** — a
profile-enabled bring-up alongside the existing ones. This converts
"31 services with zero runtime evidence" into a measured number, and it
is the smallest instrument that can.

**Work item 2 (operator-reordered: this is now SECOND, and first in
priority) — measure the graceful-degradation question, defect class B.**

Profiles-off is the *intended* default security state, so this is a
question about the system as it is meant to run, not about an optional
extra. For each active caller of a gated dependency, establish: request
behaviour, timeout/retry behaviour, caller health, capability status,
fallback, and externally visible indication. The specific failure shapes
to distinguish are bounded explicit degradation, crash/restart,
blocking/hang, retry or log storm, misleading healthy status, and silent
unsafe fallback.

**"The process stayed up" is not proof of graceful degradation.** The
unavailable capability must not masquerade as available. Profiles are
**not** to be enabled to make these measurements pass — the point is to
test the intended profiles-off core.

*(original wording of this item follows)*
`dashboard`, `agentic` and `metrics-gateway` are live and call gated
services unconditionally. What they do when those endpoints are absent is
**unmeasured today**, and by the security model that absence is the
*normal* state. This may matter more than any activation.

**Explicitly NOT recommended:** booting all 34, or moving anything into a
default profile. The first fails a P0 gate; the second is the approach
task #41's title assumes and the security model forbids.

## 6. What this census does not establish

* whether any never-started service is **superseded**. Not searched for.
* whether the profile omission is a deliberate operational posture or an
  accident of CI never having needed one. The *gating* is deliberate and
  documented; the *never-enabling* is not documented anywhere found.
* whether `core-tests`' bare `docker compose build` covers profiled
  services. Does not affect any classification above, since building is
  not starting.
* anything about runtime behaviour. Nothing was started. Every row is a
  statement about **definitions and invocations in the tree**, which is
  precisely the distinction #47 spent four CI runs learning to keep.
