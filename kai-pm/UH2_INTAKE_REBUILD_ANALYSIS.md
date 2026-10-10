# UH-2 intake — evidence, root cause, and proposed replacement

2026-08-07. **No code written. No trust-boundary change made.**
Deliverable for review before authorisation.

---

## 1. Full intake denominator

The denominator is `ADAPTER_REGISTRY` — 11 sources, derived from the
registry rather than from the 7-entry polling table, because the polling
table is itself one of the things that has drifted.

**A correction I made mid-analysis, before it reached this document:** my
first pass grepped for `adapt_weather(` etc. and reported *"10 of 11
adapters have zero callers"*. That was **false**. Adapters are dispatched
through `ADAPTER_REGISTRY[sensor]`, so a name-grep cannot see the call.
The same indirection hid `supervisor`'s `HEARTBEAT_URL` earlier today.
The denominator below is built from the registry object at runtime.

### `docker-compose.minimal.yml` — owner `agentic` on agent-net, control-net

| source | adapter | polled | service defined | reachable | verdict |
|---|---|---|---|---|---|
| calendar | yes | yes | yes | yes | **FUNCTIONAL** |
| weather | yes | yes | yes | yes | **FUNCTIONAL** |
| clipboard | yes | yes | yes | **no** (sensor-net) | UNREACHABLE |
| screen | yes | yes | yes | **no** (sensor-net) | UNREACHABLE |
| docker | yes | yes | yes | **no** (observability-net) | UNREACHABLE |
| git | yes | yes | yes | **no** (observability-net) | UNREACHABLE |
| system | yes | yes | yes | **no** (observability-net) | UNREACHABLE |
| email | yes | **no** | yes | no | NO POLL ENTRY |
| news | yes | **no** | yes | no | NO POLL ENTRY |
| telegram | yes | **no** | **no** | – | SOURCE ABSENT |
| market | yes | **no** | n/a | n/a | NOT A SERVICE |

**2 of 11 functional (18%).**

### `docker-compose.full.yml` — same owner, same networks

| verdict | count |
|---|---|
| SOURCE ABSENT (service not in this file) | 9 |
| NO POLL ENTRY (`telegram` — defined **and reachable**, never polled) | 1 |
| NOT A SERVICE (`market`) | 1 |
| **FUNCTIONAL** | **0** |

**0 of 11 functional.** And `telegram` is the mirror-image defect: the one
source the owner *can* reach has no polling entry.

Every failure is silent. `shadow.py:106` catches, logs at **`logger.debug`**,
returns `None`; the cycle then reports `sensors_unavailable` and completes
normally.

---

## 2. Root-cause classes

| class | count (minimal) | description |
|---|---|---|
| **RC-1 zone mismatch** | 5 | one high-authority poller cannot resolve services in isolated zones. Not a bug — a category error. |
| **RC-2 profile mismatch** | 9 (full) | the owner and its sources are declared in different compose files. |
| **RC-3 registry/endpoint drift** | 3 | adapter exists, no polling entry (email, news, telegram). The two tables are maintained by hand and have diverged. |
| **RC-4 in-process source** | 1 | `market` is adapted directly by `paper_trade_slice.py`; it is not a polled service and should not be counted as intake. |

**RC-1 is the architectural finding.** A single process on
`agent-net`+`control-net` polling across `sensor-net` and
`observability-net` cannot work under this segmentation model, and no
amount of fixing endpoints changes that. The current acquisition model is
incompatible with the security model it runs inside.

RC-3 is the familiar shape: two hand-maintained lists beside each other,
drifting. The remedy is the programme's standard one — derive one from
the other, or fail when they disagree.

---

## 3. Gen1 lineage — existing owners inspected before proposing anything

| component | classification | evidence |
|---|---|---|
| `PerceptionEvent` contract | **MIGRATED** | canonical, used by ingress, adapters, world state |
| `PerceptionIngress` validation | **MIGRATED** | bounds, dedup, staleness, verdicts all present |
| `EventJournal` | **MIGRATED**, with a caveat | works; is **not** an append-only security primitive (`erase_subject`, `truncate`, silent torn-line skip) |
| UH-3 reducers / evidence / claims | **MIGRATED** | deterministic, provenance-carrying |
| `ADAPTER_REGISTRY` (11 adapters) | **MIGRATED** | contract-correct; the wiring around them is not |
| `ShadowPerceptionRunner` | **LEGACY-LIVE / BROKEN** | 2-of-11; central polling across zones |
| `SENSOR_ENDPOINTS` table | **CONFLICT** | drifts from the registry; RC-3 |
| `cortex_source.py` | **DORMANT-INTENDED** | the *defined* cutover from polling to world-state, gated by `KAI_CORTEX_SOURCE`, **default `poll`** — i.e. the migration exists and has never been switched on |
| direct sensor → memu POSTs | **LEGACY-LIVE / BROKEN** | D168; never functioned |

`cortex_source.py` matters: the intended Gen1→Gen2 cutover was already
designed, with a tested fallback, and left at its legacy default. That is
the migration stalling one step from completion, not an absence of design.

---

## 4. Reuse vs retire

**Preserve unchanged:** the event contract, ingress validation, provenance
model, verdict semantics (FAIL/UNKNOWN/STALE/DUPLICATE), UH-3 reducers,
all 11 adapters, `cortex_source.py`'s switch design.

**Retire:** `ShadowPerceptionRunner`'s central cross-zone polling model,
and the hand-maintained `SENSOR_ENDPOINTS` table.

**Rebuild:** the acquisition layer only — who polls, from where, and how
the result reaches the trusted side.

---

## 5. Proposed replacement topology

Zone-local intake. One low-authority intake per zone; no process spans
zones; journals are handed over as data, not as network reach.

    sensor-net       sensors -> sensor-zone intake       -> sensor journal
    observability-net watchers -> observability-zone intake -> obs journal
    agent-net        weather, calendar (already local)   -> agent journal
                                                              |
    trusted consumer (agentic, unchanged networks)  <---- read-only ----+
      -> revalidate + reconcile each journal
      -> UH-3 world state / evidence / claims

**No new authority is created.** Each intake can reach only its own zone
and can only append to its own journal. The trusted consumer gains no
network.

**Alternative considered and rejected:** give `agentic` membership of
`sensor-net` and `observability-net`. That fixes 5 findings by attaching
the command plane to the two least-trusted zones — the largest possible
authority expansion, and precisely the inversion this analysis exists to
prevent.

**Open question I cannot decide from evidence:** whether the
observability-zone intake should be a new service or an added role on an
existing observability-net member. `metrics-gateway` is already on that
network and already aggregates. That is a reuse candidate worth checking
before creating a third container.

---

## 6. Authority and trust-boundary impact

| change | authority effect | boundary effect |
|---|---|---|
| sensor-zone intake | new low-authority principal, sensor-net only | none crossed by a process |
| observability-zone intake | new low-authority principal (or an existing member gains a bounded role) | none crossed by a process |
| trusted consumer reads journals | **none** — read-only, no new network | volume becomes the crossing; reviewed as such |
| retire central polling | **removes** the implicit expectation that a control-plane process reaches every zone | strictly reduces |

Net: authority decreases. That is the test this design has to pass, and
the reason the rejected alternative fails it.

---

## 7. Journal ownership model

One writer per journal, and the writer is the zone's intake. Sensors never
mount a journal. The trusted consumer mounts every journal **read-only**.
Journals carry perception records and nothing else — no secrets, policy,
commands, capabilities, memory or control-plane state.

Because `EventJournal` is **not** tamper-evident, the consumer needs
history-integrity state held outside the volume. Recommended: per-record
`prev_digest` chaining with a single trusted `(offset, digest)` anchor per
journal — O(1) trusted state, detects deletion, reordering and
substitution. Reconciliation failure yields `TAMPER_SUSPECT`, never a
silent clean replay.

---

## 8. Runtime proof plan

Per source, and a source counts as functional only if **every** step
passes: service exists → runs → correct zone → correct owner → owner
reaches it → adapter exists → identity correct → event journals → trusted
consumer receives it → UH-3 handles it → no bypass to memory.

The acceptance criterion is a printed table with a verdict per source and
**no verdict inferred**. Anything deliberately unsupported is marked
`INTENTIONALLY EXCLUDED` with a reason — `market` is the first candidate,
as an in-process source rather than intake.

A source must never count as functional because a URL, an adapter, a
compose variable or an in-process test event exists. Today all 11 would
pass that weaker bar; 2 pass the real one.

## 9. Negative security proof plan

From a sensor: cannot reach memu-core, cannot reach another zone, cannot
mount a journal. From an intake: cannot reach any network but its own,
cannot rewrite history undetected, cannot write another zone's journal.
From the trusted side: cannot write any journal. Each proven by a test
that **fails if the property is lost**, run after the new path works.

## 10. Rollback

Every stage is additive until the final one. Intakes can be removed from
compose with no loss because the paths they replace do not currently
function. The last stage — deleting the legacy direct-memory calls and
switching `KAI_CORTEX_SOURCE` — is a single revertible commit.

## 11. Preserved exactly

Event contract · ingress validation · provenance model · verdict
semantics · UH-3 reducers · all 11 adapters · `cortex_source.py` switch.

## 12. Rebuilt or replaced

`ShadowPerceptionRunner`'s cross-zone polling model · the
`SENSOR_ENDPOINTS` table · acquisition topology · journal handoff ·
runtime wiring.

---

## 13. What I am asking for

1. Confirm the zone-local intake shape before any code.
2. Ruling on reusing `metrics-gateway` versus a new observability intake.
3. Ruling on the hash-chain mechanism.
4. Confirm `market` as `INTENTIONALLY EXCLUDED`.

Audio and OCR are deliberately **not** in this document. The foundation is
2-of-11; adding sources to it first would be building on the defect.
