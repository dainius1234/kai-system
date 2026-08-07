# UH-2 intake — architecture redesign

2026-08-07. **No implementation. No trust-boundary change.** For
adversarial review before authorisation.

Evidence base: `scripts/security/report_perception_intake.py` (44
source-profile pairs, re-derived on every run) and the measurements below.

---

## The finding that shapes everything

**Cortex is already the correctly-placed acquisition owner, and nobody
connected it to UH-2.**

| owner | networks | reaches |
|---|---|---|
| `agentic` (current UH-2 poller) | agent-net, control-net | **4 of 9** |
| `cortex` | agent-net, **observability-net** | **7 of 9** |

Cortex polls the same sources — weather, airquality, calendar, docker,
sysmetrics, git, screen, clipboard, house-doctor — with its own
hand-written URL constants (`cortex/app.py:71-79`), into its own state
document. Two acquisition paths for one set of sources, and **the one
with the wrong network membership is the one wired into UH-2.**

Only `sensor-net` (clipboard, screen-watcher, and the 8 sensor services)
is unreachable by both. That is the single place a new component is
genuinely required.

**Second finding — reducer coverage.** 4 of 11 event types fall through
to `reduce_generic`: `clipboard_update`, `email_check`, `news_update`,
`telegram_message`. A generic fallback is not semantic coverage; it is
the reducer equivalent of `except: pass`.

---

## 1. Final authoritative source population model

Three classes, so the future gate cannot demand that history be WORKING:

* **ACTIVE / REQUIRED** — an intended perception source. Must satisfy the
  full chain. The gate's denominator.
* **INTENTIONALLY EXCLUDED** — declared, with a reason, in the tree.
  First members: `market` (in-process via `paper_trade_slice`, not
  intake); `parakeet-server` (a model server, not an observer).
* **SUPERSEDED / RETIRED** — a legacy path deliberately removed.

The classification must be **declared in the tree and read by the
instrument**, never a list beside it. Proposed: a `kai.perception.role`
label on the compose service, so the population is derived from the same
file that defines the service.

## 2. Trust-zone ownership model

| zone | owner | why |
|---|---|---|
| agent-net + observability-net | **`cortex`** (extended) | already reaches 7 of 9; already polls them |
| sensor-net | **new `sensor-intake`**, low authority, sensor-net only | nothing on that plane can reach out, and nothing may reach in |
| egress readers (email, news, telegram) | their own services, emitting | they already hold the credentials and the outward connection |
| **trusted consumption** | `agentic`, networks **unchanged** | reads journals, never polls |

`agentic` stops being an acquirer entirely. That is the redesign in one
line.

## 3. Existing component lineage

| component | decision | evidence |
|---|---|---|
| `PerceptionEvent` contract | **REUSE** | canonical, complete |
| `PerceptionIngress` validation | **REUSE** | bounds, dedup, staleness, verdicts |
| provenance model | **REUSE** | principal/purpose/provenance present |
| verdict semantics | **REUSE** | ACCEPTED/STALE/DUPLICATE/REJECTED |
| `EventJournal` | **EXTEND** | works; needs tamper-evidence (§15) |
| UH-3 reducers | **EXTEND** | 7 dedicated, 4 generic fallbacks to fill |
| 11 adapters | **REUSE** | contract-correct; the wiring around them was not |
| `cortex_source.py` | **MIGRATE, and invert** | see below |
| `cortex/app.py` polling | **EXTEND** | right networks, right sources, wrong destination |
| `ShadowPerceptionRunner` polling | **RETIRE** | 2 of 44; cross-zone by construction |
| `SENSOR_ENDPOINTS` table | **RETIRE** | hand-maintained, drifted (RC-3) |
| direct sensor→memu POSTs | **RETIRE** | D168; never functioned |

**`cortex_source.py` inverts.** Today it converts UH-2 world state *into*
Cortex's shape — UH-2 → Cortex, defaulted off. The redesign reverses the
flow: Cortex acquires and **emits PerceptionEvents into UH-2**. The
switch mechanism and its tested fallback are reused; the direction of
travel is what changes.

## 4. Reuse vs rebuild

**Rebuilt:** acquisition ownership, physical topology, journal handoff,
runtime wiring, the endpoint table.
**Reused unchanged:** every cognitive contract above.
**Newly built:** exactly one service — `sensor-intake`.

One new container for a 44-source rebuild is the measure of how much of
Gen2 was already right.

## 5. Zone-local intake topology

    sensor-net     8 sensor services -> sensor-intake  -> sensor journal
    observability  watchers          -> cortex         -> cortex journal
    agent/egress   weather, calendar, air, news, email, telegram
                                     -> cortex / self  -> same journal
                                                              |
    agentic (networks UNCHANGED)  <----- read-only ------------+
      -> revalidate -> UH-3 -> evidence -> claims

No process joins a zone it does not already belong to. **Authority
strictly decreases**: `agentic` loses its implicit expectation of
reaching every zone.

## 6. Journal ownership and handoff

One writer per journal, and the writer is the zone's owner. Sensors never
mount a journal. `agentic` mounts every journal **read-only**. Journals
carry perception records only — no secrets, policy, commands,
capabilities, memory or control-plane state, asserted by content.

The volume is the boundary and is reviewed as one. The residual risks are
stated, not waved away: the writer can rewrite, delete and flood its own
journal. §15 is the mitigation.

## 7. Authentication and provenance

Within a zone, per-service secrets (file-delivered) prove identity to the
zone's intake. The intake **overwrites** `source_type` and
`provenance.source` from the authenticated identity; the submitted values
are never read. An unknown sender is **rejected**, never defaulted.

Cortex is a special case worth naming: it *pulls*, so the provenance it
stamps is derived from which endpoint it chose to call, exactly as the
current runner does. That is sound for a puller and unsound for a
receiver — which is why `sensor-intake` needs authentication and Cortex
does not.

## 8. Adapter strategy for the 31 MISSING

**Not one by one.** The class problem is that adapters exist for pulled
services and not for push-only sensors. The strategy:

1. Sensors that already expose a status endpoint → Cortex-style pull by
   `sensor-intake`, reusing the existing adapter shape.
2. Sensors that only emit on an event (audio, wake, camera) → push to
   `sensor-intake`, which is why it needs authentication.
3. Each new adapter ships **with its reducer**, or not at all (§9).

**Audio and OCR remain frozen** and are not in this plan.

## 9. Reducer coverage strategy

A source is not covered by `reduce_generic`. Four existing gaps —
clipboard, email, news, telegram — are filled before any new source is
added, because they are the cheapest possible proof that the
adapter-with-reducer rule is real.

Rule: **no adapter may be registered without a dedicated reducer**,
enforced by extending the intake instrument to fail on generic-fallback
coverage once the four are closed.

## 10. Removing duplicate hand-maintained tables

Three exist: `SENSOR_ENDPOINTS` (shadow), the URL constants in
`cortex/app.py:71-79`, and `_service_for()` in the intake report. All
three are the same defect. Target: **one declaration**, in compose, read
by everything else — the same remedy as every other list-beside-the-thing
this fortnight.

## 11. Runtime migration sequence

1. **Phase 0** — fill the 4 generic reducers. No topology change.
2. **Phase 1** — Cortex emits PerceptionEvents alongside its state
   document. Journal written; `agentic` consumes read-only. Legacy
   polling still present but **not authoritative**.
3. **Phase 2** — retire `ShadowPerceptionRunner` polling and
   `SENSOR_ENDPOINTS`. `agentic` becomes consumer-only.
4. **Phase 3** — `sensor-intake` on sensor-net; sensor sources qualify.
5. **Phase 4** — retire the direct sensor→memu POSTs; unfreeze audio/OCR
   only if the instrument shows the architecture working.

## 12. One authoritative path

At every phase, exactly one writer owns the effect. Phase 1 runs Cortex's
state document and the journal side by side — but the state document
remains authoritative until Phase 2 switches it, and there is **no
dual-write comparison**. Note that the sensor→memu path has never
worked, so for sensors the migration is *nothing → new*, not
*old → shadow → new*.

## 13. Failure and degradation semantics

Every rejection is a **named verdict with a counter**: accepted,
rejected-invalid, rejected-identity, rejected-event-type, duplicate,
stale, rate-limited, unavailable. Journal-write failure is `503` to the
sender, never a swallowed warning. Reducer failure leaves the event in
the journal for replay. **No `logger.debug` and continue** — that is
`shadow.py:106`, and it is why this was invisible.

## 14. Observability and lineage

The chain must be answerable per source: produced → reached intake →
journalled → consumed → became evidence → rejected, with counters at each
hop. The intake instrument becomes the control surface and is run in CI.

## 15. Storage, tamper and replay

`EventJournal` is **not** an append-only security primitive — it has
`erase_subject()` and `truncate()`, and skips unparseable lines in three
places. Required: per-record `prev_digest` hash chaining with a single
trusted `(offset, digest)` anchor per journal held outside the volume;
`TAMPER_SUSPECT` on any chain break; torn-final-line distinguished from
mid-stream corruption; per-source rate limits; a volume quota with a
defined hard-full behaviour that surfaces rather than drops.

## 16. Negative security tests

Sensor cannot reach memu-core · sensor cannot mount a journal · sensor
cannot reach another zone · `sensor-intake` cannot reach any network but
its own · Cortex cannot write another zone's journal · `agentic` cannot
write any journal · ingress cannot rewrite history undetected · a forged
provenance field is overwritten, not honoured. Each must **fail if the
property is lost**, and run after the path works.

## 17. Full 44-source qualification

The instrument prints a verdict per source-profile pair. Completion is
every **ACTIVE/REQUIRED** source WORKING, with EXCLUDED and SUPERSEDED
declared in the tree and counted separately. No source counts as WORKING
because a URL, an adapter, a compose variable or an in-process test event
exists.

## 18. Gate promotion criteria

Promote from REPORT to GATE when: the population classes are declared in
the tree; every ACTIVE source is WORKING; no adapter relies on
`reduce_generic`; and the instrument has been shown to **fail** on a
deliberately broken source. Gate condition is `WORKING == ACTIVE`, not
`WORKING == total`.

## 19. Rollback

Phases 0-1 are additive; removal is deletion. Phase 2 (retiring the
poller) is revertible while `SENSOR_ENDPOINTS` remains in git history —
and it currently delivers 2 of 44, so the rollback target is weak by
measurement. Phase 4 is a single revertible commit.

## 20. Legacy retired at completion

`ShadowPerceptionRunner`'s polling loop · `SENSOR_ENDPOINTS` · the URL
constants in `cortex/app.py` · the three direct sensor→memu POSTs ·
`reduce_generic` as de facto coverage.

---

## What I am asking

1. Confirm **Cortex as the observability/agent-zone acquisition owner**
   rather than a new service. This is the load-bearing decision.
2. Confirm `agentic` becomes **consumer-only**.
3. Confirm **Phase 0 first** — four reducers, no topology change.
4. Ruling on the `kai.perception.role` label as the population
   declaration.
