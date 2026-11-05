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


---

## 0A. Cortex audit — required before promotion (direction §9)

Cortex was promoted on topology. Topology is not enough, so here is the
rest of it. **Three findings change the shape of the proposal, and one
of them is a blocker.**

| check | finding | verdict |
|---|---|---|
| network membership | `agent-net`, `observability-net` | **good** — the reason it reaches 7 of 9 |
| **defined in which profiles** | **`minimal.yml` ONLY** — absent from `full` and `sovereign` | **GAP** |
| **profile gating** | `introspection` — one of the 26 services **never started by anything** | **GAP** |
| **authentication** | **NONE.** `/state` (GET) and `/observe_turn` (POST) have no `Depends`, no token check, no auth of any kind | **BLOCKER** |
| failure semantics | `record_degradation("upstream", "cortex_fetch", exc)` at `app.py:157` | **good** — materially better than `shadow.py:106`'s `logger.debug` |
| silently drops observations? | records the degradation, then returns `None`; the caller treats `None` as absent | **partial** — visible, but the loss is still indistinguishable from "nothing to report" |
| writes to memory / world model? | no writes found; it serves state, it does not persist | **good** |
| `KAI_CORTEX_SOURCE` cutover | exists, defaults to `poll`, has a tested fallback | sound mechanism, unproven at runtime |

### The blocker

Promoting Cortex makes an **unauthenticated service the primary
perception authority** on `agent-net` and `observability-net`.
`POST /observe_turn` accepts conversation turns from anyone who can
reach it. Authentication must land **before** promotion, not after — the
whole point of the redesign is that provenance is derived from
authenticated identity, and Cortex currently has no identity to derive
from.

### A third acquisition path nobody has counted

`POST /observe_turn` — *"receive each conversation turn for bridge +
tacit learning"* — is a **push** intake into Cortex. So the system has
three acquisition paths, not two:

1. `ShadowPerceptionRunner` polling (2 of 44)
2. Cortex polling (7 of 9 reachable, feeding its own state document)
3. `POST /observe_turn` push (ungoverned, unauthenticated, uncounted)

The third is not in the 44 because the instrument counts *services that
observe*, and this is a caller pushing in. It must be brought under the
same governance or explicitly excluded, and it is neither today.

### What this does to the proposal

It does **not** kill it — Cortex is still the best-placed owner, and its
failure semantics are already better than the runner's. But "reuse
Cortex" is not free reuse:

* it must gain authentication (blocker)
* it must be **defined in `full` and `sovereign`**, which is new
  deployment surface, not reuse
* it must actually be **started**, which no profile does today
* `POST /observe_turn` must be governed or excluded

The honest summary: **Cortex is the right owner and is itself
never-executed code.** Promoting it without the audit above would move
perception onto a service with the same provenance as everything else
this programme has been finding.


## 0C. RAW OBSERVATION != CORTEX INTERPRETATION

**This is a blocker-class property and it corrects a miss in my own
audit.** §0A examined Cortex's authority, networks and failure
semantics, and never asked what it *produces*. I treated an interpreter
as a pipe.

Verified in `cortex/app.py:95-101` — Cortex's own state model:

    level2_summary: str            "plain-English situation summary <= 20 words"
    level3_implication: str        "implication + recommendation <= 30 words"
    intent_fan: List[IntentHypothesis]   probabilistic intent inference
    tacit_rules: List[str]
    sensor_credibility: Dict[str, float]

None of these is an observation. They are **derived interpretations over
observations**, and three distinct epistemic kinds are involved:

| kind | example | status |
|---|---|---|
| observation | `CPU 82%` | may become evidence |
| interpretation | "system under moderate load" | an inference *over* evidence |
| recommendation | "monitor; investigate if persistent" | an action proposal |

**These must not share epistemic status.** If Cortex is connected to
UH-2 naively — as an acquisition owner emitting PerceptionEvents — its
Level-2/Level-3 output would enter through the same door as a sensor
reading and become indistinguishable from directly observed fact.

Required separation:

    raw source        -> PerceptionEvent -> evidence
    Cortex inference  -> ATTRIBUTED INTERPRETATION
                         + explicit references to the evidence it used
                         + confidence
                         + freshness
                         + NO factual-authority upgrade for having been
                           produced by Cortex

The consequence for the redesign: Cortex has **two outputs and they take
different paths**. Its pulls become PerceptionEvents. Its inferences must
not. That is a constraint on the connection design, not a footnote.

**Checklist item for adversarial review:** *if Cortex is compromised, or
simply wrong, can its Level-2/Level-3/intent/tacit inference be mistaken
downstream for directly observed fact?* If yes, **BLOCKER**.

## 0D. `/observe_turn` — verified state mutations

Confirmed in `cortex/app.py:531-547`. Unauthenticated, and it mutates
five pieces of module-global state directly:

    _topic_history.append(keywords)
    _state.bridge_active = bridge_active
    _state.bridge_note  = bridge_note
    _tacit_msg_lengths.append(len(obs.user_message))
    _tacit_hourly_counts[hour] = _tacit_hourly_counts.get(hour, 0) + 1

So it is a **third state-changing intake path outside the canonical
spine**, not merely an unauthenticated read. Governing it therefore
cannot mean adding a token to the existing endpoint — that would leave
an authenticated-but-still-parallel authority.

Target shape:

    conversation turn
      -> authenticated caller
      -> bounded typed observation, allowed event type
      -> provenance derived from the authenticated identity
      -> governed acceptance + journal
      -> ONLY THEN Cortex derives bridge/tacit interpretation from the
         accepted event

The one-authoritative-path rule applies here exactly as it does to
sensors.

## 0E. Authentication — the precision that matters

Cortex needs **no caller authentication to perform a controlled pull**;
provenance for a pull binds to the endpoint Cortex chose to call, which
is sound for a puller and is what the current runner already relies on.

What requires authentication before promotion is Cortex's **inbound
service surface**: `GET /state` and `POST /observe_turn` need
authenticated caller identity and route-specific authorisation. `/health`
may remain a health surface.

**No promotion while an arbitrary reachable caller can read Cortex state
or inject conversation turns.**

## 0B. Denominator semantics (direction §10)

"44" is **capability x profile pairs**, not 44 distinct sensors:

    unique perception capabilities .................. 18
      adapter-backed .............................. 11
      observers with no adapter .................... 7
        audio-service, camera-service, files-service,
        monitor-service, screen-capture, vision-service,
        wake-service

    profile/deployment source instances ............ 44   <- "the 44"

Both numbers are needed and they answer different questions. **18** is
how much perception the project intends. **44** is how many
source-in-a-profile paths must each be proven, and it is the right
denominator for the gate because a capability working in `minimal` and
absent from `full` is not a working capability. The instrument will
report both from now on.

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

## 2A. Cortex has two outputs, and only one of them is intake

Following from §0C, the ownership model needs a line it did not have:

| Cortex output | path | epistemic status |
|---|---|---|
| pulled source readings | PerceptionEvent -> journal -> UH-3 evidence | observation |
| Level-2 / Level-3 / intent / tacit / credibility | **attributed interpretation**, referencing the evidence it used | inference, never fact |

"Cortex is the acquisition owner" is true of the first row only.

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

Direction accepted: reuse Cortex, connect it to UH-2, build one sensor-net
intake, remove agentic as cross-zone poller, eliminate duplicate tables.
The audit in §0A adds conditions rather than objections.

1. **Cortex authentication is a blocker, not a follow-up.** Confirm it
   lands before promotion. Promoting an unauthenticated service to
   perception authority would reproduce the defect class we are removing.
2. Confirm the scope includes **defining and starting Cortex in `full`
   and `sovereign`**. It exists in `minimal` only and has never been
   started. This is the part of "reuse" that is not free.
3. Ruling on `POST /observe_turn` — govern it under the same identity
   and event-type rules, or declare it INTENTIONALLY EXCLUDED. It is a
   third acquisition path and is currently neither.
4. Confirm **Phase 0 first** — the four generic reducers, no topology
   change.
5. **Catalogue as a JOIN, not a copy.** Refined per ruling 5, and the
   refinement is better than my proposal. Compose owns only what compose
   genuinely owns — source identity, trust zone, acquisition owner,
   acquisition method, lifecycle, required/optional/superseded, profile
   presence. Reducer and event-contract semantics stay in their own
   registries. The instrument **joins** compose + `ADAPTER_REGISTRY` +
   the event contract + `REDUCER_MAP` and **fails on disagreement or a
   missing link**.

   My version would have copied code-owned facts into compose, which is
   the same defect as `SENSOR_ENDPOINTS` with a better name — one giant
   table holding copies of facts from four subsystems. The rule is **one
   authoritative owner per fact**, and the instrument is what makes them
   agree.

---

# Revision 3 — adversarial review resolution

Every item below is either a control the plan already carried (kept, not
rewritten as a discovery), an accepted finding with a decision, or a
rejection with a reason. **One new blocker was found while resolving
them** and is stated first because it is larger than anything in the
review.

## R3.0 NEW BLOCKER — there is no memory-promotion owner to name

Q15 asked me to name the exact component that owns
`evidence -> candidate memory -> approved long-term memory`. I inspected
the tree rather than describing an intention.

**It does not exist.** `common/world_state/` contains no promotion path —
no reference to memorize, memu or long-term storage anywhere. Instead
**six services write directly** to `POST /memory/memorize`:

    agentic/app.py            agentic/introspect_app.py
    skill-hunter/app.py       house-doctor/app.py
    perception/audio/app.py   screen-capture/app.py     (the last two broken, D168)

So the "controlled downstream promotion" this plan has referred to
throughout is not a component with a gap in it. It is **six independent
writers and no governed path at all**. The evidence/claim model exists
and nothing consumes it into memory.

**Consequence:** the redesign cannot end at "evidence, then controlled
promotion downstream", because downstream is six unrelated callers.
Naming the owner is not a documentation task — the owner has to be
designed, and that is a second architecture decision of the same size as
this one.

**Recommendation:** keep memory promotion explicitly **out of scope** for
the intake rebuild, and raise it as its own decision. Phase 1-4 stop at
evidence. Nothing in this plan creates a seventh writer.

## R3.1 Historical journal confidentiality (Blocker 1) — ACCEPTED, solution reopened

The risk is real and I accept it: sensor-intake writes the journal, so a
fully compromised intake gains **historical** sensor confidentiality, not
just what it received while compromised. That distinction is the property
to defend.

The proposed remedy (FIFO/NFS because Docker lacks write-only mounts) is
**one option, not the option**. Options to evaluate against the property:

| option | prevents historical plaintext recovery? | cost |
|---|---|---|
| filesystem permissions — writer uid has write, not read | only if the OS can express append-without-read on the actual deployment; must be *proven*, not assumed | low |
| per-record encryption to a trusted-side public key | **yes** — intake holds no decryption key, so ciphertext it wrote earlier stays opaque | key management; breaks the "journal is human-readable" convenience |
| unidirectional handoff (FIFO / one-way channel) | yes | loses journal-first durability unless the writer keeps its own spool, which recreates the problem |
| rotate-and-hand-off — intake writes a short-lived segment, trusted side takes ownership and the intake loses access | partial: bounded historical window rather than none | operational complexity |

**Leading candidate: per-record encryption to a trusted-side key**, because
it satisfies the property without giving up journal-first durability, the
read-only trusted consumer, or the no-dual-homing rule. Provisional, not
decided.

**Explicitly preserved regardless of which is chosen:** no sensor→trusted
network bridge; no dual-homed process; journal-first durability; bounded
failure; trusted consumer read-only.

**Stated plainly: the volume design is not defended because we designed
it.** If none of the above satisfies the property, the handoff changes.

## R3.2 Compromised intake forging sensor identity (Blocker 2) — ACCEPTED

Correct, and the current design's provenance claim is dishonest under
intake compromise. Sensor→intake authentication protects against a
compromised *sensor*; it does nothing against a compromised *intake*.

Decision is **per source class**, and the honest split is:

**Push sensors (audio, wake, camera) — family A, end-to-end authenticity.**
The sensor authenticates the *observation*, not the connection, with a
secret the intake never holds. **Per-source HMAC is the minimum adequate
mechanism** — asymmetric signing buys key-rotation and non-repudiation
properties this deployment does not currently need, and costs a key
infrastructure it does not have. The trusted verifier checks origin; the
intake becomes an untrusted relay.

**Pull sources (weather, docker, git, sysmetrics…) — family B, downgrade
the claim.** These cannot attest anything; they answer an HTTP GET. So
the trusted side must record what it actually knows:

    NOT   "sensor X asserted Y"
    BUT   "the acquiring owner reports that endpoint X returned Y"

That distinction survives into provenance and confidence. **No false
cryptographic certainty**, and no pretending a `200 OK` is an assertion.

## R3.3 `/observe_turn` migration mechanism (Blocker 3) — ALREADY COVERED as target, ACCEPTED as gap

The target shape was already in §0D. The **mechanism** was not. Now
specified:

* **the route stays**, and becomes a thin submitter — it validates,
  authenticates the caller, and forwards a typed event to the governed
  path. It stops mutating anything.
* **direct mutation is removed in the same commit** that adds the
  governed path — not before (no gap), not later (no dual authority).
* **event type:** `conversation.turn`, with a dedicated reducer. It does
  not reuse a sensor type.
* **who may submit:** authenticated callers holding the
  `conversation.turn` grant. Today that is the dashboard and agentic;
  derived from the token map, not listed here.
* **who journals it:** the governed ingress, exactly as for any source.
* **how Cortex consumes it:** Cortex derives bridge/tacit interpretation
  **from the accepted journal event**, never from the request.
* **duplicate/retry:** stable event id over (caller identity, turn
  timestamp, message digest); resubmission returns `duplicate`, which is
  a success for the sender.
* **negative test:** a caller with a valid token but no
  `conversation.turn` grant is rejected; and `_topic_history`,
  `bridge_active`, `bridge_note`, `_tacit_msg_lengths` and
  `_tacit_hourly_counts` are **unchanged** after a rejected submission.

Until this lands, `/observe_turn` remains a **blocker to Cortex
promotion**.

## R3.4 Cortex interpretations driving action (Major 4) — RECLASSIFIED

The core risk is accepted; the proposed remedy is **rejected as too
crude**. "Every Cortex interpretation must have independent corroboration
before any action" would make low-risk reasoning require two sources,
which is a tax on correctness rather than a control.

The architecture already has the right chain:

    interpretation -> deliberation/proposal -> policy -> approval -> capability -> actuator

The hard invariant, machine-enforced rather than documented:

* **Cortex never holds direct action authority.**
* `EvidenceRecord` may not be constructed from an interpretation. A
  Level-2/Level-3/intent/tacit output that reaches evidence is a
  **BLOCKER**, and a negative test asserts it.
* a recommendation may become **input to proposal generation**, never an
  executable command.
* **policy** decides where corroboration is required, by risk and domain
  — high-risk action may demand independent evidence; low-risk reasoning
  need not.

## R3.5 Journal integrity (Major 5) — ALREADY COVERED

§15 already requires `prev_digest` chaining, a trusted `(offset, digest)`
anchor outside the volume, `TAMPER_SUSPECT` on chain break, and the
torn-tail vs mid-stream distinction. Kept as written, not restated as a
finding.

**Accepted refinement:** add a monotonic per-journal sequence number.
The chain already proves the property; the sequence number makes gap and
reorder diagnosis explicit rather than inferred, which is an operability
gain rather than a security one. Recorded as a refinement, not a fix.

## R3.6 Rate / size / DoS (Major 6) — ALREADY COVERED, clarified

§5 already carries payload bounds, per-source rate limits, volume quota,
hard-full behaviour and a `rate-limited` verdict. Not duplicated.

**Clarification accepted, and it is a real one:** limits are enforced at
the **first network-facing boundary, before parsing or storage**.
Bounding after parsing means an attacker still spends our CPU. Per source:
max event bytes, request rate and burst, storage contribution. Hard-full
must **surface**, never silently drop, and a test proves it.

## R3.7 Idempotency and dedupe (Major 7) — ACCEPTED, real gap

The plan said "stable event id, duplicate is a success" and stopped. The
rest matters:

* **no "exactly once" claim.** Target is **at-least-once delivery plus
  durable deduplication**.
* **event id ownership:** the *sender* computes it, so a retry is
  recognisable; the receiver never invents one.
* **dedupe persistence:** the dedupe set must survive restart, or the
  first retry after a crash creates a duplicate. This is the gap the
  current in-memory duplicate detection has.
* **retention window:** bounded, and the bound is stated — beyond it a
  replay is accepted as new. An unbounded dedupe set is a memory leak
  wearing a correctness argument.
* **sender behaviour on UNKNOWN:** a timeout is not a failure. The sender
  retries with the same id; the receiver answers `duplicate`.
* **replay must not double-mutate:** reducers consume by event id, so a
  journal replay produces no second evidence record.

## R3.8 Observability (Major 8) — ALREADY COVERED, strengthened

§14 already requires counters at every hop. **Accepted strengthening:** a
correlation/event id carried through every stage, so a *single* event can
be traced produced → intake → journalled → consumed → evidence, rather
than only counted at each.

## R3.9 Negative tests (Major 9) — REJECTED as written

The claim that the plan lacks concrete negative tests is **false**. §16
already mandates eight, named. Kept.

**Extended** for the newly accepted issues:

* a compromised intake cannot recover historical plaintext journal content
* a compromised intake cannot impersonate an end-to-end-authenticated sensor
* `/observe_turn` cannot mutate Cortex state before governed acceptance
* an interpretation cannot enter `EvidenceRecord` as observed fact
* a recommendation cannot directly invoke an actuator

## R3.10 Cortex full/sovereign (Major 10) — ALREADY COVERED

§0A already makes runtime proof in every required profile a promotion
condition. Kept.

## R3.11 Cortex network blast radius (Q13) — ACCEPTED, and it is worse than expected

Measured rather than assumed:

| edge | peers | what a compromised Cortex gains |
|---|---|---|
| `agent-net` | **17** — agentic, dashboard, **memu-core**, ollama, supervisor, vault-sync, verifier, skill-hunter, notify-service, tts-service, redis, … | **direct reach to memu-core**, which exposes `POST /memory/memorize` |
| `observability-net` | 7 — docker-watcher, git-watcher, sysmetrics, heartbeat, monitor-service, memu-core-introspect, dashboard | the watcher sources it needs |

**Cortex does not call memu-core today (0 references), but it can.** So
promoting Cortex to perception authority promotes a service that already
sits one HTTP call from the memory store.

`observability-net` is justified — it is the whole reason Cortex reaches
7 of 9. `agent-net` is **not** justified by acquisition: of its 17 peers,
Cortex needs weather, airquality, calendar and house-doctor. The edge
should be narrowed or the four moved, and "it reaches 7/9 therefore its
networks are right" is exactly the reasoning this review exists to stop.

## R3.12 Reuse tool-gate instead of sensor-intake (Q14) — REJECTED, with the check performed

The REUSE → EXTEND → MIGRATE → CREATE check, run once as instructed.
Every current `sensor-net` member is **itself a source**:

    audio-service  clipboard-service  files-service
    screen-watcher  vision-service  wake-service

Making any of them the intake would give one sensor authority over the
others' provenance — the confused-deputy problem, inside the least
trusted zone. There is **no low-authority sensor-local candidate**.

Putting tool-gate on `sensor-net` is rejected outright: it holds
execution and policy authority, and would save one container at the cost
of the trust model. The dedicated `sensor-intake` is retained on
evidence, not preference.

## R3.13 Minors 11/12 — ACCEPTED

Direct sensor→memu code is **deleted** at the migration point, and the
duplicate source tables are **physically removed** when the joined
catalogue becomes authoritative. No dormant escape hatches — code left
inert behind DNS isolation is code waiting for a network change to
resurrect it.

