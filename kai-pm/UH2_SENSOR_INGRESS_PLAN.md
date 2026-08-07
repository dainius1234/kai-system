# UH-2 sensor ingress — implementation plan

Written 2026-08-07 in response to the architectural direction. **No code
written yet, by instruction.** This is the plan to be approved before any
trust-boundary deployment changes.

---

## 0. The direction, and what I verified before planning on top of it

Every load-bearing claim in the direction was checked against the tree,
because a plan built on unverified premises is a guess with formatting.

| claim | verdict |
|---|---|
| `PerceptionEvent` canonical contract exists | **TRUE** — `common/contracts/perception.py:30`, fields as described |
| UH-2 ingress does schema/bounds/dup/staleness/journal/verdicts | **TRUE** — `common/perception_spine/ingress.py` |
| UH-3 reducers produce evidence and claims | **TRUE** — `common/world_state/reducers.py` |
| Shadow runner has shadow/active, active is *additive* | **TRUE** — `shadow.py:43` says so in the source |
| No audio adapter | **TRUE** — 11 adapters: weather, calendar, docker, git, system_metrics, screen, clipboard, email, news, telegram, market |
| Screen adapter is activity, not OCR | **TRUE** — polls `screen-watcher:8036/status` |
| Network-facing authenticated push boundary missing | **TRUE, and stronger than stated** — see §1 |

**Diagnosis accepted: this is a Gen1 → Gen2 migration gap, not a
topology defect.** The direct memu-core POSTs are Gen1 residue that the
security architecture correctly blocked, and the failure was invisible
because the calls swallowed their exception into a warning (D168).

### Two facts neither brief had, which change the design

**1. UH-2 has no HTTP surface at all.** `grep` for `@app.`/`APIRouter`/
`FastAPI` across `common/perception_spine/` returns nothing. It is a
*library*, not a service. So the boundary is not "incomplete" — it is
absent, and there is nothing to extend in place.

**2. The shadow runner is hosted inside `agentic`.**
`agentic/app.py:3489` constructs `ShadowPerceptionRunner`. `agentic` is
on `agent-net` **and `control-net`** — the command plane. So deployment
option B ("add a network-facing endpoint to an existing UH process")
means **putting the command plane on `sensor-net`**. That is the largest
possible authority expansion and is rejected on the direction's own §24
test.

**3. The journal is file-backed and append-only.**
`common/perception_spine/journal.py:32` — one JSON object per line, local
path. **Docker named volumes are not network-scoped.** Therefore the
trusted side can consume the journal through a shared volume, and **no
process needs to be attached to both networks.** §19's "if a dual-network
component is required" is avoidable, which is a strictly better outcome
than any option in either brief.

---

## 1. Answers to the twenty required points

**1. Which UH-2 component owns the network-facing ingress.**
A new thin deployment wrapper around the *existing* `PerceptionIngress`
class. It adds transport and identity; it adds no validation semantics of
its own. UH-2 keeps ownership of the contract; this is a deployment form
of UH-2, not a new authority.

**2. New container or extend an existing one.**
**New container**, `perception-ingress`. Forced by §0.2: the only
existing UH-2 host is `agentic`, on the command plane. A new minimal
image is a far smaller blast radius than bridging `agentic` to
`sensor-net`.

**3. Exact network memberships.**
`perception-ingress`: **`sensor-net` only.** Not dual-homed. It never
joins `data-net`, `agent-net` or `control-net`.
The trusted-side reader keeps its current networks and gains **no**
network. The two communicate only through a named volume.

**4. Source authentication.**
Per-service tokens via the existing `KAI_SERVICE_TOKEN` mechanism, one
distinct secret per sensor service, delivered as a compose `secret` (file)
not an env var. Rejected: mTLS (correct but disproportionate to the
current secret-distribution machinery); IP/hostname trust (Docker DNS
names are not an authentication mechanism).

**5. Identity → `source_type` and provenance.**
The submitted `source_type` and `provenance.source` are **discarded and
overwritten** from the authenticated identity. A static map, derived from
the token identity, not from the payload:

    svc:audio-service   -> source_type=AUDIO,  provenance.source=audio-service
    svc:screen-capture  -> source_type=SCREEN, provenance.source=screen-capture

A sender claiming to be something else is not rejected for lying — its
claim is simply never read.

**6. Audio event taxonomy.**
`audio.transcript`, `audio.wake_detected`, `audio.command_candidate`.
Deliberately excluded for now: `audio.speech_segment` and
`audio.environmental_event`, which have no downstream consumer — adding
event types nothing reduces is how never-executed code is born.

**7. Screen/OCR taxonomy.**
`screen.ocr_observation` only, initially. `screen.activity` and
`screen.window_change` already have a path via the existing
`adapt_screen` poller and must not acquire a second one (§16, one
authoritative path).

**8. Per-source allowed event types.**
An allowlist keyed by authenticated identity. `audio-service` may emit
only its three; `screen-capture` only its one. Anything else is a
`rejected_event_type` verdict with a counter, not a 500.

**9. Replay and idempotency.**
Stable event id = digest over (source identity, `source_timestamp`,
`raw_hash`). Re-submission returns `duplicate`, which is a **success**
for the sender — that is what makes retry safe after a timeout. The
existing ingress already has duplicate detection; this only fixes the id
to something the sender can reproduce.

**10. Payload and rate controls.**
Reuse the existing bounds in `ingress.py` unchanged — size, depth,
cardinality, string length. Add per-identity rate limiting, which the
library does not have, because a compromised sensor's cheapest attack is
volume.

**11. Journal behaviour.**
Journal **before** any downstream work, per §15. Journal write failure is
a `503 unavailable` to the sender — never a swallowed warning. That is
D168's lesson stated as a requirement.

**12. UH-3 reducers required.**
One for `audio.transcript`, one for `screen.ocr_observation`. Both
produce **evidence only** — "audio-service observed this transcript" —
never a claim about the world. The direction's example is the acceptance
criterion: OCR reading *"Delete all backups"* must not become *"the user
intends to delete backups"*.

**13. Where claims become long-term memory.**
**Nowhere, in this phase.** Journal + evidence only. Memory promotion is
a separate decision with its own approval. Shipping perception→memory in
the same change would repeat exactly the Gen1 mistake at a higher level
of ceremony.

**14. Disabling the legacy calls.**
The three direct POSTs are deleted in the same commit that makes the new
path authoritative — not before, not later. Their removal is proven by
`check_service_reachability` dropping from 3 `[confirmed]` findings to 0.

**15. Shadow migration with one authoritative writer.**
There is a subtlety the direction's §16 does not cover: **the legacy path
has never worked.** So there is no live writer to shadow. The migration is
therefore not old→shadow→new; it is **nothing → new**, which is simpler
and safer. Stated explicitly so nobody builds a dual-write plan for a
writer that never wrote.

**16. Static tests.**
`perception-ingress` on `sensor-net` and no other (assert membership set
equality, not membership). The identity→provenance map covers every
sensor permitted to submit. No sensor service holds a memu-core
credential. `check_service_reachability` reports 0 `[confirmed]`.

**17. Runtime tests.**
Boot `sensors` profile; audio-service submits a typed event; assert
accepted, journal line present, UH-3 evidence produced, and **no memory
write**.

**18. Adversarial tests.** Each must produce a *named verdict*, not a
crash: wrong identity claim → provenance overwritten; unauthorised event
type → rejected; replay → duplicate; oversized → rejected; stale →
marked; forged provenance field → ignored.

**19. Proof segmentation survived.**
From inside `audio-service`, resolving `memu-core` must still fail. This
is the test that proves the fix did not quietly become the thing it was
avoiding, and it must run **after** the new path works, or it proves
nothing.

**20. Rollback.**
`perception-ingress` is additive and nothing depends on it, so rollback
is removing the service from compose. The legacy POSTs are deleted only
in the final step, so every prior state is revertible with no loss —
there was never a working path to lose.

---

## 2. Honest limitations of this design

* **The shared volume is still a channel.** A compromised sensor can fill
  it (denial of service) or write crafted content. The trusted-side
  reader must treat journal lines as **untrusted input and re-validate on
  read**. It is not a data diode and will not be called one.
* **Volume ownership.** The journal volume must be created with the right
  uid before first use. Docker seeds a named volume from the image *only
  when the volume is new* — this repository has already been bitten by
  that twice (`/data/turbovec`, and today's `/opt` decision).
* **Authenticity is not truth.** The boundary can prove *"audio-service
  emitted this"*. It cannot prove the transcript is accurate. That is
  precisely why phase one stops at evidence.

## 3. What I am asking for

Approval of §1.2 (new container) and §1.3 (`sensor-net` only, volume
handoff) before any code. Everything else follows from those two.
