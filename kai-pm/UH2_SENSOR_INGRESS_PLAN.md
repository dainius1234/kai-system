# UH-2 sensor ingress — implementation plan (rev 2)

2026-08-07. **No code written. No trust-boundary change made.** Written
to be attacked with the frozen adversarial checklist before authorisation.

Rev 2 supersedes rev 1. Three corrections and one new blocker.

---

## 0. Corrections to rev 1, and one blocker found while measuring

### 0.1 CORRECTION — I called the journal "append-only" as a security property. It is not.

Verified in `common/perception_spine/journal.py`:

    line  80  append()
    line 131  erase_subject()      <- deletes records by principal
    line 182  truncate()           <- empties the file
    lines 53-57, 112-115, 151-155  except json.JSONDecodeError: continue

It is a normal writable file used append-style by convention. It supports
erasure and truncation, and it **silently skips unparseable lines in
three places**, one of them commented *"Torn line — drop it."*

So the correct language, used throughout rev 2:

> **UNTRUSTED PERCEPTION JOURNAL / CROSS-BOUNDARY TRANSPORT** — never
> "immutable append-only trusted log".

This matters because rev 1's design leaned on a property the
implementation does not provide. Claiming stronger isolation than the
code delivers is the precise failure this programme exists to remove.

### 0.2 CORRECTION — the volume *is* the trust boundary

Removing the network bridge does not remove the crossing. It relocates
it. §3 reviews the volume with the same seriousness as a network edge.

### 0.3 CORRECTION — sensors must not touch the volume

Rev 1 did not say this explicitly. `EventJournal`'s lock is
**process-local** (`threading.Lock`), so multiple writers are unsafe on
integrity grounds *before* any security argument. Single writer is
mandatory.

### 0.4 BLOCKER — UH-2's existing intake is 2-of-7 functional

Measured, per direction §19. `agentic` hosts the runner and is on
`agent-net` + `control-net`:

| source | target service | target networks | reachable from agentic? |
|---|---|---|---|
| weather | weather-service | agent-net, egress-net | **yes** |
| calendar | calendar-service | agent-net, egress-net | **yes** |
| clipboard | clipboard-service | sensor-net | **NO** |
| screen | screen-watcher | sensor-net | **NO** |
| docker | docker-watcher | observability-net | **NO** |
| git | git-watcher | observability-net | **NO** |
| system | sysmetrics | observability-net | **NO** |

`minimal.yml`: **2 reachable, 5 unreachable.**
`full.yml`: **all seven services are not defined in the file at all** —
the runner would poll seven endpoints that do not exist in that profile.

And the failure is swallowed at the quietest possible level —
`shadow.py:106`:

    except Exception as exc:
        logger.debug("sensor fetch failed: %s — %s", url, exc)
    return None

`logger.debug`, then `None`, then the cycle reports normally.

**Consequence for this plan:** UH-2 is not a working spine with a missing
sensor branch. It is a spine whose existing intake is 71% dead in the one
profile that defines it, failing silently, for the same Gen1 → Gen2
reason. Routing audio and OCR into it without fixing that means
migrating from one non-functional path to another and reporting progress.

**This is a precondition, not a sub-task.** Recommend a Phase 0 before
any of the twenty points below.

---

## 1. Rejected alternatives, and why

| # | option | verdict |
|---|---|---|
| 1 | `agentic` joins `sensor-net` | **REJECTED.** `agentic` is on `control-net` — the command plane. This attaches the least-trusted zone directly to the highest authority. Largest possible authority expansion. |
| 2 | sensor services join `data-net` | **REJECTED.** Converts a correctly-blocked dependency into a security regression. This is what the broken Gen1 code implicitly wanted. |
| 3 | dual-homed generic mediator | **INFERIOR.** A process on both networks is a pivot if compromised. The volume design removes the process-level bridge entirely. |
| 4 | sensors write the volume directly | **REJECTED.** Process-local locking makes it unsafe for integrity alone; and it lets a sensor bypass authentication by writing its own provenance. |
| 5 | leave the direct memu writes broken, only log the degradation | **NOT A MIGRATION.** That is today's state plus a better log line. Architecturally invalid code left waiting for a future network change to resurrect it. |

**Selected candidate:** sensor-net-only low-authority UH-2 ingress →
single-writer perception-only named volume → read-only trusted consumer →
trusted-side progression validation → UH-3 reduction → controlled memory
promotion.

---

## 2. The volume reviewed as a trust boundary

Answering the direction's §3 explicitly.

| question | answer |
|---|---|
| Who can mount it? | `perception-ingress` (rw) and the trusted consumer (ro). Nothing else. Sensor services do **not** mount it. |
| Who can write? | `perception-ingress` only. Single writer. |
| Who can read? | The trusted consumer, read-only. |
| Can a sensor mount it directly? | No — not declared on any sensor service. Asserted by a static test. |
| Can the trusted side write back? | No — `:ro`. Enforced by the kernel, tested adversarially (test G). |
| Can the untrusted side alter old records? | **YES.** The writer has rw on a normal file. This is the residual risk; §4 is the mitigation. |
| Can it delete history? | **YES** — `truncate()`/`erase_subject()` exist. Same mitigation. |
| Can it flood storage? | **YES** without a quota. §5. |
| Can it rewrite offsets? | **YES.** §4. |
| Can it insert torn/malformed records? | **YES**, and today they are silently skipped. §6. |
| Can it replace an observed event with a different valid one? | **YES** without progression state. §4. |
| Can it create a reverse channel? | Not through the volume (`:ro` one way). The volume must carry **only** perception records — no secrets, policy, commands, capabilities, memory, operator responses or control-plane state. Asserted by test. |

**Stated plainly: the volume is a one-way transport whose contents are
untrusted. It is not a data diode and will not be described as one.**

---

## 3. Trusted-side progression state — mechanism comparison

The requirement (direction §7): *the untrusted writer must not be able to
rewrite its own history without the trusted consumer noticing.*

| mechanism | detects rewrite? | detects deletion? | detects reorder? | cost |
|---|---|---|---|---|
| A. trusted checkpoint: last offset + last event digest, stored **outside** the volume | rewrite of the *last* record only | offset regression, yes | no | trivial |
| B. checkpoint + per-record digest recorded for every consumed record | yes, on re-read | yes | yes | O(n) trusted state |
| C. hash chain: each record binds the previous record's digest | yes | yes | yes | O(1) trusted state |

**Recommendation: C, hash chain, with A as the trusted anchor.** Each
record carries `prev_digest`; the consumer keeps only the last accepted
`(offset, digest)` outside the volume. Any deletion, reorder or
substitution breaks the chain at the next read, and the trusted state is
a single small tuple rather than a growing table.

B is rejected as O(n) trusted state for a property C gets in O(1). A
alone is rejected because it detects only tampering with the tail, and
the interesting attack is rewriting the middle.

**On reconciliation failure the state becomes `TAMPER_SUSPECT` — an
explicit degraded verdict, surfaced, never a silent clean replay.**

This is a proposal, not a decision. If a cleaner existing Kai primitive
covers it, that should win.

---

## 4. Torn vs corrupt — the semantics must differ at the boundary

Current behaviour skips any unparseable line, everywhere. At a trust
boundary that must split:

* **a malformed FINAL line** — consistent with a crash mid-write.
  Acceptable; recorded as `torn_tail`, and the consumer waits rather than
  advancing past it.
* **a malformed line ANYWHERE ELSE** — inconsistent with append-only
  operation. `TAMPER_SUSPECT`. Not skipped, not silently replayed.

This is a change to how the *consumer* reads, not to `EventJournal`,
whose skip-on-torn behaviour remains reasonable for local crash recovery.

---

## 5. Storage exhaustion is a security property

Required, none of which exists today: per-source rate limits, per-event
size limits (the bounds in `ingress.py` cover payload, not arrival rate),
a bounded volume quota, a rotation or compaction policy with the
retention decision made explicitly, backpressure to the sender,
high-watermark warning, and a defined hard-full behaviour.

**Hard-full must reject with `unavailable` and surface it. It must never
silently drop, and it must never let the journal fill the host and take
trusted services down with it.**

---

## 6. The twenty points

Each: current state / proposed change / authority effect / trust-boundary
effect / failure mode / test-proof / rollback.

**1. Ingress owner.** *Current:* no HTTP surface anywhere in
`common/perception_spine/` — it is a library. *Change:* thin transport
wrapper around the existing `PerceptionIngress`; adds identity and
transport, no new validation semantics. *Authority:* none added; UH-2
keeps contract ownership. *Boundary:* none crossed. *Failure:* wrapper
diverges from library semantics — mitigated by adding no rules of its
own. *Proof:* the wrapper imports the class rather than reimplementing.
*Rollback:* delete the wrapper.

**2. New container or extend.** *Current:* the only UH-2 host is
`agentic`. *Change:* new minimal `perception-ingress` image. *Authority:*
creates a new low-authority principal. *Boundary:* avoids attaching the
command plane to `sensor-net`. *Failure:* one more service to operate.
*Proof:* static test — `agentic` networks unchanged. *Rollback:* remove
the service.

**3. Networks.** *Current:* n/a. *Change:* `perception-ingress` on
`sensor-net` **only**; trusted consumer gains **no** network. *Authority:*
no service gains reach. *Boundary:* the network boundary is not crossed
by any process. *Failure:* someone later adds a second network "to debug".
*Proof:* assert set **equality** of memberships, not membership — a
subset test would pass after exactly that mistake. *Rollback:* n/a.

**4. Source authentication.** *Current:* none; the trusted poller stamps
provenance because it initiated the call. *Change:* per-service secrets,
one per sensor, file-delivered. *Authority:* sensors gain a proveable
identity, not a capability. *Boundary:* inside `sensor-net`. *Failure:*
secret sprawl. *Proof:* adversarial test — audio claims to be
screen-capture, is rejected. *Rollback:* the path is new; disable it.

**5. Identity → provenance.** *Current:* n/a. *Change:* submitted
`source_type` and `provenance.source` are **overwritten** from
authenticated identity, never read. *Authority:* removes the sensor's
ability to self-describe. *Boundary:* this is the load-bearing control.
*Failure:* a new sensor missing from the map — must **reject**, never
default. *Proof:* forged-provenance test asserts the stored value is the
authenticated one. *Rollback:* n/a.

**6-7. Taxonomies.** *Current:* no audio/OCR event types.
*Change:* `audio.transcript`, `audio.wake_detected`,
`audio.command_candidate`; `screen.ocr_observation` only. *Authority:*
bounded vocabulary per source. *Boundary:* n/a. *Failure:* adding types
nothing reduces — that is how never-executed code is born, so each type
ships with its reducer or not at all. *Proof:* every declared type has a
reducer. *Rollback:* remove types.

**8. Per-source allowlist.** *Change:* keyed by authenticated identity;
anything else is a named `rejected_event_type` verdict with a counter.
*Failure:* allowlist drifts from taxonomy — derive one from the other.
*Proof:* audio-service emitting `policy.changed` is rejected.

**9. Replay/idempotency.** *Change:* event id = digest over (identity,
`source_timestamp`, `raw_hash`); resubmission returns `duplicate`, which
is a **success** for the sender — that is what makes retry-after-timeout
safe. *Failure:* unstable digest turns retries into duplicates-as-new.
*Proof:* submit twice, assert one journal record.

**10. Payload/rate.** *Change:* reuse existing bounds; **add**
per-identity rate limiting, which the library lacks. *Proof:* flood test
(B) — bounded, visible, trusted side unaffected.

**11. Journal behaviour.** *Change:* journal before any downstream work;
journal failure is `503 unavailable` to the sender. *Failure:* the D168
failure — a swallowed write. *Proof:* fault-inject a journal failure,
assert 503 and a counter, not a warning.

**12. UH-3 reducers.** *Change:* one per event type, producing
**evidence only**. *Authority:* sensors cannot produce claims.
*Failure:* an over-eager reducer promoting observation to claim.
*Proof:* the direction's own criterion — OCR of *"Delete all backups"*
yields *"screen-capture observed text"*, never *"the user intends"*.

**13. Memory promotion.** *Change:* **none in this phase.** Journal and
evidence only. *Rationale:* shipping perception→memory in the same change
repeats the Gen1 mistake with better ceremony.

**14. Legacy closure.** *Change:* the three direct POSTs are **deleted**,
not left inert behind DNS isolation — otherwise architecturally invalid
code waits for a future network change to resurrect it. *Proof:*
`check_service_reachability` drops from 3 `[confirmed]` to 0.

**15. One authority.** *Current, and it changes the plan:* **the legacy
path has never worked** (D168). There is no live writer to shadow, so the
migration is **nothing → new**, not old → shadow → new. Stated so nobody
builds a dual-write comparison for a writer that never wrote.

**16-19. Tests.** Static: membership set-equality; identity map total over
permitted senders; no sensor holds a memu credential; volume mounted rw by
exactly one service. Runtime: submit → accept → journal → evidence → **no
memory write**. Adversarial: the ten in §7. Segmentation proof: from
inside `audio-service`, resolving `memu-core` **still fails** — run
*after* the new path works, or it proves nothing.

**20. Rollback.** Every step before legacy deletion is additive and
revertible with no loss, because there was never a working path to lose.
After deletion, rollback is the revert of that commit.

---

## 7. Adversarial tests (the direction's ten, plus two)

A sensor mounts the journal directly → **no access**.
B sensor floods ingress → bounded, visible, trusted side unaffected.
C ingress modifies an old record → chain break → `TAMPER_SUSPECT`.
D ingress deletes records / rewinds offset → checkpoint regression detected.
E ingress fabricates a valid event → accepted **only** as an observation
  from that source domain; never fact, never authority.
F ingress fills the volume → bounded failure, `DEGRADED` surfaced, host
  intact.
G trusted consumer writes the volume → permission denied.
H sensor connects to memu-core → still impossible.
I ingress connects to agent-net/data-net → no route.
J malformed record mid-stream → `TAMPER_SUSPECT`, not skipped.
**K** *(added)* torn **final** line after a crash → `torn_tail`, consumer
  waits; distinguishable from J.
**L** *(added)* volume contains anything other than perception records →
  fail. Guards the reverse-channel property by content, not intent.

---

## 8. Red-team lenses

**Compromise `audio-service` completely.** CAN: submit forged
observations within the audio event domain; attempt flood, bounded.
CANNOT: reach memu-core, reach the control plane, mount the volume,
execute tools, obtain capabilities, write world state, alter policy, read
trusted memory, make its observation a fact.

**Compromise `perception-ingress` completely.** CAN: forge any sensor's
observations within the sensor event domain; flood or deny perception;
attempt to rewrite journal history. CANNOT: reach any network but
`sensor-net`; read trusted memory; obtain trusted-side secrets; write
world state directly; **rewrite accepted history without detection** —
that is what §3's chain buys, and it is the whole reason the mechanism
is in the plan.

---

## 9. What I am asking for

1. Ruling on **Phase 0** — fix UH-2's 5 unreachable and 7 undefined
   polling sources first. I recommend yes; migrating onto a 2-of-7 spine
   is not a migration.
2. Approval of §1's selected candidate over the five rejected options.
3. Approval of §3's hash chain, or direction to a cleaner existing
   primitive.

Then this goes to adversarial review. No code until it survives.
