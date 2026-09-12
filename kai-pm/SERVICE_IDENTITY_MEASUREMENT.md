# Service identity — the measurement, before any identity code

**Date:** 2026-08-07 · **Status:** evidence, for review. No implementation.
**Instrument:** `scripts/security/report_service_identity.py` (registered
`kind=REPORT`), tests in `scripts/test_service_identity.py` (21 assertions).
Re-runnable: `python3 scripts/security/report_service_identity.py`.

Written against the direction *Kai Service Identity Architecture (Revised)*.
It contains **two findings the direction did not have**, both of which change
the size of the job. They are in §3 and §4.

---

## 0. What I got wrong first, so the numbers can be trusted

The first run of this instrument reported **33 endpoints across 9 services**.
Nine was wrong. The ninth service was `common`, operation `db_restore` —
matched from the *usage example inside `common/service_auth.py`'s own
docstring*. A regex counted prose as code.

That is the inverted corollary of this programme's systemic finding: a scope
**larger** than reality reports failure over things that are right. It
appeared on the instrument built to measure exactly that class, which is §3.5
— diagnostics inherit the defect they hunt.

The scanner now parses with `ast` and counts only real call nodes. Two tests
pin it: a docstring example yields nothing, a real call yields the operation.

**Corrected population: 32 endpoints across 8 services.**

---

## 1. Measured population

```
inspected: 32 endpoint(s) protected by shared-token auth (across 8 service(s))

  A (membership is enough) ....... 6
  B (identity is material) ...... 26
```

**26 of 32** protected endpoints rest on a mechanism that cannot distinguish
one caller from another. Every service holding `KAI_SERVICE_TOKEN` can act as
any other service on all 26.

The A/B split is a **declared judgement**, not an inferred one — `_CLASS` in
the report states the verdict and the reason for every operation, so a
specific line can be argued with instead of a number. Anything not declared
reports as **UNCLASSIFIED**, never silently A. That branch has no live
instance now, so a test un-declares `tool_execute` to prove it still fires.

## 2. Affected endpoints and services

### B — identity is material (26)

| service | operations |
|---|---|
| `agentic` (4) | `checkpoint_restore`, `checkpoint_delete`, `subject_erasure`, `paper_trade_slice` |
| `backup-service` (7) | `postgres_restore`, `backup_full` ×2, `backup_postgres`, `backup_redis`, `backup_memory`, `backup_ledger` |
| `monitor-service` (5) | `monitor_rule_create/update/delete/enable/disable` |
| `browser-agent` (4) | `browser_run`, `browser_navigate`, `browser_click`, `browser_type` |
| `vault-sync` (2) | `vault_ingest`, `vault_export` |
| `executor` (2) | `tool_execute`, `executor_recover` |
| `cortex` (1) | `cortex_observe_turn` |
| `telegram-bot` (1) | `telegram_alert` |

`backup_full` counts twice because the handler carries two routes
(`/backup` and `/backup/full`). Two routes are two endpoints; not a duplicate.

The sharpest three, if the list needs a priority:

* **`tool_execute`** — who requested an execution is the primary authority
  question in this system. It is currently unanswerable.
* **`cortex_observe_turn`** — the turn's provenance *is* the caller. This is
  the endpoint that blocked UH-2 and produced this measurement.
* **`monitor_rule_disable`** — silencing an alert is the most
  attribution-sensitive act in monitoring, and any token-holder can do it
  anonymously.

### A — membership is genuinely enough (6)

`cortex_state_read`, `browser_scrape`, `browser_screenshot`, `browser_search`,
`monitor_rule_check`, `monitor_alerts_clear`.

All six are read-only or clear a local display buffer. Nothing downstream is
attributed to the caller, and the response is identical whoever asks.

---

## 3. Finding 1 — Kai has **two** shared-secret mechanisms, and the second one
## already implements most of the proposed design

The direction names `KAI_SERVICE_TOKEN`. There is a second, older one:

`common/auth.py` — `INTERSERVICE_HMAC_SECRET`, used by `agentic` →
`tool-gate` and by `perception/camera`. Mounted as a Docker secret
(`/run/secrets/hmac_secret`) into **three** services in
`docker-compose.full.yml`: `agentic`, `tool-gate`, `camera-service`.

Measured against the direction's requirements, it **already has**:

| direction asks for | `common/auth.py` + `tool-gate/app.py` |
|---|---|
| canonical signed string | `actor_did\|session_id\|tool\|nonce\|ts` |
| key ID in the signature | `f"{key_id}:{digest}"` |
| timestamp skew check | `SIGNATURE_SKEW_SECONDS` (`tool-gate/app.py:547`) |
| nonce replay cache | `SEEN_NONCES` + `NONCE_TTL_SECONDS`, keyed `session:nonce` |
| **nonce cache survives restart** | `_persist_nonces()` / `_restore_nonces()` — the direction lists this as an open question; it is built |
| dual-sign rotation overlap | `sign_gate_request_bundle()` — signs with primary and previous key |
| key revocation | `INTERSERVICE_HMAC_REVOKED_IDS` |

What it does **not** have is the one thing that matters:

> `actor_did` is a **caller-supplied field**, signed with a secret that three
> services hold. Any of the three can sign as any actor.

So it is not "no identity". It is **caller-asserted identity,
cryptographically sealed** — which reads as stronger than a bare bearer token
while providing the same guarantee. That is a worse failure mode than
`KAI_SERVICE_TOKEN`, because the signature invites trust.

**Consequence for the plan:** the envelope, the replay defence, the rotation
story and the revocation list do not need to be designed or built. The defect
is one property — *the key is shared and the identity is a field* — and the
remedy is one property: **make the key per-service and derive the identity
from which key verified, never from a field.**

## 4. Finding 2 — "we already have Ed25519" is half true, and the half that is
## missing is a dependency in every service image

Verified rather than assumed:

* `scripts/auto_rotate_ed25519.py` exists and **does** generate real keypairs
  (`Ed25519PrivateKey.generate()`, D68 fixed this from two random blobs).
  It rotates key material into a JSON state file.
* **Nothing signs or verifies with it.** There is no Ed25519 signing path in
  any service. It is key management with no consumer.
* `cryptography` is pinned in **exactly one** requirements file —
  `scripts/requirements-kai-control.txt`. It is in **no service image**.
* On this host the distro `cryptography` build **panics on import**:
  `pyo3_runtime.PanicException` from `cryptography.hazmat.bindings._rust`.
  Two existing tests already skip because of it
  (`test_prod_hardening.py:279`, `hmac_rotation_drill.py`), and DECISIONS
  records the same panic twice.

I have **not** verified whether a pip-installed `cryptography` works inside
the `python:3.11-slim` service images — that can only be tested by building
one, and it is the check to run before committing to Ed25519. What is
verified is that adopting it means adding a native-extension dependency to
every service that signs or verifies, and that the one environment we can
test in today cannot import it.

HMAC-SHA256 needs `hmac` and `hashlib` — stdlib, present in every image,
already used by both existing mechanisms.

---

## 5. Architecture options

All three keep the direction's non-negotiables: no central issuer, no
cross-zone call to an identity server, receiver-derived principal, request-
bound signature, shared token retained for the A set.

### Option 1 — per-service **HMAC** keys, reusing the existing envelope

Each service gets its own key as a Docker secret. Receivers hold a read-only
`kid → (identity, key)` map, mounted from one source, integrity-checked.
Identity is **derived from which key verified the signature** — the caller
never names itself. `common/auth.py`'s payload gains method, path and body
hash; `actor_did` stops being an input to identity and becomes, at most, a
subject field that carries no authority.

* **Cost:** stdlib only. One outbound call site changes (§6). The nonce
  cache, skew check, rotation and revocation are reused as-is.
* **Weakness, stated plainly:** the verifier holds a secret capable of
  *forging* the caller's signature. This is fine when only the verifier needs
  the proof. It is **not** fine if a third party must later audit "cortex
  really did say this" — a compromised or dishonest receiver could have
  fabricated the record. That is the non-repudiation gap, and it lands
  squarely on `cortex_observe_turn`, where the whole point is durable
  provenance.

### Option 2 — per-service **Ed25519** keys (the direction as written)

Same envelope, asymmetric keys. Receivers hold only public keys.

* **Gains:** non-repudiation. A compromised receiver cannot forge a caller.
  The public-key map is not a secret, so distribution and tamper-protection
  are a checksum problem rather than a secret-management problem.
* **Cost:** `cryptography` into every signing and verifying service image,
  with the import failure in §4 unresolved. Larger signatures. A public-key
  map format and its integrity mechanism to design and gate.

### Option 3 — mTLS with per-service certificates

Rejected, briefly, so the rejection is on record. Identity would live at the
transport layer and terminate at the container edge; nothing carries the
principal into the handler where provenance is actually recorded, so we would
still need an application-layer envelope. It also needs a CA and a rotation
story for certificates. More machinery, less of what we need.

## 6. Recommendation

**Option 1 as the mechanism, with the algorithm as an explicit field in the
envelope, so Option 2 is a key-type swap and not a rewrite.**

Reasoning:

1. The envelope, replay defence and rotation already exist and are tested
   (§3). Option 1 changes *what key signs* and *where identity comes from* —
   two properties — rather than introducing a subsystem.
2. Option 2's only real gain over Option 1 is non-repudiation against a
   compromised receiver, and it is currently blocked on a dependency that
   does not import in the one environment we can test (§4).
3. Deferring is only honest if the trigger is named. **The named trigger:**
   move `cortex_observe_turn` (and any endpoint whose output is later read as
   durable provenance) to Ed25519 once a `python:3.11-slim` image is proven
   to import `cryptography` — that build test is a one-commit experiment and
   should be run before this decision is closed, not after.

I am not confident enough in point 2 to call it settled, and it is the
operator's call. If the answer is "Ed25519 everywhere from the start", the
work in §3 is unaffected — the envelope and migration are identical, and only
the signing primitive differs.

## 7. Migration blast radius — smaller than it looks

Measured, not estimated:

* **Caller side: one function.** `_auth_headers()` at
  `common/actuator_registry/mutating_handlers.py:183`. Grepping every
  non-test source file for `KAI_SERVICE_TOKEN` returns exactly three files:
  that one, the receiver library `common/service_auth.py`, and a *comment* in
  `cortex/app.py`. Every authenticated service-to-service call in Kai is
  built by `build_mutating_handler`, and every one of them gets its header
  from that single function.
* **Receiver side: one library.** `common/service_auth.py` —
  `check_token()` gains a signature path and returns a principal;
  `require_service_auth(operation)` gains a scope. 32 call sites already use
  that dependency and would not change shape.
* **Compose:** per-service secrets and a public/verification map mounted to
  receivers. `KAI_SERVICE_TOKEN` is already declared in every service that
  needs it across all three profiles — verified, zero missing — so the
  distribution pattern is proven, only the fan-out changes.
* **Profile spread, worth knowing:** `browser-agent`, `cortex`,
  `monitor-service` and `vault-sync` — **12 of the 26 B endpoints** — exist
  only in `docker-compose.minimal.yml`. They are among the 26 services that
  have never been started. Their migration cannot be verified by running
  them until that is fixed, so they must be marked *migrated, unproven*
  rather than done.

The transition per endpoint, as the direction sets out: accept signature
**or** shared token, with token-authenticated calls recording identity as
`unverified` and **never** contributing to provenance; then drop the fallback
once the single caller signs. Because there is one caller, the fallback
window can be short.

## 8. What stays on shared-token membership auth

The 6 A endpoints, indefinitely. Read-only, no attribution, identical
response for any authorised caller. Making them prove identity would add a
signing dependency and a key to services for no property gained.

`INTERSERVICE_HMAC_SECRET`'s *envelope* stays. Its *shared key* does not.

## 9. What absolutely requires verified per-service identity

In order of how badly the current state hurts:

1. **`cortex_observe_turn`** — provenance is the entire output. Caller-
   asserted identity here reproduces the UH-2 defect one layer up. This is
   the endpoint that stopped the build.
2. **`tool_execute`, `executor_recover`** — authority. "Who asked" is the
   question the tool gate exists to answer.
3. **The destructive seven** — `postgres_restore`, `checkpoint_restore`,
   `checkpoint_delete`, `subject_erasure`, and the four `backup_*`. Each
   writes an audit record whose subject line is currently a blank.
4. **`monitor_rule_disable`/`enable`/`create`/`update`/`delete`** — an
   anonymous silencer of alerts is an attacker's first move.
5. **`browser_run`/`navigate`/`click`/`type`, `telegram_alert`,
   `vault_ingest`/`export`** — outbound acts taken as Kai, and content
   carrying provenance.

## 10. Separate trust boundary — do not let these merge

**Service identity ≠ observation authenticity.** The service key proves who
called the intake. It says nothing about who originated the observation. A
compromised sensor with a valid service key can still submit fabricated
readings, and the intake would correctly record *this sensor said it* —
which is the honest answer, and is exactly why the Phase 0 attribution rule
exists: *reducers may attest what the source delivered; they may not attest
that attacker-controlled text is true.*

Per-source event signing is a distinct control with a distinct key. It is not
part of this work and must not be quietly folded into it.

---

## 11. Open questions the direction lists, with what is now known

| question | status |
|---|---|
| exact canonical request format | **open** — `common/auth.py:_payload()` is the starting shape; needs method, path, body hash, and an explicit algorithm field |
| nonce-cache persistence / restart | **answered** — built and running in `tool-gate` (`_persist_nonces`/`_restore_nonces`); needs lifting into the shared library |
| public-key map format and tamper protection | **open**, and Option-dependent — Option 1 needs a *secret* map, Option 2 a checksummed public one |
| one `ServicePrincipal` class, or alongside `dashboard_auth`? | **recommend alongside.** `dashboard_auth.DashboardPrincipal` already carries per-caller tokens, `hmac.compare_digest`, and `require_dashboard_auth(scope)`, and its own docstring records that *session is caller-supplied and therefore not an authorisation input* — the same lesson. Reuse the **shape**; keep the types apart, because merging human and service principals is how a service ends up holding an operator's authority |
| per-endpoint scope rules | **open** — 26 endpoints to name. The A/B table in §2 is the input |
| does `cryptography` import in `python:3.11-slim`? | **unverified, and it gates the Option 1 / Option 2 choice.** One image build answers it |

---

## 12. The Ed25519 availability probe — result

Authorised as the single next action. **The container half could not be
run: the Docker daemon is down in this environment** (`docker info` fails;
the CLI is present). So this is the wheel-level probe, and it answers less
than the authorised probe was meant to answer.

### The two verdicts, kept apart on purpose

> **Ed25519 library feasibility — PARTIALLY PROVEN.**
> The pinned wheel installs, imports and performs Ed25519 sign/verify on
> Python 3.11.15 / linux x86_64 in *this* environment. The wheel is
> `abi3` + `manylinux_2_28`, which is a property of the artefact and
> therefore travels: no compiler or toolchain is needed to install it
> anywhere the glibc floor is met.

> **Real service-image feasibility — UNKNOWN.**
> Nothing here shows that a `python:3.11-slim` image builds with the
> dependency added, that it imports at runtime inside that container, or
> that the service starts and serves with it present. UNKNOWN until one
> `docker build` runs on a host with a daemon.

The distance between those two verdicts is this programme's own recurring
defect (§R8): an artefact behaving correctly where it *was* tested says
nothing about the place it has never run. "Works in this Python
environment" must not be allowed to read as "works in the Kai runtime",
and the labels above exist so it cannot.

### Ran, and passed

`pip install cryptography==43.0.1` into an isolated directory, imported
with nothing else on the path, on Python **3.11.15**, linux x86_64:

```
cryptography 43.0.1  imported clean
Ed25519 keypair generated: priv 32 bytes, pub 32 bytes, sig 64 bytes
sign / verify round trip: OK
tampered message: InvalidSignature (rejected)
```

**The broken artefact was never the library.** The panic recorded in
DECISIONS and skipped by two existing tests comes from the *distro*
package at `/usr/lib/python3/dist-packages/cryptography`, whose Rust
binding is mismatched. The pinned wheel does not go near it.

### What the wheel is, which is the part that decides "practical"

```
cryptography-43.0.1-cp39-abi3-manylinux_2_28_x86_64.whl   3.99 MB
cffi-2.1.1-cp311-...-manylinux_2_17_x86_64.whl            0.22 MB
pycparser-3.0-py3-none-any.whl                            0.05 MB
installed footprint                                      ~13.7 MB
```

Three facts follow, and they are the useful ones:

* **`abi3`** — one binary works across CPython ≥ 3.9. No per-version
  rebuild, and it survives a future Python bump.
* **`manylinux_2_28`** — prebuilt. **No compiler, no Rust toolchain, no
  build-time headers.** This is the difference between a one-line
  requirements change and a multi-stage build, and it is the thing I
  expected to be the blocker. It is not.
* **~13.7 MB per image**, in 8–9 images (below), not 47.

### Blast radius of the dependency, measured

All **47** Dockerfiles are `python:3.11-slim`; each of the 9 services
involved has its own `requirements.txt`. The dependency would land in
**9 of 47 images**: the 8 services holding protected endpoints, plus
`tool-gate` if the existing envelope is unified. `common/actuator_registry`
is the only caller-side module, and it ships inside those same images.

Zero service requirements currently pull `cryptography` transitively —
checked for the usual carriers (pyjwt, paramiko, authlib, pyopenssl and
others). So this is a genuinely new dependency, not one already present.

### The half I did not prove, stated as such

`python:3.11-slim` is Debian-based and current tags are bookworm
(glibc 2.36); `manylinux_2_28` requires glibc ≥ 2.28, which bookworm
(2.36) and even bullseye (2.31) satisfy. **That is an inference from the
wheel tag, not a run.** This host is glibc 2.39, so it does not prove
the image.

The outstanding check is one `docker build` on a machine with a daemon:
add `cryptography==43.0.1` to one service's requirements, build, and
`import`. Everything else about the dependency is now known.

### What this does and does not settle

It weakens the *feasibility* objection without removing it. What was
believed — "cryptography is unavailable" — is wrong: the distro build is
broken, the library is not. What replaces it is narrower and still open:
the wheel needs no toolchain and costs ~13.7 MB, **and whether the image
builds and runs with it remains UNKNOWN.**

It does **not** settle the choice. The remaining question is the one you
framed — what key model sits under the existing envelope — and the honest
statement of the trade is:

| | Option 1 — per-service HMAC | Option 2 — per-service Ed25519 |
|---|---|---|
| new dependency | none (stdlib) | 9 images, ~13.7 MB, no toolchain |
| verifier can forge the caller | **yes** | no |
| key map contents | secrets | public keys |
| map compromise | attacker can impersonate every caller | attacker can swap identities but not forge past signatures |
| envelope changes | identical either way | identical either way |

The row that matters is the second. Under Option 1 the receiver holds a
key capable of manufacturing any caller's signature, so a record saying
"cortex asserted this" is only as trustworthy as the service that wrote
it down. For most of the 26 that is acceptable. For `cortex_observe_turn`,
where the record *is* the product, it is the same class of defect as
`actor_did` — weaker, but the same shape: a party with an interest in the
answer is trusted to produce it.

### One thing to note about "reuse the existing envelope"

REUSE is the right call, and the report should be exact about what is
being reused. The envelope in `common/auth.py` is currently used by
**one receiver** (`tool-gate`) and two senders. None of the 8 services
holding the 32 protected endpoints uses it — they use bearer tokens. So
extending it means *adopting a proven local mechanism in 8 more services*,
which is still far better than creating one, but it is adoption work, not
a no-op. The envelope also needs three additions before it can carry
identity at all: method, path and body hash are not in the signed string
today, so a valid signature can currently be replayed onto a different
route of the same service.

---

## What was verified here, and what was not

**Verified by running it:** the 32/8/26/6 counts and every row behind them;
the docstring false positive and its fix; that `common/auth.py` implements
the envelope, skew, nonce persistence, dual-sign and revocation listed in §3;
that three services mount `hmac_secret`; that `KAI_SERVICE_TOKEN` is declared
in every protected service in all three profiles; that exactly one function
sends it; that `cryptography` panics on import on this host.

**Verified by running it (probe, §12):** that the pinned `cryptography`
wheel installs and imports in isolation on Python 3.11.15 / linux x86_64;
that Ed25519 sign, verify and tamper-rejection work; that the wheel is
`abi3` + `manylinux_2_28` and needs no build toolchain; its size and
dependency set; that all 47 Dockerfiles are `python:3.11-slim`; that no
service requirement pulls `cryptography` transitively today.

**Not verified:** that the wheel imports *inside* a `python:3.11-slim`
container — the Docker daemon is down here, so the glibc claim is an
inference from the wheel's platform tag, not a run. Whether any of the 12
minimal-only B endpoints behave as their code reads, since none of them
has ever been started.

**Written and withdrawn, not in the tree:** a draft
`common/service_identity.py` was started before the "do not build
anything yet" instruction arrived. It is parked outside the repository
and is recorded here so its absence is not mistaken for it never having
existed. Two things in it are worth keeping whichever option wins, and
they are stated as proposals rather than code: a **length-prefixed**
canonical string (a delimiter can be forged from inside a field, a length
cannot), and an answer to the restart-gap question — if the nonce cache
fails to restore, refuse any timestamp older than start-up for one skew
window, because a request captured before the restart carries an older
timestamp by construction.
