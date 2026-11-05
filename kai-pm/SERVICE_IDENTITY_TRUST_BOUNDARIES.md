# Service identity — the trust boundaries, before adding any more crypto

**Date:** 2026-08-08 · **Status:** analysis, no new mechanism proposed.
Written because the instruction was *establish the actual trust boundary
first*, and because a hash or signature that lives in the same
compromise domain as the thing it protects adds ceremony, not security.

---

## 1. The key map: who can actually modify it

Traced through the deployment as it stands, not as intended.

| stage | artefact | who can write it | what a write achieves |
|---|---|---|---|
| generation | `secrets/service-identity/keymap.json`, mode 0644 | anyone with the host account that ran `make service-keys` | full control of identity |
| source of truth | the same file on the host | same account | same |
| delivery | bind mount, `:ro` | the account that can edit the host file | same |
| in-container | `/etc/kai/identity/keymap.json` | **nobody** — the mount is read-only, verified | nothing |
| other containers | not mounted anywhere else | — | — |

Measured, not assumed:

* `check_service_identity_wiring` gates that the map is mounted `:ro`,
  and its test proves a writable mount is a finding.
* `KeyMap.load` refuses a map that is group- or other-writable, tested.
* Only `cortex` mounts it. No second container has it at all, writable
  or otherwise.

### What that means for a map signature

**The host account that can edit the map is the same account that can
edit any signature or hash over it, and the same account that holds
every private key.** A checksum committed to the repository would be
verified by code that the same account can edit, in an image that the
same account builds.

So a map signature today would defend against exactly one thing:
accidental corruption. It would not defend against the compromise it
looks like it defends against, and it would *read* as though it did —
which is worse than not having it, because it converts an honest gap
into a decorated one.

**Conclusion: do not add map signing yet.** It becomes worth doing when
the map's source of truth moves into a genuinely separate trust domain —
a signing host, an HSM, or a deploy pipeline whose credentials the
runtime account does not hold. At that point the anchor is somewhere the
attacker of the runtime cannot reach, and the signature starts meaning
something.

What *is* worth having now, and is already there:

* the loader refuses ambiguity (duplicate key id, one public key
  claiming two identities, duplicated material posing as rotation);
* the map's SHA-256 is logged on load, so drift between services is
  observable by comparison rather than assumed identical.

---

## 2. Destination identity is not a hostname

The signed string contains a `destination`. Today that value is a
Compose DNS service name, and it would be easy to let that harden into
`service identity == hostname` without anyone choosing it.

**The rule, chosen deliberately:**

> `signed destination` is a **canonical logical service identity**.
> A URL or hostname is **routing configuration**.
> They coincide today because Compose DNS names are the project's
> canonical logical IDs. That is a decision, not a fact about networks.

Consequences, all currently true:

* the **caller** derives `destination` from the URL it is about to call
  (`urlparse(CORTEX_URL).hostname`), so an override changes the signed
  destination with it — the caller cannot sign for one service and send
  to another by accident;
* the **receiver** is authoritative about its own expected destination.
  It reads `KAI_SERVICE_NAME` from its own environment and never from
  the request. A caller that names a destination the receiver does not
  agree with simply fails signature verification;
* `KAI_SERVICE_NAME` is therefore **local configuration for signing and
  for the receiver's own identity**, never an assertion anyone else can
  make. `check_service_identity_wiring` asserts no code path reads an
  identity-naming header.

If routing and identity ever diverge — a service reachable under two
names, or moved behind a proxy — the signed destination must stay the
logical identity and the URL must be allowed to change underneath it.
That is the reason the two are named differently now, while they happen
to be equal.

---

## 3. What is proven, and at what level

The distinction the operator asked to keep explicit:

| claim | level | status |
|---|---|---|
| the signing/verifying primitive is correct | unit | **PROVEN** — 80 assertions |
| `/observe_turn` is governed at the route | route/test | **PROVEN** — 35 assertions through the real app |
| the wiring is coherent | static | **PROVEN** — 28 assertions, gate fires on each defect |
| the deployment path works between containers | deployment | **NOT PROVEN** |
| Ed25519 usable in real Kai service images | runtime | **UNKNOWN** |

The last two are one experiment:
`scripts/security/verify_identity_in_containers.sh`, which refuses to
report without a daemon rather than skipping quietly. Route-level proof
is not deployment proof, and this table exists so the two cannot be
conflated by a reader in a hurry — including me.

---

## 4. Chronic authentication failure is now countable

The defect that motivated this: `/observe_turn`'s only caller sent no
credentials at all, every call was refused, and `record_degradation`
swallowed it. An integration that had **never worked** was
indistinguishable from one nobody used.

What was added, and deliberately what was not:

* `common.service_auth.auth_telemetry()` — counts keyed by operation and
  outcome class (`verified:<identity>`, `no_grant:<identity>`,
  `signature_rejected`, `unsigned_but_grant_gated`,
  `shared_token_unverified`, `keymap_unavailable`,
  `backend_unavailable`, `token_rejected_<status>`);
* exposed on `GET /health`, which mutates nothing — reading it is not an
  event;
* the caller now records a refusal as a degradation with the status code
  and body, instead of discarding it.

**Not added, on purpose:** any live probe that posts a turn to prove
authentication works. That would make the watchdog a perception source
and Cortex would learn from its own monitor. Real authenticated calls
belong in CI and container tests, against disposable state.
