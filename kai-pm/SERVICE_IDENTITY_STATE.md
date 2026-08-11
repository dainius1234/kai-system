# Service identity — authoritative engineering state

**Validated checkpoint:** `773d21d`
**Date:** 2026-08-08
**Status of the next gate:** not run — no Docker daemon in this environment.

This file is the single place that says what is proven and what is not.
It exists because the distance between "the tests pass" and "it works
where it runs" is where this programme keeps finding defects, and that
distance is only visible if someone writes it down.

---

## 1. The checkpoint

```
prepush exit 0            zero EXIT GATE: FAIL
tree clean                local and remote match at 773d21d
4,168 tests passed        5 skipped
79.73% coverage           floor 60%

Secret & Restart Gate            46 / 46
Test Wiring                      19 / 19
Service identity audit           25 / 25
Service identity (ed25519)       80 / 80
/observe_turn identity slice     43 / 43
Identity wiring gate             28 / 28
Container-proof harness          35 / 35
All Unified Hunter suites        green
```

## 2. Classification — strict, and not to be blurred

### PROVEN

| claim | by |
|---|---|
| shell syntax of the container harness | `bash -n`, and every harness-test run |
| embedded Python executes across the shell boundary | stub docker executes each `python -c` for real |
| harness control flow | 35 assertions over 8 scenarios |
| request construction | all 8 snippets extracted from the file and run |
| the real `check_identity` verification path | exercised inside the stub via a fake `urlopen` |
| failure classification and exit-code semantics | PASS→0, FAIL→1, UNKNOWN→2, each asserted |
| the timeout override affects duration only | defaults still 30/2; the names appear in 2 defaults and 2 wait loops and in no check, expectation or exit path |
| **failing paths cannot emit `PROVEN`** | asserted in every non-passing scenario |

### UNKNOWN

| claim | why it is not proven |
|---|---|
| Ed25519 feasibility in the real **built image** | no image has been built with `cryptography==43.0.1` |
| actual Docker **mount and network** behaviour | both are simulated in the harness |
| container-to-container `/observe_turn` **deployment path** | never executed |

### Therefore

```
/observe_turn                 route-proven + harness-proven,
                              NOT deployment-proven
Category-B verified identity  1 / 26
Category-B shared token       25 / 26
Category-A membership-only    6
bulk migration                BLOCKED
```

No identity migration and no architecture change before the Docker
result.

## 3. The EROFS incident — permanent, do not simplify

Preserved in three places, deliberately: the fix comment in
`scripts/security/verify_identity_in_containers.sh`, **D171** in the
append-only `kai-pm/DECISIONS.md`, and the commit message of `773d21d`.

The significance is not that a bug was fixed. It is that **the old check
would have produced a false negative against a correctly mounted
read-only container**: a `:ro` bind mount raises `EROFS` (errno 30), the
check caught only `PermissionError` (`EACCES`, 13), so a correct system
would have failed the step with an uncaught `OSError`. The natural
reading of that failure is "the mount is broken" — which could have
driven a wrong architectural conclusion off a single run. A second fault
sat underneath it: as root, a mode-0444 file is writable anyway, so the
check proved nothing about permissions even when it did not crash.

Neither `bash -n` nor compiling the snippets could reach it. It took
driving the real script under a stub.

This is now the standing worked example for **I-8**.

## 4. The next and only step

```
make verify-identity-containers
```

on a Docker-capable environment. It refuses to report without a daemon
rather than skipping quietly.

**Attempted here on 2026-08-08:** `docker info` fails; the CLI is
present and the daemon is not. The script exits 2 and prints that the
status remains UNKNOWN. That is the correct behaviour and it is not
evidence of anything about the images.

### Expected classification of the result

| outcome | meaning |
|---|---|
| real Docker path passes completely | upgrade the deployment claim, with exact evidence |
| daemon unavailable / Docker absent | UNKNOWN, exit 2 |
| any contract, security, mount or network failure | FAIL, exit 1, naming the failing boundary |
| **any failing or unexecuted path** | **must never read PROVEN** |

### What to report when it runs

1. exact command
2. environment (host, Docker version, daemon)
3. image/build result
4. health/wait result
5. caller → receiver request result
6. receiver identity verification result (which identity was derived)
7. refusal-case results, each one
8. read-only mount proof (`EROFS` or `EACCES`, stated)
9. final exit code
10. whether the claim may move from harness-proven to deployment-proven
11. the exact documents and sections to update on PASS

### Documents to update on PASS

| file | section | change |
|---|---|---|
| `kai-pm/SERVICE_IDENTITY_STATE.md` | §2 | move the three UNKNOWN rows to PROVEN, with the run's evidence |
| `kai-pm/SERVICE_IDENTITY_MEASUREMENT.md` | §12 | replace the two verdict blocks — "Real service-image feasibility — UNKNOWN" becomes PROVEN |
| `kai-pm/SERVICE_IDENTITY_TRUST_BOUNDARIES.md` | §3 | the level table: deployment row PROVEN |
| `kai-pm/DECISIONS.md` | new entry | D172, the deployment proof, append-only |

Only then does the bulk migration of the remaining 25 class-B endpoints
become available, using `/observe_turn` as the template.

---

## 5. PROGRAMME HOLD — read this before touching identity

This workstream is **held**, not in progress. Nobody is waiting at a
keyboard for it. It is parked deliberately, and the conditions to
restart it are written below so this does not decay into something
somebody has to remember.

### Why it stopped

The only unresolved identity claim requires execution inside a
Docker-daemon-capable environment. This working environment has the
Docker CLI and no daemon, so **no valid runtime evidence for that
boundary can be collected here today**.

State the corollary plainly, because it is the easiest thing to get
wrong: **the absence of a daemon is not evidence for or against
deployment correctness.** It is the absence of a measurement. Nothing
about the images, the mounts, the network or the container-to-container
path is more or less likely because we could not look.

### Restart trigger

Availability of a Docker-daemon-capable environment — later hardware,
**or another approved GitHub/runtime environment**.

**Worth knowing before anyone waits for hardware:** this repository's
own CI already satisfies that condition. `.github/workflows/core-tests.yml`
runs on `ubuntu-latest` and executes `docker compose -f
docker-compose.minimal.yml up -d --build` — the exact profile the
identity slice is wired into. So the trigger is plausibly met *today*,
subject to a decision to run the proof there. That is an operator call,
not a licence to start: it changes where the proof runs, and running a
security proof inside CI has its own questions (key material generated
in a runner, secrets handling, what a green CI run is allowed to prove
about a production host).

### Restart command

```
make verify-identity-containers
```

### Result semantics

| outcome | what it means |
|---|---|
| PASS | the deployment claim may be promoted; update the four records in §4 |
| FAIL | identify the exact failing boundary **before** any remediation |
| daemon unavailable / not executed | **UNKNOWN remains UNKNOWN** |

### Prohibited while held

Not stylistic preferences — each of these would change the evidence
baseline the eventual proof is measured against:

* **no opportunistic Category-B migration.** The remaining 25 stay on
  the shared token. Migrating one without a proven deployment path
  converts it from insecure-but-working to correctly-closed-and-broken.
* **no trust-boundary redesign.** §3 of
  `SERVICE_IDENTITY_TRUST_BOUNDARIES.md` records why there is no map
  signature; revisiting that without new evidence is re-deciding a
  settled question.
* **no identity cleanup that changes the evidence baseline.** Tidying
  the harness, the key map format, the wiring or the gates would mean
  the thing eventually proven is not the thing measured here.

Bug fixes that the gates *fail on* are exempt — a red gate is new
evidence, and this hold is not a reason to leave a gate broken.
