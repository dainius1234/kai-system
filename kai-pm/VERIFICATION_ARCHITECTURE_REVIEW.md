# Verification-architecture gap review

**Date:** 2026-08-12 · **Scope:** the proof system, not Kai
**Status:** analysis and recommendation only. No implementation.

> Make the proof system itself as trustworthy as the system it is
> proving.

Everything below was measured against this tree. Where a number appears
it was computed, not recalled; where something was not run it says so.

---

## 0. What already exists — measured, not assumed

This matters first, because the request was *reuse → extend → only then
create*, and the honest finding is that **most of the architecture the
review asks for is already here and under-populated, not missing.**

| capability | evidence |
|---|---|
| instrument registry | 50 gates declared, 50 found on disk; `check_gate_registry.py --gate` exits 0 |
| declared invariants | I-1 fail-closed · I-2 denominator · I-3 prove-it-can-fail · I-4 declare-once · I-5 no-inert-rules · I-6 closures-hold · I-7 ratchets-calibrated — all 7 enforced |
| shared fail-closed library | `gate_inputs.require()` / `inspected()` — one import, not twelve edits |
| artefact identity | `gated_commit.sh`: TESTED TREE SHA == COMMITTED TREE SHA, in an isolated worktree |
| verdict integrity | conditional capture (`if cmd; then rc=0; else rc=$?; fi`) used in the gate, the collector and Claim B |
| environment fidelity | `chmod 755` on the candidate worktree, after 0700 broke an unprivileged harness |
| suppression register | `check_ci_tolerations.py` — 24 suppressions, 19 declared, owner + review date each |
| test-order determinism | `check_test_isolation.py` — a whole class of nondeterminism already gated |
| hermetic probe | Claim A runs `docker run --network none` |
| supply-chain scanning | `pip-audit` and `trivy` in `core-tests.yml`, advisory, declared with owners |
| gate self-tests | 220 `scripts/test_*.py` suites |

**The registry already has the right schema.** `Gate` carries `inputs`,
`denominator`, `probe`, `proven_by`, `ratchet`, `calibrated_by`,
`in_policy_check`, `in_workflows`, `findings`. That is close to the
evidence envelope §F/§M asks for. The gap is **population and
emission**, not design:

```
denominator      50 / 50 declared
proven_by        50 / 50 declared
probe            35 / 50   (15 carry an explicit skip reason)
calibrated_by    10 / 50
ratchets          4, and 0 of them lack calibration
```

The four ratchets — the gates whose verdict rests on a stored baseline,
and therefore the ones that can go blind and stay green — are **all
calibrated**. That is the strongest single fact in this review, and it
means the recursive question is already answered *for the population
where it bites hardest*.

---

## 1. Gap matrix

Severity is *how likely this is to make us believe a false engineering
conclusion*, not how much work it is.

| # | practice | covered? | evidence | gap | sev | reuse? | recommendation |
|---|---|---|---|---|---|---|---|
| A | instrument self-verification | **mostly** | I-3 `probe` on 35/50; 220 suites; `test_ci_tolerations.py` now asserts known-positive + known-negative for the new check | the *case list* is ad hoc. No gate declares that it has an UNKNOWN case, a malformed-input case, or a timeout case — those exist where someone thought of them | **HIGH** | extend registry | **STRENGTHEN** |
| B | mutation testing | **partly, by hand** | I-3 *is* manual mutation: inject the violation, assert it fires. 35/50 | 15 gates have no can-fail proof; no scheduled re-proof that the 35 still fire | MED | **keep ours** | **STRENGTHEN**, not adopt |
| C | hermetic verification | **partly** | `--network none` on Claim A; isolated worktree; `check_test_isolation.py` | 188 of 296 requirement lines are unpinned (`>=`) across 50 files; no locale/TZ pinning in evidence jobs; CI installs from PyPI at run time | **HIGH** | pip-tools / hashes | **STRENGTHEN** |
| D | build & image identity | **partly** | immutable Docker image IDs recorded (`sha256:b5e68a3…`) from the container Compose creates | every Dockerfile is `FROM python:3.11-slim` — **no digest anywhere**. Two runs a week apart can differ and the evidence cannot show it | **HIGH** | digest pin (free) | **STRENGTHEN** |
| E | supply-chain provenance | **scanning only** | pip-audit + trivy, both advisory with declared owners | zero SBOM, zero attestation tooling (`grep` for syft/cyclonedx/spdx/cosign/slsa across workflows and Makefile: **no matches**). Model weights and caches have no provenance record | MED | syft / CycloneDX | **LATER**, fold into the Unified Artifact Admission Gate |
| F | machine-readable evidence bundle | **no** | verdicts live in prose: CI logs, `kai-pm/*.md`, commit messages | nothing emits a structured record. Reconstructing why a claim became PROVEN means reading prose — the stated failure mode | **HIGH** | JSONL, no dependency | **STRENGTHEN — top control** |
| G | verdict provenance (`producer=`) | **ad hoc** | exists in `#47` only: `producer=probe / timeout-wrapper / gate / resolver` | not a standard envelope. Exit `124` from `timeout` and exit `124` from a probe would be indistinguishable anywhere else | **HIGH** | ours, generalised | **STRENGTHEN** |
| H | independent second channel | **partly** | Claim B asserts width 384 *and* corroborates with `dim=384` in the service log; `gated_commit.sh` re-reads the tree after commit | applied where someone remembered. Not a property of the evidence layer | MED | ours | **STRENGTHEN** (cheap: make the envelope have room for a corroborating channel) |
| I | flakiness taxonomy | **no** | no `reruns`/`flaky`/`retry` config anywhere; `check_test_isolation.py` covers *order* dependence only | deterministic FAIL, flake, timeout and environment-specific failure are all just "red". `TIMEOUT_UNKNOWN` in #47 is the only place they are separated | MED | **do not adopt reruns** | **STRENGTHEN** with a ledger, not a rerun plugin |
| J | fail-closed vs evidence-preserving | **now correct, locally** | #47: collectors exit 0 on any defined verdict; completeness judged once, last, before an `if: always()` upload | the principle is not written down as a rule, so the next collector will re-learn it. Run 2 lost Claim B to exactly this | MED | ours | **STRENGTHEN** — promote to a named rule |
| K | branch protection | **no** | no `required_status_checks` anywhere; one aspirational mention in the product spec | every gate is bypassable at merge by construction | MED | GitHub native | **LATER** — needs authorisation, and we are not merging |
| L | standard frameworks | **n/a** | — | see §3 | — | mostly **KEEP CURRENT** |
| M | instrument-health record | **declarable, not emitted** | registry has `calibrated_by`; 10/50 populated; all 4 ratchets calibrated | a claim can be promoted today with no record of whether its instrument was calibrated *for that run* | **HIGH** | extend registry + envelope | **STRENGTHEN — top control** |
| N | denominator drift | **weakest area** | `denominator` is declared 50/50 — **but it is a regex the output must match**, not a value compared against reality | it proves the gate *says* a number. It does not prove the number is right. Finding #48 is a live instance: a registry whose denominator is "what CI references", not "what exists" | **HIGH** | ours | **STRENGTHEN — top control** |

---

## 2. Classification

**KEEP CURRENT** — bespoke, and better than the standard alternative for
this system:

* the gate registry and I-1…I-7. No off-the-shelf tool encodes
  "fail closed on a missing input" plus "report a denominator" plus
  "prove it can fail" as auditable per-gate declarations.
* `gated_commit.sh`. There is no standard tool for
  TESTED TREE == COMMITTED TREE; it is four `git` plumbing commands and
  a refusal.
* `check_ci_tolerations.py`. Policy-as-code (OPA/conftest) would replace
  reasoned prose with rego and lose the reasons.
* manual I-3 probes over blanket mutation testing.

**STRENGTHEN** — A, B, C, D, F, G, H, I, J, M, N.

**ADOPT STANDARD TOOL** — two only, and both narrow:

* **digest-pinned base images.** `FROM python:3.11-slim@sha256:…`. Free,
  no dependency, and it is the single change that makes "what exact
  software produced this evidence" answerable at all.
* **`--junitxml`** on the pytest invocations. Structured test results
  for one flag, versus writing a bespoke parser.

**LATER** — E (SBOM/attestation, into the Unified Artifact Admission
Gate), K (branch protection, needs authorisation).

**NOT NEEDED** —

* mutation frameworks (`mutmut`, `cosmic-ray`) across ~140k LOC.
  Unaffordable, and low-signal against a repo whose critical properties
  are already probed by hand.
* policy-as-code engines. See above.
* `pytest-rerunfailures`. **Actively harmful here** — it institutionalises
  rerun-until-green, which the review names as the danger.
* reproducible-build toolchains (Nix/Bazel). Enormous, and digest
  pinning plus hash-pinned requirements buys most of the benefit.

---

## 3. On the standard tools, specifically

| candidate | what it would solve better | verdict |
|---|---|---|
| syft / CycloneDX SBOM | answers "what was in the image" without bespoke code; feeds the admission gate | **LATER** — real value, but it belongs to the admission gate, not a parallel subsystem |
| `actions/attest-build-provenance` (SLSA) | signed, registry-backed build provenance | **LATER, and note the blocker**: images are built locally in CI and never pushed to a registry, so there is nothing to attach an attestation to. Adopting this implies a registry decision first |
| `--junitxml` | structured results for a flag | **ADOPT** |
| `pip-compile --generate-hashes` | turns 188 unpinned lines into hash-verified installs | **STRENGTHEN** (C) — but it is a production-dependency change, and #47's constraints forbid that today |
| digest-pinned `FROM` | removes silent base-image drift under evidence | **ADOPT** |
| mutmut / cosmic-ray | automated I-3 | **NOT NEEDED** at repo scale; reconsider for `common/service_auth.py` and the tool gate only |
| OPA / conftest | declarative CI policy | **NOT NEEDED** |
| pytest-rerunfailures | flake tolerance | **REFUSE** — it is the failure mode |

---

## 4. The two recursive questions, answered as they stand today

### 1. Can we prove the instrument was calibrated and healthy when it produced the claim?

**Partly. For ratchets, yes. For everything else, no.**

All 4 ratchet gates — the ones that can go blind and stay green — name a
calibration suite, and I-7 enforces it. That is the population where the
question bites hardest and it is closed.

But `calibrated_by` is populated on **10 of 50** gates, and nothing links
a calibration *run* to a claim. The registry says a calibration suite
*exists*; no artefact says it *passed on the tree that produced this
verdict*. So today a claim can be promoted on a probe returning the
expected answer with no record of the instrument's health at that moment
— which is exactly what §M forbids.

### 2. Can we prove the instrument's scope has not silently drifted?

**No. This is the weakest control in the system.**

`denominator` is declared on 50/50 gates and is **a regex the output must
match** — it proves the gate *emits* a count. Nothing compares that count
to an independently derived expectation, so a detector whose population
shrinks from 26 to 3 still prints a denominator, still matches its regex,
and still passes.

Finding #48 is this defect, already open, already named: *the registry's
`scripts/ci/` denominator is "what CI references", not "what exists".*
That is not a hypothetical — it is the shape, found in our own
instrument, before this review.

R5 says derive the scope from the tree. I-2 currently only requires that
the scope be *stated*.

---

## 5. Top 3 controls

Chosen for leverage: each is an extension of something already here, and
each closes a way we could believe a false conclusion.

### 1. One machine-readable evidence envelope (F + G + M in one artefact)

A JSONL record per claim, emitted by the producer, carrying:

```
commit · tree SHA · run id · runner identity · image id/digest
producer · producer exit code · measurement status · claim verdict
observed value · instrument module + tree SHA · calibration result
expected denominator · observed denominator · timestamp · reason
```

Why first: it answers 5 of the 6 questions in the brief and **both**
recursive questions, and #47 already computes almost every field — they
are currently `printf`'d into prose and thrown away. This is a change of
*destination*, not of *machinery*. No dependency; JSONL and `python3`.

The rule that makes it worth having:

> **A claim may not be promoted if the envelope lacks a calibration
> result, or carries a failed one.** The probe returning the expected
> answer is not sufficient.

### 2. Expected denominator vs observed denominator (N)

Promote I-2 from "state a number" to "state a number **and** compare it
to one derived independently of the check". Not a frozen constant — a
*declared expectation* whose unexplained change fails or requires review,
with the expected side derived from the tree wherever possible.

Why second: it is the only control that catches a detector quietly
measuring a shrinking fraction of the system, and #48 proves we already
have one. It is I-8 applied to the verification system, as the brief
says.

### 3. Close the calibration gap on the 40 uncalibrated gates, and the 15 with no can-fail proof

Not by adopting a mutation framework — by finishing the mechanism that
already exists. Each of those 55 slots is a gate that could be reporting
green while measuring nothing, and we would have no way to tell.

Order matters: **do #2 before #3**, because the denominator comparison
tells you *which* gates have already drifted, and R4 says count the
population before fixing it.

---

## 6. What this review deliberately does not recommend

* No new framework. Every recommendation above extends an existing file.
* No change to production defaults, dependencies or fallbacks — #47's
  constraints stand, and the hash-pinning in §C is flagged precisely
  because it would breach them.
* No merge to `main`, and no branch-protection change. §K is analysis
  only, as instructed.
* Nothing here is scheduled. It is a gap analysis; the sequencing
  decision is the operator's.

## 7. Honest limits of this review

* Everything in §0 and §1 was measured on this tree today. The counts
  (50/50, 35/50, 10/50, 188 unpinned of 296, 4 ratchets, zero SBOM tool
  matches) come from running the registry and grepping the tree, not
  from memory.
* **Not verified:** whether the 35 declared `probe`s still fire. The
  registry records that a proof exists; I did not re-run them
  individually. That is itself an instance of the gap in §M.
* **Not verified:** GitHub branch-protection state. I did not query the
  API; the finding is that the repository contains no reference to it.
