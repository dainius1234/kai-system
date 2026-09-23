# CAI v1.0 — threat model

Source: implementation order §9. Each row names the control and the
hostile calibration case that must prove it (order §18). A threat with no
calibration case is not claimed as covered.

## In scope

| threat | control | proven by |
|---|---|---|
| executor mistake — work on the wrong baseline | WA binds exact `baseline_commit` + `baseline_tree`; admission requires ancestry and, after genesis, baseline = previous admitted candidate | A9, P-7/P-8 cases |
| executor scope expansion | complete `--no-renames` tree diff checked against WA path scope; forbidden dominates | S1–S4 |
| compromised executor credentials — forging authority | only a signature by the **pinned** key is eligible; executor cannot sign as Dainius | A2–A4, D8 |
| compromised executor credentials — namespace pollution | look-alike and unverifiable records make derivation REFUSE (P-3); ruleset design restricts tag creation to Dainius | N2–N4, D8 |
| ambiguous / lost / delayed chat | no chat input exists anywhere in the verifier | H4 |
| stale / replayed WA | expiry (A8), single use (P-6), baseline staleness (P-7), repository binding (A6) | A6, A8, A9 |
| wrong candidate | admission target must equal the IV&V evidence's exact commit | V2, D7 |
| candidate drift — same tree, different commit | commit identity, not tree identity | V3 |
| stale baseline | P-7 | A9 |
| path / mode bypass | symlink, gitlink, rename-both-ends, ambiguity, collision | S3–S7 |
| ordinary check spoofing | required-workflow pinning by exact workflow SHA — **server availability UNVERIFIED — `rulesets/PINNING_RESEARCH.md` §3** | not claimed until leg 2 |
| verifier outage | every unavailability → UNKNOWN → REFUSE | A5 |
| forked admission history | sibling successors → UNRESOLVED; orphans → REFUSE | D3–D5 |
| cross-repository replay | payload binds `repository_id` **and** full name | A6, D6 |

## ARCHITECTURE FINDING — the in-scope/out-of-scope line has collapsed

Measured 2026-09-23: the executor pushes as `dainius1234` (216391246),
the same GitHub identity as the authority. "Compromised executor
credentials" (in scope) is therefore today "a compromised Dainius GitHub
account" (out of scope). Signature pinning still prevents forgery;
rulesets cannot separate the two principals, and deletion of the head
admission would silently regress the baseline. Returned to Kai and
Dainius: `rulesets/PINNING_RESEARCH.md` §4.

## Explicitly out of scope (order §9)

* a malicious Dainius;
* a fully compromised Dainius GitHub account or signing key;
* a compromised GitHub platform.

No second human and no claimed independence is added to address these.
A root compromise of the pinned key defeats CAI by design, and that is
stated rather than mitigated with an appearance of independence.

## Residual risks stated, not solved

* **Before rulesets are live, CAI enforces nothing.** Anyone with push
  access can create tags in the authoritative namespaces. P-3 makes that
  visible; only the leg-2 rulesets make it impossible.
* **Scratch commits are not prevented.** CAI controls admission, not
  every commit (order §4.2).
* **Denial of service by namespace pollution** is accepted under P-3 in
  exchange for fail-closed integrity. Kai to rule.
