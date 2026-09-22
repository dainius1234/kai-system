# Change Authority Interlock (CAI) v1.0 — contract

**Status: IMPLEMENTATION LEG 1 — NOT ACTIVE.** Tranche `CAI-BOOT-001`.
Not a D-number. Creates no programme decision. CAI currently provides
**zero live enforcement**: at the baseline below GitHub reported
`main protected = false`, `rulesets = []`, and none of the proposed
rulesets in `rulesets/` is applied.

## 0. Specification source — stated, because it limits every claim below

The only specification available to the implementer is Kai's
implementation order `KAI → ORION — CAI v1.0 BOOTSTRAP IMPLEMENTATION
AUTHORISED BY DAINIUS` (items 1–27), relayed by Dainius on 2026-09-22.
The reviewed CAI v1.0 design that followed three DeepSeek rounds is
**not in the repository**: a case-insensitive search of all 896 tracked
files at `194db0a` (and, read-only, the D379 branch) for
`change authority interlock`, `CAI`, `kai-admitted` and `kai-wa/`
returned nothing.

Therefore:

* anything the order fixes is implemented as the order states it;
* anything the order leaves open is marked **PROPOSED (P-n)** in §9 and
  is a Kai IV&V item, not a reviewed design decision.

## 1. Purpose — narrow, and quoted

> mechanically record who authorised a change, against which exact
> baseline, under what scope; preserve HOLD/admission history; and make
> the exact independently reviewed commit recoverable as the admitted
> programme baseline without relying on chat memory.

Roles are unchanged: Dainius is final consequential authority; Kai is
independent technical adjudicator / IV&V; Orion is executor; DeepSeek is
adversarial input only. CAI adds no authority role.

## 2. Frozen design (order §4 — not redesigned here)

1. No privileged write-capable Authority Controller. The verifiers are
   pure: facts in, `PASS`/`REFUSE` + reasons out. They never mutate the
   repository, merge, admit or create authority.
2. Work branches are scratch. Executor commits carry **zero** admission
   weight until admitted.
3. Chat is not authority. Once CAI is ACTIVE only signed control records
   create Work Authority, HOLD, revoke or admission. During bootstrap,
   Dainius's explicit instruction is the authority for this tranche.
4. Control records are **signed annotated Git tags**.
5. Admission targets the **exact candidate commit object** Kai reviewed —
   never an equal tree, similar diff, merge, squash or rebase equivalent.
6. Physical `main` is integration state. Programme baseline = target
   commit of the latest VALID admission derived through the **complete**
   chain. Never HEAD, `main`, latest commit/PR/branch/tag.
7. **No executor acknowledgement in the admission predicate** (order §6).

## 3. Namespaces and ref grammar

Only these exact grammars are records. Enumeration is by exact ref
directory (`git for-each-ref refs/tags/kai-wa/ refs/tags/kai-admitted/`),
so `refs/tags/kai-wa-evil/...` is never enumerated at all.

| record | ref | tag target | P |
|---|---|---|---|
| Work Authority | `refs/tags/kai-wa/wa/<WA_ID>` | the WA's `baseline_commit` | P-1 |
| HOLD / REVOKE | `refs/tags/kai-wa/ev/<WA_ID>/<EVENT_ID>` | the WA's **tag object** | P-1, P-2 |
| Admission | `refs/tags/kai-admitted/<ADMISSION_ID>` | the exact `candidate_commit` | — |

`<ID>` = `^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$`. A ref inside the two
enumerated directories that does not match its grammar exactly is a
**look-alike**: it is never a record, it is reported, and it makes the
namespace state `REFUSE` (P-3).

Every record must additionally satisfy:

* annotated tag (a lightweight tag is not a record);
* the tag object's own `tag` header equals the ref name (a signed object
  re-pushed under another name is refused);
* the ID in the ref equals the ID in the payload;
* the tag target is the object the record type requires (table above).

## 4. Payload encoding

The tag message is exactly one canonical JSON object followed by `\n`:
keys sorted, separators `,` and `:`, UTF-8, no floats, no duplicate keys.
The verifier re-serialises what it parsed and **refuses any message
that is not byte-identical to that canonical form**, so there is one
byte representation per record. Tags are created with
`--cleanup=verbatim`.

Every payload binds the repository: `repository_id = 1004463473` **and**
`repository_full_name = dainius1234/kai-system` (order §10). The schemas
in `schemas/` are the single definition; the verifier loads and enforces
those files rather than a copy of their rules.

## 5. Signature states (order §11)

Verification uses `git verify-tag --raw` against an **isolated**
`GNUPGHOME` that the verifier builds from the pinned authority key only.
Classification is by the GnuPG status lines, never by exit code alone:

| state | condition | eligible |
|---|---|---|
| `VALID_PINNED` | exactly one `VALIDSIG` whose **primary** fingerprint equals the pinned fingerprint, and a `GOODSIG` | yes |
| `WRONG_KEY` | `VALIDSIG` for another primary key, **or** `ERRSIG … 9` / `NO_PUBKEY` (signer not pinned) | REFUSE |
| `UNSIGNED` | no signature block in the tag object | REFUSE |
| `INVALID` | `BADSIG`, `EXPSIG`, `EXPKEYSIG`, `REVKEYSIG`, or any other `ERRSIG` | REFUSE |
| `UNKNOWN` | gpg/git unavailable, timeout, no pinned key configured, or status output that matches none of the above | REFUSE |

Only OpenPGP signatures are implemented. SSH-format signatures are
**not** implemented: `ssh-keygen` is not installed in the implementation
environment, so no SSH path could be calibrated, and an uncalibrated
path is not claimed (P-4).

## 6. Path / mode semantics (order §12)

The change population is the complete baseline→candidate tree diff:

    git -c core.quotepath=false -c diff.renames=false \
        diff-tree -r -z --raw --no-renames --no-ext-diff <baseline> <candidate>

Rename detection is **disabled**. It is a similarity heuristic whose
result depends on configuration, so it is not a deterministic input to an
authority decision. Without it a rename is its deletion plus its
addition, and **both paths are checked** — which is exactly the order's
rule that old and new path must each satisfy policy.

For every changed entry, each of its paths must:

* decode as strict UTF-8 and equal its own NFC normalisation;
* contain no control character, no backslash, no empty / `.` / `..`
  component, and no component equal to `.git` case-insensitively;
* not collide, after case-folding and NFC, with any other path in the
  candidate tree;
* match an allowed exact path or an allowed prefix, and match no
  forbidden exact path or forbidden prefix (forbidden dominates).
  Prefixes must end in `/`, so `a/` never matches `a-evil/x`.

Modes `120000` (symlink) and `160000` (gitlink) in either the old or the
new mode are refused by default. Authority reads Git object paths and
modes only; `.gitattributes` working-tree conversion never enters it.

## 7. Work Authority, HOLD, revoke (order §13–14)

A WA is **ACTIVE** when its record verifies (§3–5), its baseline commit
and tree exist and match, every `governing_refs` entry resolves and
hashes to its recorded `sha256`, and it is unexpired at the evaluation
time. `note` is free text and never grants scope.

A valid HOLD or REVOKE event is **terminal**: it dominates that WA for
admission. There is no release event; recovery is a new WA. Chat text is
never read.

## 8. Admission chain (order §5, §15)

Algorithm — `max(sequence)` and "latest tag" are never used:

1. enumerate `refs/tags/kai-admitted/` exactly;
2. parse every record; 3. verify signature; 4. verify pinned key;
5. verify repository identity; 6. verify schema;
7. build the chain from genesis — exactly one record with `sequence = 1`
   and `previous_admission_tag_object = null`;
8. verify sequence continuity; 9. verify each record's
   `previous_admission_tag_object` equals its predecessor's **tag object
   SHA**;
10. two valid records claiming the same successor position →
    `UNRESOLVED` — no baseline is derived and none is chosen;
    any valid record not on the chain (skipped sequence, wrong
    predecessor) → `REFUSE`;
11. resolve `ivv_evidence_ref` to the exact committed bytes, hash them,
    compare with `ivv_evidence_sha256`; missing → REFUSE, mismatch →
    REFUSE, unresolvable → UNKNOWN/REFUSE; the evidence must name this
    `wa_id`, this `candidate_commit` and `candidate_tree`, and carry
    `disposition = ACCEPT_FOR_ADMISSION`;
12. the admission tag **target** must equal the evidence's
    `candidate_commit` (same tree / different commit → REFUSE), and the
    target commit's tree must equal `candidate_tree`.

Each admission additionally requires its WA to be valid, not dominated
by a terminal event issued at or before that admission's position
(P-2), unexpired at the admission's signed tagger time (P-5), not already
consumed by an earlier admission (P-6), and — except at genesis — based
on the previous admission's candidate (P-7); and the baseline→candidate
change population must satisfy the WA scope (§6), with the candidate a
descendant of the baseline (P-8).

**Programme baseline** = the candidate commit of the chain head, and
only when every record above verifies. Otherwise the derivation result
is `REFUSE` or `UNRESOLVED` with reasons, and no baseline is reported.

## 9. Choice points the order does not fix — PROPOSED, for Kai IV&V

* **P-1 ref layout inside `kai-wa/`.** The order fixes the namespace
  `refs/tags/kai-wa/...`. Git cannot hold `kai-wa/X` and `kai-wa/X/y` at
  once, so WA records and events are split into `wa/` and `ev/`.
* **P-2 terminal-event ordering.** An event carries
  `admission_sequence_at_issue`; it dominates admissions of its WA with
  `sequence >` that value. This orders HOLD against admission without
  trusting any clock. The alternative — terminal events retroactively
  invalidating earlier admissions — would make the programme baseline
  regress when a HOLD is issued.
* **P-3 fail closed on namespace pollution.** Any unverifiable record or
  look-alike inside an authoritative directory makes the derivation
  `REFUSE`, not "ignore and continue". Before rulesets exist anyone with
  push access can pollute the namespace; failing closed makes that
  visible at the cost of denial of service. Kai may prefer "exclude and
  report".
* **P-4 SSH signatures** not implemented (see §5).
* **P-5 expiry clock.** For admission, WA expiry is evaluated at the
  admission tag's **signed tagger time**; for current eligibility, at an
  explicit `--now` input. Wall-clock never enters admission.
* **P-6 single use.** A WA backs at most one admission (replay).
* **P-7 staleness.** For `sequence > 1`, the WA's `baseline_commit` must
  equal the previous admission's `candidate_commit`. At genesis it is
  unconstrained.
* **P-8 ancestry.** The candidate must descend from the WA baseline
  (`git merge-base --is-ancestor`); unavailable history → UNKNOWN/REFUSE.
* **P-9 `control_level`.** Required, pattern `^C[0-9]$`, but its values'
  semantics are not defined in the order, so the verifier enforces
  presence and shape only and acts on no level.
* **P-10 IV&V evidence form.** A committed JSON file conforming to
  `schemas/ivv_evidence.v1.schema.json`, referenced by
  `{kind: git-blob, commit, path}`.

## 10. What this leg does not do

No live ruleset change. No admission tag. No Work Authority tag. No
pinned production key (the committed authority configuration pins none,
so every authority check refuses with `UNKNOWN`). No claim that CAI is
ACTIVE. Activation is order §26, after Kai IV&V.
