# HOUSE-IN-ORDER-CENSUS-INSTRUMENT v1.1

Built under Dainius's authorisation of 2026-08-23, exactly within Kai's
D341 ruling. **Census v1.0 is untouched and remains immutable historical
evidence with known defects** (D341 F1–F4).

This package is not a patch of v1.0. It is a separate instrument, and
every claim it makes carries the scope it was measured in.

---

## What changed, and why each change exists

| # | change | earned by |
|---|---|---|
| 1 | `NO_WRITER` → **`NO_WRITER_WITHIN_ANALYZED_SCOPE`**, closure rules unchanged | D341 F4 + Kai's ruling: the state is logically valid, the *name* overstated its proof scope |
| 2 | **Four-leg qualification gate** (`qualify.py`) | D341 F1/F2 and H2's `REFERENCE` were one defect found three times by hand |
| 3 | Extraction artefacts **rejected before admission**, never used as exclusion witnesses | Kai: "that distinction cleans the denominator rather than merely explaining pollution" |
| 4 | Shell redirection recognised **only in shell context** (`run:` blocks, not whole YAML) | D341 preflight: `expr: vram_percent > 90` was admitted as a write |
| 5 | **URI syntax is not remote semantics** | Kai: `open("https://x.md","w")` performs no HTTP request |
| 6 | **Absolute syntax is not "outside the repository"** (R6, same class as 5) | `/home/user/repo/data/SOUL.md` *is* `data/SOUL.md` |
| 7 | `POSSIBLE_WRITER` and edge-context `OTHER` removed from declared alphabets | D341 F1/F2 — no decorative states |
| 8 | **Denominator reconciliation** emitted and asserted | Kai: report raw / rejected / admitted / sum(dispositions) and prove they reconcile |
| 9 | **Relevance and target separated** | a dynamic prefix does not erase `.md` evidence (see below) |
| 10 | **Portable and subject-bound** | D340: H2's `pass_a.py` hard-coded an absolute repo path and a session `/tmp` path |
| 11 | **RESOLVE-ONCE subject binding**, and `materialise()` refuses anything that is not an immutable object id | an execution-proven silent misbinding: a symbolic ref re-dereferenced after resolution let the census measure one commit while stamping another, with reconciliation passing |

### The four legs (Kai's D341 formulation)

1. `IMPLEMENTATION_EMITTABLE` — a real code path assigns the value.
2. `FIXTURE_REACHABLE` — a fixture actually reached it **at runtime**.
3. `CALIBRATION_DISCRIMINATING` — a **passing** assertion is about it.
4. `SUBJECT_POPULATION_APPLICABILITY` — how often it occurs on the exact
   real subject.

**Leg 4 is mandatory reporting, not pass/fail.** A legitimate state may
simply be absent from a corpus; failing it would turn a corpus accident
into an ontology. Its binding consequence: *a downstream claim about a
subject may not rely on a state whose applicability on that subject is
zero.*

---

## Running it

Nothing is hard-coded. The package resolves its own directory from
`__file__`; repository, subject ref and output path are parameters.

```sh
cd kai-pm/house_in_order_census_v11

# instrument qualification alone (subject-independent, legs 1-3)
python3 qualify.py

# a census against any repository at any ref
python3 run_census.py --repo /path/to/repo --ref <sha> --out result.json

# v1.0 <-> v1.1 reconciliation on one identical subject
python3 compare_v10_v11.py --repo /path/to/repo --ref <sha> \
    --frozen-v10 /path/to/kai-pm/house_in_order_instrument
```

Exit status is non-zero when qualification finds anything, so it can be
chained with `&&` (R3).

**Subject binding — the resolve-once invariant.** The supplied ref is
resolved to an **immutable commit id exactly once**. Tree derivation,
`ls-tree`, `git archive`, reconciliation and stamping all use that id;
nothing dereferences the symbolic ref again, and `materialise()`
**refuses** any argument that is not a 40-hex object id. The
materialisation is then verified against `git ls-tree` of the original
repository before anything is measured, and a mismatch aborts the run
(R11).

This is a repair, not a precaution. The previous version passed the
symbolic ref into `materialise()`, which re-dereferenced it. If a branch
moved in between, **both** sides of the expect/got reconciliation saw
the new commit — so they agreed, the run reported `reconciles: True`,
and the result was stamped with the old commit while containing the new
one's content. A silent MISBINDING that presents as a clean, fully
populated table, invisible to the very control meant to catch it.
Demonstrated by execution with the movement forced at a controlled
boundary, and now held by `cal_subject_binding.py`.

A symbolic ref is a **pointer**. Only an object id is an **identity**.

`--ref` still accepts a branch name for convenience; the resolution
happens once, up front, and both the invocation ref and the resolved
commit are recorded so a reader can see which was used.

---

## Results

### Qualification — 39 declared values, 0 findings

All 39 pass legs 1–3. The denominator is derived by scanning the package
for modules exporting `ALPHABETS` (R5); there is no list of values kept
beside the thing being checked.

Calibration: **182 assertions, 0 failures** across four suites —
`cal_docgraph` 127, `cal_opscan` 12, `cal_claims` 34,
`cal_subject_binding` 9 — including 60 generated crossings and 7
metamorphic groups. The suites are discovered by scanning the package
for `cal_*.py` rather than from a hand-written list, so a new suite
cannot be added and silently left unexecuted (R5).

Portable reproduction is proven **by execution, not by `sha256sum`**
(D334): the package was copied to an unrelated directory, where its
manifest verified 19/19, all suites and `qualify.py` exited 0, and a
World A census reproduced every measure of the in-tree run identically.
That check earned its place immediately — it caught `census-worldA.json`
having been generated *before* the seven-reads correction, so the
manifest was hashing stale evidence: a manifest that verified perfectly
over content that was already wrong.

**Proof the gate can fail.** On its first run it reported
`EXCLUDED_FROM_T :: NOT DISCRIMINATED` — reached six times, but every
assertion about it had been credited to the *witness* alphabet instead.
The gap was real and was closed by adding an assertion whose subject is
the disposition itself.

### World A — original H1 subject `d8aac4d4`, tree `3abc9e9d`

```
documents 272   edges 944
raw_candidate_matches         1458
  rejected EXTRACTION_ARTEFACT    117
  rejected QUOTED_STRING_CONTENT   87
  rejected COMMENT_CONTEXT         38
  rejected NOT_SHELL_CONTEXT       26
  rejected ARROW_OPERATOR           4
rejected_non_operations        272
admitted_candidate_operations 1186
sum(dispositions)             1186     reconciles: True
claims: NO_PROVEN_WRITER 267 · PROVEN_WRITE_RELATION 5
```

### World B — subject `0dcd228`, tree `89960687`

```
documents 330   edges 1389
admitted 1210 = sum(dispositions)          reconciles: True
claims: NO_PROVEN_WRITER 325 · PROVEN_WRITE_RELATION 5
```

Invoked with an **immutable commit SHA**, not with `HEAD`. The
population grew from 275 to 330 because a parallel workstream added 55
documents to the measured directory; that is a real change in the
subject, not drift in the instrument.

**World B is bound to the commit named in its own artefact, not to
"now".** A current-tree census necessarily predates the commit that
banks it, so its subject can never be the commit containing it. World A
is invariant and is the comparison subject for that reason; World B
exists only to report subject-population applicability, and its
`subject_commit` must be read with it. Re-running it on a later tree
produces a different, equally valid subject — which is why the number
travels with its commit or not at all.

### v1.0 ↔ v1.1 reconciliation on the identical World A subject

| measure | v1.0 | v1.1 | delta |
|---|---|---|---|
| documents | 272 | 272 | **identical** |
| edges | 944 | 944 | **identical** |
| operations | 1343 | 1186 | −157 |
| `RESOLVED_WRITE` | 5 | 5 | **0** |
| `RESOLVED_READ` | 15 | 8 | −7 |
| `UNRESOLVED_TARGET` | 30 | 37 | +7 |
| `UNRESOLVED_RELEVANCE` | 865 | 715 | −150 |
| `RESOLVED_NON_DOCUMENT_TARGET` | 428 | 421 | −7 |
| documents whose claim changed | — | — | **0** |

Every delta is explained, and they sum to −157:

* **−150 / −7** are non-operations rejected before admission — comments,
  quoted strings, prose arrows, comparison operators in non-shell YAML.
* **−7 `RESOLVED_READ` / +7 `UNRESOLVED_TARGET`** are the same seven
  operations, reclassified. See below.
* **Write-side evidence is byte-identical: the same 5 proven write
  relations, no erasure and no fabrication.** Every claim rests on this
  set, which is why it is the load-bearing comparison.

Edges being identical proves change 7 was declaration-only: removing a
structurally unreachable context value changed no output.

### The seven reclassified reads — `READ_TARGET_RECLASSIFIED_CONSERVATIVELY`

`(ROOT / "README.md").read_text()` has an unresolvable **prefix** and a
fixed final component ending `.md`.

**These are not seven read relations that were erased.** Every one of
the seven operations is preserved in full, and each is listed by file
and line in `applicability-world*.json` and `compare-v10-v11.json`. What
was withdrawn is the *unproven claim about which tracked document each
one touches*:

* document **relevance** — PROVEN, by the fixed `.md` component;
* exact document **target** — UNPROVEN, because of the dynamic prefix.

Hence `UNRESOLVED_TARGET`. v1.0 resolved such paths by matching the
literal suffix against the tracked tree, which asserts a target it has
not established. That is the D332 shape: `str(tmp) + "/SOUL.md"` is
structurally identical and denotes a temporary file. Same shape,
different truth — static evidence cannot separate them, so the honest
answer is that the target is unproven. Calibrated as a boundary pair.

**A regression this build introduced, and fixed.** An intermediate
version folded the AST dynamic-expression flag into the disposition
test, filing all seven under `UNRESOLVED_RELEVANCE` — which asserts we
cannot tell whether the target is a document *at all*, when the `.md` is
right there. That version really did erase the relevance, and every
calibration suite stayed green while it did. The v1.0 ↔ v1.1
reconciliation caught it, not the suites. **Relevance and target are
separate questions**, and conflating them is a misbinding.

### Leg 4 — zero-applicability on both subjects

`READ_AND_WRITE` · `URI_SYNTAX` · `NO_WRITER_WITHIN_ANALYZED_SCOPE` ·
`BASENAME_AMBIGUOUS` · `BROKEN_LINK` · op mode `RW`.

**These are not defects.** They are restrictions: none may support a
downstream claim about these subjects. In particular
`NO_WRITER_WITHIN_ANALYZED_SCOPE` remains at **0**, which is D341 F4
holding under v1.1 exactly as Kai ruled it would — the closure rules
were not relaxed by one inch to manufacture it.

### Applicability travels with the evidence (Kai's D342 freeze condition)

A leg-4 restriction that lives only in a qualification report can be
separated from the numbers it restricts: a downstream tool reads
`census-worldA.json` and never reads the report. So every census carries
a **subject applicability record**, bound both ways:

* the full record is a **top-level block inside the census**, so copying
  the file cannot strip it;
* the identical canonical bytes are written as a standalone artefact
  (`applicability-world*.json`), named in `MANIFEST.sha256` and
  referenced from the census by exact SHA-256.

Per declared state it carries `L1_IMPLEMENTATION_EMITTABLE`,
`L2_FIXTURE_REACHABLE`, `L3_CALIBRATION_DISCRIMINATING`,
`L4_SUBJECT_POPULATION_COUNT`, **`DOWNSTREAM_USABLE_ON_THIS_SUBJECT`**
and `DOWNSTREAM_RESTRICTION_REASON`. The binding rule is stated in the
record itself:

> A downstream claim about this subject MAY NOT rely on any state whose
> `DOWNSTREAM_USABLE_ON_THIS_SUBJECT` is false.

On both subjects: **39 declared states, 33 usable, 6 restricted.** It is
not duplicated into document rows.

### Repair impact is orthogonal to semantics

A repair with no current-subject effect must not become another ontology
value — that mixes two separate things. Repair impact is recorded
separately, and every figure is **measured on the subject**, never a
constant kept beside the code:

| rule | `RULE_STATUS` | `CURRENT_SUBJECT_EFFECT` |
|---|---|---|
| `URI_SYNTAX_NOT_REMOTE_SEMANTICS` | `CORRECTED_AND_QUALIFIED` | `NONE` (0 ops) |
| `ABSOLUTE_PATH_NOT_OUTSIDE_REPOSITORY` | `CORRECTED_AND_QUALIFIED` | `NONE` (0 ops) |
| `READ_TARGET_RECLASSIFIED_CONSERVATIVELY` | `CORRECTED_AND_QUALIFIED` | `OPERATIONS_RECLASSIFIED` (7 ops) |

The first two are **proven sound by boundary-paired fixtures and inert
on this subject**: no absolute path in either tree has a suffix matching
a tracked document, and `URI_SYNTAX` has zero applicability. They
prevent a future false exclusion; they repair no present number. The
measurement is recomputed on every run, so this claim cannot rot.

---

## Files

| file | role |
|---|---|
| `opscan.py` | candidate extraction and admission accounting |
| `claims.py` | dispositions, exclusion witnesses, scoped claims |
| `docgraph.py` | document reference graph, kinds and edge contexts |
| `qualify.py` | four-leg qualification gate |
| `applicability.py` | subject applicability record and its SHA binding |
| `repairs.py` | measured repair impact, orthogonal to semantics |
| `caltrace.py` | calibration trace harness (legs 2 and 3) |
| `cal_docgraph.py` `cal_opscan.py` `cal_claims.py` | calibration suites |
| `run_census.py` | portable subject-bound runner |
| `cal_subject_binding.py` | the moving-symbolic-ref regression fixture |
| `compare_v10_v11.py` | v1.0 ↔ v1.1 reconciliation |
| `census-worldA.json` `census-worldB.json` | census evidence, each embedding its applicability record |
| `applicability-worldA.json` `applicability-worldB.json` | standalone applicability artefacts, SHA-bound to the censuses |
| `compare-v10-v11.json` | reconciliation evidence, carrying the seven reclassified instances |

## Status

**PRE-FREEZE REPAIRED CANDIDATE. NOT FROZEN. NOT FREEZE-READY.**

Freeze eligibility is not a property this package can assert about
itself, and a green manifest does not confer it:

> **PACKAGE INTEGRITY ≠ INSTRUMENT VALIDITY ≠ FREEZE ELIGIBILITY.**

The manifest verifying proves the bytes match. It proves nothing about
whether the instrument may be frozen. That decision is Kai's and
Dainius's. H2 classifier v1.1 is **not authorised** and has not been
started.

The superseded pre-repair candidate identity is
`67071cce7b2fe86aa756e29d0c8efc65ec995161368f63f8892f32a50c006353`. It
remains the historical identity of the package that carried the
subject-binding defect, and it is not the identity of this one.
