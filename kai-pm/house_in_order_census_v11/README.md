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

**Subject binding.** `run_census.py` materialises the ref with
`git archive` and verifies the result against `git ls-tree` of the
original repository *before measuring anything*. v1.0 listed files from
git but read their bytes from the working tree, so a dirty checkout
silently mixed two subjects. If materialisation does not match, the run
aborts and measures nothing (R11).

---

## Results

### Qualification — 39 declared values, 0 findings

All 39 pass legs 1–3. The denominator is derived by scanning the package
for modules exporting `ALPHABETS` (R5); there is no list of values kept
beside the thing being checked.

Calibration: **173 assertions, 0 failures** across three suites —
`cal_docgraph` 127, `cal_opscan` 12, `cal_claims` 34 — including 60
generated crossings and 7 metamorphic groups.

Portable reproduction is proven **by execution, not by `sha256sum`**
(D334): the package was copied to an unrelated directory, where its
manifest verified 14/14, all suites and `qualify.py` exited 0, and a
World A census reproduced every measure of the in-tree run identically.
That check earned its place immediately — it caught `census-worldA.json`
having been generated *before* the seven-reads correction, so the
manifest was hashing stale evidence.

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

### World B — current tree

```
documents 274   edges 969
raw 1464 = rejected 272 + admitted 1192 = sum(dispositions)
claims: NO_PROVEN_WRITER 269 · PROVEN_WRITE_RELATION 5
```

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

### The seven reclassified reads — a regression this build introduced and fixed

`(ROOT / "README.md").read_text()` has an unresolvable **prefix** and a
fixed final component ending `.md`. An intermediate version of this
build folded the AST dynamic-expression flag into the disposition test,
which filed all seven under `UNRESOLVED_RELEVANCE` — asserting we could
not tell whether the target was a document, when the `.md` is right
there. That is a MISBINDING, and it silently erased seven real read
relations.

**Relevance and target are separate questions.** They are now separated:
relevance is proven (`.md`), the target is not, so the disposition is
`UNRESOLVED_TARGET`.

This is deliberately more conservative than v1.0, which resolved such
paths by matching the literal suffix against the tracked tree. That is
the D332 shape: `str(tmp) + "/SOUL.md"` has identical structure and
denotes a temporary file, not the repository document. Same shape,
different truth — so static evidence cannot decide it, and the honest
answer is that the target is unproven. Calibrated as a boundary pair.

### Leg 4 — zero-applicability on both subjects

`READ_AND_WRITE` · `URI_SYNTAX` · `NO_WRITER_WITHIN_ANALYZED_SCOPE` ·
`BASENAME_AMBIGUOUS` · `BROKEN_LINK` · op mode `RW`.

**These are not defects.** They are restrictions: none may support a
downstream claim about these subjects. In particular
`NO_WRITER_WITHIN_ANALYZED_SCOPE` remains at **0**, which is D341 F4
holding under v1.1 exactly as Kai ruled it would — the closure rules
were not relaxed by one inch to manufacture it.

### Changes proven sound but INERT on this subject

The URI and absolute corrections (5 and 6) are proven by boundary-paired
fixtures and change **nothing** on either subject: no absolute path in
the tree has a suffix matching any tracked document, and `URI_SYNTAX`
has zero applicability. They prevent a future false exclusion; they
repair no present number. Recorded here so the distinction is not lost.

---

## Files

| file | role |
|---|---|
| `opscan.py` | candidate extraction and admission accounting |
| `claims.py` | dispositions, exclusion witnesses, scoped claims |
| `docgraph.py` | document reference graph, kinds and edge contexts |
| `qualify.py` | four-leg qualification gate |
| `caltrace.py` | calibration trace harness (legs 2 and 3) |
| `cal_docgraph.py` `cal_opscan.py` `cal_claims.py` | calibration suites |
| `run_census.py` | portable subject-bound runner |
| `compare_v10_v11.py` | v1.0 ↔ v1.1 reconciliation |
| `census-worldA.json` `census-worldB.json` `compare-v10-v11.json` | run evidence |

## Status

**PROPOSED FOR FREEZE. NOT FROZEN, NOT ACCEPTED.**

Freeze and acceptance are Kai's and Dainius's to grant. H2 classifier
v1.1 is **not authorised** and has not been started.
