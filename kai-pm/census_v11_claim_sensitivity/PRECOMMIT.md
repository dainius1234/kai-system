# CLAIM-SENSITIVITY MUTATIONS — PRECOMMITTED BEFORE EXECUTION

Written and hashed BEFORE any mutant was built or run. Kai's D343 final
review requires the mutant and its expectation to be precommitted; a
prediction written after seeing the result is not a prediction.

SUBJECT   d8aac4d49e6ba997e3eb38062c0917186ee3f197 (World A, invariant)
          tree 3abc9e9d8ca11966a6f996d5f0af68072ee5b117
BASELINE  PROVEN_WRITE_RELATION 5 · NO_PROVEN_WRITER 267
          NO_WRITER_WITHIN_ANALYZED_SCOPE 0
INSTRUMENT Census v1.1, aggregate
          67071cce7b2fe86aa756e29d0c8efc65ec995161368f63f8892f32a50c006353
          UNMODIFIED. Mutations are applied to DISPOSABLE COPIES of the
          subject only. The repository subject and the candidate package
          are not touched.

## MUTATION A — POSITIVE INJECTION

Append to the tracked source file `scripts/sync_docs.py`:

    import pathlib
    pathlib.Path("docs/DEMO.md").write_text("mutation A probe")

`docs/DEMO.md` is a real tracked document in World A and is currently
NO_PROVEN_WRITER with ZERO writers of any kind. `pathlib` is used
literally because a renamed import alias would be treated as a dynamic
expression and the mutation would never reach the branch under test —
that is exactly how two earlier mutation attempts (D332) passed while
proving nothing.

EXPECTED, PREDICTED IN ADVANCE:
  A1  PROVEN_WRITE_RELATION            5 -> 6
  A2  docs/DEMO.md   NO_PROVEN_WRITER -> PROVEN_WRITE_RELATION
  A3  docs/DEMO.md   sources == ["scripts/sync_docs.py"]
  A4  NO other document changes claim  (TARGETED)
  A5  the injected operation is present in the admitted set with
      disposition RESOLVED_WRITE and target docs/DEMO.md (REACHABLE)
  A6  baseline and mutant claim tables differ (DISCRIMINATING)

## MUTATION B — POSITIVE REMOVAL

In `scripts/auto_changelog.py`, neutralise the single write site
proven by the analyser to be the only operation resolving to
CHANGELOG.md:

    line 139:  CHANGELOG.write_text(new_text)
    becomes :  pass  # neutralised for Mutation B

Lines 55 and 112 are READS of the same file and are deliberately left
intact: a read establishes no write relation, so if they were sufficient
to keep the claim alive the claim layer would be wrong.

EXPECTED, PREDICTED IN ADVANCE:
  B1  PROVEN_WRITE_RELATION            5 -> 4
  B2  CHANGELOG.md   PROVEN_WRITE_RELATION -> NO_PROVEN_WRITER
  B3  CHANGELOG.md   must NOT become NO_WRITER_WITHIN_ANALYZED_SCOPE:
      unresolved candidate writes remain, so the space is not closed
  B4  NO other document changes claim  (TARGETED)
  B5  no operation resolving to CHANGELOG.md with mode W remains
      (APPLIED), while the two READ operations survive
  B6  baseline and mutant claim tables differ (DISCRIMINATING)

## FAILURE RULE

Any expectation not met => STOP and return the evidence. No package
change, no reinterpretation of the expectation after the fact.
