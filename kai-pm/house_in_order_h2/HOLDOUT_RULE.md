# H2 independent holdout — SELECTION RULE, COMMITTED BEFORE SELECTION

Kai's requirement: a blind real-document holdout, selected **mechanically
from the 272 exact paths without using classifier labels**, by a **fixed
hash rule committed before the selected documents are opened**.

## Rule (fixed here, before any selection is computed)

For each of the 272 tracked Markdown paths at H1 subject
`d8aac4d49e6ba997e3eb38062c0917186ee3f197`:

    key = SHA256("H2-HOLDOUT-v1|" + repository_relative_path)

Sort ascending by `key` as lowercase hex. **Take the first 20.**

## Properties

* Uses only the **path string** — never content, never a classifier
  label, never anything I find interesting.
* Deterministic and independently reproducible by anyone with the 272
  paths.
* The salt `H2-HOLDOUT-v1` is fixed by this commit and must not be
  changed to obtain a different sample. A different sample requires a
  new salt **and a new committed rule**.

## Adjudication order

1. This rule is committed. *(this commit)*
2. The 20 paths are computed and published.
3. **Kai adjudicates them from raw evidence, before seeing any
   classifier verdict.**
4. Orion independently produces the classifier result.
5. The two are compared and disagreements reported.

The holdout **never replaces the 272 denominator**. All 272 are
classified regardless.
