# HOUSE-IN-ORDER-H2-CLASSIFIER v1.0 — execution record

**Separate from `HOUSE-IN-ORDER-CENSUS-INSTRUMENT v1.0`, which is
untouched.** This package consumes that frozen instrument's output.

    H1 SUBJECT  d8aac4d49e6ba997e3eb38062c0917186ee3f197
    TREE        3abc9e9d8ca11966a6f996d5f0af68072ee5b117

## Exact commands

    # Pass A — evidence only, no roles
    cd <subject-checkout>/kai-pm/house_in_order_instrument
    python3 <pkg>/pass_a.py

    # Capability contract, then Pass B
    python3 <pkg>/run_h2.py <subject-checkout> <full-history-repo> <pass_a-output.json>

## Runtime identity

    python3 3.11 (CPython) — stdlib only, no third-party imports

## Sources

* tree source    — exact-tree checkout of the H1 subject
* history source — full-history repository, positively demonstrated
  non-degenerate (README.md = 122 at the subject). **The exact-tree
  checkout is NOT an admissible history source**: it is depth-1 and
  returns 1 for every path.

## Declared limitations

* `HISTORY_BOUNDARY` — the history source is shallow-marked, boundary
  `d6e5d8cf` (2026-08-05, the import). Commit counts are **since that
  boundary**, not absolute.
* 243 claim sentences abstained as `AMBIGUOUS_SUBJECT`.
* `AUTHORITY` is `UNKNOWN` for all 272 **by construction** — H2 verifies
  no claim. Authority is earned at H3.
