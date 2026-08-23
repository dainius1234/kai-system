# Phase 2 — Documentation Sync & Drift-Control Redesign

> **STATUS: PLANNING / CONTROL-DESIGN FINDING — NO IMPLEMENTATION AUTHORISED BY THIS FILE.**
>
> Operator reminder: the repository was intended to keep front-page/project-state information synchronized after changes. Current inspection shows the present mechanism provides only partial metric regeneration plus stale-check enforcement. It does not keep the whole README/architecture truth synchronized.

## 1. What exists now

Current `scripts/sync_docs.py`:

- scans test functions/files;
- counts `make test-core` dependencies;
- counts Python LOC;
- counts compose services/files;
- derives milestone count from README content;
- patches only the README `Project Status` table;
- patches only test-target/test-count rows in `docs/PROJECT_BACKLOG.md`;
- supports `--check` to fail when those generated values drift.

Current enforcement:

- `.github/workflows/core-tests.yml` runs `make check-docs`;
- `make policy-check` also invokes `check-docs`;
- repository instructions tell agents/developers to manually run `make sync-docs` and then `make check-docs` after major code/test/service changes.

Therefore the system is currently **automatic drift detection for a narrow generated region**, not full automatic documentation synchronization after every relevant change.

## 2. Date behaviour

The README Project Status date is refreshed when `sync_docs.py` runs, but the current check deliberately ignores date age.

Reason recorded in the script: an earlier implementation treated the wall-clock date as drift, causing CI to turn red at midnight even when the repository tree had not changed. That created a noisy gate people would learn to ignore.

That correction was sensible, but it also means the displayed date is not presently a strong currentness guarantee.

Future design should avoid returning to a wall-clock freshness gate.

Better semantics should bind status to **measured repository identity**, e.g. exact commit/tree/generator revision and generation timestamp, so a reader can tell what subject the status describes without the repository becoming stale merely because a day passed.

## 3. Current control gap

The README currently duplicates volatile truth outside the generated region, including examples such as:

- badge metrics;
- Quick Reference test counts;
- service/test/LOC statements in prose;
- capability status language;
- architecture descriptions;
- hardware/deployment facts;
- diagrams/graphics.

These can contradict the generated Project Status table while `check-docs` still passes.

Thus:

> **A GREEN DOCS-SYNC GATE DOES NOT CURRENTLY MEAN THE README IS CURRENT.**

It means only that the small population owned by `sync_docs.py` matches its scanners.

This is a denominator/coverage problem, not merely an operator-discipline problem.

## 4. Production-grade target

Phase 2 should replace the current mixed manual/generated model with an explicit documentation-truth architecture.

### 4.1 Single machine source for volatile facts

Facts that can be mechanically derived should have one source and one generator, for example:

- service/profile counts;
- test suites/targets/assertions;
- compose/profile inventory;
- build/runtime versions;
- current measured commit/tree;
- generated status labels where evidence supports them;
- relevant hardware/runtime profile metadata.

Badges, status tables and Quick Reference must consume the same generated values rather than each keeping a copy.

### 4.2 Generated regions are explicit

README sections owned by automation should carry stable markers/IDs so the generator knows exactly what it owns.

The generator must refuse if an expected generated region is missing or duplicated rather than silently certifying nothing.

### 4.3 Semantic architecture is not blindly auto-generated

Not every sentence should be machine-written.

Architecture, capability descriptions and limitations require qualified evidence and design review. They should use a claim ledger / source binding rather than naive text regeneration.

Automation should detect when referenced claims become stale or their evidence subject changes, then fail/refuse until a reviewed update is made.

### 4.4 Local refresh + CI enforcement

Preferred model:

`CHANGE`
→ deterministic local docs/status refresh as part of the controlled commit/merge path
→ generated diff visible for review
→ `check-docs`/claim checks prove the committed tree is synchronized
→ CI verifies but does not silently mutate the repository.

Do not rely on an agent remembering to run `make sync-docs`.

Possible future enforcement points to evaluate:

- governed commit wrapper;
- pre-commit/pre-push hook installed by project tooling;
- merge-gate dependency;
- CI check that regenerates in a temporary tree and fails on any diff;
- generated status artefact consumed by README rendering.

The exact mechanism must be chosen during Phase 2 after checking developer/CI portability.

### 4.5 Currentness stamp

Prefer a subject-bound stamp such as:

`GENERATED_FROM_COMMIT`
`GENERATED_FROM_TREE`
`GENERATOR_REVISION`
`GENERATED_AT`

The date is informative; commit/tree identity is the correctness binding.

A new day with no tree change must not create a failure.

### 4.6 Coverage / emittability requirement

The docs control itself must declare its population:

- which files it checks;
- which README regions it owns;
- which volatile claims it derives;
- which claims remain manually reviewed;
- which linked docs are checked for currentness/authority.

A future gate must print its denominator and demonstrate known-positive drift detection.

## 5. Relationship to House-in-Order lessons

This failure shape reinforces existing engineering doctrine:

- **present != enforced** — `sync_docs.py` existing does not mean updates happen;
- **binding != enforcement** — instructions saying "always run sync-docs" are not a mechanism;
- **non-detection != absence** — a green narrow check cannot prove no README drift outside its population;
- **denominator must be closed** — the gate must state exactly what documentation truth it covers;
- **context must not become semantics** — generated counts should not derive authority/status from prose that they themselves later rewrite;
- **measurement identity matters** — dates alone are weaker than exact commit/tree binding;
- **generator output must be reproducible** — CI should be able to regenerate and compare independently.

These rules should also be considered for future A4 self-diagnosis: Kai should be able to detect when its own documentation/architecture representation no longer agrees with its measured implementation.

## 6. Phase-2 obligation

During Kingsman professionalisation:

1. inventory all current generated/manual README facts;
2. recover why the old sync mechanism changed over time;
3. define a closed population of machine-derived facts;
4. remove duplicate manually maintained copies;
5. design subject-bound currentness rather than wall-clock freshness;
6. integrate deterministic refresh into the controlled change/commit path;
7. keep CI as independent verification rather than silent mutation;
8. add known-positive/known-negative tests for docs drift;
9. connect architecture/capability prose to an evidence/claim ledger;
10. make README, architecture docs and graphics update from one qualified truth model where practical.

## 7. Plain-language conclusion

The original idea was right: after the system changes, its public/project description should not be left for somebody to remember manually.

What exists today is only a partial version of that idea. It updates a small metrics table when explicitly asked and blocks some stale numbers, but it does not own the whole README and it does not truly auto-sync every relevant change.

Phase 2 should finish the idea properly: **one truth source, deterministic refresh, visible generated diff, independent stale check, exact commit/tree binding, and no duplicate numbers scattered around the front page.**
