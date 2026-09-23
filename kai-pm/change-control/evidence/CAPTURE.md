# CAI-BOOT-001 leg 1 — evidence capture

**EVIDENCE CLASS: PRODUCER MEASUREMENT.** Orion's code, Orion's runs. The
GitHub-runner results below are the same producer's code in a different
environment — **cross-environment reproduction, not independent
corroboration.** Nothing here admits, closes or activates anything.

## 1. Subject

* branch `claude/cai-v1-bootstrap`, created from exactly `main`
  `194db0a0c13b4d5b322997fc1ceb33bdd21a77bc` (tree `11712eb3…`)
* local capture: HEAD `46c4dc57dfcc2374e47f24fb66d525c24d675faf`, working
  tree **CLEAN**, 2026-09-23T10:35:54Z — `runs/SUMMARY.txt`
* every status is `CompletedProcess.returncode` of the child that wrote
  the file beside it; every output is kept whole (`runs/*.txt`)

## 2. Local capture (`runs/`)

| step | rc | bytes | tally, read from the file |
|---|---|---|---|
| test_cai_authority | 0 | 1,475 | declared 16, executed 16, passed 16, failed 0 |
| test_cai_scope | 0 | 694 | declared 7, executed 7, passed 7, failed 0 |
| test_cai_admission | 0 | 2,073 | declared 23, executed 23, passed 23, failed 0 |
| test_cai_bootstrap | 0 | 1,206 | declared 13, executed 13, passed 13, failed 0 |
| test_cai_mutation | 0 | 3,748 | mutants declared 35, executed 35, killed 35, survived 0 (unmutated control passed first) |
| bootstrap_design | 0 | 282 | 3 rulesets, design invariants PASS, 2 placeholders |
| authority_real_repo | **1** | 290 | REFUSE — `UNKNOWN: no authority key is pinned` (correct: nothing pinned) |
| admission_real_repo | **1** | 338 | REFUSE — unpinned + EMPTY chain (correct) |
| gate_registry | **1** | 1,345 | 40 declared / 49 on disk — 9 I-4 findings, all CAI (§4) |
| test_wiring | 0 | 250 | 107 self-run suites (unchanged from baseline) |

Cases: 16 + 7 + 23 + 13 = **59 declared, 59 executed, 59 passed**.
Of these, 9 (X1–X9) and 13 (B1–B13) are supplementary to the order's
§18 minimum.

## 3. GitHub runner — CAI workflow

Run `35849261312` on `65b9b60`, runner "GitHub Actions 1000003398",
Git 2.55.0: job *Hostile calibration and ruleset design* **success**, all
10 steps success including mutation calibration. Job *Programme baseline
derivation* **skipped** (dispatch-only in leg 1, by design).

## 4. Existing workflows on this branch — every red attributed

Existing workflows trigger on `claude/**`, so they ran on this branch.
Baseline: at `194db0a` (runs of 2026-08-07) Core Tests, Python
application and Unified Hunter all **succeeded**; Policy-as-Code has no
run at that commit.

| run / commit | workflow | failing step | attributable to CAI? | evidence |
|---|---|---|---|---|
| 35849261228 / 65b9b60 (and 35847181740 / 5ccd14d) | Policy-as-Code | Instrumentation invariants (I-4 enforced) | **YES** | 9 CAI modules unregistered; resolved in a throwaway worktree by `../REGISTRY_INTEGRATION.patch` (gate PASS 49/49) |
| 35849261337 / 65b9b60 | Unified Hunter | Unified Hunter suites → `make test-uh` stops at `test-gate-registry` (Makefile:141) | **YES**, same I-4 root | reproduced locally: rc 2, 36,278 bytes; the only `***` line is test-gate-registry. Targets after it did **not run** — unmeasured, not passed. The "counts have risen … synthetic" block in that log is the hygiene survey's own known-positive calibration, not a failure |
| 35849261149 / 65b9b60 | Core Tests | Doc-drift check (README metrics) | **YES** | reproduced: `sync_docs.py --check` rc 1; README test files 208 → 213 (the 5 CAI suites), LOC ~132,913 → ~135,660. `README.md` is outside the authorised surface |
| 35849261226 / 65b9b60 (and 35785607632 / fcccc58) | Python application | Cross-file test isolation (A-05) | **NOT by change population** | `scripts/test_audio_transcribe.py: added 0 -> 1`; also fails on `fcccc58`, which changed only Markdown and JSON under `kai-pm/change-control/`. Root cause UNKNOWN; `main` has not been re-run today, so "pre-existing on main" is NOT established |
| 35785607624 / fcccc58 | Core Tests | Bring up minimal sovereign AI stack | **NOT by change population** | `memu-core-introspect` requested `huggingface.co` at runtime: "Temporary failure in name resolution". Docs-only commit. Root cause UNKNOWN beyond that log |

## 5. Baselines measured before any CAI file existed (`194db0a`)

* `check_gate_registry --gate`: PASS (40 declared / 40 on disk).
* `check_test_wiring`: PASS (107 self-run suites, 76 pytest-collected).
* `make policy-check`: **rc 2** first — `No module named 'pydantic'`
  (rules 7 and 11); environment, not code. After installing
  dependencies the way `policy-checks.yml` does, **rc 0**, full chain.
  Deviations from CI's install: `pip install --upgrade pip` skipped
  (Debian-managed pip cannot uninstall itself); 4 of 49 requirement files
  failed to install (`letta-agent`, `memu-graph`, `perception/vision`,
  `verifier`).

## 6. Environment

git 2.43.0, gpg 2.4.4, Python 3.11.15; `ssh-keygen` **absent** (hence no
SSH-signature path, contract P-4); `docs.github.com` blocked by the
session network policy; GitHub API reachable for this repository only.

## 7. Reproduce

    python3 kai-pm/change-control/evidence/capture.py
    python3 kai-pm/change-control/evidence/consumer_inventory_search.py out.json
