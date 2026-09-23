# CAI v1.0 — consumer inventory (order §16)

**Question:** which code infers accepted / programme state from the
checked-out revision (`HEAD`, `github.sha`, `main`, the current branch)?
Once CAI is active the programme baseline is the admitted chain head,
never any of those.

**Result: A = 0, B = 8 lines, C = 1 consumer (3 lines), not-a-consumer
= 6 lines; 5 documentation lines listed and excluded.** Not "complete":
complete *within the universe and patterns below*, and no further.

## 1. Universe — mechanically defined

* **Subject:** `main` at `194db0a0c13b4d5b322997fc1ceb33bdd21a77bc` —
  896 tracked files (`git ls-tree -r`).
* **Searched (572):** files that execute or configure: `.py .sh .bash
  .yml .yaml .toml .cfg .ini .json .js .mjs .ts .tsx`, any `Makefile*`,
  any `Dockerfile*`, and extension-less files.
* **Excluded (309 docs):** `.md .txt .rst .html .csv` — prose does not
  execute. Their hits are still **listed** (§4), not dropped.
* **Excluded (15 other):** everything else (images, lock files, …).
* **Out of universe by design:** the D379 branch and PR #122 (excluded
  from CAI by the order), and the CAI files themselves (they are the new
  system, not a consumer of the old one).

Reproduce: `python3 kai-pm/change-control/evidence/consumer_inventory_search.py out.json`
(re-run 2026-09-23: byte-identical JSON).

## 2. Patterns, and the calibration that changed them

| id | pattern (applied per line) |
|---|---|
| P1 | `rev-parse` … `HEAD` |
| P2 | `github.sha` / `GITHUB_SHA` |
| P3 | `github.ref`, `ref_name`, `head_ref`, `base_ref` and the env forms |
| P4 | `origin/main`, `refs/heads/main`, `checkout/switch/--branch/-b main`, `branches:` lists naming `main` |
| P5 | `rev-parse --abbrev-ref`, `symbolic-ref`, `branch --show-current` |
| P6 | `git describe` |
| P7 | `git` … `log/diff/show/rev-list/cat-file/ls-tree/archive/diff-tree` … `HEAD` |

Separators accept shell spacing **and** argv-list spelling
(`["rev-parse", "--short", "HEAD"]`).

**The first version was too narrow** — it matched shell spelling only,
returned 6 hits, and missed `git-watcher/app.py:92`, `:98` and
`scripts/sync_docs.py:105`, all argv-list form. Found by a known-positive
check, fixed by widening the separators.

**Reconciliation against a loose superset** (any line containing
`rev-parse`, `HEAD`, `GITHUB_SHA` or `github.sha`): 8 loose lines were
not matched by any tight pattern. Every one is classified below; two of
them (`git-watcher/app.py:134`, `scripts/auto_changelog.py:64`) are real
Git-HEAD consumers the tight patterns still missed, because in argv form
the word `git` is not on the line. Nothing returned was dropped.

Raw: 10 tight hits on 9 distinct lines (P1 3, P4 6, P5 1 —
`git-watcher/app.py:92` matches P1 and P5) + 8 loose = **17 distinct
lines** in **9** files (derived from the hits file; an earlier draft of
this sentence said 7, written rather than counted).

## 3. Classification — every line

A = programme-authority consumer (must eventually resolve CAI admitted
state). B = ordinary build/debug metadata (may stay HEAD-based).
C = uncertain → Kai. — = not a Git-revision consumer.

| file:line | what | class | why |
|---|---|---|---|
| `.github/workflows/core-tests.yml:8` | `push: branches: ["main","claude/**"]` | B | trigger filter: selects which pushes are *tested*; asserts nothing about acceptance |
| `.github/workflows/core-tests.yml:10` | `pull_request: branches: [main]` | B | same |
| `.github/workflows/python-app.yml:10` | push trigger naming `main` | B | same |
| `.github/workflows/python-app.yml:12` | PR trigger naming `main` | B | same |
| `.github/workflows/unified-hunter.yml:15` | push trigger naming `main` | B | same |
| `.github/workflows/unified-hunter.yml:17` | PR trigger naming `main` | B | same |
| `scripts/sync_docs.py:105` | `rev-parse --short HEAD` | B | collected into `metrics["commit"]` and **never read**: "commit" occurs exactly twice in the file, the definition and the collection |
| `scripts/auto_changelog.py:64` | `git log {ref}..HEAD` | B | changelog text from checked-out history (`Makefile:960`); authors prose, asserts no acceptance |
| `git-watcher/app.py:92` | `rev-parse --abbrev-ref HEAD` → `branch` | **C** | see below |
| `git-watcher/app.py:98` | `rev-parse --short HEAD` → `commit_hash` | **C** | see below |
| `git-watcher/app.py:134` | `rev-list … HEAD...@{upstream}` → ahead/behind | **C** | see below |
| `git-watcher/app.py:86` | `rev-parse --git-dir` | — | repository-existence probe; reads no revision |
| `common/resilience.py:77` | `{"GET","HEAD",…}` | — | HTTP verb |
| `scripts/ci/compose_probe.py:233` | `('GET','HEAD')` | — | HTTP verb |
| `scripts/security/check_image_tags.py:124` | "HEAD the manifest" | — | comment about an HTTP verb |
| `scripts/security/check_image_tags.py:174` | `method="HEAD"` | — | HTTP verb |
| `scripts/sync_docs.py:102` | docstring "short hash of HEAD" | — | docstring |

**Totals:** A 0 · B 8 · C 3 lines (1 consumer) · — 6 · = 17.

### The one C: `git-watcher`

`git-watcher` reports each watched repository's **physical** checkout
state — branch, short commit, dirty, ahead/behind — as JSON "for Kai's
situational awareness". Its consumers are reasoning components:
`agentic/app.py` (`GIT_WATCHER_URL`, `git_dirty_count`), `cortex/app.py`,
`common/perception_spine/shadow.py` (`/summary`), the actuator registry
and the service watchdog.

The code reports physical state and labels it as such. Whether any agent
**treats** "the current commit" as the approved or deployed programme
state cannot be decided from this code: that is a property of how the
agents reason, and it is exactly the confusion CAI exists to remove.
**Returned to Kai.** Order §26 item 15 requires zero unresolved
consequential consumers before activation; this is the one.

## 4. Documentation hits — excluded from the universe, listed

| file:line | text |
|---|---|
| `SESSION_BACKLOG.md:196` | "Merge branch 'origin/main' into copilot/fix-ci-on-pr-60" |
| `kai-pm/DECISIONS.md:1404` | "Merged `origin/main` (`2b17d5e`) into the feature branch …" |
| `kai-pm/ORION_FIELD_NOTES.md:148` | "`git show HEAD:file > file`, run, count, restore." |
| `kai-pm/REALITY_CHECK_2026-06-18.md:60` | "`git diff origin/main..claude/project-rework-plan-pgvp35 --stat`" |
| `kai-pm/WAYPOINTS.md:35` | "`git diff <sha>..HEAD --stat`" |

Historical narrative and operator how-to; none executes.

## 5. Limits of this inventory

* Pattern-based. Two further routes were then **checked**, not left as a
  caveat: reading Git state as files (`.git/HEAD`, `.git/refs`,
  `ORIG_HEAD`, `FETCH_HEAD`) and Git libraries (`import git`,
  `from git import`, GitPython, pygit2, dulwich, `git.Repo(`). Universe:
  the 587 tracked files outside the documentation extensions, binaries
  skipped by `git grep -I` — close to, but NOT the same as, the 572 in §1:
  **0 matches** (`git grep` exit 1, measured without a
  pipe; a first attempt piped through `cut` and printed `rc=0`, which was
  `cut`'s status, not the search's). The identical command on
  `rev-parse` returned 4 lines in 2 files, rc 0, as a known-positive. A
  route matching none of these is still outside what was measured.
* Runtime behaviour of the agents (§3, C) is not observable statically.
* One commit, one branch: `194db0a`.
