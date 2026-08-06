# Tech debt: 321 × E402 — the import layout, not the imports

**Status:** open, **non-blocking**. Nothing here breaks. No `# noqa` was
added and none should be.
**Opened:** 2026-08-06, during the lint sweep.
**Owner:** unassigned — needs a decision about repository layout, not a
cleanup pass.
**Blocking gate:** none. `flake8 --select=E9,F63,F7,F82` is at 0 and
stays there.

---

## What E402 says, and what it means here

`E402 module level import not at top of file`. Flake8 reports it when an
import follows a statement. The lint sweep left it alone deliberately;
this is the audit the operator asked for before deciding.

**All 321, classified — not a sample:**

| kind | count | files |
|---|---:|---:|
| `sys.path.insert(...)` before the import | **305** | 103 |
| `importlib` module loading | 8 | 4 |
| conditional / `try:` import | 5 | 3 |
| `os.environ[...]` set before the import | 2 | 1 |
| other (`conftest.py`) | 1 | 1 |
| **total** | **321** | **110** |

**Zero of the 321 is a misplaced import.** Every one is load-bearing:
move it up and the file stops working. A bulk `# noqa: E402` would add
321 lines of noise asserting something already true, and "fixing" them
by reordering would break 110 files.

## Why 305 of them exist

Service directories are **hyphenated**: `memu-core`, `trust-ledger`,
`tool-gate`, `broker-bridge`, `document-parser`. A hyphen is not a legal
Python identifier, so none of them can be imported as a package. The
only way to reach `memu-core/app.py` is to put its directory on
`sys.path` first and import the bare module name:

```python
sys.path.insert(0, str(ROOT / "trust-ledger"))
from ledger import FileLedger    # noqa: E402 — unavoidable, see above
```

That is also why bare module names collide (`app`, `introspect_app`) and
why `scripts/module_stubs.py` and the isolation plugin exist at all —
see `DECISIONS.md` on the three occasions that collision cost a day.

So E402 is not a lint problem. It is one visible symptom of a layout
decision: **hyphenated service directories that cannot be packages.**

## The options, when someone gets to it

1. **Leave it.** Cost: 321 lint lines that must be excluded by rule
   rather than read, and the module-name collisions stay. Free today.
2. **Underscore the directories** (`memu_core/`, `trust_ledger/`).
   Imports become ordinary; the collisions go. Cost: every Dockerfile,
   compose `build:` and `COPY`, and every `sys.path` line changes at
   once. Mechanical but wide, and it touches the deployment surface.
3. **Add a thin importable package** (`kai/services/memu_core.py`)
   re-exporting each service. Imports become ordinary for callers; the
   directories stay put. Cost: one more indirection to keep honest, and
   `check_dockerfile_coverage`-style drift between the package and the
   tree.

Option 2 is the real fix and the one worth doing before the tree grows
again. It is a scheduled change with a green CI on both sides of it, not
something to slip into a lint commit.

## What was decided now

- No `# noqa: E402` added anywhere. A suppression comment on a line that
  is correct teaches readers the rule is wrong rather than the code.
- `setup.cfg` / `.flake8` unchanged — E402 is **not** globally ignored.
  Hiding it would remove the only signal that this layout question is
  still open.
- This file is the record. Nothing in CI fails on it.
