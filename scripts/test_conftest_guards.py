"""The global test guards in conftest.py, and proof they still hold.

conftest.py blocks the ML stack during tests. That is not tidiness: on
2026-08-04 GitHub's runner had sentence-transformers installed, so
`memu-core/app.py`'s module-scope

    _st_model = _ST(EMBEDDING_MODEL_NAME)

succeeded into transformers, then torch, then the CUDA bindings, on a box
with no GPU — and pytest died of **SIGSEGV, exit 139**, mid-collection.
Not a failing test: a dead process, with no summary line, no isolation
report, and nothing naming the file that did it.

Developer machines do not have those packages, so the failure existed only
where nobody could reproduce it. The block removes the divergence by making
CI behave the way every local machine already does.

These tests exist because a guard nothing checks is a guard that gets
deleted in a tidy-up, and this one's absence is invisible until it costs
another CI run to a segfault.
"""
from __future__ import annotations

import importlib
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

# Derived from conftest, never retyped. A second copy of this list is a
# second definition, and the one that is not the source of truth drifts —
# which is how fixture dicts in this repo twice went stale against a
# structure that had grown (class D in TEST_WRITING_REVIEW.md).
sys.path.insert(0, str(ROOT))
import conftest  # noqa: E402

BLOCKED = tuple(sorted(conftest._BLOCKED))


@pytest.mark.parametrize("name", BLOCKED)
def test_a_stubbed_dependency_is_not_importable(name: str):
    """Whether or not it is installed, a test may not import it.

    Every name here is stubbed somewhere in scripts/test_*.py and absent on
    developer machines, so the suite has only ever run the stubbed branch.
    Blocking it means CI runs that branch too, instead of silently taking
    one nobody has tested.
    """
    # Ask the finder, not `import_module`. An earlier suite that left a
    # stub in sys.modules — test_broker_bridge_yfinance does exactly that,
    # and it is declared in the isolation baseline — short-circuits the
    # import before any finder is consulted, so this asserted "DID NOT
    # RAISE" about a block that was working perfectly. Order-dependence in
    # the test written to prove determinism.
    finder = next((f for f in sys.meta_path
                   if type(f).__name__ == "_BlockHeavyML"), None)
    assert finder is not None, "the block is not installed"
    with pytest.raises(ModuleNotFoundError) as excinfo:
        finder.find_spec(name)
    assert "blocked during tests" in str(excinfo.value), str(excinfo.value)


def test_the_block_covers_the_known_divergences():
    """The two that actually bit, and the datastore clients."""
    for name in ("cv2", "sentence_transformers", "torch", "psycopg2"):
        assert name in conftest._BLOCKED, f"{name} must stay blocked"


def test_locally_installed_packages_are_not_blocked():
    """Blocking these would change the local answer, not fix a divergence."""
    for name in ("psutil", "redis", "docx", "feedparser"):
        assert name not in conftest._BLOCKED, (
            f"{name} is installed on developer machines; the suite exercises "
            f"it for real and blocking it would remove coverage")


def test_the_block_names_its_escape_hatch():
    """A refusal that does not say how to proceed is a dead end."""
    finder = next(f for f in sys.meta_path
                  if type(f).__name__ == "_BlockHeavyML")
    with pytest.raises(ModuleNotFoundError) as excinfo:
        finder.find_spec("torch")
    assert "KAI_TESTS_USE_REAL_ML" in str(excinfo.value)


def test_a_submodule_is_blocked_too():
    """`import torch.nn` must not slip past a root-name check."""
    finder = next(f for f in sys.meta_path
                  if type(f).__name__ == "_BlockHeavyML")
    with pytest.raises(ModuleNotFoundError):
        finder.find_spec("torch.nn")


def test_the_block_can_be_lifted():
    """Proof the guard is a switch, not a wall — run in a clean process.

    Without this, `test_the_ml_stack_is_not_importable` would pass just as
    happily if the block were unconditional, and the escape hatch would be
    a comment rather than a mechanism.
    """
    script = (
        "import sys, os\n"
        "sys.path.insert(0, %r)\n" % str(ROOT) +
        "import conftest\n"
        "print('blocked' if any(type(f).__name__ == '_BlockHeavyML'\n"
        "                       for f in sys.meta_path) else 'open')\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True,
        cwd=str(ROOT), env={**os.environ, "KAI_TESTS_USE_REAL_ML": "true"},
        timeout=120,
    )
    assert result.stdout.strip() == "open", result.stdout + result.stderr


def test_memu_core_still_declares_a_module_scope_model_load():
    """The block is a test decision; the production property stays visible.

    If this ever stops matching, memu-core no longer loads a model at
    import — good news, and a reason to revisit the block rather than to
    quietly keep it.
    """
    source = (ROOT / "memu-core" / "app.py").read_text(encoding="utf-8")
    assert "from sentence_transformers import SentenceTransformer" in source
