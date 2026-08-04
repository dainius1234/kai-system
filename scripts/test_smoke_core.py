"""Smoke-probe tests for scripts/smoke_core.py.

The previous version of this file was three lines that ran
`scripts/smoke_core.py` and returned its exit code. Its comment said

    # When core services are not running, the script should exit nonzero

which is a real property, and one that needs no stack to check — but the
file never asserted it. It propagated the status instead, so it *failed*
whenever the services were down, which is every context except a live
box. It had no make target and pytest collects nothing from a file whose
only code is behind `if __name__ == "__main__"`, so it ran nowhere at
all, and nothing ever pointed that out.

What is testable without a stack is the property that matters: **the
probe reports failure when nothing answers.** A health check that cannot
tell "healthy" from "absent" is the boundary blindness this programme has
been removing everywhere else, and it would be worse here than most —
`smoke_core` is what says the core came up.

The live run against a real stack is `make core-smoke`, and the live
version of this file is `make test-smoke-core-live`.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import smoke_core  # noqa: E402

# Port 1 is privileged and unbound; nothing will ever answer here.
_CLOSED = "http://127.0.0.1:1/health"


def test_check_reports_false_when_nothing_answers():
    """The core assertion: absence must not read as health."""
    assert smoke_core.check(_CLOSED) is False


def test_check_reports_false_for_a_refused_connection():
    assert smoke_core.check("http://127.0.0.1:1/") is False


def test_main_exits_nonzero_with_no_core_services():
    """The property the old comment described and never asserted."""
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "smoke_core.py")],
        capture_output=True, text=True, cwd=str(ROOT), timeout=120,
    )
    assert result.returncode != 0, (
        "smoke_core reported success with no core services running:\n"
        + result.stdout
    )


def test_main_names_the_failure():
    """A nonzero exit that says nothing is not much better than a pass."""
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "smoke_core.py")],
        capture_output=True, text=True, cwd=str(ROOT), timeout=120,
    )
    assert "some core services failed" in result.stdout, result.stdout
