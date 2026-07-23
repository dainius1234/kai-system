"""
Tests for H3: Test Coverage Gate.
Verifies that the coverage gate is wired in CI and in the Makefile.
Run: pytest scripts/test_h3_coverage_gate.py -v
"""
from pathlib import Path
import re

ROOT      = Path(__file__).parent.parent
MAKEFILE  = ROOT / "Makefile"
PYAPP_YML = ROOT / ".github" / "workflows" / "python-app.yml"


def _makefile():
    return MAKEFILE.read_text(encoding="utf-8")


def _pyapp():
    return PYAPP_YML.read_text(encoding="utf-8")


# ── GitHub Actions workflow ────────────────────────────────────────────────────

class TestCIWorkflow:
    def test_pyapp_workflow_exists(self):
        assert PYAPP_YML.exists(), "python-app.yml must exist"

    def test_pytest_cov_installed_in_ci(self):
        assert "pytest-cov" in _pyapp()

    def test_cov_common_measured(self):
        assert "--cov=common" in _pyapp()

    def test_cov_fail_under_in_ci(self):
        assert "--cov-fail-under=" in _pyapp(), (
            "CI must enforce a minimum coverage threshold via --cov-fail-under"
        )

    def test_cov_fail_under_threshold_adequate(self):
        yml = _pyapp()
        match = re.search(r"--cov-fail-under=(\d+)", yml)
        assert match, "--cov-fail-under must include a numeric threshold"
        threshold = int(match.group(1))
        assert threshold >= 60, f"Coverage gate threshold {threshold}% is too low — must be ≥ 60"

    def test_cov_report_term_missing_in_ci(self):
        assert "--cov-report=term-missing" in _pyapp(), (
            "CI must emit per-file missing-line coverage report for actionable output"
        )

    def test_ignores_archive_in_ci(self):
        yml = _pyapp()
        coverage_step = yml[yml.find("--cov=common"):][:300]
        assert "--ignore=_archive" in coverage_step or "--ignore=_archive" in yml

    def test_coverage_step_is_named(self):
        yml = _pyapp()
        assert "coverage gate" in yml.lower() or "with coverage" in yml.lower(), (
            "The coverage step should have a descriptive name"
        )


# ── Makefile coverage target ───────────────────────────────────────────────────

class TestMakefileCoverageTarget:
    def test_coverage_target_exists(self):
        assert "coverage:" in _makefile()

    def _coverage_block(self):
        mk = _makefile()
        start = mk.find("coverage:")
        # next target begins at the next non-indented line after start
        rest = mk[start:]
        end = re.search(r"\n\S", rest[len("coverage:"):])
        return rest[:end.start() + len("coverage:")] if end else rest[:600]

    def test_makefile_coverage_measures_common(self):
        assert "--cov=common" in self._coverage_block()

    def test_makefile_coverage_has_fail_under(self):
        assert "--cov-fail-under=" in self._coverage_block(), (
            "make coverage must enforce --cov-fail-under to match CI gate"
        )

    def test_makefile_coverage_threshold_consistent(self):
        block = self._coverage_block()
        match = re.search(r"--cov-fail-under=(\d+)", block)
        assert match, "--cov-fail-under must include a numeric threshold in Makefile"
        mk_threshold = int(match.group(1))

        yml = _pyapp()
        ci_match = re.search(r"--cov-fail-under=(\d+)", yml)
        assert ci_match, "--cov-fail-under must exist in python-app.yml"
        ci_threshold = int(ci_match.group(1))

        assert mk_threshold == ci_threshold, (
            f"Makefile threshold ({mk_threshold}%) must match CI threshold ({ci_threshold}%)"
        )

    def test_makefile_coverage_has_html_report(self):
        assert "html" in self._coverage_block(), (
            "make coverage should emit an HTML report for developer use"
        )

    def test_makefile_coverage_has_term_missing(self):
        assert "term-missing" in self._coverage_block()


# ── Threshold sanity ──────────────────────────────────────────────────────────

class TestThresholdSanity:
    def test_threshold_not_trivially_low(self):
        """Gate must not be set so low that it provides no protection."""
        yml = _pyapp()
        match = re.search(r"--cov-fail-under=(\d+)", yml)
        assert match
        assert int(match.group(1)) >= 60

    def test_threshold_not_unrealistically_high(self):
        """Gate must not be set so high that new code always trips it."""
        yml = _pyapp()
        match = re.search(r"--cov-fail-under=(\d+)", yml)
        assert match
        assert int(match.group(1)) <= 95
