"""Tests for `check_workflow_outputs` — unbounded content, constant delimiter.

`friday-cleanup.yml` failed with

    Invalid value. Matching delimiter not found 'EOF'

because seven steps wrote arbitrary command output into `$GITHUB_OUTPUT`
inside a heredoc bounded by a literal `EOF`. Content that contains a line
`EOF` closes the block early; the runner then rejects the file.

The content is unbounded and the delimiter is a constant — the same shape
as every unbounded-read finding in H-2 and H-3. Both YAML and bash accept
it, which is why it survived until the workflow was actually dispatched.

All synthetic except the last two, which read the repository and say so.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from scripts.security import check_workflow_outputs as gate  # noqa: E402

passed = 0
failed = 0
EXPECTED_SCENARIOS = 13
executed: list = []


def check(name: str, condition: bool, detail: str = "") -> None:
    global passed, failed
    if condition:
        passed += 1
    else:
        failed += 1
        print(f"  FAIL: {name}" + (f" — {detail}" if detail else ""))


def scenario(name: str) -> None:
    executed.append(name)


def test_a_literal_delimiter_is_reported() -> None:
    scenario("literal delimiter fails")
    text = 'echo "output<<EOF" >> "$GITHUB_OUTPUT"\n'
    found = gate.findings_in(text, "w.yml")
    check("it is reported", len(found) == 1, str(found))
    check("names the key", found and "'output'" in found[0], str(found))
    check("names the delimiter", found and "'EOF'" in found[0], str(found))
    check("and the line", found and "w.yml:1" in found[0], str(found))


def test_a_generated_delimiter_passes() -> None:
    scenario("generated delimiter passes")
    for delim in ('$d', '$RANDOM', 'EOF_$RANDOM$RANDOM', '$(uuidgen)'):
        text = f'echo "output<<{delim}" >> "$GITHUB_OUTPUT"\n'
        check(f"{delim} is accepted", gate.findings_in(text, "w.yml") == [],
              str(gate.findings_in(text, "w.yml")))


def test_github_env_is_covered_too() -> None:
    """Same file-command mechanism, same failure."""
    scenario("GITHUB_ENV covered")
    text = 'echo "BODY<<EOF" >> "$GITHUB_ENV"\n'
    check("it is reported", len(gate.findings_in(text, "w.yml")) == 1,
          str(gate.findings_in(text, "w.yml")))


def test_an_unquoted_target_is_still_matched() -> None:
    """`>> $GITHUB_OUTPUT` without quotes is the same construct."""
    scenario("unquoted target")
    text = 'echo "output<<EOF" >> $GITHUB_OUTPUT\n'
    check("still reported", len(gate.findings_in(text, "w.yml")) == 1,
          str(gate.findings_in(text, "w.yml")))


def test_a_heredoc_feeding_a_command_is_not_flagged() -> None:
    """`cat <<EOF` and `python - <<'PY'` are a different construct with a
    different failure mode. Flagging them would put noise into the files
    this gate exists to keep readable — and a gate with false positives
    gets somebody to "fix" working code."""
    scenario("command heredoc ignored")
    text = ("cat <<EOF\nhello\nEOF\n"
            "python - <<'PY'\nprint(1)\nPY\n")
    check("nothing is reported", gate.findings_in(text, "w.yml") == [],
          str(gate.findings_in(text, "w.yml")))


def test_several_in_one_file_are_all_reported() -> None:
    """friday-cleanup had four; reporting one and stopping would have sent
    somebody back for three more rounds."""
    scenario("all instances reported")
    text = ('echo "a<<EOF" >> "$GITHUB_OUTPUT"\n'
            'echo "b<<END" >> "$GITHUB_OUTPUT"\n'
            'echo "c<<$d" >> "$GITHUB_OUTPUT"\n')
    found = gate.findings_in(text, "w.yml")
    check("both literals are reported", len(found) == 2, str(found))
    check("and the generated one is not",
          all("'c'" not in line for line in found), str(found))


def test_a_line_with_no_heredoc_is_ignored() -> None:
    scenario("plain output ignored")
    text = 'echo "exit_code=0" >> "$GITHUB_OUTPUT"\n'
    check("nothing reported", gate.findings_in(text, "w.yml") == [], "")


# ── the real tree ────────────────────────────────────────────────────

def test_the_repository_passes_today() -> None:
    scenario("repository passes")
    findings, seen, workflows = gate.audit()
    check("no constant delimiters", findings == [], str(findings))
    check("and something was inspected", seen > 0, str(seen))
    check("across every workflow", workflows >= 9, str(workflows))


def test_the_denominator_matches_the_known_count() -> None:
    """Seven heredocs, the number found and fixed on 2026-08-05. A
    ceiling, not an equality: adding one is fine, and losing sight of all
    seven is what this asserts against."""
    scenario("denominator holds")
    _, seen, _ = gate.audit()
    check("at least the seven that were fixed are still watched",
          seen >= 7, str(seen))


# ── the second failure mode, which the first gate missed ─────────────

def test_printf_without_a_newline_is_reported() -> None:
    """The gate passed friday-cleanup while it was still failing.

    With a unique delimiter in place it failed again on
    `printf "%b" "$list"` — printf appends no trailing newline, so the
    closing delimiter was glued onto the last line of content. Unique and
    still unfindable. Two failure modes, one error message.
    """
    scenario("printf without newline fails")
    text = 'printf "%b" "$stale_list" >> "$GITHUB_OUTPUT"\n'
    found = gate.findings_in(text, "w.yml")
    check("it is reported", len(found) == 1, str(found))
    check("names printf", found and "printf" in found[0], str(found))
    check("and says what to add", found and "\\n" in found[0], str(found))


def test_printf_with_a_newline_passes() -> None:
    scenario("printf with newline passes")
    for fmt in ('%b\\n', '%s\\n', '%b\\n%b\\n'):
        text = f'printf "{fmt}" "$x" >> "$GITHUB_OUTPUT"\n'
        check(f"{fmt} accepted", gate.findings_in(text, "w.yml") == [],
              str(gate.findings_in(text, "w.yml")))


def test_printf_elsewhere_is_not_flagged() -> None:
    """Only writes into a GitHub file command matter; printf to stdout or
    to an ordinary file has no delimiter to lose."""
    scenario("printf elsewhere ignored")
    for text in ('printf "%b" "$x"\n', 'printf "%b" "$x" >> /tmp/log\n'):
        check("not flagged", gate.findings_in(text, "w.yml") == [], text)


def test_echo_is_not_flagged() -> None:
    """`echo` appends a newline, so it cannot cause this."""
    scenario("echo not flagged")
    text = 'echo "$out" >> "$GITHUB_OUTPUT"\n'
    check("not flagged", gate.findings_in(text, "w.yml") == [], "")


def run_all() -> None:
    test_a_literal_delimiter_is_reported()
    test_a_generated_delimiter_passes()
    test_github_env_is_covered_too()
    test_an_unquoted_target_is_still_matched()
    test_a_heredoc_feeding_a_command_is_not_flagged()
    test_several_in_one_file_are_all_reported()
    test_a_line_with_no_heredoc_is_ignored()
    test_the_repository_passes_today()
    test_the_denominator_matches_the_known_count()
    test_printf_without_a_newline_is_reported()
    test_printf_with_a_newline_passes()
    test_printf_elsewhere_is_not_flagged()
    test_echo_is_not_flagged()

    check(f"all {EXPECTED_SCENARIOS} scenarios ran",
          len(executed) == EXPECTED_SCENARIOS,
          f"{len(executed)} ran: {executed}")
    check("no scenario ran twice", len(set(executed)) == len(executed),
          str(executed))


if __name__ == "__main__":
    run_all()
    print(f"\n{'=' * 60}")
    print(f"Workflow Output Tests: {passed} passed, {failed} failed")
    if failed:
        print("EXIT GATE: FAIL")
        sys.exit(1)
    print("EXIT GATE: PASS")
