#!/usr/bin/env python3
"""A `$GITHUB_OUTPUT` heredoc must not be bounded by a constant.

`friday-cleanup.yml` died on this:

    ##[error]Unable to process file command 'output' successfully.
    ##[error]Invalid value. Matching delimiter not found 'EOF'

Seven steps across `friday-cleanup.yml` and `weekly-report-card.yml`
captured arbitrary command output — flake8, pip-audit, pytest, the
behavioural scoreboard — and wrote it into `$GITHUB_OUTPUT` bounded by a
fixed `EOF`:

    echo "output<<EOF" >> "$GITHUB_OUTPUT"
    echo "$out"       >> "$GITHUB_OUTPUT"
    echo "EOF"        >> "$GITHUB_OUTPUT"

If the captured text contains a line that is exactly `EOF`, the block
never closes, the runner rejects the whole file, and the step fails with
a message about a delimiter rather than about the content.

**The content is unbounded and the delimiter is a constant.** That is
the same shape as every unbounded-read finding in H-2 and H-3: a
container sized for what someone expected rather than for what can
arrive. It is not a shell bug and not a YAML bug — both files parse
perfectly — which is why nothing here could see it until the workflow
was actually run.

The rule: a heredoc delimiter written to `$GITHUB_OUTPUT` must be
generated at run time, so it cannot collide with content produced after
it. `$RANDOM`, `uuidgen` and `openssl rand` all qualify; a literal does
not.

**There is a second way to lose the delimiter, and the first version of
this gate missed it.** With the random delimiter in place,
`friday-cleanup.yml` failed again:

    Matching delimiter not found 'EOF_60842258526385'

The content was written with `printf "%b" "$list"`. `printf` appends no
trailing newline, so the closing delimiter was glued onto the last line
of content and never appeared as a line of its own. The delimiter was
unique and still unfindable.

Two distinct failure modes, one error message — and this gate passed the
file while the workflow kept failing. So it now checks both: the
delimiter must vary, **and** anything writing content into the block must
end with a newline.

Only `$GITHUB_OUTPUT` and `$GITHUB_ENV` are in scope. A heredoc feeding
a command (`cat <<EOF`, `python - <<'PY'`) is a different construct with
a different failure mode, and flagging it here would produce noise in
exactly the files this gate is meant to keep readable.

Exit 0 = every heredoc can close.  Exit 1 = one cannot.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import List, Tuple

REPO = Path(__file__).resolve().parent.parent.parent
WORKFLOWS = REPO / ".github" / "workflows"

sys.path.insert(0, str(REPO))

from scripts.security.gate_inputs import inspected, require  # noqa: E402

#: `echo "key<<DELIM" >> "$GITHUB_OUTPUT"` — the delimiter is group 2.
_HEREDOC = re.compile(
    r'"(\w+)<<([^"]+)"\s*>>\s*"?\$(?:GITHUB_OUTPUT|GITHUB_ENV)"?')

#: A delimiter that varies per run. Anything expanded by the shell at
#: run time cannot have been present in output generated afterwards.
_GENERATED = re.compile(r"\$")

#: `printf` into a GitHub file command. Unlike `echo` it adds no trailing
#: newline, so its format string has to carry one.
_PRINTF = re.compile(
    r"printf\s+(['\"])(.*?)\1[^>]*>>\s*\"?\$(?:GITHUB_OUTPUT|GITHUB_ENV)")


def findings_in(text: str, filename: str) -> List[str]:
    out: List[str] = []
    for line_no, line in enumerate(text.splitlines(), 1):
        match = _HEREDOC.search(line)
        if not match:
            continue
        key, delimiter = match.group(1), match.group(2).strip()
        if _GENERATED.search(delimiter):
            continue
        out.append(
            f"{filename}:{line_no}: '{key}' is bounded by the literal "
            f"'{delimiter}'. Command output containing a line '{delimiter}' "
            f"leaves the block unclosed and the runner rejects the file. "
            f"Use a delimiter generated at run time.")

    for line_no, line in enumerate(text.splitlines(), 1):
        match = _PRINTF.search(line)
        if not match:
            continue
        fmt = match.group(2)
        if "\\n" in fmt:
            continue
        out.append(
            f"{filename}:{line_no}: printf writes into a GitHub file "
            f"command with format '{fmt}', which appends no trailing "
            f"newline. The closing delimiter is then glued onto the last "
            f"line of content and never found. Add \\n to the format.")
    return out


def audit() -> Tuple[List[str], int, int]:
    """Return (findings, heredocs inspected, workflows read)."""
    findings: List[str] = []
    paths = sorted(WORKFLOWS.glob("*.yml"))
    seen = 0
    for path in paths:
        text = path.read_text(encoding="utf-8")
        seen += len(_HEREDOC.findall(text))
        findings.extend(findings_in(text, path.name))
    return findings, seen, len(paths)


def main() -> int:
    require((".github/workflows",))
    findings, seen, workflows = audit()

    print(inspected(seen, "$GITHUB_OUTPUT/$GITHUB_ENV heredoc(s)",
                    f"across {workflows} workflows"))
    print()
    if findings:
        print(f"FAIL: {len(findings)} unclosable heredoc(s):\n")
        for line in findings:
            print(f"  - {line}")
        print("\n  A delimiter is lost two ways: content can contain it, or "
              "the\n  terminator can fail to start a line. `friday-cleanup.yml` "
              "hit both,\n  one after the other, with the same error message — "
              "and YAML and\n  bash accept either, so nothing else can see them.")
        return 1
    if seen == 0:
        print("PASS: no $GITHUB_OUTPUT heredocs found — nothing to check.")
        return 0
    print(f"PASS: all {seen} heredoc(s) use a generated delimiter and "
          f"a terminator that starts its own line.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
