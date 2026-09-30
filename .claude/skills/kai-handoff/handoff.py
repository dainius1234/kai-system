#!/usr/bin/env python3
"""kai-handoff — measure, check and verify the KAI continuity handoff log.

A handoff is WORKING MEMORY WITH SOURCES. This tool creates no programme
state and grants no permission. It exists so that a new session can start
from measured, sourced state instead of from a lossy summary, and so that
the rules of the handoff are enforced by a machine rather than remembered
(R18: make the lesson part of the engineering system).

Subcommands, all read-only with respect to the repository:

  measure    print the section-0 state block, every value from a command
  check      validate kai-pm/HANDOFF_LOG.md: structure, source tags,
             UNBANKED discipline, and APPEND-ONLY against a Git revision
  verify     READ mode: re-measure and compare with the last entry's
             section 0; list commits since the recorded HEAD
  selftest   calibrate `check` and `verify` against synthetic logs with
             a known answer (I-8: a known-positive and a known-negative
             for every rule)

Standard library only. No shell=True. No command is ever read from the
log and executed: `verify` re-runs this module's own fixed measurements.
"""
from __future__ import annotations

import argparse
import dataclasses
import pathlib
import re
import subprocess
import sys

LOG_REL = "kai-pm/HANDOFF_LOG.md"
DECISIONS_REL = "kai-pm/DECISIONS.md"
LEDGER_REL = "kai-pm/FAILURE_PATTERN_LEDGER.md"

# The strict decision-heading grammar (the one D385/D386 allocators used).
DECISION_RE = re.compile(r"^## D([0-9]+)( +—|$)")
INCIDENT_RE = re.compile(r"^### `(INC-\d{4}-\d{2}-\d{2}-(\d+))`")

ENTRY_RE = re.compile(
    r"^## HANDOFF (\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z) — (\S.*?) — by (\S.*)$")

# Exact section headings, in order. The number is the contract; the text
# after it is fixed too, so an entry cannot quietly rename a section away.
SECTIONS = (
    "### 0. Measured state",
    "### 1. The four states",
    "### 2. Rulings since the last handoff",
    "### 3. Authorised / Held / Forbidden",
    "### 4. Open questions",
    "### 5. Incidents and corrections",
    "### 6. Next authorised step",
    "### 7. What I am unsure of",
    "### 8. Reader's verification",
)
TAGGED_SECTIONS = range(0, 8)          # section 8 holds commands, not claims

TAG_RE = re.compile(
    r"\[(?:"
    r"GIT [0-9a-f]{7,40}"
    r"|D[0-9]+"
    r"|LEDGER INC-\d{4}-\d{2}-\d{2}-\d+"
    r"|FILE [^\]\s]+(?::\d+(?:-\d+)?)?"
    r"|CMD `[^`]+`[^\]]*"
    r"|CONVERSATION \d{4}-\d{2}-\d{2}[^\]]*"
    r")\]")
CONVERSATION_RE = re.compile(r"\[CONVERSATION \d{4}-\d{2}-\d{2}[^\]]*\]")
BANKED_RE = re.compile(r"\[D[0-9]+\]")
UNBANKED = "⚠ UNBANKED"

MEASURE_LINE_RE = re.compile(r"^- ([a-z0-9_:./-]+): (.*?)  \[CMD `")


class Refusal(Exception):
    """A prerequisite is unmet; nothing downstream is measured (R11)."""


# ── measurement ──────────────────────────────────────────────────────────
def _run(argv, cwd):
    """Run one fixed command. Returns (rc, stdout, stderr). Never a shell."""
    p = subprocess.run(argv, cwd=cwd, capture_output=True, text=True)
    return p.returncode, p.stdout, p.stderr


def repo_root(start: pathlib.Path) -> pathlib.Path:
    rc, out, err = _run(["git", "rev-parse", "--show-toplevel"], start)
    if rc != 0:
        raise Refusal(f"not inside a Git work tree: {err.strip()}")
    return pathlib.Path(out.strip())


def _decisions(text: str):
    nums = [int(m.group(1)) for line in text.splitlines()
            if (m := DECISION_RE.match(line))]
    dups = sorted({n for n in nums if nums.count(n) > 1})
    return len(nums), len(set(nums)), dups, (max(nums) if nums else None)


def _incidents(text: str):
    ids = [(int(m.group(2)), m.group(1)) for line in text.splitlines()
           if (m := INCIDENT_RE.match(line))]
    return len(ids), len({i for _, i in ids}), (max(ids)[1] if ids else None)


def _entries(text: str):
    return [i for i, line in enumerate(text.splitlines())
            if line.startswith("## HANDOFF ")]


def measure(root: pathlib.Path, *, remote: bool = True):
    """Every section-0 value, each with the command that produced it.

    Returns an ordered list of (key, value, command-text). A value that
    could not be measured is the literal string UNMEASURED(<reason>) —
    never a guess, never omitted.
    """
    rows = []

    def git(key, argv):
        rc, out, err = _run(argv, root)
        val = out.strip() if rc == 0 else f"UNMEASURED(rc={rc}: {err.strip()[:120]})"
        rows.append((key, val, " ".join(argv)))
        return rc, out

    rows.append(("utc", _run(["date", "-u", "+%FT%TZ"], root)[1].strip(),
                 "date -u +%FT%TZ"))
    git("branch", ["git", "rev-parse", "--abbrev-ref", "HEAD"])
    git("head", ["git", "rev-parse", "HEAD"])
    git("tree", ["git", "rev-parse", "HEAD^{tree}"])
    rc, out = _run(["git", "status", "--porcelain"], root)[:2]
    rows.append(("uncommitted_paths",
                 str(len([l for l in out.splitlines() if l.strip()]))
                 if rc == 0 else f"UNMEASURED(rc={rc})",
                 "git status --porcelain | count lines"))

    if remote:
        rc, out, err = _run(["git", "ls-remote", "--heads", "origin"], root)
        if rc != 0:
            rows.append(("remote", f"UNMEASURED(rc={rc}: {err.strip()[:120]})",
                         "git ls-remote --heads origin"))
        else:
            for line in sorted(out.splitlines()):
                sha, ref = line.split("\t", 1)
                rows.append((f"remote:{ref.removeprefix('refs/heads/')}", sha,
                             "git ls-remote --heads origin"))

    for rel, key, fn in ((DECISIONS_REL, "decisions", _decisions),
                         (LEDGER_REL, "ledger", _incidents)):
        p = root / rel
        if not p.is_file():
            rows.append((key, f"UNMEASURED(no file {rel})", f"read {rel}"))
            continue
        res = fn(p.read_text(encoding="utf-8"))
        if key == "decisions":
            n, d, dups, hi = res
            cmd = f"grep -E '{DECISION_RE.pattern}' {rel}"
            rows += [("decisions_headings", str(n), cmd),
                     ("decisions_distinct", str(d), cmd),
                     ("decisions_duplicates", ",".join(map(str, dups)) or "none", cmd),
                     ("decisions_highest", f"D{hi}" if hi is not None else "none", cmd)]
        else:
            n, d, hi = res
            cmd = f"grep -E '{INCIDENT_RE.pattern}' {rel}"
            rows += [("ledger_incident_headings", str(n), cmd),
                     ("ledger_incident_distinct", str(d), cmd),
                     ("ledger_highest", hi or "none", cmd)]

    log = root / LOG_REL
    rows.append(("handoff_entries",
                 str(len(_entries(log.read_text(encoding="utf-8"))))
                 if log.is_file() else "0 (no log)",
                 f"grep -c '^## HANDOFF ' {LOG_REL}"))
    return rows


def format_measure(rows):
    # A backtick inside the command would end the CMD tag early, so it is
    # written as \x60 (the ledger heading grammar contains one).
    return "\n".join(f"- {k}: {v}  [CMD `{c.replace('`', chr(92) + 'x60')}` → {v}]"
                     for k, v, c in rows)


# ── check ────────────────────────────────────────────────────────────────
@dataclasses.dataclass
class Finding:
    line: int
    rule: str
    text: str

    def __str__(self):
        return f"  line {self.line}: [{self.rule}] {self.text}"


def check_text(text: str, committed: str | None):
    """Validate a handoff log. Returns (findings, stats)."""
    findings: list[Finding] = []
    lines = text.splitlines()

    # APPEND-ONLY: the committed log must be an exact byte prefix.
    if committed is not None and not text.startswith(committed):
        # locate the first differing line for the report
        c = committed.splitlines()
        first = next((i for i, (a, b) in enumerate(zip(c, lines)) if a != b),
                     min(len(c), len(lines)))
        findings.append(Finding(first + 1, "APPEND-ONLY",
                                "an existing entry was edited or removed; "
                                "corrections must be NEW entries"))

    starts = _entries(text)
    if not starts:
        findings.append(Finding(0, "NO-ENTRY", "the log contains no entry"))
    stats = {"entries": len(starts), "claim_lines": 0, "tagged": 0,
             "unbanked": 0, "rulings": 0}

    for n, s in enumerate(starts):
        end = starts[n + 1] if n + 1 < len(starts) else len(lines)
        block = lines[s:end]
        if not ENTRY_RE.match(block[0]):
            findings.append(Finding(s + 1, "HEADER",
                                    "entry header must be '## HANDOFF "
                                    "<YYYY-MM-DDTHH:MM:SSZ> — <id> — by <producer>'"))
        # sections: exact headings, exact order, each exactly once
        found = [(i, l) for i, l in enumerate(block) if l.startswith("### ")]
        heads = [l for _, l in found]
        if heads != list(SECTIONS):
            findings.append(Finding(
                s + 1, "SECTIONS",
                f"sections must be exactly {len(SECTIONS)} in order; "
                f"found {heads}"))
            continue
        bounds = [i for i, _ in found] + [len(block)]
        for k in range(len(SECTIONS)):
            body = block[bounds[k] + 1:bounds[k + 1]]
            content = [(bounds[k] + 1 + j, l) for j, l in enumerate(body)
                       if l.strip() and l.strip() != "---"]
            if not content:
                findings.append(Finding(s + bounds[k] + 1, "EMPTY-SECTION",
                                        f"{SECTIONS[k]} is empty; write "
                                        f"'- None [source]' rather than omit"))
                continue
            if k not in TAGGED_SECTIONS:
                continue
            for j, l in content:
                ln = s + j + 1
                if l.startswith("  "):
                    continue            # continuation of the bullet above
                if not l.startswith("- "):
                    findings.append(Finding(ln, "BULLET",
                                            "claims are bullets; prose "
                                            "cannot carry a source tag"))
                    continue
                stats["claim_lines"] += 1
                if TAG_RE.search(l):
                    stats["tagged"] += 1
                else:
                    findings.append(Finding(ln, "UNTAGGED",
                                            f"no source tag: {l[:90]!r}"))
                if UNBANKED in l:
                    stats["unbanked"] += 1
                    if not CONVERSATION_RE.search(l):
                        findings.append(Finding(
                            ln, "UNBANKED-SOURCE",
                            "an UNBANKED ruling must name its "
                            "[CONVERSATION <date> ...] source"))
                if k == 2 and not l.startswith("- None"):
                    stats["rulings"] += 1
                    if not (BANKED_RE.search(l) or UNBANKED in l):
                        findings.append(Finding(
                            ln, "RULING-STATUS",
                            "a ruling is either banked [D<n>] or marked "
                            f"'{UNBANKED}'; there is no third state"))
    return findings, stats


def committed_text(root: pathlib.Path, ref: str):
    rc, out, err = _run(["git", "show", f"{ref}:{LOG_REL}"], root)
    if rc == 0:
        return out
    if "does not exist" in err or "exists on disk, but not in" in err:
        return None                                     # new file at ref
    raise Refusal(f"cannot read {LOG_REL} at {ref}: {err.strip()}")


# ── verify ───────────────────────────────────────────────────────────────
NOT_COMPARED = {"utc", "handoff_entries"}    # expected to move every time


def last_entry_state(text: str):
    starts = _entries(text)
    if not starts:
        raise Refusal("the log has no entry to verify against")
    lines = text.splitlines()
    block = lines[starts[-1]:]
    try:
        a = block.index(SECTIONS[0])
        b = block.index(SECTIONS[1])
    except ValueError:
        raise Refusal("the last entry has no section 0/1 boundary")
    state = {}
    for l in block[a + 1:b]:
        m = MEASURE_LINE_RE.match(l)
        if m:
            state[m.group(1)] = m.group(2)
    return block[0], state


def _remote_measured(state: dict, *, queried: bool = True) -> bool:
    """Were remote heads actually measured on this side?

    Not queried (--no-remote) or queried-and-failed (a `remote` key holding
    UNMEASURED) both mean NO. A branch missing from an unmeasured side is
    unknown, not deleted: reporting it as DIFFERS would claim a fact
    nobody measured (R17)."""
    return queried and "UNMEASURED" not in state.get("remote", "")


def compare(old: dict, new: dict, *, new_remote_queried: bool = True):
    """Pure comparison of two section-0 states. Returns rows of
    (status, key, old, new) with status MATCH / DIFFERS / UNMEASURED."""
    old_rm = _remote_measured(old)
    new_rm = _remote_measured(new, queried=new_remote_queried)
    rows = []
    for k in sorted(set(old) | set(new)):
        if k in NOT_COMPARED:
            continue
        o, n = old.get(k, "<absent>"), new.get(k, "<absent>")
        if k.startswith("remote:") and not (old_rm and new_rm):
            status = "UNMEASURED"
            if not new_rm:
                n = "UNMEASURED(remote not measured this run)"
            else:
                o = "UNMEASURED(remote not measured in the recorded entry)"
        elif "UNMEASURED" in o or "UNMEASURED" in n:
            status = "UNMEASURED"
        else:
            status = "MATCH" if o == n else "DIFFERS"
        rows.append((status, k, o, n))
    return rows


def verify(root: pathlib.Path, *, remote: bool = True):
    log = root / LOG_REL
    if not log.is_file():
        raise Refusal(f"{LOG_REL} is absent. Do not reconstruct one from "
                      f"memory; report its absence.")
    header, old = last_entry_state(log.read_text(encoding="utf-8"))
    new = {k: v for k, v, _ in measure(root, remote=remote)}
    print(f"VERIFY against: {header}")
    rows = compare(old, new, new_remote_queried=remote)
    diffs = 0
    for status, k, o, n in rows:
        if status != "MATCH":
            diffs += 1
        print(f"  {status:<10} {k}: {o}" + ("" if o == n else f"  →  {n}"))
    oh = old.get("head")
    if oh and oh != new.get("head") and "UNMEASURED" not in oh:
        rc, _, _ = _run(["git", "merge-base", "--is-ancestor", oh, "HEAD"], root)
        if rc == 0:
            _, out, _ = _run(["git", "log", "--oneline", f"{oh}..HEAD"], root)
            print(f"  commits since the recorded HEAD ({oh[:12]}):")
            print("".join(f"    {l}\n" for l in out.splitlines()) or "    (none)")
        else:
            print(f"  WARNING: recorded HEAD {oh[:12]} is NOT an ancestor of "
                  f"the current HEAD. History diverged or was rewritten — "
                  f"stop and report before any work.")
    counts = {s: sum(1 for r in rows if r[0] == s)
              for s in ("MATCH", "DIFFERS", "UNMEASURED")}
    print(f"VERIFY: compared={len(rows)} match={counts['MATCH']} "
          f"differs={counts['DIFFERS']} unmeasured={counts['UNMEASURED']}. The repository wins for FACTS; rulings "
          f"marked {UNBANKED} must be raised with the operator.")
    return diffs


# ── selftest (calibration) ──────────────────────────────────────────────
def _valid_entry(stamp="2026-09-30T00:00:00Z", head="a" * 40):
    body = {
        0: f"- head: {head}  [CMD `git rev-parse HEAD` → {head}]",
        1: "- physical HEAD recorded  [GIT aaaaaaa]",
        2: f"- a conversation ruling {UNBANKED}  [CONVERSATION 2026-09-25 Kai]\n"
           "- a banked ruling  [D386]",
        3: "- None authorised  [CONVERSATION 2026-09-30 Dainius]",
        4: "- open question  [FILE kai-pm/X.md:1]",
        5: "- None  [LEDGER INC-2026-09-19-38]",
        6: "- None recorded  [CONVERSATION 2026-09-30 Dainius]",
        7: "- nothing unverified  [CMD `true` → 0]",
        8: "python3 .claude/skills/kai-handoff/handoff.py verify",
    }
    out = [f"## HANDOFF {stamp} — synthetic — by selftest", ""]
    for k, h in enumerate(SECTIONS):
        out += [h, "", body[k], ""]
    return "\n".join(out) + "\n"


def selftest():
    """Each rule gets a known-positive (must fire) and the valid log is the
    known-negative (must not). The expected answer is the construction,
    never the checker's own output."""
    preamble = "# log\n\n"
    good = preamble + _valid_entry()
    cases = []

    def case(name, text, committed, expect_rule):
        f, _ = check_text(text, committed)
        rules = {x.rule for x in f}
        ok = (not f) if expect_rule is None else (expect_rule in rules)
        cases.append((name, ok, sorted(rules)))

    case("NEG valid single entry", good, None, None)
    case("NEG valid append of a second entry",
         good + "\n" + _valid_entry("2026-10-01T00:00:00Z"), good, None)
    case("POS edited earlier entry",
         good.replace("open question", "changed question"), good, "APPEND-ONLY")
    case("POS deleted log content", preamble, good, "APPEND-ONLY")
    case("POS untagged claim",
         good.replace("- open question  [FILE kai-pm/X.md:1]", "- open question"),
         None, "UNTAGGED")
    case("POS prose line in a claim section",
         good.replace("- None authorised", "None authorised"), None, "BULLET")
    case("POS missing section",
         good.replace(SECTIONS[5] + "\n", ""), None, "SECTIONS")
    case("POS sections out of order",
         good.replace(SECTIONS[4], "@@").replace(SECTIONS[5], SECTIONS[4])
             .replace("@@", SECTIONS[5]), None, "SECTIONS")
    case("POS renamed section",
         good.replace(SECTIONS[7], "### 7. Doubts"), None, "SECTIONS")
    case("POS UNBANKED without conversation source",
         good.replace("[CONVERSATION 2026-09-25 Kai]", "[FILE a.md]"),
         None, "UNBANKED-SOURCE")
    case("POS ruling neither banked nor UNBANKED",
         good.replace("- a banked ruling  [D386]",
                      "- a floating ruling  [FILE a.md]"), None, "RULING-STATUS")
    case("POS bad header",
         good.replace("— by selftest", ""), None, "HEADER")
    case("POS empty section",
         good.replace("- None  [LEDGER INC-2026-09-19-38]", ""), None,
         "EMPTY-SECTION")
    case("POS no entry", preamble, None, "NO-ENTRY")

    # verify's parser: must read section 0 of the LAST entry only
    two = good + "\n" + _valid_entry("2026-10-01T00:00:00Z", head="b" * 40)
    _, st = last_entry_state(two)
    cases.append(("NEG verify reads the LAST entry's section 0",
                  st.get("head") == "b" * 40, [st.get("head", "")[:8]]))

    # compare(): the expected status of every key is fixed by construction
    base = {"head": "h1", "decisions_highest": "D386",
            "remote:main": "m1", "remote:gone": "g1"}

    def statuses(old, new, **kw):
        return {k: s for s, k, _, _ in compare(old, new, **kw)}

    for name, old, new, kw, want in (
        ("NEG identical state", base, dict(base), {},
         {"head": "MATCH", "decisions_highest": "MATCH",
          "remote:main": "MATCH", "remote:gone": "MATCH"}),
        ("POS head moved", base, dict(base, head="h2"), {},
         {"head": "DIFFERS"}),
        ("POS remote branch deleted, remote measured both sides", base,
         {k: v for k, v in base.items() if k != "remote:gone"}, {},
         {"remote:gone": "DIFFERS", "remote:main": "MATCH"}),
        ("NEG remote NOT queried: branches unknown, not deleted", base,
         {"head": "h1", "decisions_highest": "D386"},
         {"new_remote_queried": False},
         {"remote:main": "UNMEASURED", "remote:gone": "UNMEASURED",
          "head": "MATCH"}),
        ("NEG remote query FAILED: branches unknown, not deleted", base,
         {"head": "h1", "decisions_highest": "D386",
          "remote": "UNMEASURED(rc=128: no network)"}, {},
         {"remote:main": "UNMEASURED", "remote:gone": "UNMEASURED"}),
        ("NEG recorded entry had no remote: new branches unknown-before",
         {"head": "h1", "remote": "UNMEASURED(rc=128)"},
         {"head": "h1", "remote:main": "m1"}, {},
         {"remote:main": "UNMEASURED", "head": "MATCH"}),
        ("POS value UNMEASURED on one side", base,
         dict(base, decisions_highest="UNMEASURED(no file)"), {},
         {"decisions_highest": "UNMEASURED"}),
    ):
        got = statuses(old, new, **kw)
        ok = all(got.get(k) == v for k, v in want.items())
        cases.append((f"{name.split()[0]} compare: {' '.join(name.split()[1:])}",
                      ok, sorted(f"{k}={got.get(k)}" for k in want)))

    # the tag grammar itself: known-good and known-bad tags
    for tag, want in (("[GIT 0af5d32]", True), ("[D386]", True),
                      ("[LEDGER INC-2026-09-19-38]", True),
                      ("[FILE CLAUDE.md:318]", True),
                      ("[CMD `git rev-parse HEAD` → x]", True),
                      ("[CONVERSATION 2026-09-25 Kai]", True),
                      ("[GIT xyz]", False), ("[D]", False), ("[memory]", False),
                      ("[CONVERSATION yesterday]", False)):
        cases.append((f"{'NEG' if want else 'POS'} tag grammar {tag}",
                      bool(TAG_RE.search(tag)) == want, []))

    width = max(len(n) for n, _, _ in cases)
    for name, ok, rules in cases:
        print(f"  {'ok  ' if ok else 'FAIL'}  {name:<{width}}  {rules}")
    failed = sum(1 for _, ok, _ in cases if not ok)
    print(f"Handoff selftest: {len(cases) - failed} passed, {failed} failed "
          f"(of {len(cases)})")
    return failed


# ── CLI ──────────────────────────────────────────────────────────────────
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("measure")
    m.add_argument("--no-remote", action="store_true")
    c = sub.add_parser("check")
    c.add_argument("--against", default="HEAD",
                   help="revision whose committed log must be a prefix "
                        "(default HEAD)")
    v = sub.add_parser("verify")
    v.add_argument("--no-remote", action="store_true")
    v.add_argument("--strict", action="store_true",
                   help="exit 1 if any compared value differs")
    sub.add_parser("selftest")
    a = ap.parse_args(argv)

    try:
        if a.cmd == "selftest":
            return 1 if selftest() else 0
        root = repo_root(pathlib.Path.cwd())
        if a.cmd == "measure":
            print(format_measure(measure(root, remote=not a.no_remote)))
            return 0
        if a.cmd == "check":
            log = root / LOG_REL
            if not log.is_file():
                raise Refusal(f"{LOG_REL} is absent")
            text = log.read_text(encoding="utf-8")
            findings, st = check_text(text, committed_text(root, a.against))
            for f in findings:
                print(f)
            print(f"HANDOFF CHECK: entries={st['entries']} "
                  f"claim-lines={st['claim_lines']} tagged={st['tagged']} "
                  f"rulings={st['rulings']} unbanked={st['unbanked']} "
                  f"findings={len(findings)} (append-only against "
                  f"{a.against})")
            return 1 if findings else 0
        if a.cmd == "verify":
            d = verify(root, remote=not a.no_remote)
            return 1 if (a.strict and d) else 0
    except Refusal as e:
        print(f"REFUSE: {e}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
