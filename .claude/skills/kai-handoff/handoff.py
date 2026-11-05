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
  due        is a WRITE owed? commits since the last entry's recorded
             HEAD that no entry covers (commits only; conversation-only
             rulings are invisible to it, and it says so)
  fresh      may an entry be appended here? compares this branch with
             the live handoff branch named in .claude/handoff-branch
             (fetches it; the only subcommand that touches the network
             besides verify's ls-remote)
  hook       Stop / PreCompact entry point: remind once per state, never
             loop, never block an automatic compaction

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


def ancestry(root: pathlib.Path, rev: str) -> str:
    """Is `rev` an ancestor of HEAD? YES, NO, or UNKNOWN(<reason>).

    Cloud sessions clone SHALLOW (about 50 commits). A recorded HEAD older
    than the clone depth is ABSENT from the object store, which
    `merge-base --is-ancestor` also reports as non-zero. Absent is not
    diverged: reading it as NO would tell a session to stop over a fact
    nobody measured (R17). Only exit 1 with the commit present means NO."""
    if _run(["git", "cat-file", "-e", f"{rev}^{{commit}}"], root)[0] != 0:
        rc, out, _ = _run(["git", "rev-parse", "--is-shallow-repository"], root)
        why = ("not in this SHALLOW clone's history; `git fetch --unshallow` "
               "to measure" if out.strip() == "true" else "commit not found")
        return f"UNKNOWN({why})"
    rc = _run(["git", "merge-base", "--is-ancestor", rev, "HEAD"], root)[0]
    return {0: "YES", 1: "NO"}.get(rc, f"UNKNOWN(merge-base rc={rc})")


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
    _check_verbatim(lines, starts, committed, findings, stats)
    return findings, stats


# ── VERBATIM INTEGRITY (R18 control) ─────────────────────────────────────
# The round-trip of every verbatim block used to be a SEPARATE step run
# beside `check` -- and twice it could not stop the commit it guarded:
# entry 46 (a check that only printed) and entry 61 (the check piped
# through `grep -c`, so the chain saw grep's exit status). A check that
# has to be chained correctly is a check that will one day be chained
# wrongly. It now lives INSIDE `check`, so anything that runs `check`
# runs it. The expected answer comes from the declaration the entry
# itself makes (bytes, sha256, final LF), never from the block.
VERBATIM_BEGIN = "    BEGIN-VERBATIM "
VERBATIM_END = "    END-VERBATIM "
DECLARATION_RE = re.compile(
    r"(\d[\d,]*) bytes, sha256 ([0-9a-f]{64}), final LF (True|False)")
ANY_SHA_RE = re.compile(r"(?<![0-9a-f])[0-9a-f]{64}(?![0-9a-f])")


def _check_verbatim(lines, starts, committed, findings, stats):
    """Every BEGIN-VERBATIM block must close, and must reconstruct to what
    its declaration line (the nearest preceding '- ' bullet) says.

    NEW entries (not in the committed log) are held to the full grammar
    '<N> bytes, sha256 <hex>, final LF <True|False>' and an exact match.
    COMMITTED entries are immutable history written under older grammars
    (measured 2026-10-04: 159 blocks; 114 full grammar, 36 without the
    final-LF field, 9 sha-only, 1 of those with no full sha): for them a
    full sha on the declaration line must match under either LF variant,
    and an undeclared legacy block is counted, not failed.
    """
    import hashlib
    n_committed = len(_entries(committed)) if committed is not None else 0
    stats.update(verbatim_blocks=0, verbatim_verified=0,
                 verbatim_legacy_undeclared=0)
    entry_of = []                      # line index -> entry ordinal
    for k, s in enumerate(starts):
        end = starts[k + 1] if k + 1 < len(starts) else len(lines)
        entry_of += [k] * (end - len(entry_of))
    i = 0
    while i < len(lines):
        if not lines[i].startswith(VERBATIM_BEGIN):
            i += 1
            continue
        b, name = i, lines[i][len(VERBATIM_BEGIN):]
        stats["verbatim_blocks"] += 1
        new = b < len(entry_of) and entry_of[b] >= n_committed
        e = next((j for j in range(b + 1, len(lines))
                  if lines[j].startswith((VERBATIM_BEGIN, VERBATIM_END))),
                 None)
        if e is None or lines[e] != VERBATIM_END + name:
            findings.append(Finding(b + 1, "VERBATIM-UNTERMINATED",
                                    f"BEGIN-VERBATIM {name} has no matching "
                                    f"END-VERBATIM {name} before the next "
                                    f"verbatim marker"))
            i = b + 1
            continue
        body = lines[b + 1:e]
        raw = "\n".join(l[4:] if l.startswith("    ") else l for l in body)
        unindented = [b + 2 + k for k, l in enumerate(body)
                      if l and not l.startswith("    ")]
        if unindented:
            findings.append(Finding(unindented[0], "VERBATIM-INDENT",
                                    f"{name}: {len(unindented)} line(s) "
                                    f"lack the 4-space verbatim indent"))
        d = b - 1
        while d >= 0 and not lines[d].startswith("- "):
            d -= 1
        decl = lines[d] if d >= 0 else ""
        m = DECLARATION_RE.search(decl)
        if m:
            want_n, want_sha = int(m[1].replace(",", "")), m[2]
            data = (raw + ("\n" if m[3] == "True" else "")).encode("utf-8")
            if len(data) != want_n or \
                    hashlib.sha256(data).hexdigest() != want_sha:
                findings.append(Finding(
                    b + 1, "VERBATIM-MISMATCH",
                    f"{name}: reconstructs to {len(data)} bytes sha256 "
                    f"{hashlib.sha256(data).hexdigest()[:16]}…, declared "
                    f"{want_n} bytes {want_sha[:16]}… final LF {m[3]}"))
            else:
                stats["verbatim_verified"] += 1
        elif new:
            findings.append(Finding(
                b + 1, "VERBATIM-UNDECLARED",
                f"{name}: a new verbatim block needs a declaration "
                f"'<N> bytes, sha256 <hex>, final LF <True|False>' on its "
                f"bullet line"))
        else:
            shas = set(ANY_SHA_RE.findall(decl))
            got = {hashlib.sha256((raw + t).encode("utf-8")).hexdigest()
                   for t in ("", "\n")}
            if shas and not (shas & got):
                findings.append(Finding(
                    b + 1, "VERBATIM-MISMATCH",
                    f"{name}: no sha256 on its declaration line matches "
                    f"the block under either final-LF variant"))
            elif shas:
                stats["verbatim_verified"] += 1
            else:
                stats["verbatim_legacy_undeclared"] += 1
        i = e + 1


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
        anc = ancestry(root, oh)
        if anc == "YES":
            _, out, _ = _run(["git", "log", "--oneline", f"{oh}..HEAD"], root)
            print(f"  commits since the recorded HEAD ({oh[:12]}):")
            print("".join(f"    {l}\n" for l in out.splitlines()) or "    (none)")
        elif anc == "NO":
            print(f"  WARNING: recorded HEAD {oh[:12]} is NOT an ancestor of "
                  f"the current HEAD. History diverged or was rewritten — "
                  f"stop and report before any work.")
        else:
            print(f"  UNMEASURED ancestry of recorded HEAD {oh[:12]}: {anc}. "
                  f"Divergence is NOT established; commits since it are "
                  f"not listed.")
    counts = {s: sum(1 for r in rows if r[0] == s)
              for s in ("MATCH", "DIFFERS", "UNMEASURED")}
    print(f"VERIFY: compared={len(rows)} match={counts['MATCH']} "
          f"differs={counts['DIFFERS']} unmeasured={counts['UNMEASURED']}. The repository wins for FACTS; rulings "
          f"marked {UNBANKED} must be raised with the operator.")
    return diffs


# ── due: is a WRITE owed? ────────────────────────────────────────────────
# Measured signal only: COMMITS after the last entry's recorded HEAD that no
# entry covers. A commit touching the log is a WRITE commit and covers
# itself (only WRITE appends to it). Conversation-only rulings leave no
# trace in Git, so no hook can detect them; the output says so every time.
DUE_BLIND = ("commits only: a conversation-only ruling is invisible to "
             "this check; judge those yourself")


def classify_due(commits, *, recorded_head, ancestry_state, wt_entries,
                 head_entries):
    """Pure. `commits` = [(sha, subject, [paths])] in recorded_head..HEAD.
    Returns (status, detail, uncovered) with status one of
    DUE / NOT-DUE / WRITTEN-UNCOMMITTED / DIVERGED / UNKNOWN."""
    if wt_entries > head_entries:
        return ("WRITTEN-UNCOMMITTED",
                f"{wt_entries - head_entries} entry(ies) appended in the "
                f"working tree, not yet committed (commit follows authority)",
                [])
    if not recorded_head or "UNMEASURED" in recorded_head:
        return ("UNKNOWN", "the last entry recorded no measured HEAD", [])
    if ancestry_state == "NO":
        return ("DIVERGED", f"recorded HEAD {recorded_head[:12]} is not an "
                            f"ancestor of HEAD", [])
    if ancestry_state != "YES":
        return ("UNKNOWN", f"recorded HEAD {recorded_head[:12]}: "
                           f"{ancestry_state}", [])
    uncovered = [c for c in commits if LOG_REL not in c[2]]
    if uncovered:
        return ("DUE", f"{len(uncovered)} commit(s) since the last entry's "
                       f"recorded HEAD {recorded_head[:12]} are covered by no "
                       f"entry", uncovered)
    return ("NOT-DUE", f"no uncovered commit since {recorded_head[:12]}", [])


def due_state(root: pathlib.Path):
    log = root / LOG_REL
    if not log.is_file():
        raise Refusal(f"{LOG_REL} is absent")
    wt = log.read_text(encoding="utf-8")
    committed = committed_text(root, "HEAD")
    wt_n = len(_entries(wt))
    head_n = len(_entries(committed)) if committed else 0
    _, st = last_entry_state(committed if committed and head_n else wt)
    rh = st.get("head", "")
    anc = ancestry(root, rh) if rh and "UNMEASURED" not in rh else "UNKNOWN"
    commits = []
    if anc == "YES":
        _, out, _ = _run(["git", "log", "--format=@@%H %s", "--name-only",
                          f"{rh}..HEAD"], root)
        for line in out.splitlines():
            if line.startswith("@@"):
                sha, _, subj = line[2:].partition(" ")
                commits.append((sha, subj, []))
            elif line.strip() and commits:
                commits[-1][2].append(line.strip())
    _, head, _ = _run(["git", "rev-parse", "HEAD"], root)
    return classify_due(commits, recorded_head=rh, ancestry_state=anc,
                        wt_entries=wt_n, head_entries=head_n), head.strip()


def format_due(res):
    status, detail, uncovered = res
    lines = [f"WRITE-DUE: {status} — {detail} ({DUE_BLIND})"]
    lines += [f"    {sha[:7]} {subj}" for sha, subj, _ in uncovered[:8]]
    if len(uncovered) > 8:
        lines.append(f"    … and {len(uncovered) - 8} more "
                     f"(listed {min(8, len(uncovered))} of {len(uncovered)})")
    return "\n".join(lines)


# ── fresh: one live log, one writer ─────────────────────────────────────
POINTER_REL = ".claude/handoff-branch"


def live_branch(root: pathlib.Path):
    p = root / POINTER_REL
    if not p.is_file():
        return None
    name = p.read_text(encoding="utf-8").strip()
    return name or None


def classify_fresh(*, current, live, head, remote_sha, remote_ancestry):
    """Pure. May an entry be appended here without losing another
    writer's? Returns (status, detail): FRESH / AHEAD / STALE / NOT-LIVE /
    UNKNOWN."""
    if not live:
        return ("UNKNOWN", f"no live branch named in {POINTER_REL}")
    if current != live:
        return ("NOT-LIVE", f"this branch is {current}; the live handoff log "
                            f"is on {live} [FILE {POINTER_REL}]. An entry "
                            f"committed here reaches the live log only when "
                            f"merged there")
    if not remote_sha or "UNMEASURED" in remote_sha:
        return ("UNKNOWN", f"origin/{live} not measured: {remote_sha}")
    if remote_sha == head:
        return ("FRESH", f"HEAD equals origin/{live}")
    if remote_ancestry == "YES":
        return ("AHEAD", f"HEAD is ahead of origin/{live}; no other writer "
                         f"pushed")
    if remote_ancestry == "NO":
        return ("STALE", f"origin/{live} ({remote_sha[:12]}) has commits this "
                         f"HEAD lacks: another writer pushed. `git pull "
                         f"--ff-only origin {live}` BEFORE appending")
    return ("UNKNOWN", f"ancestry of origin/{live}: {remote_ancestry}")


def fresh_state(root: pathlib.Path, *, timeout=45):
    live = live_branch(root)
    _, cur, _ = _run(["git", "rev-parse", "--abbrev-ref", "HEAD"], root)
    _, head, _ = _run(["git", "rev-parse", "HEAD"], root)
    remote_sha, anc = "UNMEASURED(not queried)", "UNKNOWN"
    if live and cur.strip() == live:
        try:
            p = subprocess.run(["git", "fetch", "--quiet", "origin",
                                f"refs/heads/{live}"], cwd=root,
                               capture_output=True, text=True, timeout=timeout)
            if p.returncode == 0:
                remote_sha = _run(["git", "rev-parse", "FETCH_HEAD"],
                                  root)[1].strip()
                anc = ancestry(root, remote_sha)
            else:
                remote_sha = f"UNMEASURED(fetch rc={p.returncode}: " \
                             f"{p.stderr.strip()[:100]})"
        except subprocess.TimeoutExpired:
            remote_sha = f"UNMEASURED(fetch timed out after {timeout}s)"
    return classify_fresh(current=cur.strip(), live=live, head=head.strip(),
                          remote_sha=remote_sha, remote_ancestry=anc)


# ── hook: Stop and PreCompact decisions ─────────────────────────────────
OWED = ("DUE", "DIVERGED")


def hook_action(event, payload, status, seen):
    """Pure. What a hook does, given the event, its input JSON, the due
    status and whether this exact reminder was already given. Returns
    ("silent",) / ("context", why) / ("block", why).

    Stop: never re-fires inside its own continuation (stop_hook_active),
    and fires once per state, so it cannot loop or nag.
    PreCompact: blocks MANUAL /compact once per state. Never blocks AUTO:
    the hooks docs state that when compaction was "triggered to recover
    from a context-limit error already returned by the API", blocking it
    makes "the current request fail[s]"; the input cannot tell that case
    from a proactive one. After an auto-compaction the SessionStart:compact
    hook reports WRITE-DUE instead."""
    if status not in OWED or seen:
        return ("silent",)
    if event == "stop":
        if payload.get("stop_hook_active"):
            return ("silent",)
        return ("context", "remind")
    if event == "precompact":
        if payload.get("trigger") != "manual":
            return ("silent",)
        return ("block", "remind")
    return ("silent",)


def _marker(payload):
    import os
    import tempfile
    sid = re.sub(r"[^A-Za-z0-9_-]", "_", str(payload.get("session_id")
                                             or "nosession"))[:80]
    d = pathlib.Path(os.environ.get("TMPDIR") or tempfile.gettempdir()) \
        / "kai-handoff-hooks"
    d.mkdir(parents=True, exist_ok=True)
    return d / sid


def run_hook(event, root: pathlib.Path):
    import json
    raw = sys.stdin.read() if not sys.stdin.isatty() else ""
    try:
        payload = json.loads(raw) if raw.strip() else {}
    except ValueError:
        payload = {}
    (status, detail, uncovered), head = due_state(root)
    key = f"{event}:{head}:{status}"
    mk = _marker(payload)
    seen = mk.is_file() and key in mk.read_text(encoding="utf-8").split()
    act = hook_action(event, payload, status, seen)
    if act[0] == "silent":
        return 0
    with mk.open("a", encoding="utf-8") as f:
        f.write(key + "\n")
    text = format_due((status, detail, uncovered))
    if act[0] == "context":
        msg = (f"kai-handoff (Stop hook, fires once per HEAD): {text}\n"
               f"Before ending: run kai-handoff WRITE "
               f"(.claude/skills/kai-handoff/SKILL.md), or tell the operator "
               f"in one line why not (for example: committing it is not "
               f"authorised).")
        print(json.dumps({"hookSpecificOutput": {
            "hookEventName": "Stop", "additionalContext": msg}}))
        return 0
    print(f"kai-handoff: compaction blocked ONCE because a handoff WRITE is "
          f"owed.\n{text}\nAsk Claude to run kai-handoff WRITE, then /compact "
          f"again. Running /compact again now proceeds without it.",
          file=sys.stderr)
    return 2


# ── commit gate (R18 control, PreToolUse) ────────────────────────────────
# The verbatim round-trip moved INTO `check`; this gate moves `check` out
# of my command chain. Whatever a Bash command looks like -- `;`, a pipe,
# `|| true`, a forgotten step -- if it would run `git commit` while the
# handoff log differs from HEAD, `check` runs first and a failure BLOCKS
# the command (exit 2). The chain's syntax can no longer decide whether
# the gate is a gate. R9: the gate inspects the proposed command text,
# never a running process, so it cannot observe itself.
GIT_COMMIT_RE = re.compile(r"\bgit\b[^\n;&|]*?\bcommit\b")


def gate_action(command: str, log_dirty: bool, check_rc: int | None):
    """Pure decision: 'allow' or 'block'. check_rc is None when not run."""
    if not GIT_COMMIT_RE.search(command or ""):
        return "allow"
    if not log_dirty:
        return "allow"
    return "allow" if check_rc == 0 else "block"


def run_gate(stdin_text: str):
    import json
    try:
        payload = json.loads(stdin_text or "{}")
        command = (payload.get("tool_input") or {}).get("command") or ""
    except Exception:                                        # noqa: BLE001
        return 0                 # not a Bash payload we understand: allow
    if not GIT_COMMIT_RE.search(command):
        return 0
    try:
        root = repo_root(pathlib.Path(__file__).resolve().parent)
        rc, out, err = _run(["git", "status", "--porcelain=v1", "--",
                             LOG_REL], root)
        if rc != 0:
            raise Refusal(f"git status failed: {err.strip()}")
        dirty = bool(out.strip())
        if gate_action(command, dirty, 0) == "allow" and not dirty:
            return 0
        text = (root / LOG_REL).read_text(encoding="utf-8")
        findings, st = check_text(text, committed_text(root, "HEAD"))
    except Exception as e:                                   # noqa: BLE001
        # FAIL CLOSED, but only here: a commit is proposed AND the log is
        # (or may be) dirty, and the gate could not establish that it
        # passes. Every other path above allows.
        print(f"kai-handoff gate: BLOCKED `git commit` — could not run the "
              f"handoff check: {e!r}", file=sys.stderr)
        return 2
    if gate_action(command, True, 1 if findings else 0) == "allow":
        return 0
    print(f"kai-handoff gate: BLOCKED `git commit` — {LOG_REL} is modified "
          f"and fails `handoff.py check` ({len(findings)} finding(s)):",
          file=sys.stderr)
    for f in findings[:20]:
        print(str(f), file=sys.stderr)
    if len(findings) > 20:
        print(f"  … {len(findings) - 20} more; run handoff.py check for all",
              file=sys.stderr)
    return 2


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

    # VERBATIM INTEGRITY: the expected digest is computed HERE from the
    # payload, independently of the checker.
    import hashlib
    payload = "line one\n  indented two\n\nlast\n"
    sha = hashlib.sha256(payload.encode()).hexdigest()

    def vb(decl, body=payload, end=True, name="P"):
        ind = "\n".join("    " + l for l in body.rstrip("\n").split("\n"))
        blk = f"{decl}  [CMD `sha256sum p` → x]\n    BEGIN-VERBATIM {name}\n{ind}\n"
        blk += f"    END-VERBATIM {name}\n" if end else ""
        return good.replace("- None  [LEDGER INC-2026-09-19-38]",
                            "- None  [LEDGER INC-2026-09-19-38]\n" + blk)
    ok_decl = f"- EVIDENCE P p: {len(payload)} bytes, sha256 {sha}, final LF True"
    case("NEG verbatim block matching its declaration", vb(ok_decl), None, None)
    case("POS verbatim body altered by one byte",
         vb(ok_decl, payload.replace("two", "tw0")), None, "VERBATIM-MISMATCH")
    case("POS verbatim declared final LF wrong",
         vb(ok_decl.replace("final LF True", "final LF False")), None,
         "VERBATIM-MISMATCH")
    case("POS verbatim declared byte count wrong",
         vb(ok_decl.replace(f"{len(payload)} bytes", f"{len(payload) + 1} bytes")),
         None, "VERBATIM-MISMATCH")
    case("POS verbatim block without END marker", vb(ok_decl, end=False), None,
         "VERBATIM-UNTERMINATED")
    case("POS new verbatim block without a declaration",
         vb("- EVIDENCE P p"), None, "VERBATIM-UNDECLARED")
    case("POS verbatim line missing its indent",
         vb(ok_decl).replace("    last\n", "last\n"), None, "VERBATIM-INDENT")
    legacy = vb("- EVIDENCE P p, verbatim")
    case("NEG committed legacy block with no declaration is counted, not failed",
         legacy, legacy, None)
    bad_legacy = vb(f"- EVIDENCE P p, verbatim, sha256 {'0' * 64}")
    case("POS committed legacy block whose declared sha does not match",
         bad_legacy, bad_legacy, "VERBATIM-MISMATCH")
    sha_nolf = hashlib.sha256(payload.rstrip("\n").encode()).hexdigest()
    old_ok = vb(f"- EVIDENCE P p, verbatim, sha256 {sha_nolf}")
    case("NEG committed legacy sha-only block (either LF variant) verifies",
         old_ok, old_ok, None)

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

    # classify_due(): expected status fixed by construction
    code = ("c1", "fix x", ["a.py"])
    entry = ("c2", "handoff", [LOG_REL])
    both = ("c3", "entry+code", [LOG_REL, "b.py"])
    for name, kw, want, n_unc in (
        ("NEG nothing since the entry", dict(commits=[]), "NOT-DUE", 0),
        ("NEG only the entry's own commit", dict(commits=[entry]),
         "NOT-DUE", 0),
        ("POS a code commit after the entry", dict(commits=[entry, code]),
         "DUE", 1),
        ("POS a code commit before the entry commit",
         dict(commits=[code, entry]), "DUE", 1),
        ("NEG a WRITE commit that also touched code covers itself",
         dict(commits=[both]), "NOT-DUE", 0),
        ("NEG entry appended but uncommitted",
         dict(commits=[code], wt_entries=5), "WRITTEN-UNCOMMITTED", 0),
        ("POS history diverged", dict(commits=[], ancestry_state="NO"),
         "DIVERGED", 0),
        ("NEG shallow clone: absent is UNKNOWN, not DIVERGED",
         dict(commits=[], ancestry_state="UNKNOWN(not in this SHALLOW clone)"),
         "UNKNOWN", 0),
        ("NEG no recorded HEAD", dict(commits=[code], recorded_head=""),
         "UNKNOWN", 0),
    ):
        args = dict(recorded_head="a" * 40, ancestry_state="YES",
                    wt_entries=4, head_entries=4)
        args.update(kw)
        st, _, unc = classify_due(**args)
        cases.append((f"{name.split()[0]} due: {' '.join(name.split()[1:])}",
                      st == want and len(unc) == n_unc, [st, len(unc)]))

    # classify_fresh()
    h, r = "h" * 40, "r" * 40
    for name, kw, want in (
        ("NEG head equals remote", dict(remote_sha=h), "FRESH"),
        ("NEG local ahead of remote",
         dict(remote_sha=r, remote_ancestry="YES"), "AHEAD"),
        ("POS another writer pushed",
         dict(remote_sha=r, remote_ancestry="NO"), "STALE"),
        ("POS not on the live branch", dict(current="other"), "NOT-LIVE"),
        ("NEG remote unmeasured is UNKNOWN, not FRESH",
         dict(remote_sha="UNMEASURED(x)"), "UNKNOWN"),
        ("NEG no pointer is UNKNOWN", dict(live=None), "UNKNOWN"),
    ):
        args = dict(current="live", live="live", head=h, remote_sha=h,
                    remote_ancestry="UNKNOWN")
        args.update(kw)
        st, _ = classify_fresh(**args)
        cases.append((f"{name.split()[0]} fresh: {' '.join(name.split()[1:])}",
                      st == want, [st]))

    # hook_action(): loop safety and the never-block-auto rule
    for name, ev, pl, st, seen, want in (
        ("POS stop, due, first time", "stop", {}, "DUE", False, "context"),
        ("NEG stop inside its own continuation", "stop",
         {"stop_hook_active": True}, "DUE", False, "silent"),
        ("NEG stop, same state already reminded", "stop", {}, "DUE", True,
         "silent"),
        ("NEG stop, not due", "stop", {}, "NOT-DUE", False, "silent"),
        ("NEG stop, written but uncommitted", "stop", {},
         "WRITTEN-UNCOMMITTED", False, "silent"),
        ("POS manual compact, due", "precompact", {"trigger": "manual"},
         "DUE", False, "block"),
        ("NEG manual compact, second attempt proceeds", "precompact",
         {"trigger": "manual"}, "DUE", True, "silent"),
        ("NEG AUTO compact is never blocked", "precompact",
         {"trigger": "auto"}, "DUE", False, "silent"),
        ("NEG unknown ancestry never blocks", "precompact",
         {"trigger": "manual"}, "UNKNOWN", False, "silent"),
    ):
        got = hook_action(ev, pl, st, seen)[0]
        cases.append((f"{name.split()[0]} hook: {' '.join(name.split()[1:])}",
                      got == want, [got]))

    # gate_action(): the commit gate's decision, every branch by construction
    for name, cmd, dirty, rc, want in (
        ("NEG no commit in the command", "git status && ls", True, 1, "allow"),
        ("NEG commit, log clean", "git add x && git commit -m m", False, None,
         "allow"),
        ("NEG commit, log dirty, check passes", "git commit -q -F -", True, 0,
         "allow"),
        ("POS commit, log dirty, check fails", "git commit -q -F -", True, 1,
         "block"),
        ("POS piped round-trip before commit (entry 61's shape)",
         "x.py | grep -c True && git add L && git commit -q -F -", True, 1,
         "block"),
        ("POS `;` past the gate (R3's shape)",
         "handoff.py check ; git commit -m x", True, 1, "block"),
        ("POS commit with -C path", "git -C /r commit -m x", True, 1, "block"),
        ("NEG the word commit outside a git command", "echo commit", True, 1,
         "allow"),
    ):
        got = gate_action(cmd, dirty, rc)
        cases.append((f"{name.split()[0]} gate: {' '.join(name.split()[1:])}",
                      got == want, [got]))

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
    sub.add_parser("due", help="is a WRITE owed? exit 1 if DUE/DIVERGED")
    sub.add_parser("fresh", help="may an entry be appended here? exit 1 if "
                                 "STALE/NOT-LIVE, 2 if UNKNOWN")
    hk = sub.add_parser("hook", help="Stop/PreCompact hook entry point")
    hk.add_argument("event", choices=("stop", "precompact"))
    sub.add_parser("gate", help="PreToolUse(Bash) entry point: block a "
                                "`git commit` while the log fails `check`")
    a = ap.parse_args(argv)

    if a.cmd == "gate":
        return run_gate(sys.stdin.read())
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
                  f"verbatim={st['verbatim_blocks']}"
                  f"/verified={st['verbatim_verified']}"
                  f"/legacy-undeclared={st['verbatim_legacy_undeclared']} "
                  f"findings={len(findings)} (append-only against "
                  f"{a.against})")
            return 1 if findings else 0
        if a.cmd == "verify":
            d = verify(root, remote=not a.no_remote)
            return 1 if (a.strict and d) else 0
        if a.cmd == "due":
            res, _ = due_state(root)
            print(format_due(res))
            return 1 if res[0] in OWED else 0
        if a.cmd == "fresh":
            st, detail = fresh_state(root)
            print(f"FRESH: {st} — {detail}")
            return {"FRESH": 0, "AHEAD": 0, "UNKNOWN": 2}.get(st, 1)
        if a.cmd == "hook":
            # A hook must never break a session: any internal failure is
            # reported on stderr and exits 0 (non-blocking).
            try:
                return run_hook(a.event, root)
            except Exception as e:                    # noqa: BLE001
                print(f"kai-handoff hook {a.event}: internal error, "
                      f"nothing enforced: {e!r}", file=sys.stderr)
                return 0
    except Refusal as e:
        if a.cmd == "hook":
            print(f"kai-handoff hook: REFUSE {e}; nothing enforced",
                  file=sys.stderr)
            return 0
        print(f"REFUSE: {e}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
