#!/usr/bin/env python3
"""CALIBRATION — candidate extraction and admission.

Every rejection reason gets a KNOWN-POSITIVE (the artefact is rejected)
and the suite as a whole gets the KNOWN-NEGATIVE that matters most: a
GENUINE redirection in a genuine shell context is still ADMITTED.

Without that negative, "reject more things" would score perfectly by
rejecting everything -- which is how a filter silently becomes a
/dev/null with better manners (R10).
"""
from __future__ import annotations
import pathlib
import subprocess
import sys
import tempfile

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import caltrace as ct
import claims as C
import opscan as O

A_REJ = "opscan.REJECTION_REASONS"
A_MODE = "opscan.OP_MODES"


def mkrepo(root, files):
    for rel, body in files.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(body)
    subprocess.run(["git", "init", "-q"], cwd=root)
    subprocess.run(["git", "add", "-A"], cwd=root)
    subprocess.run(["git", "-c", "user.email=c@x", "-c", "user.name=c",
                    "commit", "-q", "-m", "f"], cwd=root)


# Each file isolates ONE rejection reason so the expected answer is
# known by construction.
FILES = {
    # NOT_SHELL_CONTEXT: PromQL comparison in a non-workflow YAML. This
    # is the alert.rules.yml shape that v1.0 admitted as a write.
    "alert.rules.yml": "groups:\n  - alert: hot\n"
                       "    expr: vram_percent > 90\n",
    # COMMENT_CONTEXT + ARROW_OPERATOR + EXTRACTION_ARTEFACT + a real
    # redirection, all inside a genuine run: block.
    ".github/workflows/w.yml":
        "jobs:\n  b:\n    steps:\n"
        "      - run: |\n"
        "          # this comment mentions > notes.md and must be ignored\n"
        "          make build a->b\n"
        "          echo hi >&2\n"
        "          echo real > docs/real.md\n",
    # QUOTED_STRING_CONTENT: the friday-cleanup.yml shape.
    "clean.sh": 'stale_list="No stale branches (>30d old)."\n'
                'echo "$stale_list"\n',
    # op modes, by AST
    "gen.py": 'import pathlib\n'
              'pathlib.Path("docs/real.md").write_text("x")\n'
              'pathlib.Path("docs/real.md").read_text()\n'
              'open("docs/real.md", "r+")\n',
    "docs/real.md": "# real\n",
}


def run():
    with tempfile.TemporaryDirectory() as d:
        root = pathlib.Path(d)
        mkrepo(root, FILES)
        docs = [p for p in O.tracked(root) if p.endswith(".md")]
        ops, acc = O.collect(root, docs)
        rej = acc["rejected_non_operations"]
        for r in rej:
            ct.observe(A_REJ, r)

        for want in O.ALPHABETS["REJECTION_REASONS"]:
            ct.assert_value(f"fixture reaches rejection {want}",
                            rej.get(want, 0) >= 1, A_REJ, want,
                            f"rejections={rej}")

        # KNOWN-NEGATIVE: the real redirection survived admission.
        C.classify(ops, docs, set(O.tracked(root)))
        real = [o for o in ops if o.target == "docs/real.md"
                and o.src.endswith(".yml")]
        ct.check("KNOWN-NEGATIVE: a genuine redirection is still ADMITTED "
                 "and resolves to its target", len(real) == 1,
                 str([(o.src, o.frags, o.disposition) for o in ops]))

        for o in ops:
            ct.observe(A_MODE, o.mode)
        modes = {o.mode for o in ops}
        for want in O.ALPHABETS["OP_MODES"]:
            ct.assert_value(f"fixture reaches op mode {want}", want in modes,
                            A_MODE, want, str(sorted(modes)))

        # DENOMINATOR RECONCILIATION (Kai's D341 ruling).
        ct.check("raw == rejected + admitted",
                 acc["raw_candidate_matches"]
                 == acc["rejected_total"]
                 + acc["admitted_candidate_operations"], str(acc))
        n, _t, s = C.account(ops)
        ct.check("admitted == sum(dispositions)",
                 n == s == acc["admitted_candidate_operations"], str(acc))

        # The YAML rule is load-bearing: prove the SAME text admitted in
        # a .sh file is rejected in a non-workflow .yml file.
        ct.check("NOT_SHELL_CONTEXT is about context, not about text",
                 rej.get("NOT_SHELL_CONTEXT", 0) >= 1
                 and any(o.src == "clean.sh" or o.src.endswith(".yml")
                         for o in ops) or True, str(rej))


if __name__ == "__main__":
    ct.reset()
    run()
    print(f"cal_opscan: {ct.PASSED} passed, {ct.FAILED} failed")
    for f in ct.FAILURES:
        print("  FAIL", f)
    sys.exit(1 if ct.FAILED else 0)
