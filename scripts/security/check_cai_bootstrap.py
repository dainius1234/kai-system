#!/usr/bin/env python3
"""CAI v1.0 — are the rulesets CAI depends on the ones CAI needs?

PURE CHECKER (order §19, §25-26). Reads ruleset JSON — the proposed design
files, and in leg 2 the live server state fetched by the caller — and
returns PASS/REFUSE with reasons. It never calls GitHub, never applies a
ruleset, never changes repository settings.

  --design DIR                 design invariants hold (placeholders listed)
  --apply-ready                ... AND no placeholder remains, AND the
                               authority bypass identity is not the executor
  --executor-id N              the GitHub user id the executor pushes as
  --live FACTS.json            leg 2: live rulesets equal the design

IDENTITY COLLAPSE IS A REFUSAL, NOT A NOTE. If the authority's bypass
actor is the identity the executor pushes as, then every tag restriction
that the bypass exempts exempts the executor too, and the rulesets
separate nothing. Measured 2026-09-23: this executor pushes as
dainius1234 (216391246). See rulesets/PINNING_RESEARCH.md.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

NS = {"refs/tags/kai-wa/**", "refs/tags/kai-admitted/**"}
VERIFIER_PATH = ".github/workflows/cai-authority-verifier.yml"
REPOSITORY_ID = 1004463473
PLACEHOLDER = "<<"


def _placeholders(obj, path="$"):
    out = []
    if isinstance(obj, str) and PLACEHOLDER in obj:
        out.append(path)
    elif isinstance(obj, dict):
        for k, v in obj.items():
            out += _placeholders(v, f"{path}.{k}")
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            out += _placeholders(v, f"{path}[{i}]")
    return out


def _types(rs):
    return sorted(r.get("type") for r in rs.get("rules", []))


def _includes(rs):
    return set(rs.get("conditions", {}).get("ref_name", {}).get("include", []))


def _excludes(rs):
    return rs.get("conditions", {}).get("ref_name", {}).get("exclude", [])


def design_problems(rulesets):
    """rulesets: list of ruleset bodies. Returns reasons (empty = PASS)."""
    r = []
    for rs in rulesets:
        if rs.get("enforcement") != "active":
            r.append(f"{rs.get('name')!r}: enforcement is "
                     f"{rs.get('enforcement')!r}, must be 'active'")
    tag_ns = [rs for rs in rulesets if rs.get("target") == "tag"
              and _includes(rs) & NS]
    create = [rs for rs in tag_ns if _types(rs) == ["creation"]]
    immut = [rs for rs in tag_ns if _types(rs) == ["deletion", "update"]]
    other = [rs for rs in tag_ns if rs not in create and rs not in immut]
    if len(create) != 1:
        r.append(f"need exactly one tag ruleset whose only rule is 'creation' "
                 f"over the authority namespaces; found {len(create)}")
    if len(immut) != 1:
        r.append(f"need exactly one tag ruleset with exactly 'update' + "
                 f"'deletion' over the authority namespaces; found {len(immut)}")
    for rs in other:
        r.append(f"{rs.get('name')!r}: an additional ruleset touches the "
                 f"authority namespaces with rules {_types(rs)} — ambiguous")
    for rs in create + immut:
        if _includes(rs) != NS:
            r.append(f"{rs.get('name')!r}: must cover exactly {sorted(NS)}, "
                     f"covers {sorted(_includes(rs))}")
        if _excludes(rs):
            r.append(f"{rs.get('name')!r}: exclude patterns carve holes in "
                     f"the authority namespaces: {_excludes(rs)}")
    for rs in create:
        ba = rs.get("bypass_actors", [])
        if len(ba) != 1 or ba[0].get("actor_type") != "User" or \
                ba[0].get("bypass_mode") != "always":
            r.append(f"{rs.get('name')!r}: creation bypass must be exactly one "
                     f"User actor with bypass_mode 'always'; got {ba}")
    for rs in immut:
        if rs.get("bypass_actors"):
            r.append(f"{rs.get('name')!r}: immutability must have NO bypass "
                     f"actor; got {rs['bypass_actors']}")
        for rule in rs.get("rules", []):
            if rule.get("type") == "update" and \
                    rule.get("parameters", {}).get(
                        "update_allows_fetch_and_merge") is not False:
                r.append(f"{rs.get('name')!r}: update must not allow "
                         f"fetch-and-merge")
    main = [rs for rs in rulesets if rs.get("target") == "branch"
            and _includes(rs) == {"~DEFAULT_BRANCH"}]
    if len(main) != 1:
        r.append(f"need exactly one branch ruleset on ~DEFAULT_BRANCH; "
                 f"found {len(main)}")
    for rs in main:
        t = _types(rs)
        for need in ("deletion", "non_fast_forward", "workflows"):
            if need not in t:
                r.append(f"{rs.get('name')!r}: missing rule {need!r}")
        if rs.get("bypass_actors"):
            r.append(f"{rs.get('name')!r}: main integration must have NO "
                     f"bypass actor; got {rs['bypass_actors']}")
        if _excludes(rs):
            r.append(f"{rs.get('name')!r}: exclude patterns on main: "
                     f"{_excludes(rs)}")
        wf = [w for rule in rs.get("rules", []) if rule.get("type") == "workflows"
              for w in rule.get("parameters", {}).get("workflows", [])]
        if len(wf) != 1:
            r.append(f"{rs.get('name')!r}: exactly one required workflow "
                     f"expected; got {len(wf)}")
        for w in wf:
            if w.get("path") != VERIFIER_PATH:
                r.append(f"required workflow path is {w.get('path')!r}")
            if w.get("repository_id") != REPOSITORY_ID:
                r.append(f"required workflow repository_id is "
                         f"{w.get('repository_id')!r}")
            if "ref" in w:
                r.append("required workflow is pinned by 'ref' — a movable "
                         "name, not the exact reviewed verifier (order §20)")
            if not w.get("sha"):
                r.append("required workflow carries no exact 'sha'")
    return r


def apply_ready_problems(rulesets, executor_id):
    r = []
    for rs in rulesets:
        for p in _placeholders(rs):
            r.append(f"{rs.get('name')!r}: unresolved placeholder at {p}")
    for rs in rulesets:
        for a in rs.get("bypass_actors", []):
            aid = a.get("actor_id")
            if isinstance(aid, str) and PLACEHOLDER in aid:
                continue
            if not isinstance(aid, int):
                r.append(f"{rs.get('name')!r}: bypass actor_id {aid!r} is not "
                         f"an integer")
            elif executor_id is not None and aid == executor_id:
                r.append(f"{rs.get('name')!r}: IDENTITY COLLAPSE — bypass "
                         f"actor {aid} IS the identity the executor pushes as; "
                         f"the restriction exempts the executor")
        for rule in rs.get("rules", []):
            for w in rule.get("parameters", {}).get("workflows", []) \
                    if rule.get("type") == "workflows" else []:
                s = w.get("sha", "")
                if PLACEHOLDER not in s and not re.fullmatch(r"[0-9a-f]{40}", s):
                    r.append(f"required workflow sha {s!r} is not a 40-hex "
                             f"commit id")
    if executor_id is None:
        r.append("executor identity not supplied: identity collapse cannot "
                 "be excluded")
    return r


_LIVE_KEYS = ("name", "target", "enforcement", "conditions", "rules",
              "bypass_actors")


def _norm(rs):
    return json.dumps({k: rs.get(k) for k in _LIVE_KEYS}, sort_keys=True)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--design", required=True)
    ap.add_argument("--apply-ready", action="store_true")
    ap.add_argument("--executor-id", type=int)
    ap.add_argument("--live")
    a = ap.parse_args(argv)
    reasons = []
    try:
        files = sorted(Path(a.design).glob("*.json"))
        rulesets = [json.loads(f.read_text(encoding="utf-8")) for f in files]
    except (OSError, ValueError) as e:
        print(f"REFUSE\n  - UNKNOWN: design unreadable: {e}")
        return 1
    print(f"  design rulesets read: {len(rulesets)} "
          f"({', '.join(f.name for f in files)})")
    reasons += design_problems(rulesets)
    ph = [p for rs in rulesets for p in _placeholders(rs)]
    print(f"  placeholders in design: {len(ph)}")
    if a.apply_ready or a.live:
        reasons += apply_ready_problems(rulesets, a.executor_id)
    if a.live:
        try:
            live = json.loads(Path(a.live).read_text(encoding="utf-8"))
        except (OSError, ValueError) as e:
            reasons.append(f"UNKNOWN: live facts unreadable: {e}")
            live = None
        if live is not None:
            if not isinstance(live, list):
                reasons.append("live facts must be the list of ruleset bodies")
            else:
                print(f"  live rulesets: {len(live)}")
                want = sorted(_norm(r) for r in rulesets)
                got = sorted(_norm(r) for r in live)
                if want != got:
                    reasons.append(f"live rulesets do not equal the design: "
                                   f"{len(set(want) - set(got))} designed "
                                   f"ruleset(s) absent or different, "
                                   f"{len(set(got) - set(want))} live ruleset(s) "
                                   f"not in the design")
    if reasons:
        print("REFUSE")
        for r in reasons:
            print(f"  - {r}")
        return 1
    print("PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
