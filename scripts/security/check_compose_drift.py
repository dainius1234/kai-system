#!/usr/bin/env python3
"""P0-PR-09 CI gate: hardening may differ between profiles, but only upward.

Originally this compared `docker-compose.full.yml` against
`docker-compose.minimal.yml` and nothing else. A third profile —
`docker-compose.sovereign.yml` — was added later and the comparison was
never revisited, so the profile named *sovereign* was the only one never
drift-checked.

Extending it exposed why that mattered, and also why a naive extension
would have been worse than none: **equality is the wrong test.**

    executor (sovereign):  runtime: gvisor, read_only, cap_drop,
                           security_opt += apparmor:executor-aa
                           restart: on-failure

Sovereign deliberately hardens its executor beyond what `full` does. An
equality-based drift check reports that as a violation, and the cheapest
way to make it green is to *weaken sovereign*. A gate that pushes toward
less security is worse than no gate at all.

So drift here is **directional**, the same ratchet shape used everywhere
else in this programme — hygiene debt may only fall, assertion counts may
only rise, hardening may only increase:

  - **stricter**  a superset of the baseline → allowed, and recorded, so
                  it cannot silently regress later
  - **weaker**    missing something the baseline has → violation
  - **absent**    not set at all → violation, in every direction

The third is the important one, and it is the same defect as boundary
blindness one layer down: an unset `security_opt` is not *different* from
the baseline, it is *unguarded*, and comparing it as a difference invites
the reading "well, profiles differ".

`restart` is deliberately **presence-required but value-free**.
`on-failure` versus `unless-stopped` is a containment-versus-availability
choice a profile is entitled to make; having no policy at all is not.

Exit 0 = clean.  Exit 1 = violations found, or an input is missing.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Tuple

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from scripts.security.gate_inputs import inspected, require  # noqa: E402

# `full` is the baseline: the superset profile every other one is a
# subset of. Comparisons run against it, and against each profile's own
# declared x-defaults.
BASELINE = "docker-compose.full.yml"
PROFILES = (
    BASELINE,
    "docker-compose.minimal.yml",
    "docker-compose.sovereign.yml",
)

# Settings whose values form a strength ordering: more is stricter.
ORDERED_SECURITY = ("security_opt", "cap_drop")

# Settings that must be *set*, whose value is the profile's own call.
PRESENCE_REQUIRED = ("restart",)


def _defaults(data: dict) -> dict:
    for key in data:
        if key.startswith("x-") and isinstance(data[key], dict):
            return data[key]
    return {}


def _as_set(value) -> frozenset:
    if value is None:
        return frozenset()
    if isinstance(value, str):
        return frozenset({value})
    return frozenset(value)


def check_drift(root: Path = None) -> Tuple[List[str], List[str], List[str], int]:
    """Return (violations, hardening, anchor_bypasses, services_compared)."""
    paths = require(PROFILES, root=root) if root else require(PROFILES)
    data: Dict[str, dict] = {}
    for path in paths:
        try:
            data[path.name] = yaml.safe_load(path.read_text()) or {}
        except Exception as exc:
            return [f"{path.name}: failed to parse: {exc}"], [], [], 0

    violations: List[str] = []
    hardening: List[str] = []
    bypassed: List[str] = []
    compared = 0

    # ── The x-defaults anchors must themselves agree ─────────────────
    base_defaults = _defaults(data[BASELINE])
    for name in PROFILES[1:]:
        other = _defaults(data[name])
        if other.get("restart") != base_defaults.get("restart"):
            violations.append(
                f"{name}: x-defaults restart differs — "
                f"{other.get('restart')!r} vs baseline "
                f"{base_defaults.get('restart')!r}")
        if other.get("logging") != base_defaults.get("logging"):
            violations.append(f"{name}: x-defaults logging differs from baseline")
        for setting in ORDERED_SECURITY:
            base_val = _as_set(base_defaults.get(setting))
            weaker = base_val - _as_set(other.get(setting))
            if weaker:
                violations.append(
                    f"{name}: x-defaults {setting} is weaker than baseline — "
                    f"missing {sorted(weaker)}")

    # ── Every service, in every profile, must be guarded ─────────────
    #
    # Not just shared ones. The old shared-only comparison never looked
    # at the 7 services sovereign has and full does not, so a service
    # present in exactly one profile was guaranteed unchecked.
    floor = {s: _as_set(base_defaults.get(s)) for s in ORDERED_SECURITY}

    for name in PROFILES:
        services = data[name].get("services") or {}
        profile_defaults = _defaults(data[name])
        for svc in sorted(services):
            cfg = services[svc] or {}
            compared += 1

            for setting in PRESENCE_REQUIRED:
                if cfg.get(setting) is None:
                    violations.append(
                        f"{name}: service '{svc}' has no {setting} policy — "
                        f"unset is not a profile choice")

            # The floor is the **baseline's** x-defaults, not each
            # profile's own. Sovereign's anchor is far stricter
            # (`cap_drop: [ALL]`, `read_only`, `user`, `tmpfs`), and
            # imposing that as a hard requirement on a service that does
            # not use the anchor would demand `cap_drop: ALL` on
            # Postgres — which needs SETUID/SETGID to drop from root at
            # startup. The gate would then be pushing a change that
            # breaks the profile it is meant to protect.
            #
            # So a profile choosing to be stricter is *hardening*, and a
            # service skipping that profile's anchor is reported in its
            # own category rather than folded in with the floor breaches.
            # Two different defects, two different fixes, named apart.
            for setting in ORDERED_SECURITY:
                actual = _as_set(cfg.get(setting))
                required = floor[setting]
                if required:
                    if not actual:
                        violations.append(
                            f"{name}: service '{svc}' has no {setting} — "
                            f"absent is unguarded, not merely different")
                    elif required - actual:
                        violations.append(
                            f"{name}: service '{svc}' {setting} is weaker "
                            f"than the baseline floor — missing "
                            f"{sorted(required - actual)}")
                    elif actual - required:
                        hardening.append(
                            f"{name}: service '{svc}' {setting} adds "
                            f"{sorted(actual - required)}")

                own = _as_set(profile_defaults.get(setting))
                if own - required and not (own - required) & actual:
                    bypassed.append(
                        f"{name}: service '{svc}' does not inherit this "
                        f"profile's own {setting}={sorted(own - required)}")

    # ── An internal network must not stop being internal ─────────────
    #
    # The old version looped over networks-only-in-full and did nothing
    # at all — `if cfg.get("internal"): pass`. A rule that looks present
    # and does nothing is how a gate silently omits part of itself.
    #
    # The fix is *not* "every network must exist everywhere". `minimal`
    # legitimately has no `execution-net` because it runs no executor,
    # and flagging that would be a false positive of exactly the kind
    # that invites someone to add a network nothing uses. Absence is a
    # profile's business; **downgrade is not.** So the rule compares only
    # networks declared in both places, and additionally requires that a
    # network a service actually attaches to is declared at all.
    base_nets = data[BASELINE].get("networks") or {}
    for name in PROFILES[1:]:
        nets = data[name].get("networks") or {}
        for net in sorted(set(base_nets) & set(nets)):
            if (base_nets[net] or {}).get("internal") and not (nets[net] or {}).get("internal"):
                violations.append(
                    f"{name}: network '{net}' is internal in the baseline "
                    f"but not here — isolation downgraded")

    for name in PROFILES:
        nets = set((data[name].get("networks") or {}))
        for svc, cfg in sorted((data[name].get("services") or {}).items()):
            for net in (cfg or {}).get("networks") or []:
                if isinstance(net, str) and net not in nets:
                    violations.append(
                        f"{name}: service '{svc}' attaches to undeclared "
                        f"network '{net}'")

    return violations, hardening, bypassed, compared


def shared_container_names(root: Path = None) -> List[str]:
    """`container_name` values claimed by more than one profile.

    Docker container names are **global to the daemon**, not scoped to a
    compose project. Two profiles that both declare
    `container_name: sovereign-memu-core` cannot both be up: the second
    `docker compose up` fails with

        Conflict. The container name "/sovereign-memu-core" is already
        in use

    Six such names are shared between `docker-compose.minimal.yml` and
    `docker-compose.sovereign.yml`, which is survivable only because CI
    happens to tear each stack down before starting the next, and every
    teardown is `if: always()`. That is a property of the workflow, not
    of the compose files, and it is one edit away from not being true.

    Not hypothetical: `DECISIONS.md` line 1414 records this exact class
    biting before, with `sovereign-memu-core` and
    `sovereign-memu-core-introspect` colliding inside a single profile.

    Reported, not failed. Renaming a container is a change with reach —
    `docs/sovereign_ai_spec.md` and `kai-pm/PHASE1_READINESS.md` both
    name these containers — and a gate that turns red before anyone has
    decided what the names should be is a gate people learn to ignore.
    """
    import yaml
    root = root or Path(__file__).resolve().parent.parent.parent
    claimed: dict = {}
    # dict.fromkeys, not a set: BASELINE is also listed in PROFILES, and
    # iterating the same file twice made it collide with itself — a false
    # finding caught by this function's own tests before it was ever run
    # against the tree. Order is kept so the message is stable.
    unsurveyed: List[str] = []
    for profile in dict.fromkeys((BASELINE,) + tuple(PROFILES)):
        path = root / profile
        try:
            doc = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        except OSError as exc:
            # I-1, flagged by the meta-check's boundary-blindness scan the
            # first time this ran: skipping an unreadable profile would
            # make the survey cover less than it claims to, and say
            # nothing about it. Unreachable in practice — `check_drift`
            # calls `require(PROFILES)` before this — but "unreachable
            # today" is the argument that ages worst.
            unsurveyed.append(
                f"{profile}: not surveyed for container names ({exc.strerror or exc}) "
                f"— this list is therefore incomplete")
            continue
        for service, cfg in sorted((doc.get("services") or {}).items()):
            name = (cfg or {}).get("container_name")
            if name:
                claimed.setdefault(name, []).append(f"{profile}:{service}")
    return unsurveyed + [f"{name} claimed by {', '.join(owners)}"
                         for name, owners in sorted(claimed.items())
                         if len(owners) > 1]


def main() -> int:
    violations, hardening, bypassed, compared = check_drift()
    shared = shared_container_names()

    print(inspected(compared, "service definitions",
                    f"across {len(PROFILES)} profiles"))

    if shared:
        print(f"\n  Container names claimed by more than one profile "
              f"({len(shared)}) — reported, not failed. Docker container "
              f"names are global to\n  the daemon, so these profiles cannot "
              f"both be up. KAI-GATE-031:")
        for line in shared:
            print(f"    ! {line}")

    if hardening:
        print(f"\n  Hardening above the floor ({len(hardening)}) — allowed, "
              f"recorded so it cannot regress:")
        for line in hardening:
            print(f"    + {line}")

    if bypassed:
        print(f"\n  Services skipping their own profile's stricter anchor "
              f"({len(bypassed)}) — reported, not failed: fixing these needs "
              f"per-service\n  capability analysis, not a blanket edit.")
        for line in bypassed:
            print(f"    ~ {line}")

    if violations:
        print(f"\nFAIL: {len(violations)} compose drift issue(s):\n")
        for v in violations:
            print(f"  - {v}")
        print("\n  Hardening may differ between profiles, but only upward.")
        return 1

    print("\nPASS: no profile is weaker than the baseline, and every "
          "service is guarded.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
