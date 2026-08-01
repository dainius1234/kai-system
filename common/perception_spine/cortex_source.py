"""World-state source for Cortex — the defined cutover for legacy polling.

Closes UH tracker gap G-02b.  Cortex currently learns about the world by
consuming a state document produced by point-to-point polling.  The
perception spine already builds a scoped, provenance-carrying world state
from the same sensors, so this converts that world state into the shape
Cortex consumes.

The switch is explicit and defaults to the legacy path::

    KAI_CORTEX_SOURCE=poll          # default — legacy polling
    KAI_CORTEX_SOURCE=world_state   # read from the perception spine

Retiring the legacy path is therefore a config change with a tested
fallback, rather than a code change made under pressure.

**Fallback is deliberate.**  When world-state mode is selected but the
store holds nothing — a cold start, or the spine still in shadow mode —
this returns None so the caller keeps its polled state.  A perception
layer that goes blank because a migration flag was set early is worse
than one that quietly keeps working.
"""
from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

SOURCE_ENV = "KAI_CORTEX_SOURCE"

# Claim domains that carry situational meaning, in the order they should
# be presented as raw facts.
_FACT_DOMAIN_ORDER = [
    "system", "docker", "git", "calendar",
    "weather", "market", "screen", "general",
]


def cortex_source() -> str:
    value = os.getenv(SOURCE_ENV, "poll").strip().lower()
    return value if value in {"poll", "world_state"} else "poll"


def _claim_line(claim: Any) -> str:
    text = getattr(claim, "claim_text", "") or ""
    freshness = getattr(claim, "freshness", None)
    marker = ""
    if freshness is not None and getattr(freshness, "value", "") == "stale":
        marker = " [stale]"
    return f"{text}{marker}".strip()


def build_state_from_world(
    store,
    max_facts: int = 40,
) -> Optional[Dict[str, Any]]:
    """Render active world-state claims into Cortex's state shape.

    Returns None when there is nothing to report, so the caller can keep
    whatever it already had.
    """
    if store is None:
        return None

    try:
        claims = list(store.active_claims())
    except Exception:
        return None

    if not claims:
        return None

    by_domain: Dict[str, List[Any]] = {}
    for claim in claims:
        by_domain.setdefault(getattr(claim, "domain", "general"), []).append(claim)

    facts: List[str] = []
    for domain in _FACT_DOMAIN_ORDER:
        for claim in by_domain.pop(domain, []):
            line = _claim_line(claim)
            if line:
                facts.append(f"{domain}: {line}")
    # Anything in a domain not named above still gets reported.
    for domain in sorted(by_domain):
        for claim in by_domain[domain]:
            line = _claim_line(claim)
            if line:
                facts.append(f"{domain}: {line}")

    facts = facts[:max_facts]
    if not facts:
        return None

    conflicts = []
    try:
        conflicts = list(store.conflicts())
    except Exception:
        conflicts = []

    summary = f"{len(claims)} active claims across {len(set(c.domain for c in claims))} domains"
    if conflicts:
        summary += f"; {len(conflicts)} unresolved conflict(s)"

    implication = (
        "Conflicting sources disagree — treat affected domains as uncertain."
        if conflicts else
        "No contradictions between independent sources."
    )

    return {
        "level1_facts": facts,
        "level2_summary": summary,
        "level3_implication": implication,
        "refresh_count": len(facts),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "source": "world_state",
        "conflict_count": len(conflicts),
    }


def resolve_cortex_state(
    polled_state: Optional[Dict[str, Any]],
    store=None,
    mode: Optional[str] = None,
) -> Dict[str, Any]:
    """Pick the state Cortex should consume.

    In ``poll`` mode the polled state is returned untouched, so the
    default path is byte-for-byte what it was before this existed.
    """
    effective = mode or cortex_source()
    if effective != "world_state":
        return polled_state or {}

    derived = build_state_from_world(store)
    if derived is None:
        # Cold start or shadow-mode spine — keep the legacy state rather
        # than handing Cortex an empty world.
        return polled_state or {}
    return derived
