#!/usr/bin/env python3
"""Classify a service's model path from OBSERVATIONS, not from its name.

KAI-GATE-048 needs an answer for `memu-graph` that cannot collapse into
the three answers we already have. The operator's requirement, verbatim:

    The instrument must be able to distinguish memu-graph from the
    already-known cases. Specifically, it must not collapse:
      * memu-core-introspect — direct pre-readiness external attempt,
        broken contract
      * memu-core — pre-readiness load with complete local/offline
        contract
      * ollama-pull — fetch-looking behaviour delegated to an
        egress-capable peer
      * memu-graph — third-party loader whose actual timing/transport is
        what we are measuring

    If the detector classifies based only on syntax such as: package
    import, --model, "pull", model-looking string, internal network —
    then stop and fix the instrument before trusting the result.

So this module takes no source text, no service name and no image tag.
It takes an `Observations` record — facts a collector measured against a
running container and a built image — and returns one verdict. The name
of the service is not an input at all, which is the strongest available
guarantee that it cannot be recognising one.

Today already produced two over-reporting failures from matching
constructs instead of behaviour. This is the third instrument in two days
written specifically against that class.

THE ORDER OF THE TESTS IS THE ARGUMENT
======================================

Delegation is checked BEFORE egress, because `ollama-pull` runs a fetch
verb with no egress of its own and is nonetheless correct — the rule
"no egress + a fetch = broken" reports a working design as a defect.
Timing is checked BEFORE the asset contract, because a missing asset on a
lazy path is a first-request failure and a missing asset on an import
path is a readiness failure; those need different remedies and must not
share a verdict.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

# ── verdict vocabulary, from the task ────────────────────────────────
PRE_LOCAL = "PRE-READINESS LOCAL"
PRE_EXTERNAL = "PRE-READINESS EXTERNAL"
PRE_DELEGATED = "PRE-READINESS DELEGATED"
LAZY = "REQUEST-TIME / LAZY"
NO_LOAD = "NO MODEL LOAD OBSERVED"
UNKNOWN = "UNKNOWN"

VERDICTS = frozenset({PRE_LOCAL, PRE_EXTERNAL, PRE_DELEGATED, LAZY,
                      NO_LOAD, UNKNOWN})

# ── evidence levels, kept apart on purpose ───────────────────────────
# A source prediction may not inherit runtime status because it agrees
# with another service. `fusion-engine` shares `agentic`'s source
# mechanism and has no runtime proof; that distinction is load-bearing
# and is carried in the data rather than in a footnote.
SOURCE, CONFIG, IMAGE, RUNTIME = "SOURCE", "CONFIG", "IMAGE", "RUNTIME"
LEVELS = (SOURCE, CONFIG, IMAGE, RUNTIME)


@dataclass
class Observations:
    """What a collector measured. Every field is a fact, not a name.

    `None` means NOT MEASURED and propagates to UNKNOWN rather than
    being read as False. That is the whole reason these are tri-state:
    "we did not look" and "we looked and it was absent" are different
    findings, and the second is the only one that supports a verdict.
    """

    #: RUNTIME. Did the container reach its declared readiness criterion?
    reached_ready: Optional[bool] = None
    #: RUNTIME. Was the model library loaded in the serving process
    #: BEFORE readiness was reached? Measured, not inferred from imports.
    loaded_before_ready: Optional[bool] = None
    #: RUNTIME. Was it loaded at all, at any point observed?
    loaded_at_all: Optional[bool] = None
    #: RUNTIME. Did the process attempt to reach a model registry?
    external_resolution_attempted: Optional[bool] = None
    #: RUNTIME/IMAGE. Was the asset already present locally when needed?
    asset_present_locally: Optional[bool] = None
    #: CONFIG. The model is fetched/served by another service this one
    #: calls, rather than resolved in-process.
    delegated_to: Optional[str] = None
    #: CONFIG/RUNTIME. That delegate can reach the outside world.
    delegate_has_egress: Optional[bool] = None
    #: RUNTIME. Proven by probe, not inferred from `internal: true`.
    egress_available: Optional[bool] = None

    #: Which level each conclusion rests on, for the report.
    evidence_level: str = UNKNOWN
    notes: List[str] = field(default_factory=list)


def classify(obs: Observations) -> tuple[str, str]:
    """(verdict, why). Never raises; UNKNOWN is a real answer."""

    # R11. No subject, no observation. A container that never reached
    # readiness cannot support a statement about what happens before
    # readiness *in the normal case* -- unless the load itself is what
    # stopped it, which is a separate, stronger finding handled below.
    if obs.reached_ready is False:
        if obs.loaded_before_ready and obs.external_resolution_attempted:
            return (PRE_EXTERNAL,
                    "the process attempted external model resolution before "
                    "readiness and the service never became ready")
        return (UNKNOWN,
                "the service did not reach readiness; what its model path "
                "does in a healthy run was NOT MEASURED by this run")

    if obs.loaded_at_all is None and obs.loaded_before_ready is None:
        return (UNKNOWN, "no observation of the model path was recorded")

    # Delegation FIRST. `ollama-pull` runs a fetch verb on a network with
    # no egress and is correct, because the egress belongs to the peer it
    # calls. Checking egress before delegation inverts this into a defect.
    if obs.delegated_to:
        if obs.delegate_has_egress is None:
            return (UNKNOWN,
                    f"the fetch is delegated to `{obs.delegated_to}`, whose "
                    f"egress contract was NOT MEASURED")
        if obs.loaded_before_ready:
            return (PRE_DELEGATED,
                    f"acquisition happens before readiness but is performed "
                    f"by `{obs.delegated_to}`"
                    + ("" if obs.delegate_has_egress else
                       " — WHICH ALSO HAS NO EGRESS, so the delegation does "
                       "not resolve the constraint"))

    if obs.loaded_at_all is False:
        return (NO_LOAD,
                "no model load was observed at any point in the window")

    if obs.loaded_before_ready is None:
        return (UNKNOWN,
                "a load was observed but its position relative to the "
                "readiness boundary was not established")

    # Timing before contract: a missing asset on a lazy path is a
    # first-request failure; on an import path it is a readiness failure.
    if not obs.loaded_before_ready:
        tail = ""
        if obs.asset_present_locally is False:
            tail = (" — and the asset is NOT local, so the first request "
                    "that needs it performs the resolution")
            if obs.egress_available is False:
                tail += " on a container with no proven egress"
        elif obs.asset_present_locally is None:
            tail = " — local asset availability was NOT MEASURED"
        return (LAZY, "the load happens after readiness" + tail)

    if obs.external_resolution_attempted:
        return (PRE_EXTERNAL,
                "the process reached for a model registry before readiness")
    if obs.asset_present_locally:
        return (PRE_LOCAL,
                "the model was resolved from a local asset before readiness, "
                "with no registry round-trip observed")
    if obs.asset_present_locally is False:
        return (UNKNOWN,
                "loaded before readiness with no local asset and no observed "
                "external attempt — these cannot all be true; re-measure")
    return (UNKNOWN,
            "loaded before readiness, but neither the local asset nor an "
            "external attempt was measured")


def summarise(name: str, obs: Observations) -> str:
    verdict, why = classify(obs)
    lines = [
        f"  service                {name}",
        f"  verdict                {verdict}",
        f"  because                {why}",
        f"  evidence level         {obs.evidence_level}",
        f"  reached readiness      {_t(obs.reached_ready)}",
        f"  loaded before ready    {_t(obs.loaded_before_ready)}",
        f"  loaded at all          {_t(obs.loaded_at_all)}",
        f"  external attempt       {_t(obs.external_resolution_attempted)}",
        f"  asset present locally  {_t(obs.asset_present_locally)}",
        f"  delegated to           {obs.delegated_to or '(none)'}",
        f"  delegate has egress    {_t(obs.delegate_has_egress)}",
        f"  egress proven          {_t(obs.egress_available)}",
    ]
    lines += [f"  note                   {n}" for n in obs.notes]
    return "\n".join(lines)


def _t(v: Optional[bool]) -> str:
    return "NOT MEASURED" if v is None else ("YES" if v else "NO")
