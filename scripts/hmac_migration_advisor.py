#!/usr/bin/env python3
"""Advise when to migrate interservice auth from shared HMAC to asymmetric service identity.

The script is intentionally dependency-free so it can run in constrained environments.
"""

from __future__ import annotations

import os
from dataclasses import dataclass


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


@dataclass
class MigrationSignal:
    name: str
    triggered: bool
    detail: str


def collect_signals() -> list[MigrationSignal]:
    services = _env_int("AUTH_SERVICES", 3)
    teams = _env_int("AUTH_TEAMS", 1)
    key_rotations = _env_int("HMAC_ROTATIONS_PER_QUARTER", 1)
    incidents = _env_int("HMAC_INCIDENTS_90D", 0)
    external = _env_int("EXTERNAL_VERIFIER_DEPENDENCIES", 0)
    zero_trust = os.getenv("ZERO_TRUST_TARGET", "false").lower() in {"1", "true", "yes"}
    audit_score = _env_float("AUDITABILITY_SCORE", 0.5)

    return [
        MigrationSignal(
            "Service scale",
            services >= 8,
            f"AUTH_SERVICES={services} (trigger >= 8)",
        ),
        MigrationSignal(
            "Organizational scale",
            teams >= 3,
            f"AUTH_TEAMS={teams} (trigger >= 3)",
        ),
        MigrationSignal(
            "Rotation overhead",
            key_rotations >= 4,
            f"HMAC_ROTATIONS_PER_QUARTER={key_rotations} (trigger >= 4)",
        ),
        MigrationSignal(
            "Security incidents",
            incidents >= 1,
            f"HMAC_INCIDENTS_90D={incidents} (trigger >= 1)",
        ),
        MigrationSignal(
            "Third-party verification",
            external >= 1,
            f"EXTERNAL_VERIFIER_DEPENDENCIES={external} (trigger >= 1)",
        ),
        MigrationSignal(
            "Zero-trust mandate",
            zero_trust,
            f"ZERO_TRUST_TARGET={zero_trust}",
        ),
        MigrationSignal(
            "Auditability pressure",
            audit_score >= 0.8,
            f"AUDITABILITY_SCORE={audit_score:.2f} (trigger >= 0.80)",
        ),
    ]


def summarize(signals: list[MigrationSignal]) -> tuple[str, int]:
    score = sum(1 for s in signals if s.triggered)
    if score >= 3:
        return (
            "MIGRATE NEXT PHASE: begin phased rollout to asymmetric service identity "
            "(mTLS SPIFFE/SPIRE or per-service Ed25519 with key IDs).",
            score,
        )
    if score == 2:
        return (
            "PREPARE NOW: keep HMAC this phase, but run migration design + pilot in parallel.",
            score,
        )
    return (
        "STAY ON HMAC: current topology is still well-served by shared-secret auth with rotation drills.",
        score,
    )


def main() -> int:
    signals = collect_signals()
    recommendation, score = summarize(signals)

    print("HMAC Migration Advisor")
    print("======================")
    for signal in signals:
        marker = "TRIGGERED" if signal.triggered else "ok"
        print(f"- [{marker}] {signal.name}: {signal.detail}")

    print("\nResult")
    print("------")
    print(f"Trigger score: {score}/{len(signals)}")
    print(recommendation)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
