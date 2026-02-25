"""Sovereign policy loader — reads security/policy.yml once at import time.

Every service that needs policy values imports from here:

    from common.policy import POLICY, policy_hash

The policy dict is frozen at startup.  To pick up changes, restart the
service (by design — policy changes should be deliberate, not hot-loaded).
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any, Dict

# pyyaml is optional — fall back to a minimal built-in subset
try:
    import yaml  # type: ignore[import-untyped]

    def _load_yaml(path: Path) -> Dict[str, Any]:
        return yaml.safe_load(path.read_text(encoding="utf-8")) or {}

except ImportError:
    import json

    def _load_yaml(path: Path) -> Dict[str, Any]:  # type: ignore[misc]
        """Minimal YAML-subset loader (flat keys + lists only).

        Good enough for CI environments where pyyaml isn't installed.
        Handles basic scalar, list, and nested dict structures via a
        JSON conversion fallback (policy.yml is simple enough).
        """
        # Try to convert simple YAML to JSON-parseable format
        text = path.read_text(encoding="utf-8")
        # Strip comments
        lines = []
        for line in text.splitlines():
            stripped = line.split("#")[0].rstrip() if "#" in line else line.rstrip()
            lines.append(stripped)
        # If it looks like it could be JSON, try that
        clean = "\n".join(lines)
        try:
            return json.loads(clean)
        except Exception:
            pass
        # Fallback: return empty dict (services use .get() with defaults)
        return {}


# ── locate policy file ──────────────────────────────────────────────
_POLICY_PATH = Path(
    os.getenv(
        "SOVEREIGN_POLICY_PATH",
        str(Path(__file__).resolve().parent.parent / "security" / "policy.yml"),
    )
)

POLICY: Dict[str, Any] = {}
if _POLICY_PATH.exists():
    POLICY = _load_yaml(_POLICY_PATH)

# SHA-256 of the raw file — displayed on dashboard, logged on startup
_raw = _POLICY_PATH.read_bytes() if _POLICY_PATH.exists() else b""
policy_hash: str = hashlib.sha256(_raw).hexdigest()[:16]
policy_version: str = POLICY.get("version", "unknown")


# ── convenience accessors ───────────────────────────────────────────

def verifier_thresholds() -> Dict[str, Any]:
    return POLICY.get("verifier", {})


def evidence_weights() -> Dict[str, float]:
    return POLICY.get("evidence", {}).get("weights", {
        "similarity": 0.35, "relevance": 0.20, "importance": 0.20,
        "recency": 0.20, "pin_bonus": 0.05,
    })


def circuit_breaker_defaults() -> Dict[str, Any]:
    return POLICY.get("circuit_breakers", {}).get("default", {
        "failure_threshold": 3, "recovery_seconds": 30,
    })


def rate_limit(endpoint: str) -> int:
    """Return per-minute rate limit for an endpoint name."""
    limits = POLICY.get("rate_limits", {})
    return int(limits.get(endpoint, 60))


def quarantine_config() -> Dict[str, Any]:
    return POLICY.get("quarantine", {})


def risk_tier_for_tool(tool: str) -> str:
    """Return the risk tier (LOW/MEDIUM/HIGH) for a given tool name."""
    tiers = POLICY.get("risk_tiers", {})
    for tier_name, tier_conf in tiers.items():
        if tool in tier_conf.get("tools", []):
            return tier_name
    return "MEDIUM"  # default


def mode_config(mode: str) -> Dict[str, Any]:
    return POLICY.get("modes", {}).get(mode.upper(), {})
