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

    # Resolved here, not inside the function: an incomplete pyyaml would
    # otherwise import fine, let `_load_yaml` be defined, and fail at the
    # first call — long after the fallback below could have been chosen.
    _safe_load = yaml.safe_load

    def _load_yaml(path: Path) -> Dict[str, Any]:
        return _safe_load(path.read_text(encoding="utf-8")) or {}

except Exception:
    import json

    def _load_yaml(path: Path) -> Dict[str, Any]:  # type: ignore[misc]
        """There is no fallback. This exists to say so, loudly.

        It used to describe itself as a *"minimal YAML-subset loader
        (flat keys + lists only) … good enough for CI environments where
        pyyaml isn't installed"*. It is `json.loads` pointed at a YAML
        document. `security/policy.yml` begins

            version: "1.0.0"

        which is not JSON and never will be, so this path could only
        ever return `{}`. The docstring described an intention; the code
        was a stub that always failed.

        The consequence, proven in CI on 2026-08-06 when the sovereign
        profile first started tool-gate:

            JSONDecodeError: Expecting value: line 14 column 1 (char 13)
            POLICY FILE CORRUPT OR UNREADABLE — failing closed: all
            permissions will use their most restrictive defaults.

        35 service images ship `common/` and none of them declared
        pyyaml. `security/policy.yml` calls itself *"the single source of
        truth — every runtime decision reads from this file"*, and no
        container had ever read it. Today's pattern with the subject
        changed: not code that never executed, but configuration that
        was never loaded.

        Fail-closed is the right direction and it did behave that way —
        the system was restrictive, not open. But a policy file that has
        never been parsed is not policy, and the reason it went unnoticed
        for so long is that the message said CORRUPT, which sends the
        reader to the file rather than to the missing dependency.
        """
        text = path.read_text(encoding="utf-8")
        try:
            # Kept only because a *genuinely* JSON policy document would
            # still load. It is not a YAML fallback and does not pretend
            # to be one.
            return json.loads(text)
        except Exception as _exc:
            record_degradation("policy", "parse_policy_document", _exc)
            _bootstrap_logger = __import__("logging").getLogger("kai.policy")
            _bootstrap_logger.error(
                "POLICY NOT LOADED: pyyaml is not installed in this image, "
                "so %s could not be parsed. This is a missing dependency, "
                "not a corrupt file — add pyyaml to this service's "
                "requirements.txt. Continuing with an empty policy, which "
                "means every permission takes its most restrictive "
                "default.", path)
        return {}


# ── locate policy file ──────────────────────────────────────────────
_POLICY_PATH = Path(
    os.getenv(
        "SOVEREIGN_POLICY_PATH",
        str(Path(__file__).resolve().parent.parent / "security" / "policy.yml"),
    )
)

import logging as _logging
from common.degraded import record_degradation
_policy_logger = _logging.getLogger("kai.policy")

#: A file that exists and parses to nothing is an impossible result, and
#: this refuses to run on it. Set only where a deployment genuinely has
#: no policy and the hardcoded defaults are the intended behaviour.
_STRICT = os.getenv("KAI_POLICY_ALLOW_EMPTY", "").lower() not in ("1", "true", "yes")

POLICY: Dict[str, Any] = {}
if _POLICY_PATH.exists():
    try:
        _loaded = _load_yaml(_POLICY_PATH)
        if not isinstance(_loaded, dict) or not _loaded:
            raise ValueError("policy file parsed to empty or non-dict")
        POLICY = _loaded
    except Exception as _exc:
        # Three states, three messages. The old code had one, and it said
        # CORRUPT for all of them.
        #
        # On 2026-08-06 the sovereign profile started tool-gate for the
        # first time and this printed
        #
        #     POLICY FILE CORRUPT OR UNREADABLE — failing closed
        #
        # The file was not corrupt. pyyaml was not installed, so the
        # fallback loader ran `json.loads` over a YAML document and could
        # only return {}. 35 images shipped `common/` and none declared
        # pyyaml, so this had been true in every container since the
        # loader was written. The message sent every reader to the file
        # instead of to the dependency, which is why it survived.
        #
        # **Why it now refuses rather than degrades.** Fail-closed was the
        # right *direction* and it worked — the system was restrictive,
        # not open. The defect is that "running on the declared policy"
        # and "running on hardcoded defaults because nothing could read
        # the policy" were indistinguishable from outside. A file that
        # exists, has bytes, and parses to nothing is not a policy
        # decision; it is an impossible output, and a service that
        # continues on it is asserting a configuration nobody wrote.
        #
        # The suggestion came from DeepSeek, asked for a second opinion on
        # where a static gate should stop: a gate cannot decide whether a
        # guarded import's fallback works, but the service *can* decide
        # that its own output is nonsense. Undecidable statically,
        # decidable at runtime.
        #
        # An absent file is a different case and is still allowed — see
        # the `exists()` guard above. This is only about a file that is
        # there and yielded nothing.
        _detail = (
            f"POLICY NOT LOADED from {_POLICY_PATH} ({_exc}). The file "
            f"exists and is {_POLICY_PATH.stat().st_size} bytes, so this "
            f"is not a missing file and almost certainly not a corrupt "
            f"one: the usual cause is that pyyaml is not installed in "
            f"this image, leaving a fallback loader that cannot parse "
            f"YAML. Add pyyaml to this service's requirements.txt."
        )
        if _STRICT:
            _policy_logger.critical("%s Refusing to start.", _detail)
            raise RuntimeError(
                _detail + " Refusing to start on an empty policy: every "
                "permission would silently take its most restrictive "
                "default, which is indistinguishable from the policy "
                "having been applied. Set KAI_POLICY_ALLOW_EMPTY=1 to "
                "run on hardcoded defaults deliberately."
            ) from _exc
        _policy_logger.critical(
            "%s Continuing on hardcoded defaults because "
            "KAI_POLICY_ALLOW_EMPTY is set — this deployment is NOT "
            "running the policy in %s.", _detail, _POLICY_PATH)
        # POLICY stays {} — all accessors fall back to hardcoded safe defaults

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
