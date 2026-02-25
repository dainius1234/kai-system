"""Sliding-window rate limiter — reads limits from security/policy.yml.

Usage:
    from common.rate_limit import check_rate_limit

    # In a FastAPI endpoint:
    check_rate_limit("gate_request")  # raises HTTPException(429) if exceeded

Limits are defined in security/policy.yml under rate_limits:
    gate_request: 20        # per minute
    memory_memorize: 60
    memory_retrieve: 120
    execute: 10
    burst_multiplier: 2.0   # short bursts allowed up to 2x

All counters are per-process (not distributed).  For multi-replica
deployments, use Redis-backed counters instead.
"""
from __future__ import annotations

import time
from collections import defaultdict
from typing import Dict, List

from fastapi import HTTPException

from common.policy import rate_limit as _policy_rate_limit

# sliding window: list of timestamps per endpoint
_windows: Dict[str, List[float]] = defaultdict(list)

# burst multiplier from policy
try:
    from common.policy import POLICY
    BURST_MULTIPLIER = float(POLICY.get("rate_limits", {}).get("burst_multiplier", 2.0))
except Exception:
    BURST_MULTIPLIER = 2.0

WINDOW_SECONDS = 60.0  # 1-minute sliding window


def _prune(endpoint: str, now: float) -> None:
    """Remove entries older than the window."""
    cutoff = now - WINDOW_SECONDS
    entries = _windows[endpoint]
    # find the first entry within the window
    i = 0
    while i < len(entries) and entries[i] < cutoff:
        i += 1
    if i > 0:
        _windows[endpoint] = entries[i:]


def check_rate_limit(endpoint: str) -> None:
    """Check and record a request against the rate limit.

    Raises HTTPException(429) if the limit is exceeded.
    Does nothing if no limit is configured for this endpoint.
    """
    limit = _policy_rate_limit(endpoint)
    if limit <= 0:
        return  # no limit configured

    now = time.time()
    _prune(endpoint, now)

    count = len(_windows[endpoint])

    # allow bursts up to burst_multiplier * limit within a short window
    burst_limit = int(limit * BURST_MULTIPLIER)

    if count >= burst_limit:
        raise HTTPException(
            status_code=429,
            detail=f"rate limit exceeded for {endpoint}: "
                   f"{count}/{limit} per minute (burst cap {burst_limit})",
        )

    _windows[endpoint].append(now)


def rate_limit_snapshot() -> Dict[str, Dict[str, int]]:
    """Return current request counts per endpoint (for /metrics)."""
    now = time.time()
    result = {}
    for endpoint in list(_windows.keys()):
        _prune(endpoint, now)
        limit = _policy_rate_limit(endpoint)
        result[endpoint] = {
            "current": len(_windows[endpoint]),
            "limit_per_minute": limit,
        }
    return result
