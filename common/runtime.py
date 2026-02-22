from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import time
from collections import deque
from logging.handlers import TimedRotatingFileHandler
from typing import Deque, Dict, Optional, Tuple


def setup_json_logger(name: str, log_path: str) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    handler = TimedRotatingFileHandler(
        log_path,
        when="midnight",
        interval=1,
        backupCount=30,
        encoding="utf-8",
    )
    handler.setFormatter(
        logging.Formatter(
            '{"time":"%(asctime)s","level":"%(levelname)s","service":"%(name)s","msg":"%(message)s"}'
        )
    )
    logger.handlers = [handler]
    logger.propagate = False
    return logger


def detect_device() -> str:
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def sanitize_string(value: str, max_len: int = 1024) -> str:
    sanitized = re.sub(r"[;|&]", "", value)
    return sanitized[:max_len]


class ErrorBudget:
    def __init__(self, window_seconds: int = 300) -> None:
        self.window_seconds = window_seconds
        self.samples: Deque[Tuple[float, int]] = deque()

    def record(self, code: int) -> None:
        now = time.time()
        self.samples.append((now, code))
        self._prune(now)

    def _prune(self, now: float) -> None:
        while self.samples and now - self.samples[0][0] > self.window_seconds:
            self.samples.popleft()

    def snapshot(self) -> Dict[str, float]:
        now = time.time()
        self._prune(now)
        total = len(self.samples)
        if total == 0:
            return {"error_ratio": 0.0, "total": 0}
        errors = sum(1 for _, code in self.samples if code in {429, 500, 408})
        return {"error_ratio": errors / total, "total": total}


class AuditStream:
    """Append-only Redis hash-chain logger with startup integrity validation."""

    def __init__(self, service: str, redis_url: Optional[str] = None, required: bool = False) -> None:
        self.service = service
        self.redis_url = redis_url or os.getenv("AUDIT_REDIS_URL", "")
        self.required = required
        self._client = None
        if self.redis_url:
            try:
                import redis

                self._client = redis.from_url(self.redis_url, decode_responses=True)
                self._client.ping()
                self.verify_or_halt()
            except Exception:
                self._client = None
                if self.required:
                    raise

    def enabled(self) -> bool:
        return self._client is not None

    def verify_or_halt(self) -> bool:
        if not self._client:
            return True
        try:
            entries = self._client.xrange("audit:logs", min="-", max="+")
        except Exception:
            if self.required:
                raise
            self._client = None
            return False
        prev_hash = "genesis"
        for _, fields in entries:
            entry = {
                "ts": fields.get("ts"),
                "service": fields.get("service"),
                "level": fields.get("level"),
                "msg": fields.get("msg"),
            }
            expected = hashlib.sha256((json.dumps(entry, sort_keys=True) + prev_hash).encode("utf-8")).hexdigest()
            current = fields.get("hash")
            if current != expected:
                raise RuntimeError("audit chain integrity violation")
            prev_hash = current
        return True

    def log(self, level: str, msg: str) -> None:
        if not self._client:
            return
        try:
            prev_hash = self._client.get("audit:last_hash") or "genesis"
            entry = {"ts": str(time.time()), "service": self.service, "level": level, "msg": msg}
            entry_hash = hashlib.sha256((json.dumps(entry, sort_keys=True) + prev_hash).encode("utf-8")).hexdigest()
            self._client.xadd("audit:logs", {**entry, "hash": entry_hash})
            self._client.set("audit:last_hash", entry_hash)
        except Exception:
            if self.required:
                raise
            self._client = None


class CircuitBreaker:
    def __init__(self, failure_threshold: int = 3, recovery_seconds: int = 30) -> None:
        self.failure_threshold = max(1, failure_threshold)
        self.recovery_seconds = max(1, recovery_seconds)
        self.failures = 0
        self.state = "closed"
        self.opened_at = 0.0

    def allow(self) -> bool:
        if self.state == "open":
            if time.time() - self.opened_at >= self.recovery_seconds:
                self.state = "half_open"
                return True
            return False
        return True

    def record_success(self) -> None:
        self.failures = 0
        self.state = "closed"
        self.opened_at = 0.0

    def record_failure(self) -> None:
        self.failures += 1
        if self.failures >= self.failure_threshold:
            self.state = "open"
            self.opened_at = time.time()

    def snapshot(self) -> Dict[str, float | str | int]:
        cooldown = 0.0
        if self.state == "open":
            cooldown = max(self.recovery_seconds - (time.time() - self.opened_at), 0.0)
        return {"state": self.state, "failures": self.failures, "recovery_seconds": self.recovery_seconds, "cooldown_seconds": round(cooldown, 2)}


class ErrorBudgetCircuitBreaker:
    def __init__(self, *, warn_ratio: float = 0.05, open_ratio: float = 0.10, window_seconds: int = 300, recovery_seconds: int = 60) -> None:
        self.budget = ErrorBudget(window_seconds=window_seconds)
        self.warn_ratio = max(0.0, warn_ratio)
        self.open_ratio = max(self.warn_ratio, open_ratio)
        self.recovery_seconds = max(1, recovery_seconds)
        self.state = "closed"
        self.opened_at = 0.0

    def record(self, status_code: int) -> None:
        self.budget.record(status_code)
        ratio = float(self.budget.snapshot().get("error_ratio", 0.0))
        now = time.time()
        if ratio >= self.open_ratio:
            self.state = "open"
            self.opened_at = now
            return
        if ratio >= self.warn_ratio:
            if self.state == "open" and (now - self.opened_at) >= self.recovery_seconds:
                self.state = "half_open"
            elif self.state != "open":
                self.state = "half_open"
            return
        self.state = "closed"

    def allow(self) -> bool:
        if self.state != "open":
            return True
        if (time.time() - self.opened_at) >= self.recovery_seconds:
            self.state = "half_open"
            return True
        return False

    def snapshot(self) -> Dict[str, float | str | int]:
        budget = self.budget.snapshot()
        return {
            "state": self.state,
            "error_ratio": float(budget.get("error_ratio", 0.0)),
            "samples": int(budget.get("total", 0)),
            "warn_ratio": self.warn_ratio,
            "open_ratio": self.open_ratio,
            "recovery_seconds": self.recovery_seconds,
        }
