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


def sanitize_string(value: Optional[str], max_len: int = 1024) -> str:
    if value is None:
        return ""
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


class AuditLog:
    def __init__(self, service: str) -> None:
        self.service = service
        self.redis_url = os.getenv("REDIS_URL", "")
        self.stream_key = os.getenv("AUDIT_STREAM_KEY", "audit:logs")
        self.last_hash_key = os.getenv("AUDIT_LAST_HASH_KEY", "audit:last_hash")
        self.halt_on_broken = os.getenv("AUDIT_HALT_ON_BROKEN", "false").lower() == "true"
        self._redis = None
        if self.redis_url:
            try:
                import redis

                self._redis = redis.from_url(self.redis_url, decode_responses=True)
            except Exception:
                self._redis = None

    def _get_prev_hash(self) -> str:
        if not self._redis:
            return "GENESIS"
        prev = self._redis.get(self.last_hash_key)
        return prev or "GENESIS"

    def write(self, level: str, message: str) -> str:
        payload = {
            "timestamp": str(time.time()),
            "service": self.service,
            "level": level,
            "msg": sanitize_string(message, max_len=2048),
        }
        prev_hash = self._get_prev_hash()
        chain_material = f"{prev_hash}|{json.dumps(payload, sort_keys=True)}"
        entry_hash = hashlib.sha256(chain_material.encode("utf-8")).hexdigest()
        payload["hash"] = entry_hash
        if self._redis:
            self._redis.xadd(self.stream_key, payload)
            self._redis.set(self.last_hash_key, entry_hash)
        return entry_hash

    def verify_chain(self, max_entries: int = 2000) -> bool:
        if not self._redis:
            return True
        entries = self._redis.xrange(self.stream_key, count=max_entries)
        prev_hash = "GENESIS"
        for _, item in entries:
            material = {
                "timestamp": item.get("timestamp", ""),
                "service": item.get("service", ""),
                "level": item.get("level", ""),
                "msg": item.get("msg", ""),
            }
            expected = hashlib.sha256(f"{prev_hash}|{json.dumps(material, sort_keys=True)}".encode("utf-8")).hexdigest()
            if item.get("hash") != expected:
                return False
            prev_hash = expected
        return True


def init_audit_or_exit(service: str, logger: logging.Logger) -> AuditLog:
    audit = AuditLog(service=service)
    ok = audit.verify_chain()
    if not ok:
        logger.error("Audit chain verification failed on startup")
        audit.write("ERROR", "audit chain verification failed")
        if audit.halt_on_broken:
            raise RuntimeError("Audit chain verification failed")
    return audit
