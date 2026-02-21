from __future__ import annotations

import logging
import re
import time
from collections import deque
from logging.handlers import TimedRotatingFileHandler
from typing import Deque, Dict, Tuple


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
