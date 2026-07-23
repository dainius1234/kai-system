"""C10: A/B query logger — model name + response quality per query.

Appends one JSONL line per LLM query to AB_LOG_PATH (default:
logs/ab_query_log.jsonl).  Disabled when AB_LOG_ENABLED=false.

Each line:
{
  "ts": "<ISO-8601>",
  "specialist": "DeepSeek-V4",
  "model": "deepseek-v4",
  "source": "live" | "stub" | "error",
  "latency_ms": 120.4,
  "prompt_hash": "abc12345",       -- first 8 hex chars of sha256(prompt[:200])
  "session_id": "s-abc" | null,
  "word_count": 47,
  "lexical_diversity": 0.85,
  "uncertainty_penalty": 0.12,
  "net_quality_signal": 0.73,
  "input_tokens": 123,
  "output_tokens": 47
}

Usage:
    from common.ab_log import log_ab_entry
    log_ab_entry(response, prompt, session_id="s1")
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from common.llm import LLMResponse

logger = logging.getLogger(__name__)

AB_LOG_ENABLED = os.getenv("AB_LOG_ENABLED", "true").lower() not in {"0", "false", "no"}
AB_LOG_PATH = Path(os.getenv("AB_LOG_PATH", "logs/ab_query_log.jsonl"))

_lock = threading.Lock()


def _quality_fields(text: str) -> dict:
    """Compute response quality metrics without importing conviction."""
    import re
    _PUNCT = re.compile(r"[^\w\s]")
    _HEDGE = re.compile(
        r"\b(maybe|perhaps|possibly|might|could|uncertain|not sure|unclear|approximately|roughly)\b",
        re.IGNORECASE,
    )
    words = re.findall(r"\w+", text) if text else []
    word_count = len(words)
    unique_ratio = len(set(w.lower() for w in words)) / max(word_count, 1)
    hedge_count = len(_HEDGE.findall(text)) if text else 0
    uncertainty = round(min(hedge_count * 0.05, 0.5), 3)
    diversity = round(unique_ratio, 3)
    return {
        "word_count": word_count,
        "lexical_diversity": diversity,
        "uncertainty_penalty": uncertainty,
        "net_quality_signal": round(diversity - uncertainty, 3),
    }


def log_ab_entry(
    response: "LLMResponse",
    prompt: str = "",
    *,
    session_id: Optional[str] = None,
) -> None:
    """Write one JSONL entry to the A/B log.  Never raises."""
    if not AB_LOG_ENABLED:
        return
    try:
        prompt_hash = hashlib.sha256(prompt[:200].encode()).hexdigest()[:8]
        quality = _quality_fields(response.text)
        usage = response.usage or {}
        row = {
            "ts": datetime.now(tz=timezone.utc).isoformat(),
            "specialist": response.specialist,
            "model": response.model or response.specialist,
            "source": response.source,
            "latency_ms": response.latency_ms,
            "prompt_hash": prompt_hash,
            "session_id": session_id,
            **quality,
            "input_tokens": usage.get("prompt_tokens", 0),
            "output_tokens": usage.get("completion_tokens", 0),
        }
        line = json.dumps(row, ensure_ascii=False)
        with _lock:
            AB_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
            with AB_LOG_PATH.open("a", encoding="utf-8") as f:
                f.write(line + "\n")
    except Exception as exc:
        logger.debug("ab_log write failed: %s", exc)
