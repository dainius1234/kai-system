from __future__ import annotations
# Renamed from config.py to avoid shadowing the installed langgraph package.
# Import as: from kai_config import build_saver

import hashlib
import json
import os
import re
import time
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol

import redis


# ── Failure Taxonomy ─────────────────────────────────────────────────
# Every failed episode gets a failure_class explaining WHY it failed.
# The adversary uses these for targeted warnings and the planner uses
# them to extract metacognitive rules ("if X, never Y").

class FailureClass(str, Enum):
    DATA_INSUFFICIENT = "data_insufficient"
    POLICY_BLOCKED = "policy_blocked"
    CONFIDENCE_LOW = "confidence_low"
    OPERATOR_OVERRIDDEN = "operator_overridden"
    SERVICE_UNAVAILABLE = "service_unavailable"
    CONTRADICTED_BY_EVIDENCE = "contradicted_by_evidence"
    TIME_EXPIRED = "time_expired"
    SCOPE_EXCEEDED = "scope_exceeded"
    UNKNOWN = "unknown"


def classify_failure(
    episode: Dict[str, Any],
    gate_decision: Optional[Dict[str, Any]] = None,
) -> FailureClass:
    """Classify an episode's failure into a taxonomy category.

    Uses signals from the episode and gate decision to determine
    the root cause.  Order matters — more specific checks first.
    """
    outcome = float(episode.get("outcome_score", 0.5))
    if outcome >= 0.5:
        return FailureClass.UNKNOWN  # not a failure

    # Check gate decision signals
    if gate_decision:
        status = str(gate_decision.get("status", "")).lower()
        reason = str(gate_decision.get("reason", "")).lower()
        if status == "blocked" and "policy" in reason:
            return FailureClass.POLICY_BLOCKED
        if status == "blocked" and "circuit" in reason:
            return FailureClass.SERVICE_UNAVAILABLE
        if status == "unavailable":
            return FailureClass.SERVICE_UNAVAILABLE
        if not gate_decision.get("approved", True):
            return FailureClass.POLICY_BLOCKED

    # Check conviction / rethink signals
    rethink_count = int(episode.get("rethink_count", 0))
    conviction = float(episode.get("final_conviction", episode.get("conviction_score", 5.0)))
    if conviction < 6.0 and rethink_count >= 2:
        return FailureClass.CONFIDENCE_LOW

    # Check verifier signals
    verdict = str(episode.get("verifier_verdict", "")).upper()
    if verdict in ("FAIL_CLOSED", "REPAIR"):
        return FailureClass.CONTRADICTED_BY_EVIDENCE

    # Check for data insufficiency (low context coverage)
    context_used = int(episode.get("offline_context_used", -1))
    if context_used == 0:
        return FailureClass.DATA_INSUFFICIENT

    # Check for operator override
    if episode.get("conviction_override"):
        return FailureClass.OPERATOR_OVERRIDDEN

    # Check for scope/timeout signals
    if episode.get("scope_exceeded") or episode.get("time_expired"):
        return FailureClass.SCOPE_EXCEEDED if episode.get("scope_exceeded") else FailureClass.TIME_EXPIRED

    return FailureClass.CONFIDENCE_LOW if conviction < 8.0 else FailureClass.UNKNOWN


def extract_metacognitive_rule(
    episode: Dict[str, Any],
    failure_class: FailureClass,
) -> Optional[str]:
    """Extract an 'if X, never Y' rule from a failed episode.

    Uses the failure classification + episode content to generate
    a concrete, reusable constraint for future planning.
    Returns None if no meaningful rule can be extracted.
    """
    if failure_class == FailureClass.UNKNOWN:
        return None

    user_input = str(episode.get("input", ""))[:200]
    # Extract key topic words for rule specificity
    topic_words = sorted(set(re.findall(r"\w{4,}", user_input.lower())))[:5]
    topic = ", ".join(topic_words) if topic_words else "this topic"

    rules = {
        FailureClass.DATA_INSUFFICIENT: (
            f"if topic=[{topic}], always check memu-core for existing data "
            f"before asserting — last attempt had zero context chunks"
        ),
        FailureClass.POLICY_BLOCKED: (
            f"if topic=[{topic}], verify tool-gate policy allows the required "
            f"tool/action before planning — last attempt was policy-blocked"
        ),
        FailureClass.CONFIDENCE_LOW: (
            f"if topic=[{topic}], gather more evidence before proceeding — "
            f"conviction stayed below threshold after {episode.get('rethink_count', 0)} rethinks"
        ),
        FailureClass.OPERATOR_OVERRIDDEN: (
            f"if topic=[{topic}], note that operator overrode this result — "
            f"review operator's correction pattern before attempting similar tasks"
        ),
        FailureClass.SERVICE_UNAVAILABLE: (
            f"if topic=[{topic}], pre-check service health before depending on "
            f"external calls — last attempt failed due to service unavailability"
        ),
        FailureClass.CONTRADICTED_BY_EVIDENCE: (
            f"if topic=[{topic}], always cross-check with verifier before "
            f"asserting — last attempt was contradicted by evidence"
        ),
        FailureClass.TIME_EXPIRED: (
            f"if topic=[{topic}], set tighter time bounds or break into "
            f"smaller sub-tasks — last attempt ran out of time"
        ),
        FailureClass.SCOPE_EXCEEDED: (
            f"if topic=[{topic}], decompose into smaller scope — "
            f"last attempt exceeded actionable scope"
        ),
    }
    return rules.get(failure_class)


def compute_learning_value(
    conviction: float,
    outcome_score: float,
    rethink_count: int = 0,
) -> float:
    """SELAUR: Uncertainty-Aware Learning Value.

    Scales how valuable an episode is for future learning based on
    uncertainty and outcome.  The key insight: a failure when Kai was
    50% confident is MORE valuable than a failure when Kai was 90%
    confident.  The uncertain failure maps the edge of competence.

    Returns 0.0-1.0 learning value.

    Scoring logic:
    - Uncertainty = |conviction - 5.0| / 5.0 inverted: closer to 5.0 = more uncertain
    - High uncertainty + failure = highest learning value (frontier of growth)
    - High uncertainty + success = moderately valuable (frontier validated)
    - Low uncertainty + failure = valuable (calibration error — overconfident)
    - Low uncertainty + success = low value (already knew this would work)
    - Rethinks add value: the system worked hard → more to learn
    """
    # Uncertainty: 0.0 = very certain (conviction near 0 or 10), 1.0 = maximally uncertain (conviction near 5)
    uncertainty = 1.0 - abs(conviction - 5.0) / 5.0

    failed = outcome_score < 0.5
    overconfident = conviction >= 7.0 and failed

    if overconfident:
        # Calibration error: system was confident but wrong — high value
        value = 0.7 + 0.2 * uncertainty
    elif failed and uncertainty > 0.5:
        # Uncertain failure: the frontier of growth — maximum value
        value = 0.8 + 0.2 * uncertainty
    elif failed:
        # Certain failure: something went wrong despite confidence
        value = 0.5 + 0.2 * uncertainty
    elif uncertainty > 0.5:
        # Uncertain success: frontier validated — moderate value
        value = 0.4 + 0.3 * uncertainty
    else:
        # Certain success: low learning value, already competent here
        value = 0.1 + 0.2 * uncertainty

    # Rethinks add value: the harder it was, the more to learn
    rethink_bonus = min(rethink_count * 0.1, 0.2)
    return round(min(value + rethink_bonus, 1.0), 3)


class EpisodeSaver(Protocol):
    def save_episode(self, payload: Dict[str, Any]) -> None: ...

    def recall(self, user_id: str, days: int = 7) -> List[Dict[str, Any]]: ...

    def decay(self, user_id: str, days: int = 30, score_threshold: float = 0.2) -> int: ...


class RedisSaver:
    def __init__(self, redis_url: str) -> None:
        self.redis = redis.from_url(redis_url, decode_responses=True)

    def save_episode(self, payload: Dict[str, Any]) -> None:
        key = f"episodes:{payload['user_id']}"
        self.redis.lpush(key, json.dumps(payload))

    def recall(self, user_id: str, days: int = 7) -> List[Dict[str, Any]]:
        key = f"episodes:{user_id}"
        min_ts = time.time() - (days * 24 * 3600)
        items = self.redis.lrange(key, 0, 200)
        out: List[Dict[str, Any]] = []
        for raw in items:
            episode = json.loads(raw)
            if float(episode.get("ts", 0)) >= min_ts:
                out.append(episode)
        return out

    def decay(self, user_id: str, days: int = 30, score_threshold: float = 0.2) -> int:
        key = f"episodes:{user_id}"
        archive = f"episodes:archive:{user_id}"
        max_age = time.time() - (days * 24 * 3600)
        moved = 0
        kept: List[str] = []
        for raw in self.redis.lrange(key, 0, 1000):
            episode = json.loads(raw)
            too_old = float(episode.get("ts", 0)) < max_age
            low_score = float(episode.get("outcome_score", 0)) < score_threshold
            if too_old and low_score:
                self.redis.lpush(archive, raw)
                moved += 1
            else:
                kept.append(raw)
        pipe = self.redis.pipeline()
        pipe.delete(key)
        if kept:
            pipe.rpush(key, *reversed(kept))
        pipe.execute()
        return moved


class InMemorySaver:
    def __init__(self) -> None:
        self._episodes: Dict[str, List[Dict[str, Any]]] = {}
        self._archive: Dict[str, List[Dict[str, Any]]] = {}

    def save_episode(self, payload: Dict[str, Any]) -> None:
        user_id = str(payload.get("user_id", "keeper"))
        bucket = self._episodes.setdefault(user_id, [])
        bucket.insert(0, dict(payload))

    def recall(self, user_id: str, days: int = 7) -> List[Dict[str, Any]]:
        min_ts = time.time() - (days * 24 * 3600)
        return [e for e in self._episodes.get(user_id, []) if float(e.get("ts", 0)) >= min_ts][:200]

    def decay(self, user_id: str, days: int = 30, score_threshold: float = 0.2) -> int:
        max_age = time.time() - (days * 24 * 3600)
        moved = 0
        kept: List[Dict[str, Any]] = []
        archive = self._archive.setdefault(user_id, [])
        for episode in self._episodes.get(user_id, []):
            too_old = float(episode.get("ts", 0)) < max_age
            low_score = float(episode.get("outcome_score", 0)) < score_threshold
            if too_old and low_score:
                archive.append(episode)
                moved += 1
            else:
                kept.append(episode)
        self._episodes[user_id] = kept
        return moved


class ChecksummedSpoolSaver(InMemorySaver):
    def __init__(self, spool_path: str, max_bytes: int = 0) -> None:
        super().__init__()
        self.spool_path = Path(spool_path)
        self.spool_path.parent.mkdir(parents=True, exist_ok=True)
        # max_bytes=0 means use the env var or default 10 MB
        self.max_bytes = max_bytes or int(os.getenv("SPOOL_MAX_BYTES", str(10 * 1024 * 1024)))
        self._load_from_spool()

    def _line_for(self, payload: Dict[str, Any]) -> str:
        raw = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        checksum = hashlib.sha256(raw.encode("utf-8")).hexdigest()
        return json.dumps({"checksum": checksum, "payload": payload}, sort_keys=True)

    def _load_from_spool(self) -> None:
        if not self.spool_path.exists():
            return
        for line in self.spool_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
                payload = obj["payload"]
                checksum = str(obj.get("checksum", ""))
                raw = json.dumps(payload, sort_keys=True, separators=(",", ":"))
                expected = hashlib.sha256(raw.encode("utf-8")).hexdigest()
                if checksum != expected:
                    continue
                super().save_episode(payload)
            except Exception:
                continue

    def save_episode(self, payload: Dict[str, Any]) -> None:
        self._rotate_if_needed()
        line = self._line_for(payload)
        with self.spool_path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
            f.flush()
            os.fsync(f.fileno())
        super().save_episode(payload)

    def _rotate_if_needed(self) -> None:
        """Rotate spool when it exceeds max_bytes.

        Keeps the newest half of lines so the in-memory state stays
        approximately warm.  The rotated portion is written to a
        timestamped archive file alongside the spool.
        """
        if not self.spool_path.exists():
            return
        size = self.spool_path.stat().st_size
        if size <= self.max_bytes:
            return
        lines = self.spool_path.read_text(encoding="utf-8").splitlines()
        if not lines:
            return
        # keep the newest half
        keep = max(len(lines) // 2, 1)
        archive_lines = lines[:-keep]
        kept_lines = lines[-keep:]
        # write archive
        archive_path = self.spool_path.with_suffix(f".{int(time.time())}.archive")
        archive_path.write_text("\n".join(archive_lines) + "\n", encoding="utf-8")
        # rewrite spool with kept lines
        self.spool_path.write_text("\n".join(kept_lines) + "\n", encoding="utf-8")


def build_saver() -> EpisodeSaver:
    redis_url = os.getenv("REDIS_URL", "redis://redis:6379")
    prefer_memory = os.getenv("EPISODE_STORE", "redis").lower() == "memory"
    spool_path = os.getenv("EPISODE_SPOOL_PATH", "/tmp/langgraph_episode_spool.log")
    if prefer_memory:
        return ChecksummedSpoolSaver(spool_path=spool_path)

    try:
        saver = RedisSaver(redis_url=redis_url)
        saver.redis.ping()
        return saver
    except Exception:
        return ChecksummedSpoolSaver(spool_path=spool_path)
