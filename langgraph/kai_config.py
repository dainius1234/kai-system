from __future__ import annotations
# Renamed from config.py to avoid shadowing the installed langgraph package.
# Import as: from kai_config import build_saver

import hashlib
import json
import os
import re
import time
from dataclasses import dataclass
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


# ── GEM: Cognitive Alignment / Preference Extraction (P5) ───────────
# When the operator corrects Kai, extract a preference pattern.
# Format: "keeper prefers X over Y" or "keeper wants X when Y"
# These get stored in memu-core as high-importance preferences.

def extract_preference(
    original_output: str,
    correction: str,
    user_input: str,
) -> Optional[str]:
    """Extract an operator preference from a correction.

    Compares what Kai said vs what the operator corrected to,
    and generates a preference statement.  Returns None if no
    meaningful preference can be extracted.
    """
    if not correction or not original_output:
        return None

    # extract topic words from the user's original question
    topic_words = sorted(set(re.findall(r"\w{4,}", user_input.lower())))[:5]
    topic = ", ".join(topic_words) if topic_words else "general"

    # detect what changed
    orig_lower = original_output.lower()[:300]
    corr_lower = correction.lower()[:300]

    # find words unique to correction (what operator added/changed)
    orig_words = set(re.findall(r"\w{3,}", orig_lower))
    corr_words = set(re.findall(r"\w{3,}", corr_lower))
    new_words = corr_words - orig_words
    removed_words = orig_words - corr_words

    if not new_words and not removed_words:
        return None

    if new_words and removed_words:
        return (
            f"when topic=[{topic}], keeper prefers "
            f"'{' '.join(sorted(new_words)[:4])}' over "
            f"'{' '.join(sorted(removed_words)[:4])}'"
        )
    elif new_words:
        return (
            f"when topic=[{topic}], keeper wants emphasis on: "
            f"{' '.join(sorted(new_words)[:5])}"
        )
    else:
        return (
            f"when topic=[{topic}], keeper does NOT want: "
            f"{' '.join(sorted(removed_words)[:5])}"
        )


# ── Knowledge Boundary Tracker (P6) ─────────────────────────────────
# Aggregate episode outcomes per topic to map Kai's competence frontier.
# Topics with high failure rates or low conviction = knowledge gaps.

@dataclass
class TopicBoundary:
    """Competence snapshot for one topic cluster."""
    topic: str
    total_episodes: int = 0
    successes: int = 0
    failures: int = 0
    avg_conviction: float = 0.0
    avg_learning_value: float = 0.0
    is_gap: bool = False
    probe_question: str = ""


def build_knowledge_boundary(
    episodes: List[Dict[str, Any]],
    min_episodes: int = 2,
) -> List[TopicBoundary]:
    """Build a competence map from episode history.

    Groups episodes by topic keywords, computes success/failure rates,
    and flags gaps where Kai struggles or has low confidence.
    """
    # cluster episodes by topic words
    topic_clusters: Dict[str, List[Dict[str, Any]]] = {}
    for ep in episodes:
        input_text = str(ep.get("input", ""))
        words = sorted(set(re.findall(r"\w{4,}", input_text.lower())))[:3]
        if not words:
            continue
        topic_key = "+".join(words)
        topic_clusters.setdefault(topic_key, []).append(ep)

    boundaries: List[TopicBoundary] = []
    for topic_key, eps in topic_clusters.items():
        if len(eps) < min_episodes:
            continue

        total = len(eps)
        successes = sum(1 for e in eps if float(e.get("outcome_score", 0)) >= 0.5)
        failures = total - successes
        avg_conv = sum(float(e.get("final_conviction", e.get("conviction_score", 5.0))) for e in eps) / total
        avg_lv = sum(float(e.get("learning_value", 0.5)) for e in eps) / total
        success_rate = successes / total

        is_gap = success_rate < 0.5 or avg_conv < 6.0

        # generate a probing question for gaps
        probe = ""
        if is_gap:
            readable = topic_key.replace("+", ", ")
            if failures > successes:
                probe = f"I've struggled with [{readable}] — can you give me more context or examples?"
            else:
                probe = f"My confidence on [{readable}] is low — any corrections or guidance?"

        boundaries.append(TopicBoundary(
            topic=topic_key,
            total_episodes=total,
            successes=successes,
            failures=failures,
            avg_conviction=round(avg_conv, 2),
            avg_learning_value=round(avg_lv, 3),
            is_gap=is_gap,
            probe_question=probe,
        ))

    # sort: gaps first, then by failure count
    boundaries.sort(key=lambda b: (-int(b.is_gap), -b.failures))
    return boundaries


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


# ── P13: Recursive Self-Improvement Gate ─────────────────────────────
# Before any self-modification, snapshot current performance metrics.
# After the change, compare.  If performance degrades beyond tolerance,
# the change is flagged for revert.  This prevents Kai from accidentally
# making itself worse when tweaking its own rules or thresholds.

IMPROVEMENT_TOLERANCE = float(os.getenv("IMPROVEMENT_TOLERANCE", "0.1"))
IMPROVEMENT_SNAPSHOT_PATH = Path(
    os.getenv("IMPROVEMENT_SNAPSHOT_PATH", "/tmp/kai_improvement_snapshot.json")
)


@dataclass
class PerformanceSnapshot:
    """Point-in-time capture of system performance metrics."""
    timestamp: float
    avg_conviction: float
    avg_outcome: float
    failure_rate: float
    total_episodes: int
    rethink_rate: float
    label: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "avg_conviction": self.avg_conviction,
            "avg_outcome": self.avg_outcome,
            "failure_rate": self.failure_rate,
            "total_episodes": self.total_episodes,
            "rethink_rate": self.rethink_rate,
            "label": self.label,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "PerformanceSnapshot":
        return cls(
            timestamp=float(d.get("timestamp", 0)),
            avg_conviction=float(d.get("avg_conviction", 0)),
            avg_outcome=float(d.get("avg_outcome", 0)),
            failure_rate=float(d.get("failure_rate", 0)),
            total_episodes=int(d.get("total_episodes", 0)),
            rethink_rate=float(d.get("rethink_rate", 0)),
            label=str(d.get("label", "")),
        )


def capture_snapshot(
    episodes: List[Dict[str, Any]],
    label: str = "",
) -> PerformanceSnapshot:
    """Capture current performance metrics from episode history."""
    if not episodes:
        return PerformanceSnapshot(
            timestamp=time.time(),
            avg_conviction=0.0,
            avg_outcome=0.0,
            failure_rate=0.0,
            total_episodes=0,
            rethink_rate=0.0,
            label=label,
        )

    total = len(episodes)
    convictions = [float(e.get("final_conviction", e.get("conviction_score", 5.0))) for e in episodes]
    outcomes = [float(e.get("outcome_score", 0.5)) for e in episodes]
    failures = sum(1 for o in outcomes if o < 0.5)
    rethinks = sum(1 for e in episodes if int(e.get("rethink_count", 0)) > 0)

    return PerformanceSnapshot(
        timestamp=time.time(),
        avg_conviction=round(sum(convictions) / total, 3),
        avg_outcome=round(sum(outcomes) / total, 3),
        failure_rate=round(failures / total, 3),
        total_episodes=total,
        rethink_rate=round(rethinks / total, 3),
        label=label,
    )


def save_snapshot(snapshot: PerformanceSnapshot) -> None:
    """Persist a performance snapshot to disk."""
    try:
        existing: List[Dict[str, Any]] = []
        if IMPROVEMENT_SNAPSHOT_PATH.exists():
            existing = json.loads(
                IMPROVEMENT_SNAPSHOT_PATH.read_text(encoding="utf-8")
            )
        existing.append(snapshot.to_dict())
        # keep last 50 snapshots
        existing = existing[-50:]
        IMPROVEMENT_SNAPSHOT_PATH.write_text(
            json.dumps(existing, indent=2), encoding="utf-8"
        )
    except Exception:
        pass


def load_latest_snapshot() -> Optional[PerformanceSnapshot]:
    """Load the most recent performance snapshot from disk."""
    try:
        if not IMPROVEMENT_SNAPSHOT_PATH.exists():
            return None
        data = json.loads(
            IMPROVEMENT_SNAPSHOT_PATH.read_text(encoding="utf-8")
        )
        if not data:
            return None
        return PerformanceSnapshot.from_dict(data[-1])
    except Exception:
        return None


@dataclass
class ImprovementVerdict:
    """Result of comparing before/after performance snapshots."""
    approved: bool
    delta_conviction: float
    delta_outcome: float
    delta_failure_rate: float
    degraded_metrics: List[str]
    improved_metrics: List[str]
    recommendation: str


def evaluate_improvement(
    before: PerformanceSnapshot,
    after: PerformanceSnapshot,
    tolerance: float = IMPROVEMENT_TOLERANCE,
) -> ImprovementVerdict:
    """Compare two snapshots and decide if the change helped or hurt.

    A metric is considered degraded if it worsened beyond the tolerance.
    For failure_rate and rethink_rate, LOWER is better (inverted check).
    """
    delta_conv = after.avg_conviction - before.avg_conviction
    delta_out = after.avg_outcome - before.avg_outcome
    delta_fail = after.failure_rate - before.failure_rate
    delta_rethink = after.rethink_rate - before.rethink_rate

    degraded: List[str] = []
    improved: List[str] = []

    # conviction: higher is better
    if delta_conv < -tolerance:
        degraded.append("avg_conviction")
    elif delta_conv > tolerance:
        improved.append("avg_conviction")

    # outcome: higher is better
    if delta_out < -tolerance:
        degraded.append("avg_outcome")
    elif delta_out > tolerance:
        improved.append("avg_outcome")

    # failure_rate: lower is better (inverted)
    if delta_fail > tolerance:
        degraded.append("failure_rate")
    elif delta_fail < -tolerance:
        improved.append("failure_rate")

    # rethink_rate: lower is better (inverted)
    if delta_rethink > tolerance:
        degraded.append("rethink_rate")
    elif delta_rethink < -tolerance:
        improved.append("rethink_rate")

    approved = len(degraded) == 0

    if not approved:
        recommendation = (
            f"Revert recommended: {', '.join(degraded)} degraded beyond tolerance ({tolerance})."
        )
    elif improved:
        recommendation = f"Change approved: {', '.join(improved)} improved."
    else:
        recommendation = "Change neutral: no significant metric shifts."

    return ImprovementVerdict(
        approved=approved,
        delta_conviction=round(delta_conv, 3),
        delta_outcome=round(delta_out, 3),
        delta_failure_rate=round(delta_fail, 3),
        degraded_metrics=degraded,
        improved_metrics=improved,
        recommendation=recommendation,
    )


# ── P15: Dream State — Offline Consolidation Engine ─────────────────
# When Kai is idle, it consolidates knowledge: cross-referencing episodes,
# deduplicating metacognitive rules, clustering failures, synthesizing
# new insights, and recalibrating the knowledge boundary.

DREAM_MIN_EPISODES = int(os.getenv("DREAM_MIN_EPISODES", "5"))
DREAM_RULE_SIMILARITY_THRESHOLD = float(os.getenv("DREAM_RULE_SIMILARITY", "0.6"))
DREAM_INSIGHT_PATH = Path(os.getenv("DREAM_INSIGHT_PATH", "/tmp/kai_dream_insights.json"))


@dataclass
class DreamInsight:
    """A single insight produced during a dream cycle."""
    insight_type: str       # "pattern", "rule_merge", "boundary_shift", "failure_cluster", "contradiction"
    description: str
    confidence: float       # 0.0–1.0
    source_episodes: int    # how many episodes contributed
    actionable: bool        # whether this insight suggests a behaviour change

    def to_dict(self) -> Dict[str, Any]:
        return {
            "insight_type": self.insight_type,
            "description": self.description,
            "confidence": round(self.confidence, 3),
            "source_episodes": self.source_episodes,
            "actionable": self.actionable,
        }


@dataclass
class DreamCycle:
    """Results of a complete dream consolidation cycle."""
    cycle_id: str
    ts: float
    episodes_analysed: int
    insights: List[DreamInsight]
    merged_rules: int
    failure_clusters: Dict[str, int]
    boundary_shifts: List[Dict[str, Any]]
    duration_ms: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "cycle_id": self.cycle_id,
            "ts": self.ts,
            "episodes_analysed": self.episodes_analysed,
            "insights": [i.to_dict() for i in self.insights],
            "merged_rules": self.merged_rules,
            "failure_clusters": self.failure_clusters,
            "boundary_shifts": self.boundary_shifts,
            "duration_ms": round(self.duration_ms, 1),
        }


def _extract_words(text: str) -> set:
    """Extract normalised word tokens from text."""
    return set(re.findall(r"\w{3,}", text.lower()))


def _word_overlap(a: str, b: str) -> float:
    """Jaccard similarity between word sets of two strings."""
    wa, wb = _extract_words(a), _extract_words(b)
    if not wa or not wb:
        return 0.0
    return len(wa & wb) / len(wa | wb)


def deduplicate_rules(rules: List[str], threshold: float = DREAM_RULE_SIMILARITY_THRESHOLD) -> List[str]:
    """Merge near-duplicate metacognitive rules.

    Keeps the longest version of each cluster (most specific).
    Returns the deduplicated list.
    """
    if not rules:
        return []
    kept: List[str] = []
    for rule in sorted(rules, key=len, reverse=True):
        if not any(_word_overlap(rule, k) >= threshold for k in kept):
            kept.append(rule)
    return kept


def cluster_failures(episodes: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """Group failed episodes by their failure class.

    Returns a dict of failure_class → list of episode summaries.
    """
    clusters: Dict[str, List[Dict[str, Any]]] = {}
    for ep in episodes:
        fc = ep.get("failure_class")
        if not fc:
            continue
        if fc not in clusters:
            clusters[fc] = []
        clusters[fc].append({
            "input": ep.get("input", "")[:200],
            "conviction": ep.get("final_conviction", ep.get("conviction_score", 0)),
            "ts": ep.get("ts", 0),
        })
    return clusters


def synthesize_patterns(episodes: List[Dict[str, Any]]) -> List[DreamInsight]:
    """Discover recurring patterns across episodes.

    Looks for:
    1. Topics that consistently have low conviction (struggling areas)
    2. Topics with high rethink rates (complex areas requiring iteration)
    3. Conviction improvement trends (learning signals)
    """
    insights: List[DreamInsight] = []
    if len(episodes) < DREAM_MIN_EPISODES:
        return insights

    # Group by topic keywords (first 3 significant words)
    topic_groups: Dict[str, List[Dict[str, Any]]] = {}
    for ep in episodes:
        words = sorted(_extract_words(ep.get("input", "")))[:3]
        key = " ".join(words) if words else "unknown"
        if key not in topic_groups:
            topic_groups[key] = []
        topic_groups[key].append(ep)

    for topic, eps in topic_groups.items():
        if len(eps) < 2:
            continue

        convictions = [float(e.get("final_conviction", e.get("conviction_score", 0))) for e in eps]
        rethinks = [int(e.get("rethink_count", 0)) for e in eps]
        avg_conv = sum(convictions) / len(convictions)
        avg_rethink = sum(rethinks) / len(rethinks)

        # Struggling topic: consistently low conviction
        if avg_conv < 6.0 and len(eps) >= 3:
            insights.append(DreamInsight(
                insight_type="pattern",
                description=f"Struggling with '{topic}': avg conviction {avg_conv:.1f}/10 across {len(eps)} episodes.",
                confidence=min(0.5 + len(eps) * 0.1, 0.95),
                source_episodes=len(eps),
                actionable=True,
            ))

        # Complex topic: high rethink rate
        if avg_rethink >= 1.5:
            insights.append(DreamInsight(
                insight_type="pattern",
                description=f"Topic '{topic}' requires frequent rethinking: avg {avg_rethink:.1f} rethinks/episode.",
                confidence=min(0.4 + len(eps) * 0.1, 0.9),
                source_episodes=len(eps),
                actionable=True,
            ))

        # Learning trend: conviction improving over time
        if len(eps) >= 3:
            sorted_eps = sorted(eps, key=lambda e: e.get("ts", 0))
            early = [float(e.get("final_conviction", 0)) for e in sorted_eps[:len(sorted_eps)//2]]
            late = [float(e.get("final_conviction", 0)) for e in sorted_eps[len(sorted_eps)//2:]]
            if early and late:
                early_avg = sum(early) / len(early)
                late_avg = sum(late) / len(late)
                if late_avg - early_avg > 1.0:
                    insights.append(DreamInsight(
                        insight_type="pattern",
                        description=f"Learning detected for '{topic}': conviction improved from {early_avg:.1f} to {late_avg:.1f}.",
                        confidence=0.8,
                        source_episodes=len(eps),
                        actionable=False,
                    ))

    return insights


def detect_rule_contradictions(rules: List[str]) -> List[DreamInsight]:
    """Find metacognitive rules that contradict each other.

    Simple heuristic: two rules about the same topic with opposing advice
    (e.g. "always X" vs "never X").
    """
    insights: List[DreamInsight] = []
    for i, r1 in enumerate(rules):
        for r2 in rules[i + 1:]:
            overlap = _word_overlap(r1, r2)
            if overlap < 0.3:
                continue
            # Check for opposing signals
            r1_lower, r2_lower = r1.lower(), r2.lower()
            has_conflict = (
                ("always" in r1_lower and "never" in r2_lower)
                or ("never" in r1_lower and "always" in r2_lower)
                or ("should" in r1_lower and "should not" in r2_lower)
                or ("should not" in r1_lower and "should" in r2_lower)
            )
            if has_conflict:
                insights.append(DreamInsight(
                    insight_type="contradiction",
                    description=f"Contradictory rules detected: '{r1[:80]}' vs '{r2[:80]}'",
                    confidence=overlap,
                    source_episodes=0,
                    actionable=True,
                ))
    return insights


def run_dream_cycle(
    episodes: List[Dict[str, Any]],
    cycle_id: Optional[str] = None,
) -> DreamCycle:
    """Execute a complete dream consolidation cycle.

    Phases:
    1. Cluster failures by class
    2. Extract and deduplicate metacognitive rules
    3. Synthesize recurring patterns
    4. Detect rule contradictions
    5. Recalibrate knowledge boundary
    6. Package insights

    This is meant to run during idle periods (triggered by heartbeat
    when the operator goes quiet for > 30 minutes).
    """
    start = time.monotonic()
    cycle_id = cycle_id or hashlib.sha256(str(time.time()).encode()).hexdigest()[:12]
    insights: List[DreamInsight] = []

    # Phase 1: Cluster failures
    failure_clusters = cluster_failures(episodes)
    cluster_counts = {k: len(v) for k, v in failure_clusters.items()}
    for fc, eps in failure_clusters.items():
        if len(eps) >= 3:
            insights.append(DreamInsight(
                insight_type="failure_cluster",
                description=f"Recurring failure cluster '{fc}': {len(eps)} episodes. Dominant pattern needs attention.",
                confidence=min(0.5 + len(eps) * 0.1, 0.95),
                source_episodes=len(eps),
                actionable=True,
            ))

    # Phase 2: Deduplicate metacognitive rules
    all_rules = [ep.get("metacognitive_rule", "") for ep in episodes if ep.get("metacognitive_rule")]
    deduped_rules = deduplicate_rules(all_rules)
    merged_count = len(all_rules) - len(deduped_rules)

    if merged_count > 0:
        insights.append(DreamInsight(
            insight_type="rule_merge",
            description=f"Merged {merged_count} duplicate metacognitive rules (kept {len(deduped_rules)}).",
            confidence=0.9,
            source_episodes=len(all_rules),
            actionable=False,
        ))

    # Phase 3: Synthesize patterns
    pattern_insights = synthesize_patterns(episodes)
    insights.extend(pattern_insights)

    # Phase 4: Detect rule contradictions
    contradiction_insights = detect_rule_contradictions(deduped_rules)
    insights.extend(contradiction_insights)

    # Phase 5: Recalibrate knowledge boundary
    boundary = build_knowledge_boundary(episodes)
    boundary_shifts: List[Dict[str, Any]] = []
    for b in boundary:
        if b.is_gap:
            boundary_shifts.append({
                "topic": b.topic,
                "success_rate": round(b.successes / max(b.total_episodes, 1), 2),
                "avg_conviction": round(b.avg_conviction, 1),
                "probe_question": b.probe_question,
            })
            insights.append(DreamInsight(
                insight_type="boundary_shift",
                description=f"Knowledge gap in '{b.topic}': {b.successes}/{b.total_episodes} success rate, avg conviction {b.avg_conviction:.1f}.",
                confidence=min(0.6 + b.total_episodes * 0.05, 0.95),
                source_episodes=b.total_episodes,
                actionable=True,
            ))

    elapsed = (time.monotonic() - start) * 1000

    cycle = DreamCycle(
        cycle_id=cycle_id,
        ts=time.time(),
        episodes_analysed=len(episodes),
        insights=insights,
        merged_rules=merged_count,
        failure_clusters=cluster_counts,
        boundary_shifts=boundary_shifts,
        duration_ms=elapsed,
    )

    # Persist the dream cycle
    save_dream_cycle(cycle)

    return cycle


def save_dream_cycle(cycle: DreamCycle) -> None:
    """Persist a dream cycle to disk."""
    try:
        if DREAM_INSIGHT_PATH.exists():
            data = json.loads(DREAM_INSIGHT_PATH.read_text(encoding="utf-8"))
        else:
            data = {"cycles": []}
        data["cycles"].append(cycle.to_dict())
        # Keep last 20 cycles
        data["cycles"] = data["cycles"][-20:]
        DREAM_INSIGHT_PATH.write_text(json.dumps(data, indent=2), encoding="utf-8")
    except Exception:
        pass


def load_dream_cycles() -> List[Dict[str, Any]]:
    """Load all persisted dream cycles."""
    try:
        if DREAM_INSIGHT_PATH.exists():
            data = json.loads(DREAM_INSIGHT_PATH.read_text(encoding="utf-8"))
            return data.get("cycles", [])
    except Exception:
        pass
    return []
