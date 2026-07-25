"""D98: Cognitive Fingerprinting — operator thinking-style model.

Builds a model of how the operator thinks: preferred reasoning style,
risk tolerance, time horizon, decision velocity, abstraction level.

Two phases:
  Phase 0 (NOW): Collect InteractionSample records from every chat interaction.
                 Write to /data/cognitive_fingerprint.jsonl. Target: 90+ samples.
  Phase 1 (GPU): Cluster samples → infer stable thinking-style dimensions →
                 inject fingerprint into each agentic context for style matching.

can_infer() returns False until ≥90 samples are collected. Collecting starts
immediately because the fingerprint only improves with more samples.

Feature flag: FF_COGNITIVE_FINGERPRINT (default True — collecting now)
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("kai.cognitive_fingerprint")

FINGERPRINT_LOG = Path("/data/cognitive_fingerprint.jsonl")
INFERENCE_THRESHOLD = 90  # minimum samples before style inference is meaningful


@dataclass
class InteractionSample:
    query: str
    response_length_preference: str    # short | medium | long (inferred from follow-up)
    decision_made: bool                # did operator make a concrete decision?
    abstraction_level: str             # concrete | mixed | abstract
    time_horizon: str                  # immediate | near | long
    risk_signal: str                   # risk_averse | neutral | risk_tolerant
    query_type: str                    # question | directive | exploration | feedback
    timestamp: float = field(default_factory=time.time)
    session_id: str = ""
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CognitiveFingerprint:
    """Inferred thinking-style model — populated only after ≥90 samples."""
    dominant_style: str = "unknown"
    risk_tolerance: float = 0.5        # 0=averse, 1=tolerant
    preferred_abstraction: str = "mixed"
    typical_time_horizon: str = "near"
    decision_velocity: float = 0.5     # 0=deliberate, 1=fast
    sample_count: int = 0
    confidence: float = 0.0


class CognitiveFingerprintCollector:
    """Collects interaction samples and (eventually) infers thinking style."""

    def __init__(self) -> None:
        self._sample_count: Optional[int] = None

    def record(self, sample: InteractionSample) -> None:
        """Append one interaction sample to the fingerprint log."""
        try:
            FINGERPRINT_LOG.parent.mkdir(parents=True, exist_ok=True)
            with FINGERPRINT_LOG.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(asdict(sample)) + "\n")
            self._sample_count = None  # invalidate cached count
            logger.debug("Cognitive fingerprint sample recorded")
        except Exception as exc:
            logger.debug("Could not write fingerprint sample: %s", exc)

    def sample_count(self) -> int:
        if self._sample_count is None:
            try:
                if FINGERPRINT_LOG.exists():
                    self._sample_count = sum(1 for _ in FINGERPRINT_LOG.open())
                else:
                    self._sample_count = 0
            except Exception:
                self._sample_count = 0
        return self._sample_count

    def can_infer(self) -> bool:
        """True when enough samples exist AND FF_COGNITIVE_FINGERPRINT is enabled."""
        try:
            from feature_flags import is_enabled
            if not is_enabled("COGNITIVE_FINGERPRINT"):
                return False
        except ImportError:
            pass
        return self.sample_count() >= INFERENCE_THRESHOLD

    def infer(self) -> CognitiveFingerprint:
        """Return inferred fingerprint (stub until Phase 1 GPU inference)."""
        count = self.sample_count()
        fp = CognitiveFingerprint(sample_count=count)
        if not self.can_infer():
            logger.debug(
                "D98 fingerprint inference pending: %d/%d samples collected",
                count, INFERENCE_THRESHOLD,
            )
            fp.dominant_style = "stub_pending_inference"
            return fp
        # Phase 1: cluster samples → PCA/k-means → map to style dimensions
        fp.dominant_style = "stub_pending_gpu_clustering"
        fp.confidence = 0.0
        return fp

    def progress(self) -> Dict[str, Any]:
        count = self.sample_count()
        return {
            "samples_collected": count,
            "inference_threshold": INFERENCE_THRESHOLD,
            "ready_for_inference": count >= INFERENCE_THRESHOLD,
            "progress_pct": round(min(100.0, count / INFERENCE_THRESHOLD * 100), 1),
        }


# Module-level singleton — import and call .record() from the chat handler
collector = CognitiveFingerprintCollector()


def quick_sample(
    query: str,
    session_id: str = "",
    query_type: str = "question",
) -> InteractionSample:
    """Build a minimal InteractionSample from a raw query string.

    Heuristics infer fields from surface query features.
    Call collector.record(quick_sample(query)) from the chat handler.
    """
    q_lower = query.lower()
    return InteractionSample(
        query=query[:200],
        response_length_preference=(
            "short" if len(query) < 40 else "long" if len(query) > 200 else "medium"
        ),
        decision_made=any(w in q_lower for w in ("do it", "go ahead", "yes", "ok", "sure", "proceed")),
        abstraction_level=(
            "abstract" if any(w in q_lower for w in ("concept", "theory", "why", "how does")) else
            "concrete" if any(w in q_lower for w in ("show me", "example", "how to", "step")) else
            "mixed"
        ),
        time_horizon=(
            "immediate" if any(w in q_lower for w in ("now", "today", "quick", "fast")) else
            "long" if any(w in q_lower for w in ("future", "plan", "eventually", "roadmap")) else
            "near"
        ),
        risk_signal=(
            "risk_averse" if any(w in q_lower for w in ("safe", "careful", "avoid", "risk")) else
            "risk_tolerant" if any(w in q_lower for w in ("aggressive", "bold", "try", "experiment")) else
            "neutral"
        ),
        query_type=query_type,
        session_id=session_id,
    )
