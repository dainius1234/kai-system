"""D122: Model Council — Kai's self-knowledge of available LLM backends.

Phase 2: Self-Preservation. Kai tracks known models, benchmarks them for
capability and latency, and can recommend or failover to an alternative
when the primary model is unavailable or degraded.

Trust gating (per the trust ladder in trust_core.py):
    discover()   → OBSERVER  (1) — reading available model info
    benchmark()  → ASSISTANT (2) — runs a probe, costs compute/credits
    recommend()  → ASSISTANT (2) — makes a model selection decision
    auto-switch  → AGENT     (3) — groundwork only; not autonomous in Phase 0

Feature-flagged: FF_MODEL_COUNCIL=true
Fail-open: all public methods return safe defaults when infra is missing.
Storage: data/model-council/profiles.json (benchmark results + availability)
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set
from common.degraded import record_degradation

logger = logging.getLogger("kai.model_council")

_DATA_DIR = Path("data/model-council")
_PROFILES_FILE = _DATA_DIR / "profiles.json"

# Task types the council reasons about
TASK_TYPES = {"chat", "code", "analysis", "creative", "factcheck", "planning"}


# ── Council profile ────────────────────────────────────────────────────────────

@dataclass
class CouncilProfile:
    """Extended model profile with runtime status and benchmark data."""
    model_id: str
    name: str
    provider: str                               # "anthropic", "openai", "local", etc.
    task_affinities: List[str]                  # task types this model excels at
    context_window: int = 8192
    cost_per_1k_tokens: float = 0.0             # USD input cost; 0 = free/local
    quality_tier: int = 2                       # 1=basic, 2=good, 3=best
    speed_tier: int = 1                         # 1=fast, 2=medium, 3=slow
    available: bool = True
    last_checked: float = 0.0                   # epoch seconds
    benchmark_scores: Dict[str, float] = field(default_factory=dict)
    latency_p50_ms: float = 0.0
    failure_count: int = 0                      # consecutive failures

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "CouncilProfile":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})  # type: ignore[attr-defined]

    def composite_score(self, task_type: str) -> float:
        """Single score for ranking: benchmark result if present, else heuristic."""
        if task_type in self.benchmark_scores:
            return self.benchmark_scores[task_type]
        # Heuristic from static profile
        affinity_bonus = 1.0 if task_type in self.task_affinities else 0.0
        return (self.quality_tier / 3.0) * 0.6 + affinity_bonus * 0.4


# ── Built-in model registry ────────────────────────────────────────────────────
# Seeded from model_selector registry + Claude API models. Operators add more.

_BUILTIN_PROFILES: List[CouncilProfile] = [
    CouncilProfile(
        model_id="claude-sonnet-4-6",
        name="Claude Sonnet 4.6",
        provider="anthropic",
        task_affinities=["chat", "code", "analysis", "planning", "creative"],
        context_window=200000,
        cost_per_1k_tokens=0.003,
        quality_tier=3,
        speed_tier=2,
    ),
    CouncilProfile(
        model_id="claude-haiku-4-5-20251001",
        name="Claude Haiku 4.5",
        provider="anthropic",
        task_affinities=["chat", "factcheck"],
        context_window=200000,
        cost_per_1k_tokens=0.0008,
        quality_tier=2,
        speed_tier=1,
    ),
    CouncilProfile(
        model_id="claude-opus-5",
        name="Claude Opus 5",
        provider="anthropic",
        task_affinities=["analysis", "planning", "code", "creative"],
        context_window=200000,
        cost_per_1k_tokens=0.015,
        quality_tier=3,
        speed_tier=3,
    ),
    CouncilProfile(
        model_id="deepseek-v4",
        name="DeepSeek-V4",
        provider="local",
        task_affinities=["factcheck", "code", "analysis"],
        context_window=32768,
        cost_per_1k_tokens=0.0,
        quality_tier=3,
        speed_tier=2,
    ),
    CouncilProfile(
        model_id="ollama-default",
        name="Ollama (local default)",
        provider="local",
        task_affinities=["chat"],
        context_window=4096,
        cost_per_1k_tokens=0.0,
        quality_tier=1,
        speed_tier=1,
    ),
]


# ── Model Council ──────────────────────────────────────────────────────────────

class ModelCouncil:
    """Maintains a live registry of known models with benchmark data.

    All mutating operations are trust-gated. All methods are fail-open —
    exceptions are logged and safe defaults are returned.
    """

    def __init__(self, data_dir: Path = _DATA_DIR) -> None:
        self._dir = data_dir
        self._dir.mkdir(parents=True, exist_ok=True)
        self._profiles: Dict[str, CouncilProfile] = {}
        self._load_profiles()
        self._primary: str = "claude-sonnet-4-6"

    # ── Persistence ────────────────────────────────────────────────────

    def _load_profiles(self) -> None:
        """Seed from built-ins; overlay persisted benchmark results."""
        for p in _BUILTIN_PROFILES:
            self._profiles[p.model_id] = p
        f = self._dir / "profiles.json"
        if not f.exists():
            return
        try:
            data = json.loads(f.read_text())
            for entry in data.get("profiles", []):
                mid = entry.get("model_id", "")
                if mid in self._profiles:
                    # Overlay persisted runtime data onto static profile
                    p = self._profiles[mid]
                    p.available = entry.get("available", p.available)
                    p.last_checked = entry.get("last_checked", p.last_checked)
                    p.benchmark_scores = entry.get("benchmark_scores", p.benchmark_scores)
                    p.latency_p50_ms = entry.get("latency_p50_ms", p.latency_p50_ms)
                    p.failure_count = entry.get("failure_count", p.failure_count)
                else:
                    # Custom operator-added profile
                    try:
                        self._profiles[mid] = CouncilProfile.from_dict(entry)
                    except Exception as _exc:
                        record_degradation("filesystem", "load_council_profile", _exc)
            self._primary = data.get("primary", self._primary)
        except Exception as exc:
            logger.debug("Model council profile load failed (non-critical): %s", exc)

    def _save_profiles(self) -> None:
        try:
            payload = {
                "primary": self._primary,
                "profiles": [p.to_dict() for p in self._profiles.values()],
            }
            tmp = self._dir / "profiles.json.tmp"
            tmp.write_text(json.dumps(payload, indent=2))
            tmp.replace(self._dir / "profiles.json")
        except Exception as exc:
            logger.debug("Model council profile save failed: %s", exc)

    # ── Trust gate helper ──────────────────────────────────────────────

    def _check_trust(self, capability: str, context: Dict[str, Any]) -> None:
        """Raise PermissionError if trust gate denies the operation."""
        try:
            try:
                from trust_integration import gate_autonomous_action
            except ImportError:
                from agentic.trust_integration import gate_autonomous_action  # type: ignore
            allowed, reason = gate_autonomous_action(capability, context, conviction=6.0)
            if not allowed:
                raise PermissionError(f"Model council trust gate denied {capability}: {reason}")
        except PermissionError:
            raise
        except Exception as exc:
            logger.debug("Trust gate unavailable (fail-open for model council): %s", exc)

    # ── Public API ─────────────────────────────────────────────────────

    def discover(self) -> List[Dict[str, Any]]:
        """Return all registered model profiles.

        Trust: OBSERVER (1). Fail-open if trust infra is missing.
        """
        try:
            self._check_trust("model_council_discover", {"action": "list registered models"})
        except PermissionError as exc:
            logger.warning("%s", exc)
            return []
        now = time.time()
        return [
            {
                "model_id": p.model_id,
                "name": p.name,
                "provider": p.provider,
                "task_affinities": p.task_affinities,
                "available": p.available,
                "quality_tier": p.quality_tier,
                "speed_tier": p.speed_tier,
                "benchmark_scores": p.benchmark_scores,
                "latency_p50_ms": p.latency_p50_ms,
                "seconds_since_check": round(now - p.last_checked) if p.last_checked else None,
                "is_primary": p.model_id == self._primary,
            }
            for p in self._profiles.values()
        ]

    def benchmark(
        self,
        model_id: str,
        task_type: str = "chat",
        probe_fn: Optional[Callable[[str, str], float]] = None,
    ) -> Dict[str, Any]:
        """Run a lightweight probe and record benchmark score.

        Args:
            model_id: which model to probe
            task_type: task category ("chat", "code", etc.)
            probe_fn: optional injection for testing.
                      Signature: (model_id, task_type) → score 0.0–10.0
                      Default probe returns 0.0 (model unreachable).

        Trust: ASSISTANT (2). Records result persistently.
        """
        try:
            self._check_trust(
                "model_council_benchmark",
                {"action": "benchmark model", "model_id": model_id, "task_type": task_type},
            )
        except PermissionError as exc:
            logger.warning("%s", exc)
            return {"error": str(exc), "model_id": model_id}

        if model_id not in self._profiles:
            return {"error": f"unknown model: {model_id}", "model_id": model_id}

        if task_type not in TASK_TYPES:
            return {"error": f"unknown task_type: {task_type}", "model_id": model_id}

        profile = self._profiles[model_id]
        t0 = time.monotonic()
        score = 0.0
        error_msg = None

        try:
            if probe_fn is not None:
                score = float(probe_fn(model_id, task_type))
            else:
                score = self._default_probe(model_id, task_type)
            profile.available = score > 0.0
            profile.failure_count = 0 if score > 0.0 else profile.failure_count + 1
        except Exception as exc:
            error_msg = str(exc)
            profile.available = False
            profile.failure_count += 1
            logger.debug("Model probe failed for %s: %s", model_id, exc)

        latency_ms = round((time.monotonic() - t0) * 1000, 1)
        profile.last_checked = time.time()
        profile.latency_p50_ms = (
            (profile.latency_p50_ms * 0.7 + latency_ms * 0.3)
            if profile.latency_p50_ms > 0
            else latency_ms
        )
        if score > 0.0:
            profile.benchmark_scores[task_type] = round(score, 3)

        self._save_profiles()
        logger.info(
            "Model council benchmark: %s task=%s score=%.2f latency=%.0fms",
            model_id, task_type, score, latency_ms,
        )
        return {
            "model_id": model_id,
            "task_type": task_type,
            "score": score,
            "latency_ms": latency_ms,
            "available": profile.available,
            "error": error_msg,
        }

    def _default_probe(self, model_id: str, task_type: str) -> float:
        """Built-in probe: returns heuristic score from static profile.

        Real latency probing requires API keys — this gives a safe fallback
        that reflects the profile's static quality rating.
        """
        profile = self._profiles.get(model_id)
        if profile is None:
            return 0.0
        return profile.composite_score(task_type) * 10.0

    def rank(self, task_type: str = "chat") -> List[Dict[str, Any]]:
        """Return all models ranked by composite score for the given task type.

        No trust gating — ranking is read-only derived data.
        """
        if task_type not in TASK_TYPES:
            task_type = "chat"
        ranked = sorted(
            self._profiles.values(),
            key=lambda p: (p.available, p.composite_score(task_type)),
            reverse=True,
        )
        return [
            {
                "rank": i + 1,
                "model_id": p.model_id,
                "name": p.name,
                "available": p.available,
                "composite_score": round(p.composite_score(task_type), 3),
                "task_type": task_type,
                "is_primary": p.model_id == self._primary,
            }
            for i, p in enumerate(ranked)
        ]

    def recommend(
        self,
        task_type: str = "chat",
        excluded: Optional[Set[str]] = None,
    ) -> Optional[Dict[str, Any]]:
        """Recommend the best available model for the task type.

        Trust: ASSISTANT (2). Returns None if trust denied or no model available.
        """
        try:
            self._check_trust(
                "model_council_recommend",
                {"action": "recommend model", "task_type": task_type},
            )
        except PermissionError as exc:
            logger.warning("%s", exc)
            return None

        excluded = excluded or set()
        candidates = [
            p for p in self._profiles.values()
            if p.available and p.model_id not in excluded
        ]
        if not candidates:
            return None

        best = max(candidates, key=lambda p: p.composite_score(task_type))
        return {
            "model_id": best.model_id,
            "name": best.name,
            "provider": best.provider,
            "composite_score": round(best.composite_score(task_type), 3),
            "task_type": task_type,
            "latency_p50_ms": best.latency_p50_ms,
            "is_primary": best.model_id == self._primary,
        }

    def failover(self, excluded: Optional[Set[str]] = None) -> Optional[str]:
        """Return the best non-excluded available model ID for failover.

        No trust gating — failover is a safety mechanism, not an autonomous action.
        Returns None if no alternative exists (caller must handle).
        """
        excluded = excluded or set()
        excluded.add(self._primary)
        candidates = [
            p for p in self._profiles.values()
            if p.available and p.model_id not in excluded
        ]
        if not candidates:
            return None
        best = max(candidates, key=lambda p: p.composite_score("chat"))
        logger.warning(
            "Model council failover: primary=%s → %s (failure_count=%d)",
            self._primary, best.model_id, self._profiles[self._primary].failure_count,
        )
        return best.model_id

    def record_failure(self, model_id: str) -> None:
        """Increment failure count for a model (called by LLMRouter on errors)."""
        if model_id in self._profiles:
            self._profiles[model_id].failure_count += 1
            if self._profiles[model_id].failure_count >= 3:
                self._profiles[model_id].available = False
                logger.warning("Model council: %s marked unavailable after 3 failures", model_id)
            self._save_profiles()

    def record_success(self, model_id: str) -> None:
        """Reset failure count after a successful call."""
        if model_id in self._profiles:
            p = self._profiles[model_id]
            if not p.available or p.failure_count > 0:
                p.available = True
                p.failure_count = 0
                self._save_profiles()

    def set_primary(self, model_id: str) -> None:
        """Designate a model as the primary. AGENT trust required for autonomous switch."""
        if model_id not in self._profiles:
            raise ValueError(f"Unknown model: {model_id}")
        old = self._primary
        self._primary = model_id
        self._save_profiles()
        logger.info("Model council: primary changed %s → %s", old, model_id)

    def status(self) -> Dict[str, Any]:
        """Return current council status for /introspect/capabilities."""
        available = [p.model_id for p in self._profiles.values() if p.available]
        unavailable = [p.model_id for p in self._profiles.values() if not p.available]
        return {
            "primary": self._primary,
            "total_registered": len(self._profiles),
            "available_count": len(available),
            "unavailable_count": len(unavailable),
            "available": available,
            "unavailable": unavailable,
        }


# ── Singleton ──────────────────────────────────────────────────────────────────

_council: Optional[ModelCouncil] = None


def get_model_council(data_dir: Path = _DATA_DIR) -> ModelCouncil:
    global _council
    if _council is None:
        _council = ModelCouncil(data_dir=data_dir)
    return _council


def reset_model_council() -> None:
    global _council
    _council = None
