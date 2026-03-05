"""HP2 — MoE-Style Model Selector.

Routes tasks to the best-fit model based on task type, complexity,
and model capability profiles. Acts as the intelligence layer between
the specialist router (which classifies intent) and the LLM router
(which dispatches HTTP calls).

When GPU arrives with multiple models loaded, this selects the optimal
model per query. In CPU/single-model mode, always returns the default.

Usage:
    from model_selector import select_model, ModelProfile
    model = select_model(route="FACT_CHECK", complexity=0.8, available=["Ollama", "DeepSeek-V4"])
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class ModelProfile:
    """Capability profile for a single model."""
    name: str
    strengths: List[str]         # task types this model excels at
    max_context: int = 4096      # context window tokens
    speed_tier: int = 1          # 1=fast, 2=medium, 3=slow
    quality_tier: int = 2        # 1=basic, 2=good, 3=best
    moe_expert_count: int = 0    # 0 = dense model, >0 = MoE
    vram_gb: float = 0.0         # VRAM footprint (0 = CPU-only stub)


# ── Model Registry ───────────────────────────────────────────────────

_PROFILES: Dict[str, ModelProfile] = {
    "Ollama": ModelProfile(
        name="Ollama",
        strengths=["GENERAL_CHAT", "MEMORY_RECALL"],
        max_context=4096,
        speed_tier=1,
        quality_tier=1,
        moe_expert_count=0,
        vram_gb=0.4,
    ),
    "DeepSeek-V4": ModelProfile(
        name="DeepSeek-V4",
        strengths=["EXECUTE_ACTION", "FACT_CHECK", "MULTI_SIGNAL", "TAX_ADVISORY"],
        max_context=32768,
        speed_tier=2,
        quality_tier=3,
        moe_expert_count=256,
        vram_gb=12.0,
    ),
    "Kimi-2.5": ModelProfile(
        name="Kimi-2.5",
        strengths=["GENERAL_CHAT", "REFLECT", "PROACTIVE_REVIEW"],
        max_context=131072,
        speed_tier=2,
        quality_tier=3,
        moe_expert_count=0,
        vram_gb=8.0,
    ),
    "Dolphin": ModelProfile(
        name="Dolphin",
        strengths=["GENERAL_CHAT", "REFLECT"],
        max_context=8192,
        speed_tier=1,
        quality_tier=2,
        moe_expert_count=0,
        vram_gb=4.0,
    ),
}

# ── Task complexity heuristics ───────────────────────────────────────

_COMPLEX_KEYWORDS = {
    "analyse", "analyze", "compare", "evaluate", "design", "architect",
    "debug", "refactor", "optimise", "optimize", "strategy", "multi-step",
    "trade-off", "tradeoff", "risk assessment", "business plan",
}


def estimate_complexity(user_input: str) -> float:
    """Estimate task complexity from 0.0 (trivial) to 1.0 (hard)."""
    words = user_input.lower().split()
    word_count = len(words)

    score = 0.0
    # Length factor
    if word_count > 50:
        score += 0.3
    elif word_count > 20:
        score += 0.2
    elif word_count > 10:
        score += 0.1

    # Keyword factor
    text_lower = user_input.lower()
    hits = sum(1 for kw in _COMPLEX_KEYWORDS if kw in text_lower)
    score += min(hits * 0.15, 0.5)

    # Question mark = uncertainty
    if "?" in user_input:
        score += 0.1

    # Multiple sentences = multi-step
    sentences = [s.strip() for s in user_input.replace("!", ".").replace("?", ".").split(".") if s.strip()]
    if len(sentences) > 3:
        score += 0.2

    return min(score, 1.0)


def select_model(
    route: str,
    user_input: str = "",
    available: Optional[List[str]] = None,
    prefer_speed: bool = False,
) -> str:
    """Select the best model for a given route and input.

    Args:
        route: Classification route (e.g. "FACT_CHECK", "GENERAL_CHAT").
        user_input: Raw user input for complexity estimation.
        available: List of available model names. If None, uses all registered.
        prefer_speed: If True, prefer faster models over higher quality.

    Returns:
        Name of the selected model.
    """
    candidates = available if available else list(_PROFILES.keys())
    if not candidates:
        return "Ollama"

    # If only one model available, no choice
    if len(candidates) == 1:
        return candidates[0]

    complexity = estimate_complexity(user_input) if user_input else 0.5

    scored: List[tuple] = []
    for name in candidates:
        profile = _PROFILES.get(name)
        if profile is None:
            continue

        score = 0.0

        # Route match: model's strengths align with task type
        if route in profile.strengths:
            score += 3.0

        # Quality premium for complex tasks
        if complexity > 0.5:
            score += profile.quality_tier * 1.0
        else:
            score += profile.quality_tier * 0.5

        # Speed bonus when preferred or for simple tasks
        if prefer_speed or complexity < 0.3:
            score += (4 - profile.speed_tier) * 1.0

        # MoE bonus for complex routing tasks
        if profile.moe_expert_count > 0 and complexity > 0.5:
            score += 1.5

        # Context window bonus for long inputs
        if len(user_input) > 2000 and profile.max_context > 16384:
            score += 1.0

        scored.append((name, score))

    if not scored:
        return "Ollama"

    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[0][0]


def get_profile(name: str) -> Optional[ModelProfile]:
    """Get the capability profile for a model."""
    return _PROFILES.get(name)


def register_model(profile: ModelProfile) -> None:
    """Register or update a model profile."""
    _PROFILES[profile.name] = profile


def list_models() -> List[str]:
    """List all registered model names."""
    return list(_PROFILES.keys())
