"""Model registry + token counting — the plug-and-play chassis layer.

Central source of truth for model capabilities, context windows, and
token counting.  When a new model is added, define it here — everything
else (context budget, prompt templates, timeouts) adapts automatically.

Usage:
    from common.model_registry import count_tokens, get_model_spec, active_model

    tokens = count_tokens("Hello world")            # accurate count
    spec = get_model_spec("qwen2:0.5b")             # context window, etc.
    name = active_model()                             # current OLLAMA_MODEL
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, Optional

# ── tiktoken with graceful fallback ──────────────────────────────────
try:
    import tiktoken
    _ENC = tiktoken.get_encoding("cl100k_base")  # GPT-4/3.5 tokeniser
    _HAS_TIKTOKEN = True
except Exception:
    _ENC = None
    _HAS_TIKTOKEN = False


@dataclass(frozen=True)
class ModelSpec:
    """Immutable capability card for one model."""
    name: str
    context_window: int     # max tokens (input + output)
    output_reserve: int     # tokens reserved for completion
    speed_tier: int         # 1=fast, 2=medium, 3=slow
    quality_tier: int       # 1=basic, 2=good, 3=best
    supports_json: bool     # native JSON mode
    supports_vision: bool   # multimodal image input
    timeout_s: float        # per-request timeout
    tiktoken_encoding: str  # encoding name for token counting


# ── Known model specs ────────────────────────────────────────────────
# Add new models here — the rest of the system adapts automatically.
_REGISTRY: Dict[str, ModelSpec] = {
    "qwen2:0.5b": ModelSpec(
        name="qwen2:0.5b", context_window=4096, output_reserve=1024,
        speed_tier=1, quality_tier=1, supports_json=False,
        supports_vision=False, timeout_s=30, tiktoken_encoding="cl100k_base",
    ),
    "qwen2:1.5b": ModelSpec(
        name="qwen2:1.5b", context_window=4096, output_reserve=1024,
        speed_tier=1, quality_tier=1, supports_json=False,
        supports_vision=False, timeout_s=60, tiktoken_encoding="cl100k_base",
    ),
    "qwen2.5:7b": ModelSpec(
        name="qwen2.5:7b", context_window=32768, output_reserve=4096,
        speed_tier=2, quality_tier=2, supports_json=True,
        supports_vision=False, timeout_s=120, tiktoken_encoding="cl100k_base",
    ),
    "qwen2.5:14b": ModelSpec(
        name="qwen2.5:14b", context_window=32768, output_reserve=4096,
        speed_tier=2, quality_tier=3, supports_json=True,
        supports_vision=False, timeout_s=180, tiktoken_encoding="cl100k_base",
    ),
    "qwen2.5:32b": ModelSpec(
        name="qwen2.5:32b", context_window=32768, output_reserve=4096,
        speed_tier=3, quality_tier=3, supports_json=True,
        supports_vision=False, timeout_s=240, tiktoken_encoding="cl100k_base",
    ),
    "llama3:8b": ModelSpec(
        name="llama3:8b", context_window=8192, output_reserve=2048,
        speed_tier=2, quality_tier=2, supports_json=False,
        supports_vision=False, timeout_s=120, tiktoken_encoding="cl100k_base",
    ),
    "llama3.1:8b": ModelSpec(
        name="llama3.1:8b", context_window=131072, output_reserve=4096,
        speed_tier=2, quality_tier=2, supports_json=False,
        supports_vision=False, timeout_s=120, tiktoken_encoding="cl100k_base",
    ),
    "llama3.3:70b": ModelSpec(
        name="llama3.3:70b", context_window=131072, output_reserve=4096,
        speed_tier=3, quality_tier=3, supports_json=True,
        supports_vision=False, timeout_s=300, tiktoken_encoding="cl100k_base",
    ),
    "deepseek-v4": ModelSpec(
        name="deepseek-v4", context_window=32768, output_reserve=4096,
        speed_tier=2, quality_tier=3, supports_json=True,
        supports_vision=False, timeout_s=180, tiktoken_encoding="cl100k_base",
    ),
    "kimi-2.5": ModelSpec(
        name="kimi-2.5", context_window=131072, output_reserve=8192,
        speed_tier=2, quality_tier=3, supports_json=True,
        supports_vision=True, timeout_s=180, tiktoken_encoding="cl100k_base",
    ),
    "dolphin-mistral": ModelSpec(
        name="dolphin-mistral", context_window=8192, output_reserve=2048,
        speed_tier=1, quality_tier=2, supports_json=False,
        supports_vision=False, timeout_s=60, tiktoken_encoding="cl100k_base",
    ),
    "gemma2:9b": ModelSpec(
        name="gemma2:9b", context_window=8192, output_reserve=2048,
        speed_tier=2, quality_tier=2, supports_json=False,
        supports_vision=False, timeout_s=120, tiktoken_encoding="cl100k_base",
    ),
    "phi3:mini": ModelSpec(
        name="phi3:mini", context_window=4096, output_reserve=1024,
        speed_tier=1, quality_tier=1, supports_json=False,
        supports_vision=False, timeout_s=30, tiktoken_encoding="cl100k_base",
    ),
    "mistral:7b": ModelSpec(
        name="mistral:7b", context_window=8192, output_reserve=2048,
        speed_tier=2, quality_tier=2, supports_json=False,
        supports_vision=False, timeout_s=120, tiktoken_encoding="cl100k_base",
    ),
}

# Fallback for unknown models — conservative 4K assumption
_DEFAULT_SPEC = ModelSpec(
    name="unknown", context_window=4096, output_reserve=1024,
    speed_tier=2, quality_tier=1, supports_json=False,
    supports_vision=False, timeout_s=120, tiktoken_encoding="cl100k_base",
)


# ── Public API ───────────────────────────────────────────────────────

def active_model() -> str:
    """Return the currently configured Ollama model name."""
    return os.getenv("OLLAMA_MODEL", "qwen2:0.5b")


def get_model_spec(model: Optional[str] = None) -> ModelSpec:
    """Look up capabilities for a model. Falls back to conservative defaults."""
    name = (model or active_model()).lower().strip()
    if name in _REGISTRY:
        return _REGISTRY[name]
    # Try prefix match (e.g. "qwen2.5:7b-q4_K_M" matches "qwen2.5:7b")
    for key in _REGISTRY:
        if name.startswith(key):
            return _REGISTRY[key]
    return _DEFAULT_SPEC


def count_tokens(text: str, model: Optional[str] = None) -> int:
    """Count tokens accurately with tiktoken, fallback to heuristic.

    The heuristic (chars/3.5) is tuned to over-estimate slightly,
    which is safer for context budget calculations.
    """
    if not text:
        return 0
    if _HAS_TIKTOKEN:
        try:
            return len(_ENC.encode(text))
        except Exception:
            pass
    # Fallback: ~3.5 chars per token (slightly conservative for safety)
    return max(len(text) * 10 // 35, 1)


def count_messages_tokens(messages: list, model: Optional[str] = None) -> int:
    """Count tokens for an OpenAI-format message list.

    Includes per-message overhead (~4 tokens for role/separators).
    """
    total = 0
    for msg in messages:
        total += 4  # role + separators overhead
        total += count_tokens(msg.get("content", ""), model)
    total += 2  # reply priming
    return total


def context_budget(model: Optional[str] = None) -> int:
    """Max input tokens = context_window - output_reserve.

    This is the budget for system prompt + conversation history.
    The output_reserve ensures the model has room to generate a response.
    """
    spec = get_model_spec(model)
    return spec.context_window - spec.output_reserve


def model_timeout(model: Optional[str] = None) -> float:
    """Per-model timeout in seconds."""
    return get_model_spec(model).timeout_s


def list_models() -> Dict[str, ModelSpec]:
    """Return the full model registry."""
    return dict(_REGISTRY)
