"""Unified LLM backend abstraction — local-first, sovereign-safe.

Provides a single interface to query any of the 3 local LLM specialists
(DeepSeek-V4, Kimi-2.5, Dolphin) via OpenAI-compatible HTTP endpoints.
Falls back to a stub response when no live backend is configured, so
the full pipeline can be exercised in dev/test without GPU hardware.

Configuration via environment variables:
    LLM_DEEPSEEK_URL  — e.g. http://llm-deepseek:11434
    LLM_KIMI_URL      — e.g. http://llm-kimi:11434
    LLM_DOLPHIN_URL   — e.g. http://llm-dolphin:11434
    LLM_TIMEOUT        — request timeout in seconds (default: 30)

Usage:
    from common.llm import LLMRouter, query_specialist

    router = LLMRouter()
    resp = await router.query("DeepSeek-V4", "explain risk assessment")
    # or use the module-level shortcut:
    resp = await query_specialist("Dolphin", "tell me a joke")
"""
from __future__ import annotations

import hashlib
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class LLMResponse:
    """Result from a single LLM query."""
    specialist: str
    text: str
    latency_ms: float
    source: str           # "live", "stub", or "error"
    model: str = ""
    usage: Dict[str, int] = field(default_factory=dict)


# ── Specialist URL registry ─────────────────────────────────────────
_DEFAULT_URLS: Dict[str, str] = {
    "DeepSeek-V4": os.getenv("LLM_DEEPSEEK_URL", ""),
    "Kimi-2.5": os.getenv("LLM_KIMI_URL", ""),
    "Dolphin": os.getenv("LLM_DOLPHIN_URL", ""),
}

LLM_TIMEOUT = float(os.getenv("LLM_TIMEOUT", "30"))


class LLMRouter:
    """Route prompts to the right local LLM via OpenAI-compatible API.

    Maintains per-specialist retry/fallback state.  Thread-safe for use
    behind an async FastAPI app (each call creates its own httpx client).
    """

    def __init__(self, backends: Optional[Dict[str, str]] = None) -> None:
        self.backends: Dict[str, str] = {}
        for name, url in (backends or _DEFAULT_URLS).items():
            url = url.strip()
            if url:
                self.backends[name] = url

    @property
    def available(self) -> List[str]:
        """Names of specialists with configured endpoints."""
        return list(self.backends.keys())

    @property
    def stub_mode(self) -> bool:
        """True when no live backends are configured."""
        return len(self.backends) == 0

    async def query(
        self,
        specialist: str,
        prompt: str,
        *,
        system: str = "You are a helpful assistant.",
        temperature: float = 0.3,
        max_tokens: int = 512,
    ) -> LLMResponse:
        """Send a prompt to a specialist and return the response.

        Falls back to a deterministic stub if the specialist has no
        configured URL or if the call fails.
        """
        url = self.backends.get(specialist, "")
        start = time.monotonic()

        if url:
            return await self._live_query(specialist, url, prompt, system, temperature, max_tokens, start)

        return self._stub_response(specialist, prompt, start)

    async def query_multi(
        self,
        specialists: List[str],
        prompt: str,
        **kwargs: Any,
    ) -> List[LLMResponse]:
        """Query multiple specialists in parallel."""
        import asyncio
        tasks = [self.query(name, prompt, **kwargs) for name in specialists]
        return list(await asyncio.gather(*tasks))

    # ── internals ────────────────────────────────────────────────────

    async def _live_query(
        self,
        specialist: str,
        url: str,
        prompt: str,
        system: str,
        temperature: float,
        max_tokens: int,
        start: float,
    ) -> LLMResponse:
        import httpx

        payload = {
            "model": specialist,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        try:
            async with httpx.AsyncClient(timeout=LLM_TIMEOUT) as client:
                resp = await client.post(f"{url}/v1/chat/completions", json=payload)
                resp.raise_for_status()
                data = resp.json()

            choice = data.get("choices", [{}])[0]
            text = choice.get("message", {}).get("content", "")
            usage = data.get("usage", {})
            model = data.get("model", specialist)
            latency = (time.monotonic() - start) * 1000

            return LLMResponse(
                specialist=specialist,
                text=text,
                latency_ms=round(latency, 1),
                source="live",
                model=model,
                usage=usage,
            )
        except Exception as exc:
            latency = (time.monotonic() - start) * 1000
            return LLMResponse(
                specialist=specialist,
                text=f"[error: {str(exc)[:100]}]",
                latency_ms=round(latency, 1),
                source="error",
            )

    @staticmethod
    def _stub_response(specialist: str, prompt: str, start: float) -> LLMResponse:
        """Deterministic stub — reproducible across runs."""
        h = hashlib.sha256(f"{specialist}:{prompt[:200]}".encode()).hexdigest()[:8]
        text = (
            f"[{specialist} stub-{h}] This is a local stub response. "
            f"Wire LLM_{'DEEPSEEK' if 'Deep' in specialist else 'KIMI' if 'Kimi' in specialist else 'DOLPHIN'}_URL "
            f"to enable live inference."
        )
        latency = (time.monotonic() - start) * 1000
        return LLMResponse(
            specialist=specialist,
            text=text,
            latency_ms=round(latency, 1),
            source="stub",
        )


# ── module-level convenience ─────────────────────────────────────────
_router = LLMRouter()


async def query_specialist(specialist: str, prompt: str, **kwargs: Any) -> LLMResponse:
    """Module-level shortcut: query a specialist via the global router."""
    return await _router.query(specialist, prompt, **kwargs)


async def query_multi(specialists: List[str], prompt: str, **kwargs: Any) -> List[LLMResponse]:
    """Module-level shortcut: query multiple specialists in parallel."""
    return await _router.query_multi(specialists, prompt, **kwargs)
