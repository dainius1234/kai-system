"""Unified LLM backend abstraction — local-first, sovereign-safe.

Provides a single interface to query any of the 3 local LLM specialists
(DeepSeek-V4, Kimi-2.5, Dolphin) via OpenAI-compatible HTTP endpoints.
Falls back to a stub response when no live backend is configured, so
the full pipeline can be exercised in dev/test without GPU hardware.

Configuration via environment variables:
    LLM_DEEPSEEK_URL         — e.g. http://llm-deepseek:11434
    LLM_KIMI_URL             — e.g. http://llm-kimi:11434
    LLM_DOLPHIN_URL          — e.g. http://llm-dolphin:11434
    LLM_TIMEOUT              — request timeout in seconds (default: 120)
    STREAM_HEARTBEAT_TIMEOUT — kill stream if no token for N seconds (default: 30)
    MODEL_TAGS_CACHE_TTL     — seconds to cache Ollama /api/tags response (default: 60)
    LLM_WARMUP_ENABLED       — send a warm prompt on startup (default: true)
    OLLAMA_AUTO_PULL         — pull missing model automatically on warmup (default: false)

Usage:
    from common.llm import LLMRouter, query_specialist

    router = LLMRouter()
    resp = await router.query("DeepSeek-V4", "explain risk assessment")
    # or use the module-level shortcut:
    resp = await query_specialist("Dolphin", "tell me a joke")
"""
from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


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
# Ollama is the primary local backend — add it as the default
_OLLAMA_URL = os.getenv("OLLAMA_URL", "http://ollama:11434")
_OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:0.5b")

_DEFAULT_URLS: Dict[str, str] = {
    "Ollama": _OLLAMA_URL,
    "DeepSeek-V4": os.getenv("LLM_DEEPSEEK_URL", ""),
    "Kimi-2.5": os.getenv("LLM_KIMI_URL", ""),
    "Dolphin": os.getenv("LLM_DOLPHIN_URL", ""),
    # Cloud fallback backends — only active when an API key is configured.
    # URL stored WITHOUT /v1 suffix so _live_query's /v1/chat/completions append works.
    "Groq": (os.getenv("GROQ_API_URL", "https://api.groq.com/openai")
             if os.getenv("GROQ_API_KEY") else ""),
    "OpenRouter": (os.getenv("OPENROUTER_API_URL", "https://openrouter.ai/api")
                   if os.getenv("OPENROUTER_API_KEY") else ""),
}

# Maps specialist names to their Ollama model identifiers.
# When a specialist has a configured URL it uses the OpenAI-compatible
# /v1/chat/completions endpoint.  For Ollama specifically, the model
# field is overridden by OLLAMA_MODEL so any pulled model works.
_MODEL_MAP: Dict[str, str] = {
    "Ollama": _OLLAMA_MODEL,
    "DeepSeek-V4": os.getenv("LLM_DEEPSEEK_MODEL", "deepseek-v4"),
    "Kimi-2.5": os.getenv("LLM_KIMI_MODEL", "kimi-2.5"),
    "Dolphin": os.getenv("LLM_DOLPHIN_MODEL", "dolphin-mistral"),
    "Groq": os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
    "OpenRouter": os.getenv("OPENROUTER_MODEL", "meta-llama/llama-3.3-70b-instruct:free"),
}

# API keys for cloud backends (empty string = not configured).
_API_KEY_MAP: Dict[str, str] = {
    "Groq": os.getenv("GROQ_API_KEY", ""),
    "OpenRouter": os.getenv("OPENROUTER_API_KEY", ""),
}

LLM_TIMEOUT = float(os.getenv("LLM_TIMEOUT", "120"))
LLM_CONNECT_TIMEOUT = float(os.getenv("LLM_CONNECT_TIMEOUT", "10"))
LLM_READ_TIMEOUT = float(os.getenv("LLM_READ_TIMEOUT", "120"))
LLM_MAX_RETRIES = int(os.getenv("LLM_MAX_RETRIES", "3"))
LLM_RETRY_BACKOFF = float(os.getenv("LLM_RETRY_BACKOFF", "1.0"))

# HTTP status codes that warrant a retry (rate-limited or temporarily unavailable).
_RETRY_STATUS_CODES: frozenset[int] = frozenset({429, 503})

# ── C5: Ollama /api/tags pre-flight cache ────────────────────────────
# In-process cache; TTL read at call time so tests can override via env.
_model_tags_cache: Optional[List[str]] = None
_model_tags_cache_ts: float = 0.0

# ── model-aware token counting (replaces 4-char heuristic) ──────────
try:
    from common.model_registry import (
        count_tokens as _count_tokens,
        model_timeout as _model_timeout,
        get_model_spec,
    )
except ImportError:
    def _count_tokens(text: str, model=None) -> int:  # type: ignore[misc]
        return max(len(text) * 10 // 35, 1)

    def _model_timeout(model=None) -> float:  # type: ignore[misc]
        return LLM_TIMEOUT

    def get_model_spec(model=None):  # type: ignore[misc]
        return None


def _validate_llm_response(text: str) -> str:
    """Validate LLM response — reject empty or error-only output."""
    if not text or not text.strip():
        return "[empty response from model]"
    # Detect common Ollama error patterns
    if text.strip().startswith("{\"error\""):
        return f"[model error: {text[:200]}]"
    return text


# ── C5: Ollama /api/tags pre-flight helpers ──────────────────────────

def _check_model_in_tags(tags: List[str], model_name: str) -> bool:
    """Return True if *model_name* (or a prefix match) appears in *tags*.

    Supports partial matching so ``qwen2:0.5b`` matches a tag like
    ``qwen2:0.5b-instruct-q4_0``.
    """
    model_lower = model_name.lower()
    model_base = model_lower.split(":")[0]
    for tag in tags:
        tag_lower = tag.lower()
        if tag_lower == model_lower:
            return True
        # Prefix match on the base name (before ":")
        tag_base = tag_lower.split(":")[0]
        if model_base and model_base == tag_base:
            return True
    return False


async def ensure_model_available(model_name: str) -> bool:
    """Check whether *model_name* is pulled and available in Ollama.

    Queries ``{OLLAMA_URL}/api/tags`` and does a prefix-aware name match.
    The tag list is cached in-process for ``MODEL_TAGS_CACHE_TTL`` seconds
    (default 60) to avoid hammering Ollama on every request.

    Returns:
        True  — model is available OR Ollama is unreachable (fail-open so
                the existing circuit breaker handles real outages).
        False — Ollama responded but the model is not in the tag list.
    """
    global _model_tags_cache, _model_tags_cache_ts

    cache_ttl = float(os.getenv("MODEL_TAGS_CACHE_TTL", "60"))
    ollama_base = os.getenv("OLLAMA_URL", "http://ollama:11434")
    now = time.monotonic()

    # Serve from cache if still fresh
    if _model_tags_cache is not None and (now - _model_tags_cache_ts) < cache_ttl:
        return _check_model_in_tags(_model_tags_cache, model_name)

    import httpx
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(f"{ollama_base}/api/tags")
            resp.raise_for_status()
            data = resp.json()
            tags = [m.get("name", "") for m in data.get("models", [])]
            _model_tags_cache = tags
            _model_tags_cache_ts = now
            available = _check_model_in_tags(tags, model_name)
            if not available:
                logger.warning(
                    "model_unavailable: model=%s not found in Ollama tags (%d models)",
                    model_name, len(tags),
                )
            return available
    except Exception:
        # Ollama unreachable, returned an error, or any other unexpected
        # failure (e.g. JSON decode) — fail open; the circuit breaker
        # handles real persistent outages.
        return True


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

        model = _MODEL_MAP.get(specialist, specialist)
        is_cloud = url.startswith("https://")

        # C5: verify model is loaded before routing; skip for cloud APIs
        if not is_cloud and not await ensure_model_available(model):
            fallback = _OLLAMA_MODEL
            logger.warning(
                "model_fallback: requested=%s not available, using default=%s",
                model, fallback,
            )
            model = fallback

        # Auth header for cloud APIs
        headers: Dict[str, str] = {}
        api_key = _API_KEY_MAP.get(specialist, "")
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        if specialist == "OpenRouter":
            headers["HTTP-Referer"] = "https://github.com/dainius1234/kai-system"

        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False,
        }

        # C3: Retry with exponential backoff on 429/503 and transient connection errors.
        last_exc: Optional[Exception] = None
        for attempt in range(max(1, LLM_MAX_RETRIES)):
            try:
                # C1: Use model-aware timeout from registry instead of hardcoded LLM_TIMEOUT
                model_name = _MODEL_MAP.get(specialist, specialist)
                query_timeout = _model_timeout(model_name)
                timeout = httpx.Timeout(query_timeout, connect=LLM_CONNECT_TIMEOUT, read=query_timeout)
                async with httpx.AsyncClient(timeout=timeout, headers=headers) as client:
                    resp = await client.post(f"{url}/v1/chat/completions", json=payload)

                # Retry before raise_for_status so we can log and sleep first
                if resp.status_code in _RETRY_STATUS_CODES and attempt < LLM_MAX_RETRIES - 1:
                    wait = LLM_RETRY_BACKOFF * (2 ** attempt)
                    logger.warning(
                        "llm_retry: specialist=%s status=%d attempt=%d/%d wait=%.1fs",
                        specialist, resp.status_code, attempt + 1, LLM_MAX_RETRIES, wait,
                    )
                    await asyncio.sleep(wait)
                    last_exc = Exception(f"HTTP {resp.status_code}")
                    continue

                resp.raise_for_status()
                data = resp.json()
                choice = data.get("choices", [{}])[0]
                text = choice.get("message", {}).get("content", "")
                text = _validate_llm_response(text)
                usage = data.get("usage", {})
                model_out = data.get("model", specialist)
                latency = (time.monotonic() - start) * 1000
                return LLMResponse(
                    specialist=specialist,
                    text=text,
                    latency_ms=round(latency, 1),
                    source="live",
                    model=model_out,
                    usage=usage,
                )
            except (httpx.ConnectError, httpx.TimeoutException) as exc:
                last_exc = exc
                if attempt < LLM_MAX_RETRIES - 1:
                    wait = LLM_RETRY_BACKOFF * (2 ** attempt)
                    logger.warning(
                        "llm_retry: specialist=%s error=%s attempt=%d/%d wait=%.1fs",
                        specialist, type(exc).__name__, attempt + 1, LLM_MAX_RETRIES, wait,
                    )
                    await asyncio.sleep(wait)
                    continue
                break
            except Exception as exc:
                last_exc = exc
                break

        latency = (time.monotonic() - start) * 1000
        return LLMResponse(
            specialist=specialist,
            text=f"[error: {str(last_exc)[:100]}]",
            latency_ms=round(latency, 1),
            source="error",
        )

    @staticmethod
    def _stub_response(specialist: str, prompt: str, start: float) -> LLMResponse:
        """Deterministic stub — reproducible across runs."""
        h = hashlib.sha256(f"{specialist}:{prompt[:200]}".encode()).hexdigest()[:8]
        text = (
            f"[{specialist} stub-{h}] This is a local stub response. "
            f"Wire OLLAMA_URL to enable live inference (Ollama is the default backend)."
        )
        latency = (time.monotonic() - start) * 1000
        return LLMResponse(
            specialist=specialist,
            text=text,
            latency_ms=round(latency, 1),
            source="stub",
        )

    async def stream(
        self,
        specialist: str,
        messages: List[Dict[str, str]],
        *,
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ):
        """Yield text chunks from a streaming LLM call.

        *messages* is a full conversation in OpenAI format:
          [{"role": "system", "content": ...}, {"role": "user", ...}, ...]

        Yields plain-text token strings as they arrive.
        Falls back to a single stub yield when no backend is configured.

        C2: wraps per-token reads in ``asyncio.wait_for`` with
        ``STREAM_HEARTBEAT_TIMEOUT`` (default 30 s).  If no token arrives
        within the window the stream is cancelled and a stall message is
        emitted so the caller can surface it to the operator.
        """
        url = self.backends.get(specialist, "")
        if not url:
            h = hashlib.sha256(str(messages[-1:]).encode()).hexdigest()[:8]
            yield f"[{specialist} stub-{h}] Wire OLLAMA_URL or LLM env vars to enable live responses."
            return

        import httpx

        model = _MODEL_MAP.get(specialist, specialist)
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": True,
        }
        import json as _json

        # C2: read heartbeat timeout at call time so tests can override via env
        heartbeat_timeout = float(os.getenv("STREAM_HEARTBEAT_TIMEOUT", "30"))
        stream_start = time.monotonic()

        # Auth header for cloud streaming APIs
        stream_headers: Dict[str, str] = {}
        _stream_key = _API_KEY_MAP.get(specialist, "")
        if _stream_key:
            stream_headers["Authorization"] = f"Bearer {_stream_key}"
        if specialist == "OpenRouter":
            stream_headers["HTTP-Referer"] = "https://github.com/dainius1234/kai-system"

        try:
            timeout = httpx.Timeout(LLM_TIMEOUT, connect=LLM_CONNECT_TIMEOUT, read=LLM_READ_TIMEOUT)
            async with httpx.AsyncClient(timeout=timeout, headers=stream_headers) as client:
                async with client.stream("POST", f"{url}/v1/chat/completions", json=payload) as resp:
                    resp.raise_for_status()
                    aiter = resp.aiter_lines()
                    while True:
                        try:
                            line = await asyncio.wait_for(
                                aiter.__anext__(),
                                timeout=heartbeat_timeout,
                            )
                        except asyncio.TimeoutError:
                            elapsed = time.monotonic() - stream_start
                            logger.warning(
                                "stream_stall: model=%s no_token_for=%.0fs elapsed=%.1fs",
                                model, heartbeat_timeout, elapsed,
                            )
                            yield (
                                f"[stream stalled — no token for "
                                f"{heartbeat_timeout:.0f}s, cutting]"
                            )
                            return
                        except StopAsyncIteration:
                            break
                        if not line.startswith("data: "):
                            continue
                        data_str = line[6:].strip()
                        if data_str == "[DONE]":
                            break
                        try:
                            chunk = _json.loads(data_str)
                            delta = chunk.get("choices", [{}])[0].get("delta", {})
                            token = delta.get("content", "")
                            if token:
                                yield token
                        except Exception:
                            continue
        except Exception as exc:
            yield f"\n[LLM error: {str(exc)[:200]}]"


# ── module-level convenience ─────────────────────────────────────────
_router = LLMRouter()


async def query_specialist(specialist: str, prompt: str, **kwargs: Any) -> LLMResponse:
    """Module-level shortcut: query a specialist via the global router."""
    return await _router.query(specialist, prompt, **kwargs)


async def query_multi(specialists: List[str], prompt: str, **kwargs: Any) -> List[LLMResponse]:
    """Module-level shortcut: query multiple specialists in parallel."""
    return await _router.query_multi(specialists, prompt, **kwargs)


# ── C9: Model warm-up / pre-load ─────────────────────────────────────

async def _pull_model(model: str, ollama_base_url: str) -> None:
    """Stream an Ollama /api/pull for *model*, logging progress at INFO."""
    import httpx
    import json as _json
    try:
        async with httpx.AsyncClient(timeout=300.0) as client:
            async with client.stream(
                "POST", f"{ollama_base_url}/api/pull", json={"name": model}
            ) as resp:
                async for line in resp.aiter_lines():
                    if line:
                        try:
                            status = _json.loads(line).get("status", "")
                            if status:
                                logger.info("LLM pull [%s]: %s", model, status)
                        except Exception:
                            logger.debug("LLM pull [%s]: failed to parse progress line: %r", model, line)
    except Exception as exc:
        logger.warning("LLM pull failed for model=%s: %s", model, exc)


async def llm_warmup(
    router: Optional["LLMRouter"] = None,
    model: Optional[str] = None,
    specialist: Optional[str] = None,
    ollama_base_url: Optional[str] = None,
) -> None:
    """Warm up the LLM to reduce first-request cold-start latency.

    Steps:
    1. Check model availability via ``ensure_model_available``.
    2. If absent and ``OLLAMA_AUTO_PULL=true``: stream ``/api/pull``.
    3. Send a single warm prompt to force model load into RAM.
    4. Log completion time.

    Controlled by ``LLM_WARMUP_ENABLED`` (default ``true``).
    All parameters default to their corresponding env vars so callers
    only need to pass what they want to override.
    """
    if os.getenv("LLM_WARMUP_ENABLED", "true").lower() != "true":
        return

    model = model or os.getenv("OLLAMA_MODEL", "qwen2.5:0.5b")
    specialist = specialist or os.getenv("DEFAULT_SPECIALIST", "Ollama")
    ollama_base_url = ollama_base_url or os.getenv("OLLAMA_URL", "http://ollama:11434")
    router = router or _router

    start = time.monotonic()
    logger.info("LLM warmup: starting for model=%s specialist=%s", model, specialist)

    available = await ensure_model_available(model)

    if not available:
        auto_pull = os.getenv("OLLAMA_AUTO_PULL", "false").lower() == "true"
        if auto_pull:
            logger.info("LLM warmup: model=%s not found, pulling…", model)
            await _pull_model(model, ollama_base_url)
        else:
            logger.warning(
                "LLM warmup: model=%s not available and OLLAMA_AUTO_PULL=false — skipping warm prompt",
                model,
            )
            return

    try:
        resp = await router.query(specialist, "warmup", system="You are helpful.", max_tokens=1)
        elapsed = time.monotonic() - start
        logger.info("LLM warmup complete in %.1fs (source=%s)", elapsed, resp.source)
    except Exception as exc:
        logger.warning("LLM warmup: warm prompt failed: %s", exc)
