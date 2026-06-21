"""GitHub Models client — CI/tests-only LLM backend.

Not used by any production service (agentic, dashboard, memu-core, etc.)
and must never be wired into ``common/llm.py``'s ``LLMRouter``. It exists
solely so CI test runs can get a real, coherent model response instead of
relying only on the local ``qwen2:0.5b`` stub, which is too small to give
meaningful feedback on response quality. See kai-pm/DECISIONS.md D36.

GitHub's own ``gh-models`` docs state the service "is not designed for
production use cases" (rate-limited per minute/day) — consistent with
that, this client is opt-in via the presence of ``GITHUB_TOKEN`` and is
only ever called from test scripts, never from request-serving code.

Requires the calling GitHub Actions workflow to declare
``permissions: { models: read }`` (see .github/workflows/core-tests.yml).
Uses the automatic ``GITHUB_TOKEN`` — no new secret needed. Locally, or on
PRs from forks where ``GITHUB_TOKEN`` lacks model access, ``is_available()``
returns False and callers should skip rather than fail.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

GITHUB_MODELS_ENDPOINT = "https://models.github.ai/inference"

# microsoft/phi-4-mini-instruct: small, fast, free-tier-friendly — a good
# fit for CI test loops that need a real response, not flagship quality.
DEFAULT_MODEL = os.getenv("GH_MODELS_MODEL", "microsoft/phi-4-mini-instruct")


@dataclass
class GitHubModelsResponse:
    text: str
    model: str
    source: str  # "live", "unavailable", or "error"


def is_available() -> bool:
    """True only when GITHUB_TOKEN is present. Makes no network call."""
    return bool(os.getenv("GITHUB_TOKEN"))


def query(
    prompt: str,
    *,
    system: str = "You are a helpful assistant.",
    model: Optional[str] = None,
    temperature: float = 0.3,
    max_tokens: int = 512,
    timeout: float = 30.0,
) -> GitHubModelsResponse:
    """Query GitHub Models. CI/tests only — see module docstring.

    Returns ``source="unavailable"`` (no exception) when GITHUB_TOKEN is
    unset, so callers can skip cleanly when run outside GitHub Actions.
    """
    model = model or DEFAULT_MODEL
    token = os.getenv("GITHUB_TOKEN")
    if not token:
        return GitHubModelsResponse(text="", model=model, source="unavailable")

    import requests

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
    }
    try:
        resp = requests.post(
            f"{GITHUB_MODELS_ENDPOINT}/chat/completions",
            json=payload, headers=headers, timeout=timeout,
        )
        resp.raise_for_status()
        data = resp.json()
        text = data["choices"][0]["message"]["content"]
        return GitHubModelsResponse(text=text, model=model, source="live")
    except Exception as exc:
        return GitHubModelsResponse(text=f"[error: {exc}]", model=model, source="error")
