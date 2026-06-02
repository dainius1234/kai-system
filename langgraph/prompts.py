from __future__ import annotations

from agentic import prompts as _agentic_prompts
from agentic.prompts import (
    AGENTS_PATH,
    INJECTION_RE,
    SOUL_PATH,
    _KAI_CORE_IDENTITY,
    _SYSTEM_PROMPTS,
    _load_agents,
    _load_soul,
)


def __getattr__(name: str):
    return getattr(_agentic_prompts, name)
