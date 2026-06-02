"""agentic.prompts — Kai's system prompt constants.

Re-exports all public prompt names so any module can do:
    from agentic.prompts import _KAI_CORE_IDENTITY, _SYSTEM_PROMPTS
"""

from agentic.prompts.system import _KAI_CORE_IDENTITY, _SYSTEM_PROMPTS

__all__ = ["_KAI_CORE_IDENTITY", "_SYSTEM_PROMPTS"]
