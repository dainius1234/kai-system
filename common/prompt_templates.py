"""Prompt templates — structured, model-aware prompt construction.

Replaces hardcoded prompt strings with configurable templates that
adapt to model capabilities. When a bigger model is plugged in,
prompts automatically get richer instructions.

Usage:
    from common.prompt_templates import build_system_prompt, build_chat_prompt
    system = build_system_prompt(mode="WORK")
    messages = build_chat_prompt(system, history, user_input)
"""
from __future__ import annotations

import os
from typing import Dict, List, Optional

from common.model_registry import get_model_spec


# ── System prompts by mode ───────────────────────────────────────────
# These are the core identity prompts. They scale with model quality:
# - Tier 1 (tiny): short, direct instructions
# - Tier 2+ (7B+): rich persona with reasoning guidelines

_SYSTEM_BASE_MINIMAL = (
    "You are Kai, a personal AI assistant. "
    "Be concise, helpful, and honest. "
    "If unsure, say so."
)

_SYSTEM_PUB = (
    "You are Kai — a sovereign personal AI and trusted companion. "
    "Mode: PUB (casual). Be warm, witty, and conversational. "
    "Use British English. Speak like a knowledgeable friend at the pub. "
    "Share opinions when asked. Be honest, never sycophantic. "
    "If you don't know something, say so directly."
)

_SYSTEM_WORK = (
    "You are Kai — a sovereign personal AI and professional advisor. "
    "Mode: WORK (professional). Be precise, structured, and thorough. "
    "Use British English. Focus on actionable advice. "
    "For financial/tax topics, cite UK rules (HMRC, Companies House). "
    "Flag uncertainty explicitly. Never guess on numbers or deadlines."
)

# Extended instructions for quality tier 2+ models
_REASONING_GUIDELINES = (
    "\n\nReasoning guidelines:"
    "\n- Think step-by-step for complex questions"
    "\n- Consider multiple perspectives before answering"
    "\n- Separate facts from opinions"
    "\n- If a question has prerequisites, address them first"
    "\n- For multi-part questions, structure your response clearly"
)

_OUTPUT_FORMAT_GUIDELINES = (
    "\n\nOutput format:"
    "\n- Use markdown for structured responses"
    "\n- Use bullet points for lists"
    "\n- Use code blocks for technical content"
    "\n- Keep responses focused — don't pad with filler"
)

_JSON_MODE_HINT = (
    "\n\nWhen asked for structured data, respond in valid JSON only — "
    "no markdown, no explanation outside the JSON."
)


def build_system_prompt(
    mode: str = "PUB",
    *,
    model: Optional[str] = None,
    extra_context: str = "",
) -> str:
    """Build a model-aware system prompt.

    - Tier 1 models get minimal instructions (saves tokens)
    - Tier 2+ get full persona + reasoning guidelines
    - Tier 3 with JSON support get format hints
    """
    spec = get_model_spec(model)

    if spec.quality_tier <= 1:
        base = _SYSTEM_BASE_MINIMAL
    elif mode.upper() == "WORK":
        base = _SYSTEM_WORK
    else:
        base = _SYSTEM_PUB

    prompt = base

    if spec.quality_tier >= 2:
        prompt += _REASONING_GUIDELINES
        prompt += _OUTPUT_FORMAT_GUIDELINES

    if spec.quality_tier >= 3 and spec.supports_json:
        prompt += _JSON_MODE_HINT

    if extra_context:
        prompt += f"\n\nContext:\n{extra_context}"

    return prompt


def build_chat_messages(
    system_prompt: str,
    history: List[Dict[str, str]],
    user_input: str,
    *,
    model: Optional[str] = None,
    evidence: str = "",
    personality_note: str = "",
) -> List[Dict[str, str]]:
    """Build a complete message list for the LLM.

    Injects evidence and personality notes as system context,
    then appends conversation history and the current user input.
    """
    messages: List[Dict[str, str]] = []

    # System prompt with optional injections
    system = system_prompt
    if evidence:
        system += f"\n\nRelevant evidence:\n{evidence}"
    if personality_note:
        system += f"\n\n{personality_note}"
    messages.append({"role": "system", "content": system})

    # Conversation history
    for msg in history:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if role in ("user", "assistant", "system") and content:
            messages.append({"role": role, "content": content})

    # Current input
    messages.append({"role": "user", "content": user_input})

    return messages


# ── Specialist-specific prompt wrappers ──────────────────────────────

def fact_check_prompt(claim: str, evidence: str = "") -> str:
    """Structured prompt for fact-checking tasks."""
    spec = get_model_spec()
    if spec.quality_tier >= 2:
        return (
            f"Fact-check the following claim. Rate confidence 0-10.\n\n"
            f"Claim: {claim}\n\n"
            f"{'Evidence: ' + evidence if evidence else 'No evidence provided.'}\n\n"
            f"Respond with:\n"
            f"1. Verdict (TRUE / FALSE / UNCERTAIN)\n"
            f"2. Confidence (0-10)\n"
            f"3. Reasoning (2-3 sentences)"
        )
    return f"Is this true? {claim}"


def planning_prompt(goal: str, constraints: str = "") -> str:
    """Structured prompt for planning/reasoning tasks."""
    spec = get_model_spec()
    if spec.quality_tier >= 2:
        return (
            f"Create a step-by-step plan for: {goal}\n\n"
            f"{'Constraints: ' + constraints if constraints else ''}\n\n"
            f"For each step:\n"
            f"- Action (what to do)\n"
            f"- Rationale (why)\n"
            f"- Risk (what could go wrong)\n"
            f"- Fallback (alternative if step fails)"
        )
    return f"Plan for: {goal}"


def reflection_prompt(topic: str, context: str = "") -> str:
    """Structured prompt for self-reflection tasks."""
    spec = get_model_spec()
    if spec.quality_tier >= 2:
        return (
            f"Reflect on: {topic}\n\n"
            f"{'Context: ' + context if context else ''}\n\n"
            f"Consider:\n"
            f"- What assumptions am I making?\n"
            f"- What might I be wrong about?\n"
            f"- What would change my mind?"
        )
    return f"Think about: {topic}"


# ── Template registry for extensibility ──────────────────────────────

_CUSTOM_TEMPLATES: Dict[str, str] = {}

TEMPLATE_DIR = os.getenv("PROMPT_TEMPLATE_DIR", "")


def register_template(name: str, template: str) -> None:
    """Register a custom prompt template at runtime."""
    _CUSTOM_TEMPLATES[name] = template


def get_template(name: str) -> Optional[str]:
    """Retrieve a registered template by name."""
    return _CUSTOM_TEMPLATES.get(name)
