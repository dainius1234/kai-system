"""Specialist Router — zero-LLM request classification for kai-system.

Classifies incoming user messages into one of eight routes so that
only genuine reasoning tasks burn LLM tokens.  Everything else is
dispatched directly to the appropriate service.

Routes:
    MEMORY_RECALL    — answered by Memu-Core vector search (no LLM)
    TAX_ADVISORY     — answered by kai-advisor rules engine (no LLM)
    FACT_CHECK       — answered by Verifier pipeline (no LLM)
    PROACTIVE_REVIEW — answered by Memu-Core proactive nudges (no LLM)
    REFLECT          — answered by Memu-Core reflection (no LLM)
    EXECUTE_ACTION   — needs plan + conviction gate → Executor
    MULTI_SIGNAL     — needs Fusion-Engine multi-LLM consensus
    GENERAL_CHAT     — default fallback → Ollama LLM

Usage:
    from router import classify
    decision = classify("what tax do I owe?", session_context={})
    # RouteDecision(route="TAX_ADVISORY", confidence=0.9, ...)
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class RouteDecision:
    """Result of classifying a user message."""
    route: str
    confidence: float       # 0.0–1.0
    reason: str             # human-readable explanation
    bypass_llm: bool        # True → route does NOT need an LLM call
    matched_keywords: List[str] = field(default_factory=list)


# ── Route definitions ────────────────────────────────────────────────
# Each route has: name, keywords/patterns, bypass_llm flag, min confidence
# Keywords are checked as whole-word boundaries where possible.

_ROUTES: List[Dict[str, Any]] = [
    {
        "name": "MEMORY_RECALL",
        "bypass_llm": True,
        "patterns": [
            re.compile(r"\b(remember|recall|last\s+time|what\s+did\s+(i|we)|find\s+my|search\s+for|look\s+up)\b", re.I),
            re.compile(r"\b(my\s+notes?|my\s+memories?|past\s+(conversation|interaction|session))\b", re.I),
            re.compile(r"\b(have\s+i\s+(ever|already)|did\s+(i|we)\s+(ever|already|previously))\b", re.I),
            re.compile(r"\b(what\s+was\s+(the|that|my)|when\s+did\s+(i|we))\b", re.I),
        ],
        "min_confidence": 0.7,
    },
    {
        "name": "TAX_ADVISORY",
        "bypass_llm": True,
        "patterns": [
            re.compile(r"\b(tax|vat|mtd|hmrc|self[- ]employ|national\s+insurance)\b", re.I),
            re.compile(r"\b(expense|invoice|receipt|deduct|allowance|mileage)\b", re.I),
            re.compile(r"\b(sa100|sa103|31\s+january|payment\s+on\s+account)\b", re.I),
            re.compile(r"\b(trading\s+income|turnover|profit\s+and\s+loss|threshold)\b", re.I),
            re.compile(r"\b(gnucash|bookkeep|accounting)\b", re.I),
        ],
        "min_confidence": 0.75,
    },
    {
        "name": "FACT_CHECK",
        "bypass_llm": True,
        "patterns": [
            re.compile(r"\b(is\s+it\s+true|verify|fact[- ]check|check\s+if|confirm\s+that)\b", re.I),
            re.compile(r"\b(is\s+this\s+(correct|accurate|right|valid))\b", re.I),
            re.compile(r"\b(double[- ]check|cross[- ]check|validate)\b", re.I),
        ],
        "min_confidence": 0.8,
    },
    {
        "name": "PROACTIVE_REVIEW",
        "bypass_llm": True,
        "patterns": [
            re.compile(r"\b(what\s+should\s+i\s+know|any\s+(reminders?|alerts?|nudges?|pending))\b", re.I),
            re.compile(r"\b(what('s|\s+is)\s+coming\s+up|upcoming|due\s+soon)\b", re.I),
            re.compile(r"\b(brief\s+me|status\s+update|morning\s+brief|daily\s+digest)\b", re.I),
            re.compile(r"\b(anything\s+(urgent|important|new)|heads\s+up)\b", re.I),
        ],
        "min_confidence": 0.75,
    },
    {
        "name": "REFLECT",
        "bypass_llm": True,
        "patterns": [
            re.compile(r"\b(summarise|summarize)\s+(my|this|the)\s+(week|day|month|session)\b", re.I),
            re.compile(r"\b(what\s+have\s+i\s+been\s+working\s+on|consolidate|reflect)\b", re.I),
            re.compile(r"\b(weekly\s+(review|summary|digest)|end\s+of\s+(day|week)\s+summary)\b", re.I),
        ],
        "min_confidence": 0.8,
    },
    {
        "name": "EXECUTE_ACTION",
        "bypass_llm": False,  # needs plan construction
        "patterns": [
            re.compile(r"\b(run|execute|deploy|build|create|delete|install|start|stop|restart)\b", re.I),
            re.compile(r"\b(write\s+a?\s*(file|script|code)|make\s+a?\s*(backup|snapshot))\b", re.I),
            re.compile(r"\b(send\s+(email|message|alert)|push\s+to|pull\s+from)\b", re.I),
            re.compile(r"\b(update\s+(the\s+)?(config|setting|database|server))\b", re.I),
        ],
        "min_confidence": 0.65,
    },
    {
        "name": "MULTI_SIGNAL",
        "bypass_llm": False,
        "patterns": [
            re.compile(r"\b(compare|multiple\s+opinions?|consensus|second\s+opinion)\b", re.I),
            re.compile(r"\b(ask\s+(all|every|multiple)|cross[- ]reference|triangulate)\b", re.I),
            re.compile(r"\b(debate|pros?\s+and\s+cons?|weigh\s+(up|the))\b", re.I),
        ],
        "min_confidence": 0.7,
    },
]

# GENERAL_CHAT is the fallback — no patterns needed


def classify(user_input: str, session_context: Optional[Dict[str, Any]] = None) -> RouteDecision:
    """Classify a user message into a route.

    Rule-based, deterministic, zero-LLM. Runs in microseconds.

    Args:
        user_input: The raw user message.
        session_context: Optional dict with session metadata (mode, recent
            routes, etc.) for context-aware routing.

    Returns:
        RouteDecision with the chosen route, confidence, and explanation.
    """
    ctx = session_context or {}
    text = user_input.strip()

    if not text:
        return RouteDecision(
            route="GENERAL_CHAT",
            confidence=1.0,
            reason="empty input",
            bypass_llm=False,
        )

    best_route: Optional[str] = None
    best_confidence: float = 0.0
    best_keywords: List[str] = []
    best_bypass: bool = False

    for route_def in _ROUTES:
        name = route_def["name"]
        bypass = route_def["bypass_llm"]
        min_conf = route_def["min_confidence"]
        matched: List[str] = []

        for pattern in route_def["patterns"]:
            match = pattern.search(text)
            if match:
                matched.append(match.group(0).lower())

        if not matched:
            continue

        # confidence scales with match count + specificity
        # 1 match → min_confidence, each additional +0.05, cap at 0.99
        confidence = min(min_conf + (len(matched) - 1) * 0.05, 0.99)

        # context boost: if recent route was the same, slightly boost
        if ctx.get("last_route") == name:
            confidence = min(confidence + 0.05, 0.99)

        # priority boost: MEMORY_RECALL wins ties when explicit recall
        # verbs are present ("remember", "search for", "find my", etc.)
        if name == "MEMORY_RECALL":
            confidence += 0.06  # ensure recall beats other routes
            confidence = min(confidence, 0.99)

        if confidence > best_confidence:
            best_route = name
            best_confidence = confidence
            best_keywords = matched
            best_bypass = bypass

    if best_route:
        reason = f"matched {len(best_keywords)} signal(s): {', '.join(best_keywords[:5])}"
        return RouteDecision(
            route=best_route,
            confidence=round(best_confidence, 3),
            reason=reason,
            bypass_llm=best_bypass,
            matched_keywords=best_keywords,
        )

    # Fallback: GENERAL_CHAT
    return RouteDecision(
        route="GENERAL_CHAT",
        confidence=0.5,
        reason="no strong specialist signal — general conversation",
        bypass_llm=False,
    )


# ── Service dispatch helpers ─────────────────────────────────────────
# These map route decisions to actual service calls.  Used by app.py.

async def dispatch_memory_recall(query: str, user_id: str = "keeper", top_k: int = 5) -> str:
    """Dispatch to Memu-Core /memory/retrieve — zero LLM cost."""
    import httpx
    import os
    memu_url = os.getenv("MEMU_URL", "http://memu-core:8001")
    try:
        async with httpx.AsyncClient(timeout=8.0) as client:
            resp = await client.get(
                f"{memu_url}/memory/retrieve",
                params={"query": query, "user_id": user_id, "top_k": top_k},
            )
            resp.raise_for_status()
            records = resp.json()

        if not records:
            return "I searched my memory but didn't find anything matching that. Could you give me more detail?"

        # format results as a natural response
        parts = []
        for i, r in enumerate(records[:5], 1):
            content = r.get("content", {})
            text = content.get("text", "") or content.get("result", "") or content.get("query", "")
            if text:
                cat = r.get("category", "general")
                parts.append(f"{i}. [{cat}] {text[:300]}")

        if parts:
            return "Here's what I found in my memory:\n\n" + "\n".join(parts)
        return "I found some records but they didn't have readable content. Try rephrasing?"

    except Exception:
        return "I couldn't reach my memory system right now. Please try again in a moment."


async def dispatch_tax_advisory(query: str) -> str:
    """Dispatch to kai-advisor rules engine — zero LLM cost."""
    import os
    from common.self_emp_advisor import advise, load_expenses, load_income_total, thresholds

    self_emp_root = os.getenv("SELF_EMP_ROOT", "/data/self-emp")
    income_csv = os.getenv("INCOME_CSV", f"{self_emp_root}/Accounting/income.csv")
    expenses_log = os.getenv("EXPENSES_LOG", f"{self_emp_root}/Accounting/expenses.log")

    income_total = load_income_total(income_csv)
    expenses_lines = load_expenses(expenses_log)
    suggestions = advise(income_total=income_total, expenses_lines=expenses_lines)
    th = thresholds()

    parts = [f"**Tax Advisory (offline UK rules)**\n"]
    parts.append(f"- Recorded income: £{income_total:,.2f}")
    parts.append(f"- MTD threshold: £{th.get('mtd_start', 50000):,.0f}")
    parts.append(f"- VAT threshold: £{th.get('vat_threshold', 90000):,.0f}")

    if suggestions:
        parts.append("\n**Suggestions:**")
        for s in suggestions:
            parts.append(f"- {s}")
    else:
        parts.append("\nNo immediate action items detected.")

    return "\n".join(parts)


async def dispatch_fact_check(claim: str) -> str:
    """Dispatch to Verifier /verify — zero LLM cost."""
    import httpx
    import os
    verifier_url = os.getenv("VERIFIER_URL", "http://verifier:8003")
    try:
        async with httpx.AsyncClient(timeout=8.0) as client:
            resp = await client.post(
                f"{verifier_url}/verify",
                json={"claim": claim, "source": "router-fact-check"},
            )
            resp.raise_for_status()
            result = resp.json()

        verdict = result.get("verdict", "UNKNOWN")
        evidence = result.get("evidence_summary", "No additional evidence.")
        return f"**Fact Check Result:** {verdict}\n\n{evidence}"

    except Exception:
        return "I couldn't reach the verification service right now. Please try again."


async def dispatch_proactive_review() -> str:
    """Dispatch to Memu-Core /memory/proactive — zero LLM cost."""
    import httpx
    import os
    memu_url = os.getenv("MEMU_URL", "http://memu-core:8001")
    try:
        async with httpx.AsyncClient(timeout=8.0) as client:
            resp = await client.get(f"{memu_url}/memory/proactive")
            resp.raise_for_status()
            data = resp.json()

        nudges = data.get("nudges", [])
        if not nudges:
            return "All clear — nothing time-sensitive in my memory right now."

        parts = [f"**{len(nudges)} reminder(s) for you:**\n"]
        for n in nudges:
            msg = n.get("nudge_message", "")
            signals = ", ".join(n.get("time_signals", []))
            parts.append(f"- {msg}")
            if signals:
                parts[-1] += f" (signals: {signals})"

        return "\n".join(parts)

    except Exception:
        return "I couldn't check my reminders right now. Please try again."


async def dispatch_reflect() -> str:
    """Dispatch to Memu-Core /memory/reflect — zero LLM cost."""
    import httpx
    import os
    memu_url = os.getenv("MEMU_URL", "http://memu-core:8001")
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.post(f"{memu_url}/memory/reflect")
            resp.raise_for_status()
            data = resp.json()

        status = data.get("status", "unknown")
        if status == "skipped":
            return f"Not enough recent memories to reflect on ({data.get('reason', 'unknown')})."

        insights = data.get("insights", [])
        keywords = data.get("keyword_themes", [])
        analyzed = data.get("memories_analyzed", 0)

        parts = [f"**Reflection complete** — analyzed {analyzed} recent memories.\n"]
        if insights:
            parts.append("**Insights:**")
            for insight in insights:
                parts.append(f"- {insight}")
        if keywords:
            parts.append(f"\n**Trending themes:** {', '.join(keywords)}")

        return "\n".join(parts)

    except Exception:
        return "I couldn't run reflection right now. Please try again."


async def dispatch_route(decision: RouteDecision, user_input: str, session_id: str = "default") -> Optional[str]:
    """Dispatch a classified request to the right handler.

    Returns the response string for zero-LLM routes.
    Returns None for routes that need LLM processing (GENERAL_CHAT,
    EXECUTE_ACTION, MULTI_SIGNAL) — caller should fall through to
    the existing LLM pipeline.
    """
    route = decision.route

    if route == "MEMORY_RECALL":
        return await dispatch_memory_recall(user_input)
    elif route == "TAX_ADVISORY":
        return await dispatch_tax_advisory(user_input)
    elif route == "FACT_CHECK":
        return await dispatch_fact_check(user_input)
    elif route == "PROACTIVE_REVIEW":
        return await dispatch_proactive_review()
    elif route == "REFLECT":
        return await dispatch_reflect()

    # GENERAL_CHAT, EXECUTE_ACTION, MULTI_SIGNAL → return None
    # to signal "use the LLM pipeline"
    return None


# ═══════════════════════════════════════════════════════════════════════
# J7: SKILLS AUTO-INSTALL HUB
#  Reads .md skill files from /skills/ directory, parses trigger patterns,
#  and registers them as additional routes.
#
#  Skill file format:
#    # Skill: <Name>
#    ## Trigger patterns
#    - "pattern1"
#    - "pattern2"
#    ## Action
#    <instruction text or endpoint URL>
#    ## Response template
#    <optional response format>
#
#  Hot-reload: call load_skills() or POST /skills/reload
# ═══════════════════════════════════════════════════════════════════════

import os  # noqa: E402
from pathlib import Path as _Path  # noqa: E402


@dataclass
class Skill:
    """A loaded skill definition from a .md file."""
    name: str
    trigger_patterns: List[re.Pattern]
    trigger_strings: List[str]
    action: str
    response_template: str
    source_file: str


_SKILLS_DIR = _Path(os.getenv("SKILLS_DIR", "/skills"))
_LOCAL_SKILLS_DIR = _Path("data/skills")  # fallback for dev
_loaded_skills: List[Skill] = []


def _parse_skill_file(path: _Path) -> Optional[Skill]:
    """Parse a skill .md file into a Skill object."""
    try:
        text = path.read_text(encoding="utf-8")
    except Exception:
        return None

    name = ""
    triggers: List[str] = []
    action = ""
    response_template = ""
    section = ""

    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("# Skill:"):
            name = stripped[8:].strip()
        elif stripped.lower().startswith("## trigger"):
            section = "trigger"
        elif stripped.lower().startswith("## action"):
            section = "action"
        elif stripped.lower().startswith("## response"):
            section = "response"
        elif stripped.startswith("## "):
            section = ""
        elif section == "trigger" and stripped.startswith("- "):
            # Extract pattern from "- "pattern"" or "- pattern"
            pat = stripped[2:].strip().strip('"').strip("'")
            if pat:
                triggers.append(pat)
        elif section == "action" and stripped:
            action += stripped + "\n"
        elif section == "response" and stripped:
            response_template += stripped + "\n"

    if not name or not triggers:
        return None

    patterns = []
    for t in triggers:
        try:
            patterns.append(re.compile(r"\b" + re.escape(t) + r"\b", re.IGNORECASE))
        except re.error:
            patterns.append(re.compile(re.escape(t), re.IGNORECASE))

    return Skill(
        name=name,
        trigger_patterns=patterns,
        trigger_strings=triggers,
        action=action.strip(),
        response_template=response_template.strip(),
        source_file=str(path),
    )


def load_skills() -> List[Skill]:
    """Scan skills directories and load all .md skill files."""
    global _loaded_skills
    skills: List[Skill] = []

    for skills_dir in [_SKILLS_DIR, _LOCAL_SKILLS_DIR]:
        if not skills_dir.exists():
            continue
        for md_file in sorted(skills_dir.glob("*.md")):
            skill = _parse_skill_file(md_file)
            if skill:
                skills.append(skill)

    _loaded_skills = skills
    return skills


def match_skill(user_input: str) -> Optional[Skill]:
    """Check if user input matches any loaded skill trigger patterns."""
    for skill in _loaded_skills:
        for pattern in skill.trigger_patterns:
            if pattern.search(user_input):
                return skill
    return None


def list_skills() -> List[Dict[str, Any]]:
    """Return summary of all loaded skills."""
    return [
        {
            "name": s.name,
            "triggers": s.trigger_strings,
            "action": s.action[:200],
            "source": s.source_file,
        }
        for s in _loaded_skills
    ]


# Load skills on import
load_skills()
