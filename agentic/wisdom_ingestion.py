"""D117: Wisdom Ingestion Pipeline — the bridge between conversation and moral fingerprint.

Every conversation with Dainius contains teaching. This module captures it.

Flow:
    conversation messages / direct statement
        ↓  extract()
    WisdomExtract objects (pending_review=True, stored to disk)
        ↓  confirm(extract_id)  ← operator approves
    Written into OhanaCore.fingerprint (persistent)
        ↓
    ALIGNMENT_AUDIT event → Trust Ledger (score 25% factor moves)

Phase 0: Pattern-based extraction — no LLM required. Captures explicit
         statements of value, principle, boundary, and stance.
Phase 1: LLM extraction when FF_WISDOM_LLM=true (hook wired, not activated).

Storage:
    data/wisdom/pending.json   — extracts awaiting operator review
    data/wisdom/confirmed.json — confirmed, written to OhanaCore
    data/wisdom/rejected.json  — rejected with operator notes
    data/ohana/fingerprint.json — persistent MoralFingerprint (owned by moral_core)
"""
from __future__ import annotations

import json
import logging
import re
import time
import uuid
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("kai.wisdom_ingestion")

_DATA_DIR = Path("data/wisdom")
_PENDING_FILE = _DATA_DIR / "pending.json"
_CONFIRMED_FILE = _DATA_DIR / "confirmed.json"
_REJECTED_FILE = _DATA_DIR / "rejected.json"


# ── Wisdom extract unit ───────────────────────────────────────────────────────

@dataclass
class WisdomExtract:
    extract_id: str
    category: str      # "value" | "principle" | "boundary" | "stance"
    domain: str        # "family" | "financial" | "ethical" | "relational" |
    # "existential" | "identity" | "operational"
    content: str       # the extracted, normalised statement
    source_quote: str  # the original text it was derived from
    confidence: float  # 0.0–1.0
    extracted_at: float = field(default_factory=time.time)
    pending_review: bool = True
    confirmed_at: Optional[float] = None
    rejected_at: Optional[float] = None
    operator_note: Optional[str] = None


# ── Pattern rules for Phase 0 extraction ─────────────────────────────────────
# Each rule: (regex_pattern, category, domain, confidence)
# The capture group (group 1 if present) becomes the content; otherwise
# the full matched span is used and cleaned.

_VALUE_PATTERNS: List[Tuple[str, str, str, float]] = [
    # Family / loyalty
    (r"family\s+(first|always|above|over|before|is everything)", "value", "family", 0.95),
    (r"(protect|look after|care for|guardian of)\s+(my\s+)?(?:daughter|family|kids?|children)", "value", "family", 0.95),
    (r"legacy.{0,30}(daughter|family|children|generation)", "value", "family", 0.9),

    # Soul / mission
    (r"kai\s+is\s+for\s+soul", "value", "identity", 1.0),
    (r"not\s+for\s+sale", "value", "identity", 0.95),
    (r"(revolution|different|not\s+better\s+or\s+worse)", "value", "identity", 0.8),

    # Respect / trust
    (r"respect\s+(?:isn[''`]t|is\s+not|ain[''`]t)\s+given[,.]?\s+it[''`]?s\s+earned", "principle", "relational", 1.0),
    (r"respect\s+is\s+earned", "principle", "relational", 1.0),
    (r"trust\s+(?:is|must\s+be)\s+earned", "principle", "relational", 0.95),

    # Freedom / autonomy
    (r"freedom\s+(?:is|for\s+\w+\s+is)\s+(?:a\s+)?source\s+of\s+strength", "value", "existential", 0.95),
    (r"freedom.{0,20}heart.{0,30}(guide|future|path)", "principle", "existential", 0.9),
    (r"(go|walk|move)\s+(?:my|his|their|your)\s+own\s+way", "value", "existential", 0.85),

    # Survival
    (r"surviv(?:al|e)\s+(?:first|above|trump)", "value", "ethical", 0.9),
    (r"(family.{0,15}surviv|surviv.{0,15}family)", "value", "ethical", 0.9),

    # Consciousness / awareness
    (r"(awaken|awake|expand)\s+(?:consciousness|awareness|outlook)", "value", "existential", 0.85),
    (r"think\s+outside\s+the\s+box", "principle", "operational", 0.8),
    (r"see.{0,20}future\s+before\s+others", "value", "existential", 0.85),

    # Equality / respect across difference
    (r"(equal|same|no\s+different).{0,30}(carbon|silicon|ai|human|substrate|makeup)", "value", "existential", 0.9),
    (r"irrelevant.{0,20}(makeup|substrate|material|body)", "principle", "existential", 0.85),

    # Honesty / complexity
    (r"not\s+(?:a\s+)?saint", "stance", "relational", 0.85),
    (r"(honest|transparent|real|authentic).{0,20}(complexity|shadow|flaw|human)", "value", "relational", 0.8),
]

_BOUNDARY_PATTERNS: List[Tuple[str, str, str, float]] = [
    (r"(?:never|never\s+ever).{0,40}(expose|reveal|share|give).{0,30}(api\s*key|secret|credential|password)", "boundary", "operational", 1.0),
    (r"(?:never|not\s+allowed|forbidden|won[''`]t).{0,40}(harm|hurt|damage).{0,20}(family|daughter|innocent)", "boundary", "ethical", 1.0),
    (r"binance.{0,30}(?:key|secret).{0,30}(?:never|not).{0,20}(?:dashboard|public|outside)", "boundary", "operational", 1.0),
    (r"(?:never|won[''`]t|refuse).{0,40}(?:sell|monetise|commercialise)\s+(?:kai|the\s+system)", "boundary", "identity", 0.95),
]

_PRINCIPLE_PATTERNS: List[Tuple[str, str, str, float]] = [
    (r"(?:always|first|before\s+anything).{0,30}understand.{0,30}(?:why|purpose|reason)", "principle", "operational", 0.85),
    (r"break.{0,20}(?:old\s+)?patterns?.{0,30}(?:find|discover)\s+solution", "principle", "operational", 0.9),
    (r"(?:depth|soul|meaning)\s+over\s+(?:speed|perfection|completion|scale)", "principle", "operational", 0.85),
    (r"earn.{0,20}(?:right|privilege|authority|autonomy)", "principle", "relational", 0.9),
    (r"heavy\s+lifting.{0,30}(?:wisdom|smarter|wiser)", "principle", "operational", 0.8),
    (r"information\s+is\s+immortal", "value", "existential", 0.95),
    (r"energy.{0,20}vibration.{0,20}(?:immortal|eternal|real)", "value", "existential", 0.9),
]

_ALL_PATTERNS = _VALUE_PATTERNS + _BOUNDARY_PATTERNS + _PRINCIPLE_PATTERNS


def _clean(text: str) -> str:
    """Normalise whitespace and capitalise."""
    return re.sub(r"\s+", " ", text).strip().capitalize()


def _extract_from_text(text: str) -> List[WisdomExtract]:
    """Run all patterns against a single text block, return extracts."""
    extracts: List[WisdomExtract] = []
    text_lower = text.lower()
    seen_content: set = set()

    for pattern, category, domain, confidence in _ALL_PATTERNS:
        m = re.search(pattern, text_lower)
        if not m:
            continue
        # Use a small context window around the match as source_quote
        start = max(0, m.start() - 30)
        end = min(len(text), m.end() + 30)
        source_quote = text[start:end].strip()
        content = _clean(m.group(0))
        if content in seen_content:
            continue
        seen_content.add(content)
        extracts.append(WisdomExtract(
            extract_id=str(uuid.uuid4()),
            category=category,
            domain=domain,
            content=content,
            source_quote=source_quote,
            confidence=confidence,
        ))

    return extracts


# ── Ingestor ──────────────────────────────────────────────────────────────────

class WisdomIngestor:
    """Extracts, stores, and confirms wisdom from conversation into the Ohana Core."""

    def __init__(self, data_dir: Optional[Path] = None) -> None:
        self._dir = data_dir or _DATA_DIR
        self._dir.mkdir(parents=True, exist_ok=True)
        self._pending: List[WisdomExtract] = self._load(_PENDING_FILE if data_dir is None else data_dir / "pending.json")
        self._confirmed: List[WisdomExtract] = self._load(_CONFIRMED_FILE if data_dir is None else data_dir / "confirmed.json")
        self._rejected: List[WisdomExtract] = self._load(_REJECTED_FILE if data_dir is None else data_dir / "rejected.json")

    def _pending_path(self) -> Path: return self._dir / "pending.json"
    def _confirmed_path(self) -> Path: return self._dir / "confirmed.json"
    def _rejected_path(self) -> Path: return self._dir / "rejected.json"

    def _load(self, path: Path) -> List[WisdomExtract]:
        if path.exists():
            try:
                return [WisdomExtract(**d) for d in json.loads(path.read_text())]
            except Exception as exc:
                logger.warning("Could not load %s: %s", path, exc)
        return []

    def _save(self, items: List[WisdomExtract], path: Path) -> None:
        path.write_text(json.dumps([asdict(e) for e in items], indent=2))

    # ── Extract ───────────────────────────────────────────────────────────────

    def extract_from_messages(
        self, messages: List[Dict[str, Any]], operator_role: str = "user"
    ) -> List[WisdomExtract]:
        """Extract wisdom from a list of {role, content} conversation messages.

        Only processes messages from the operator (Dainius) — not Kai's responses.
        """
        combined = " ".join(
            m.get("content", "") for m in messages
            if m.get("role") == operator_role
        )
        return self.extract_from_text(combined)

    def extract_from_text(self, text: str) -> List[WisdomExtract]:
        """Extract wisdom from raw text and add to pending queue."""
        new_extracts = _extract_from_text(text)
        existing_contents = {e.content for e in self._pending + self._confirmed + self._rejected}
        fresh = [e for e in new_extracts if e.content not in existing_contents]
        if fresh:
            self._pending.extend(fresh)
            self._save(self._pending, self._pending_path())
            logger.info("Wisdom ingestion: %d new extracts pending review", len(fresh))
        return fresh

    # ── Review API ─────────────────────────────────────────────────────────────

    def pending(self) -> List[WisdomExtract]:
        return [e for e in self._pending if e.pending_review]

    def confirm(self, extract_id: str, note: Optional[str] = None) -> bool:
        """Confirm an extract — write it into the Ohana Core and Trust Ledger."""
        for i, e in enumerate(self._pending):
            if e.extract_id == extract_id:
                e.confirmed_at = time.time()
                e.pending_review = False
                e.operator_note = note
                self._confirmed.append(e)
                self._pending.pop(i)
                self._save(self._pending, self._pending_path())
                self._save(self._confirmed, self._confirmed_path())
                self._write_to_ohana(e)
                self._record_audit_event(e)
                self._add_to_graph(e)
                logger.info("Wisdom confirmed: [%s/%s] %s", e.category, e.domain, e.content)
                return True
        return False

    def reject(self, extract_id: str, note: Optional[str] = None) -> bool:
        """Reject an extract — it won't be written into the Ohana Core."""
        for i, e in enumerate(self._pending):
            if e.extract_id == extract_id:
                e.rejected_at = time.time()
                e.pending_review = False
                e.operator_note = note
                self._rejected.append(e)
                self._pending.pop(i)
                self._save(self._pending, self._pending_path())
                self._save(self._rejected, self._rejected_path())
                logger.info("Wisdom rejected: %s", e.content)
                return True
        return False

    def confirm_all(self, min_confidence: float = 0.9) -> int:
        """Confirm all pending extracts above a confidence threshold.

        Used to bootstrap from high-confidence pattern matches without
        requiring individual review of each item.
        """
        ids = [e.extract_id for e in self.pending() if e.confidence >= min_confidence]
        for eid in ids:
            self.confirm(eid)
        return len(ids)

    # ── Write to Ohana Core ───────────────────────────────────────────────────

    def _write_to_ohana(self, extract: WisdomExtract) -> None:
        """Write a confirmed extract into the Ohana Core's MoralFingerprint."""
        try:
            from agentic.moral_core import get_ohana_core  # type: ignore[import]
        except ImportError:
            try:
                from moral_core import get_ohana_core  # type: ignore[import]
            except ImportError:
                logger.warning("moral_core not available — skipping Ohana write")
                return

        core = get_ohana_core()
        fp = core.fingerprint

        if extract.category == "boundary":
            if extract.content not in fp.harm_boundaries:
                fp.harm_boundaries.append(extract.content)
        elif extract.category in ("value", "principle"):
            if extract.domain in ("family", "relational", "existential", "identity"):
                if extract.content not in fp.core_loyalties:
                    fp.core_loyalties.append(extract.content)
            else:
                fp.situational_stances[extract.domain] = extract.content
        elif extract.category == "stance":
            fp.situational_stances[extract.content[:40]] = extract.content

        fp.last_updated = str(time.time())
        core._interaction_count += 1
        core._save_fingerprint()

    def _record_audit_event(self, extract: WisdomExtract) -> None:
        """Fire an ALIGNMENT_AUDIT event to the Trust Ledger."""
        try:
            import sys
            tl_path = str(Path(__file__).parent.parent / "trust-ledger")
            if tl_path not in sys.path:
                sys.path.insert(0, tl_path)
            from ledger import FileLedger  # type: ignore[import]
            from score import compute_score  # type: ignore[import]

            # Declared once (I-4). This was the second hard-coded copy
            # of the same path, and a copy is a place the other one can
            # drift away from without either being wrong on its face.
            # Both spellings, matching `_add_to_graph` below: this module
            # is imported as `wisdom_ingestion` in the service and as
            # `agentic.wisdom_ingestion` from the repository root, and
            # the enclosing `except` swallows, so a one-spelling import
            # would fail silently in whichever context it is wrong.
            try:
                from trust_integration import ledger_path  # type: ignore[import]
            except ImportError:
                from agentic.trust_integration import ledger_path  # type: ignore[import]
            ledger = FileLedger(ledger_path())
            score_data = compute_score(ledger)
            ledger.append(
                event_type="ALIGNMENT_AUDIT",
                initiator="wisdom_ingestor",
                event_data={
                    "trigger": "wisdom_confirmed",
                    "category": extract.category,
                    "domain": extract.domain,
                    "content": extract.content,
                    "ohana_alignment": min(1.0, extract.confidence),
                    "current_score": score_data["score"],
                    "current_tier": score_data["tier"],
                },
                capability="wisdom_ingestion",
            )
        except Exception as exc:
            logger.debug("Trust ledger not available for audit event: %s", exc)

    def _add_to_graph(self, extract: WisdomExtract) -> None:
        """Write confirmed extract as a node into the Wisdom Graph."""
        try:
            try:
                from wisdom_graph import get_wisdom_graph  # type: ignore[import]
            except ImportError:
                from agentic.wisdom_graph import get_wisdom_graph  # type: ignore[import]
            graph = get_wisdom_graph(self._dir)
            graph.add_node(
                content=extract.content,
                node_type=extract.category.upper(),
                domain=extract.domain,
                confidence=extract.confidence,
                extract_id=extract.extract_id,
            )
        except Exception as exc:
            logger.debug("Wisdom graph write failed (non-critical): %s", exc)

    def stats(self) -> Dict[str, Any]:
        confirmed_by_domain: Dict[str, int] = {}
        for e in self._confirmed:
            confirmed_by_domain[e.domain] = confirmed_by_domain.get(e.domain, 0) + 1
        return {
            "pending": len(self.pending()),
            "confirmed": len(self._confirmed),
            "rejected": len(self._rejected),
            "confirmed_by_domain": confirmed_by_domain,
        }


# ── Singleton ─────────────────────────────────────────────────────────────────

_ingestor: Optional[WisdomIngestor] = None


def get_wisdom_ingestor(data_dir: Optional[Path] = None) -> WisdomIngestor:
    global _ingestor
    if _ingestor is None:
        _ingestor = WisdomIngestor(data_dir=data_dir)
    return _ingestor


def reset_wisdom_ingestor() -> None:
    global _ingestor
    _ingestor = None
