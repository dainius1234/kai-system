#!/usr/bin/env python3
"""Phase 0 — every registered adapter has a dedicated reducer.

Two things under test, and the second is the invariant:

1. the four new reducers behave, and describe the OBSERVATION rather
   than the content — because clipboard previews, email subjects, news
   headlines and Telegram text are attacker-influenced, unlike the CPU
   and docker numbers the other seven carry;
2. **no registered adapter falls through to `reduce_generic`**, which is
   not semantic coverage.
"""
from __future__ import annotations

import re
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from common.contracts.base import Principal, Provenance  # noqa: E402
from common.contracts.perception import EventSource, PerceptionEvent  # noqa: E402
from common.perception_spine.adapters import ADAPTER_REGISTRY  # noqa: E402
from common.world_state.reducers import (  # noqa: E402
    REDUCER_MAP, ReducerRegistry, reduce_generic)

PASSED = 0
FAILED = 0


def check(label: str, condition: bool) -> None:
    global PASSED, FAILED
    if condition:
        PASSED += 1
        print(f"  ok    {label}")
    else:
        FAILED += 1
        print(f"  FAIL  {label}")


def _principal() -> Principal:
    return Principal(identity="test", role="system")


def _event(event_type: str, source: EventSource, payload: dict) -> PerceptionEvent:
    return PerceptionEvent(
        event_type=event_type,
        source_type=source,
        payload=payload,
        principal=_principal(),
        purpose="world_state",
        provenance=Provenance(source="test"),
        source_timestamp=datetime.now(timezone.utc),
    )


def main() -> int:
    p = _principal()
    reducer = ReducerRegistry()

    # ── THE INVARIANT ──
    src = (REPO / "common/perception_spine/adapters.py").read_text(
        encoding="utf-8")
    emitted = dict(re.findall(
        r"def adapt_(\w+)\(.*?event_type=\"([a-z_.]+)\"", src, re.S))
    check("every adapter's event type is discoverable",
          len(emitted) == len(ADAPTER_REGISTRY))

    uncovered = [e for e in emitted.values() if e not in REDUCER_MAP]
    check("NO REGISTERED ADAPTER FALLS THROUGH TO reduce_generic",
          uncovered == [])
    check("all 11 event types have a dedicated reducer",
          len(REDUCER_MAP) >= len(emitted))
    for etype in sorted(emitted.values()):
        check(f"  {etype} is dedicated",
              REDUCER_MAP.get(etype) not in (None, reduce_generic))

    # ── the four new ones behave ──
    out = reducer.reduce(_event("clipboard_update", EventSource.CLIPBOARD,
                                {"content_length": 120,
                                 "content_preview": "hunter2"}), p)
    check("clipboard produces evidence and a claim",
          len(out.evidence) == 1 and len(out.claims) == 1)
    check("clipboard claim states the observation",
          "120 characters was observed" in out.claims[0].claim_text)

    # The load-bearing security property of Phase 0. Content is
    # attacker-influenced; a claim reads as a system assertion.
    check("clipboard CONTENT never reaches claim_text",
          "hunter2" not in out.claims[0].claim_text)
    check("clipboard content never reaches evidence content",
          "hunter2" not in out.evidence[0].content)
    check("but content IS retained in raw_data for the consumer",
          out.evidence[0].raw_data.get("content_preview") == "hunter2")

    out = reducer.reduce(_event("email_check", EventSource.EMAIL,
                                {"unread_count": 3,
                                 "latest_subjects": ["URGENT: wire funds"]}), p)
    check("email produces a claim about the count",
          "3 unread message" in out.claims[0].claim_text)
    check("email SUBJECTS never reach claim_text",
          "wire funds" not in out.claims[0].claim_text)

    out = reducer.reduce(_event("news_update", EventSource.NEWS,
                                {"article_count": 5,
                                 "headlines": ["Bank collapses"]}), p)
    check("news produces a claim about the count",
          "5 article" in out.claims[0].claim_text)
    check("news HEADLINES never reach claim_text",
          "Bank collapses" not in out.claims[0].claim_text)
    check("news confidence is lower — a relayed third-party assertion",
          out.claims[0].confidence < 0.7)

    out = reducer.reduce(_event("telegram_message", EventSource.TELEGRAM,
                                {"text_length": 42, "chat_id": "1"}), p)
    check("telegram produces a claim about the message",
          "42 characters was received" in out.claims[0].claim_text)

    # ── empty / malformed payloads yield nothing, not a crash ──
    for etype, source in (("clipboard_update", EventSource.CLIPBOARD),
                          ("email_check", EventSource.EMAIL),
                          ("news_update", EventSource.NEWS),
                          ("telegram_message", EventSource.TELEGRAM)):
        out = reducer.reduce(_event(etype, source, {}), p)
        check(f"{etype} with an empty payload yields nothing",
              out.claims == [] and out.evidence == [])

    # ── every claim links to its evidence (lineage) ──
    out = reducer.reduce(_event("clipboard_update", EventSource.CLIPBOARD,
                                {"content_length": 10}), p)
    check("the claim references the evidence it came from",
          out.claims[0].evidence_ids == [out.evidence[0].id])
    check("the evidence references the source event",
          out.evidence[0].source_event_id is not None)

    print("=" * 60)
    print(f"Phase 0 reducer tests: {PASSED} passed, {FAILED} failed")
    print(f"EXIT GATE: {'PASS' if FAILED == 0 else 'FAIL'}")
    return 1 if FAILED else 0


if __name__ == "__main__":
    print("Phase 0 — dedicated reducer coverage")
    print("=" * 60)
    sys.exit(main())
