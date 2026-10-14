"""KAI-GATE-048 C: recover each attempt's CONTRACT, then classify its response.

Pure stdlib. Calibratable on a host with no cognee, no container, no model
and — deliberately — no instructor and no jsonschema, because the analyser
must state honestly what it could and could not establish in each case.

WHAT D222 ESTABLISHED, AND WHY THIS FILE CHANGED
================================================

Read out of the pinned wheels the image installs (instructor 1.15.1):

* `response_model` is a NAMED PARAMETER of instructor's patched create
  (`core/patch.py:147`) and is returned separately by
  `handle_response_model` (`processing/response.py:409`). It NEVER enters
  the kwargs that reach `Completions.create`. Fabricating it into a
  production row would be inventing evidence.
* Under `Mode.JSON`, `handle_json_modes`
  (`providers/openai/utils.py:491`) puts `response_format` =
  `{"type": "json_object"}` — a MODE DIRECTIVE carrying no contract — and
  appends `json.dumps(response_model.model_json_schema(), indent=2)` to
  the SYSTEM MESSAGE.
* Retries (`reask_md_json`, `providers/openai/utils.py:151`) APPEND the
  failed reply and a repair instruction. They do not touch `messages[0]`.

So the contract for an attempt is carried BY THAT ATTEMPT, inside
`messages[0]["content"]`. It is recovered from the row, never reached
down from the outer caller.

TWO INDEPENDENT LABELS THAT MUST NOT MERGE
==========================================

`REQUIRED_FIELDS_PRESENT` is not `VALID_INSTANCE`. Checking that the
top-level required keys exist is not JSON Schema validation: it ignores
types, formats, nested objects, enums, `additionalProperties` and every
other constraint. When no validator is available the analyser says the
narrower thing and does not round up. That is the same defect as D216's
vacuous predicate, one level less obvious.

THE QUESTIONS, ASKED INDEPENDENTLY AND IN ORDER
===============================================

    1. was the CONTRACT recovered?      -> CONTRACT_UNMEASURED
    2. is the response parseable JSON?  -> NOT_JSON / NOT_JSON_OBJECT
    3. is it the schema definition?     -> SCHEMA_ECHO
    4. does it VALIDATE as an instance? -> VALID_INSTANCE
                                           / INSTANCE_INVALID
       (no validator available)         -> REQUIRED_FIELDS_PRESENT
                                           / REQUIRED_FIELDS_MISSING
    5. if not, what failed?             -> carried in `why`

Unknown at any prerequisite leaves every dependent predicate UNMEASURED.
"""
from __future__ import annotations

import hashlib
import json
import re
from typing import Any, Dict, List, Optional, Tuple

# ── verdicts ─────────────────────────────────────────────────────────
CONTRACT_UNMEASURED = "CONTRACT_UNMEASURED"
NO_RESPONSE = "NO RESPONSE"
NOT_JSON = "NOT JSON"
NOT_JSON_OBJECT = "NOT A JSON OBJECT"
SCHEMA_ECHO = "SCHEMA ECHO"
VALID_INSTANCE = "VALID INSTANCE"
INSTANCE_INVALID = "INSTANCE INVALID"
REQUIRED_FIELDS_PRESENT = "REQUIRED FIELDS PRESENT"
REQUIRED_FIELDS_MISSING = "REQUIRED FIELDS MISSING"

# Only this one says "the response satisfied the captured JSON Schema".
PASSING_VERDICTS = frozenset({VALID_INSTANCE})

# ── layers (D215) ────────────────────────────────────────────────────
ADAPTER_INPUT = "ADAPTER_INPUT"
INSTRUCTOR_RETURN = "INSTRUCTOR_RETURN"
RAW_MODEL_REQUEST = "RAW_MODEL_REQUEST"
RAW_MODEL_RESPONSE = "RAW_MODEL_RESPONSE"
VALIDATION_RESULT = "VALIDATION_RESULT"

LAYERS = frozenset({ADAPTER_INPUT, INSTRUCTOR_RETURN, RAW_MODEL_REQUEST,
                    RAW_MODEL_RESPONSE, VALIDATION_RESULT})
LAYERS_LICENSING_RAW_RESPONSE_CLAIMS = frozenset({RAW_MODEL_RESPONSE})

_SCHEMA_KEYS = frozenset({
    "$schema", "$defs", "$ref", "properties", "required",
    "additionalProperties", "patternProperties", "definitions",
})


# ── identity: two different questions, two different functions ───────
def sha256_bytes(text: Optional[str]) -> str:
    """EXACT byte identity of the string as returned. Not canonical."""
    if text is None:
        return ""
    return hashlib.sha256(text.encode("utf-8", "replace")).hexdigest()


def canonical(obj: Any) -> str:
    """Order-independent JSON serialisation.

    Plain dicts serialise as VALID JSON — never `str(dict)`, whose single
    quotes are Python repr and not parseable. Run 17 recorded
    `response_format` that way, so the analyser could not read back its
    own record (D222 §5.3).
    """
    try:
        return json.dumps(obj, sort_keys=True, separators=(",", ":"),
                          default=str)
    except (TypeError, ValueError):
        return json.dumps(str(obj))


def sha256_canonical(obj: Any) -> str:
    """CANONICAL identity — key order irrelevant. Never call this byte identity."""
    return hashlib.sha256(canonical(obj).encode("utf-8", "replace")).hexdigest()


# ── is a real JSON Schema validator available? ───────────────────────
def validator_status() -> Dict[str, Any]:
    """Recorded, so a weaker claim is never mistaken for a stronger one."""
    try:
        import jsonschema  # noqa: F401
        try:                      # __version__ is deprecated in 4.26
            from importlib.metadata import version as _v
            ver = _v("jsonschema")
        except Exception:  # noqa: BLE001
            ver = None
        return {"available": True, "library": "jsonschema", "version": ver}
    except Exception as exc:  # noqa: BLE001
        return {"available": False, "library": None, "version": None,
                "why": f"{type(exc).__name__}: {exc}",
                "consequence": "no verdict stronger than "
                               f"{REQUIRED_FIELDS_PRESENT} is possible"}


# ── where the extraction rule comes from ─────────────────────────────
def _markers_from_source(src: str) -> List[str]:
    """The literal, non-interpolated lines of instructor's json-mode prose.

    Derived from the INSTALLED implementation's source text, so a change
    upstream changes this too. The wording moved once already (upstream
    issue #1514), which is exactly why it is not copied in here.
    """
    body = src.split("dedent(", 1)[-1]
    body = body.split('"""', 2)[1] if '"""' in body else body
    out = []
    for line in body.splitlines():
        line = line.strip()
        if line and "{" not in line and "}" not in line:
            out.append(line)
    return out


def extraction_rule_provenance() -> Dict[str, Any]:
    """WHERE the rule came from, recorded on every run (I-8).

    The property the analyser needs is *this region of the system message
    is the JSON Schema instructor generated for this request* — not *these
    English strings were found*. So the primary rule is STRUCTURAL and
    works with no instructor present; when instructor IS importable its
    source is read and used as an independent corroboration.
    """
    prov: Dict[str, Any] = {
        "primary_rule": "structural: the unique schema-shaped JSON object "
                        "embedded in this attempt's system message",
        "instructor_available": False,
        "instructor_version": None,
        "source": None,
        "source_sha256": None,
        "markers": [],
    }
    try:
        import inspect
        import instructor
        from instructor.providers.openai import utils as iu
        src = inspect.getsource(iu.handle_json_modes)
        prov.update({
            "instructor_available": True,
            "instructor_version": getattr(instructor, "__version__", None),
            "source": "instructor.providers.openai.utils.handle_json_modes",
            "source_sha256": hashlib.sha256(src.encode()).hexdigest(),
            "markers": _markers_from_source(src),
            "corroboration": "the recovered region must fall between the "
                             "markers read from this source",
        })
    except Exception as exc:  # noqa: BLE001
        prov["instructor_import_error"] = f"{type(exc).__name__}: {exc}"
        prov["corroboration"] = ("NONE — instructor not importable here, so "
                                 "the structural rule stands alone")
    return prov


# ── recovering the contract from the attempt itself ──────────────────
def _system_text(messages: Any) -> Optional[str]:
    """The system message's text, or None. Never guesses at another role."""
    if not isinstance(messages, list) or not messages:
        return None
    system = None
    for msg in messages:
        if isinstance(msg, dict) and msg.get("role") == "system":
            system = msg
            break
    if system is None:
        return None
    content = system.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [p.get("text") for p in content
                 if isinstance(p, dict) and isinstance(p.get("text"), str)]
        return "\n".join(parts) if parts else None
    return None


def _json_objects_in(text: str) -> List[Tuple[int, int, Dict[str, Any]]]:
    """Every balanced-brace region that parses as a JSON object.

    A scanner rather than a regex, because a JSON Schema nests braces and
    contains strings that may hold braces of their own.
    """
    found = []
    depth = 0
    start = -1
    in_str = False
    escape = False
    for i, ch in enumerate(text):
        if in_str:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            if depth:
                depth -= 1
                if depth == 0 and start >= 0:
                    chunk = text[start:i + 1]
                    try:
                        obj = json.loads(chunk)
                    except ValueError:
                        pass
                    else:
                        if isinstance(obj, dict):
                            found.append((start, i + 1, obj))
                    start = -1
    return found


def is_schema_shaped(obj: Any) -> bool:
    """A JSON Schema *describing* an object, rather than being one."""
    if not isinstance(obj, dict):
        return False
    keys = set(obj)
    if "properties" in keys and isinstance(obj.get("properties"), dict):
        return True
    if obj.get("type") == "object" and (keys & _SCHEMA_KEYS):
        return True
    return bool(keys & {"$schema", "$defs"})


def recover_contract(messages: Any,
                     provenance: Optional[Dict[str, Any]] = None
                     ) -> Tuple[Optional[Dict[str, Any]], str, Dict[str, Any]]:
    """(schema, why, detail) — the contract THIS attempt actually carried.

    Never reaches upward to the outer `response_model`: D222 §1 shows that
    object cannot reach this boundary, so anything found up there would be
    a different request's contract wearing this one's row.

    Ambiguity is refused, not resolved. Zero candidate regions and two
    candidate regions both yield no contract, because picking one would be
    a guess dressed as a measurement.
    """
    prov = provenance if provenance is not None else extraction_rule_provenance()
    detail: Dict[str, Any] = {"candidates": 0, "corroborated": None}

    text = _system_text(messages)
    if not text:
        return None, ("no system message with readable text on this attempt, "
                      "so the contract it was sent under cannot be read"), detail

    candidates = [(a, b, o) for a, b, o in _json_objects_in(text)
                  if is_schema_shaped(o)]
    detail["candidates"] = len(candidates)
    if not candidates:
        return None, ("the system message carries no schema-shaped JSON "
                      "object, so no contract was conveyed in it"), detail
    if len(candidates) > 1:
        return None, (f"{len(candidates)} schema-shaped regions in the system "
                      f"message — ambiguous, and choosing between them would "
                      f"be a guess"), detail

    start, end, schema = candidates[0]

    # Independent corroboration, only when the installed source is readable.
    markers = prov.get("markers") or []
    if markers:
        before, after = text[:start], text[end:]
        lead = [m for m in markers if m in before]
        tail = [m for m in markers if m in after]
        detail["corroborated"] = bool(lead and tail)
        detail["markers_before"] = lead
        detail["markers_after"] = tail
        if not (lead and tail):
            return None, ("a schema-shaped region was found, but it does not "
                          "sit between the markers read from the installed "
                          "instructor source — the region cannot be confirmed "
                          "as the generated contract"), detail

    return schema, (f"recovered from this attempt's system message "
                    f"(chars {start}-{end})"), detail


def required_fields_of(schema: Any) -> List[str]:
    """Required field names, read out of the schema THIS attempt carried."""
    if not isinstance(schema, dict):
        return []
    req = schema.get("required")
    if isinstance(req, list):
        return [str(r) for r in req]
    props = schema.get("properties")
    if isinstance(props, dict):
        return sorted(str(k) for k in props)
    return []


def _validate(instance: Any, schema: Dict[str, Any]) -> Tuple[str, str]:
    """Real JSON Schema validation.

    Returns one of four statuses, because they have DIFFERENT OWNERS:

      unavailable     — the library is not importable after all; the caller
                        must degrade to the narrower claim, never pass
      schema_unusable — the CONTRACT is not a valid JSON Schema. That is a
                        defect in what was sent, not in what came back, and
                        reporting it as an invalid instance would blame the
                        model for the request's problem
      invalid         — the response fails a usable contract
      valid           — the response satisfies it
    """
    try:
        import jsonschema
    except Exception as exc:  # noqa: BLE001
        return "unavailable", f"validator not importable after all: {exc}"
    validator = jsonschema.Draft202012Validator
    try:
        validator.check_schema(schema)
    except Exception as exc:  # noqa: BLE001
        return "schema_unusable", (f"the recovered contract is not a valid "
                                   f"JSON Schema: {type(exc).__name__}")
    try:
        errors = sorted(validator(schema).iter_errors(instance),
                        key=lambda e: list(e.path))
    except Exception as exc:  # noqa: BLE001
        return "schema_unusable", (f"the recovered contract could not be "
                                   f"applied: {type(exc).__name__}")
    if not errors:
        return "valid", ""
    first = errors[0]
    where = "/".join(str(p) for p in first.path) or "(root)"
    return "invalid", (f"{len(errors)} violation(s); first at {where}: "
                       f"{first.message}")


def classify(raw: Optional[str], schema: Any,
             validator: Optional[Dict[str, Any]] = None) -> Tuple[str, str]:
    """(verdict, why). `schema` is the contract RECOVERED FROM THIS ATTEMPT.

    Never returns a passing verdict on an unrecovered contract, and never
    promotes a required-field check to schema validation.
    """
    val = validator if validator is not None else validator_status()

    # 1 — was the contract recovered? Asked FIRST, so nothing downstream
    # can be trivially true.
    #
    # "A non-empty dict" is NOT enough. `{"type": "json_object"}` is a
    # non-empty dict and is the MODE DIRECTIVE — with it as the contract,
    # `required_fields_of` returns [] and "carries every required field"
    # becomes trivially true. That is D216's vacuous predicate returning
    # in new clothes; the calibration caught it here rather than in a run.
    if not isinstance(schema, dict) or not schema or not is_schema_shaped(schema):
        return CONTRACT_UNMEASURED, (
            "CONTRACT NOT RECOVERED for this attempt — no schema-shaped "
            "contract was established, so neither 'is it an instance' nor "
            "'is it the schema' can be asked. A mode directive such as "
            "{\"type\": \"json_object\"} is not a contract. This is not a pass")

    # 2 — is there a response, and is it a JSON object?
    if raw is None or not str(raw).strip():
        return NO_RESPONSE, "the model returned nothing to classify"
    try:
        obj = json.loads(raw)
    except ValueError as exc:
        return NOT_JSON, f"not JSON at all ({exc.__class__.__name__})"
    if not isinstance(obj, dict):
        return NOT_JSON_OBJECT, (f"JSON, but a {type(obj).__name__}, not an "
                                 f"object")

    required = required_fields_of(schema)

    # 3 — is it the schema definition itself?
    if is_schema_shaped(obj):
        same = canonical(obj) == canonical(schema)
        described = sorted(obj.get("properties", {})) if isinstance(
            obj.get("properties"), dict) else []
        return SCHEMA_ECHO, (
            f"this is the SCHEMA, not an instance of it — it describes "
            f"{described} instead of carrying {required}"
            + ("; canonically IDENTICAL to the contract this attempt was "
               "sent" if same else "; schema-shaped but not the contract "
                                   "this attempt was sent"))

    # 4 — does it VALIDATE? Only a real validator licenses VALID_INSTANCE.
    if val.get("available"):
        status, why = _validate(obj, schema)
        if status == "valid":
            return VALID_INSTANCE, (
                f"validates against the contract this attempt carried "
                f"({val.get('library')} {val.get('version')})")
        if status == "invalid":
            return INSTANCE_INVALID, f"fails the captured schema: {why}"
        if status == "schema_unusable":
            # The REQUEST is malformed, not the reply. Calling this an
            # invalid instance would hand the finding to the wrong owner.
            return CONTRACT_UNMEASURED, (
                f"{why} — so the response cannot be judged against it. This "
                f"is a defect in the contract sent, not in what came back, "
                f"and it is not a pass")
        # "unavailable" — the library vanished between the status check and
        # here. Fall through to the narrower claim rather than guess.

    # 4b — NO VALIDATOR. Say the narrower thing; never round up.
    #
    # And refuse outright when the contract names nothing to check: with no
    # required fields a presence test cannot fail, and a check that cannot
    # fail is not a check. Real validation (4) is immune to this, which is
    # why the guard lives only on the key-presence path.
    if not required:
        return CONTRACT_UNMEASURED, (
            "the recovered contract names no required fields and no "
            "properties, so a top-level presence check would be trivially "
            "true, and no validator was available to test anything else. "
            "This is not a pass")
    missing = [f for f in required if f not in obj]
    if missing:
        return REQUIRED_FIELDS_MISSING, (
            f"missing {', '.join(missing)}. NOTE: no JSON Schema validator "
            f"available, so only top-level required-key presence was tested")
    return REQUIRED_FIELDS_PRESENT, (
        f"carries every required top-level field ({', '.join(required)}). "
        f"THIS IS NOT SCHEMA VALIDATION — no validator was available, so "
        f"types, nested objects and every other constraint went untested. "
        f"It must not be read as {VALID_INSTANCE}")


def licenses_raw_response_claim(layer: Optional[str]) -> bool:
    """May evidence from `layer` support a claim about the MODEL's reply?"""
    return layer in LAYERS_LICENSING_RAW_RESPONSE_CLAIMS


# ── Q6: grouping attempts into logical calls ─────────────────────────
# The correlation id an attempt would need to be grouped with the other
# attempts of ITS structured-output call. Absent from run 17's rows.
LOGICAL_CALL_KEYS = ("logical_call_id", "invocation_id", "correlation_id")

FORBIDDEN_GROUPING_SIGNALS = (
    "adjacency", "elapsed time", "prompt hash", "schema hash",
    "response similarity",
)


# ── correlation lifecycle states ─────────────────────────────────────
CORRELATION_VALID = "CORRELATION_VALID"
CORRELATION_INCOMPLETE = "CORRELATION_INCOMPLETE"
CORRELATION_CONTRADICTORY = "CORRELATION_CONTRADICTORY"
CORRELATION_UNMEASURED = "CORRELATION_UNMEASURED"

# Only this one licenses grouping attempts into logical calls for Q6.
CORRELATION_LICENSING_GROUPING = frozenset({CORRELATION_VALID})


def validate_correlation(rows: List[Dict[str, Any]],
                         phase: str = "capture") -> Dict[str, Any]:
    """Is the correlation metadata EVIDENCE, or just a field that exists?

    An id proves nothing on its own: it is produced by the same instrument
    that reports it. What makes it evidence is an observed lifecycle —

        LOGICAL_CALL_ENTER -> id minted -> attempts under that id, indices
        rising -> LOGICAL_CALL_EXIT -> CONTEXT_RESET_CONFIRMED

    — so this refuses a group whenever the lifecycle is missing or
    self-contradictory, rather than grouping on the label.

    `context-reset-confirmed` must show the context RETURNED to the value
    live before the invocation, not merely that a reset ran. Otherwise
    nesting correctness is assumed, which is the whole thing at issue.
    """
    enters, exits, resets, attempts = {}, {}, {}, {}
    dup = {"enter": set(), "exit": set(), "reset": set()}
    ordered_boundaries: List[tuple] = []
    saw_any = False

    for i, r in enumerate(rows):
        if not isinstance(r, dict):
            continue
        cid = r.get("logical_call_id")
        ev = r.get("event")
        seq = r.get("seq", i)
        if cid is not None and ev in ("logical-call-enter", "logical-call-exit",
                                      "context-reset-confirmed", "llm-call"):
            saw_any = True
        if r.get("phase") != phase:
            continue                      # selftest rows never contaminate
        if ev == "logical-call-enter":
            (dup["enter"].add(cid) if cid in enters else None)
            enters.setdefault(cid, dict(r, seq=seq))
            ordered_boundaries.append((seq, "enter", cid,
                                       r.get("parent_logical_call_id")))
        elif ev == "logical-call-exit":
            (dup["exit"].add(cid) if cid in exits else None)
            exits.setdefault(cid, dict(r, seq=seq))
            ordered_boundaries.append((seq, "exit", cid, None))
        elif ev == "context-reset-confirmed":
            (dup["reset"].add(cid) if cid in resets else None)
            resets.setdefault(cid, dict(r, seq=seq))
        elif ev == "llm-call":
            attempts.setdefault(cid, []).append(dict(r, seq=seq))

    if not saw_any:
        # Same shape as every other return. A function whose result keys
        # depend on which branch it took makes its callers guess.
        return {"state": CORRELATION_UNMEASURED,
                "why": ["no row carries correlation metadata"],
                "calls": {}, "nesting_faults": [],
                "licenses_grouping": False}

    # nesting: enters/exits must form a LIFO stack, and each enter's
    # declared parent must be whatever was open at that moment.
    stack: List[str] = []
    nesting_faults: List[str] = []
    for _seq, kind, cid, parent in sorted(ordered_boundaries,
                                          key=lambda t: t[0]):
        if kind == "enter":
            expected_parent = stack[-1] if stack else None
            if parent != expected_parent:
                nesting_faults.append(
                    f"{cid}: declared parent {parent!r}, but {expected_parent!r} "
                    f"was open")
            stack.append(cid)
        else:
            if not stack:
                nesting_faults.append(f"{cid}: exit with nothing open")
            elif stack[-1] != cid:
                nesting_faults.append(
                    f"{cid}: exit out of order — {stack[-1]!r} was innermost")
                stack.remove(cid) if cid in stack else None
            else:
                stack.pop()
    if stack:
        nesting_faults.append(f"never exited: {', '.join(map(str, stack))}")

    calls: Dict[str, Dict[str, Any]] = {}
    for cid in sorted(set(enters) | set(exits) | set(resets) | set(attempts),
                      key=lambda c: (enters.get(c, {}).get("seq", 1 << 30))):
        missing, contradictions = [], []
        en, ex, rs = enters.get(cid), exits.get(cid), resets.get(cid)
        rows_ = sorted(attempts.get(cid, []), key=lambda r: r["seq"])

        if en is None:
            missing.append("no LOGICAL_CALL_ENTER")
        if ex is None:
            missing.append("no LOGICAL_CALL_EXIT")
        if rs is None:
            missing.append("no CONTEXT_RESET_CONFIRMED")

        for kind in ("enter", "exit", "reset"):
            if cid in dup[kind]:
                contradictions.append(
                    f"the id was reused — more than one {kind.upper()}")

        if en and rows_:
            early = [r for r in rows_ if r["seq"] < en["seq"]]
            if early:
                contradictions.append(
                    f"{len(early)} attempt(s) recorded BEFORE the boundary "
                    f"was entered")
        if ex and rows_:
            late = [r for r in rows_ if r["seq"] > ex["seq"]]
            if late:
                contradictions.append(
                    f"{len(late)} attempt(s) recorded AFTER the boundary exited")

        idx = [r.get("attempt_index") for r in rows_]
        if rows_:
            if any(i is None for i in idx):
                contradictions.append("an attempt carries no attempt_index")
            elif idx != list(range(1, len(idx) + 1)):
                if len(set(idx)) != len(idx):
                    contradictions.append(f"duplicate attempt_index: {idx}")
                elif 0 in idx:
                    contradictions.append(f"attempt_index is zero-based: {idx}")
                elif sorted(idx) != list(range(1, len(idx) + 1)):
                    contradictions.append(f"attempt_index not 1..N: {idx}")
                else:
                    contradictions.append(f"attempt_index non-monotonic: {idx}")
        if ex is not None:
            declared = ex.get("attempts_observed")
            if declared != len(rows_):
                contradictions.append(
                    f"the boundary declared {declared} attempt(s) but "
                    f"{len(rows_)} row(s) carry this id")
        if rs is not None and not rs.get("confirmed"):
            contradictions.append(
                f"context NOT restored: expected {rs.get('expected_after')!r}, "
                f"observed {rs.get('context_after')!r}")
        for fault in nesting_faults:
            if str(cid) in fault:
                contradictions.append(f"nesting: {fault}")

        if contradictions:
            state = CORRELATION_CONTRADICTORY
        elif missing:
            state = CORRELATION_INCOMPLETE
        else:
            state = CORRELATION_VALID
        calls[str(cid)] = {"state": state, "attempts": len(rows_),
                           "missing": missing, "contradictions": contradictions}

    states = {c["state"] for c in calls.values()}
    if CORRELATION_CONTRADICTORY in states:
        overall = CORRELATION_CONTRADICTORY
    elif CORRELATION_INCOMPLETE in states:
        overall = CORRELATION_INCOMPLETE
    elif states == {CORRELATION_VALID}:
        overall = CORRELATION_VALID
    else:
        overall = CORRELATION_UNMEASURED
    return {"state": overall, "calls": calls,
            "nesting_faults": nesting_faults,
            "licenses_grouping": overall in CORRELATION_LICENSING_GROUPING}


def logical_call_grouping(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Can these attempts be grouped into their logical calls? Usually no.

    `max_retries=2` (cognee `ollama/adapter.py:130`) bounds a logical call
    at two raw attempts, so five raw rows span at least three separate
    calls. Grouping them by anything else — order, timing, or hash
    similarity — would manufacture a denominator, so this refuses and
    names what would be needed instead.
    """
    key = next((k for k in LOGICAL_CALL_KEYS
                if all(isinstance(r, dict) and r.get(k) is not None
                       for r in rows) and rows), None)
    if key:
        groups: Dict[str, int] = {}
        for r in rows:
            groups[str(r[key])] = groups.get(str(r[key]), 0) + 1
        return {"available": True, "key": key, "groups": groups}
    return {
        "available": False,
        "why": "no attempt carries a logical-call identifier",
        "looked_for": list(LOGICAL_CALL_KEYS),
        "refused_signals": list(FORBIDDEN_GROUPING_SIGNALS),
        "next_measurement_requirement":
            "an explicit correlation id minted at the outer structured-output "
            "invocation and carried into every underlying retry attempt",
    }
