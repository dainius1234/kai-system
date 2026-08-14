"""KAI-GATE-048 C: what KIND of object did the model return, and at what LAYER?

Pure stdlib, no I/O — calibratable on the host with no cognee, no
container and no model.

THE DISTINCTION THIS EXISTS TO MAKE
===================================

    a JSON **instance** conforming to a schema
    vs
    the **schema definition itself**

WHY THIS FILE WAS REWRITTEN (D216)
==================================

The previous version took a `required_fields` list and asked one fused
question. In run 15 that list arrived **empty** — the schema had been
located inside message prose, so `json.loads` failed and the list
collapsed to `[]`. With no required fields, "carries every required
field" is **trivially true**, and every response classified as
`VALID INSTANCE` — including, had one appeared, a schema echo.

That is a vacuous predicate: a check that cannot fail is not a check.
The repair is not a special case for an empty list. The questions are now
asked **independently and in an order where vacuity is impossible**:

    0. is there a response at all?                  -> NO RESPONSE
    1. is it JSON, and an object?                   -> OTHER INVALID
    2. CAN THE SCHEMA'S REQUIREMENTS BE ESTABLISHED?
                                                    -> CLASSIFIER_UNMEASURED
    3. is it the schema definition itself?          -> SCHEMA ECHO
    4. is it a valid instance of THAT schema?       -> VALID INSTANCE
    5. anything else                                -> OTHER INVALID

Step 2 comes **before** any instance test. An unestablished schema can
therefore never produce a success verdict — it produces an explicit
"not measured", which is what run 15 should have said.

LAYERS
======

D215's requirement: every captured record declares the layer it came
from, and a claim may only be made by evidence from a layer that
licenses it. `INSTRUCTOR_RETURN = VALID INSTANCE` must never satisfy a
question about `RAW_MODEL_RESPONSE`, because instructor's retry and
validation machinery sits between them.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, List, Optional, Tuple

# ── verdicts ─────────────────────────────────────────────────────────
VALID_INSTANCE = "VALID INSTANCE"
SCHEMA_ECHO = "SCHEMA ECHO"
OTHER_INVALID = "OTHER INVALID STRUCTURE"
NO_RESPONSE = "NO RESPONSE"
CLASSIFIER_UNMEASURED = "CLASSIFIER_UNMEASURED"

# ── layers (D215) ────────────────────────────────────────────────────
ADAPTER_INPUT = "ADAPTER_INPUT"
INSTRUCTOR_RETURN = "INSTRUCTOR_RETURN"
RAW_MODEL_REQUEST = "RAW_MODEL_REQUEST"
RAW_MODEL_RESPONSE = "RAW_MODEL_RESPONSE"
VALIDATION_RESULT = "VALIDATION_RESULT"

LAYERS = frozenset({ADAPTER_INPUT, INSTRUCTOR_RETURN, RAW_MODEL_REQUEST,
                    RAW_MODEL_RESPONSE, VALIDATION_RESULT})

# Only this layer licenses a claim about what the MODEL returned.
# Everything above it has passed through instructor's retry, parsing and
# validation, any of which can turn a malformed completion into a clean
# object — which is exactly how run 15 misreported.
LAYERS_LICENSING_RAW_RESPONSE_CLAIMS = frozenset({RAW_MODEL_RESPONSE})

# Keys that only ever appear in a JSON Schema, never in an instance of
# the models cognee asks for. From the JSON Schema vocabulary, not from
# the one response we happened to see.
_SCHEMA_KEYS = frozenset({
    "$schema", "$defs", "$ref", "properties", "required",
    "additionalProperties", "patternProperties", "definitions",
})


def sha256(text: Optional[str]) -> str:
    """Byte-level identity, so 'identical' is measured, not eyeballed."""
    if text is None:
        return ""
    return hashlib.sha256(text.encode("utf-8", "replace")).hexdigest()


def canonical(obj: Any) -> str:
    """Order-independent serialisation, for CANONICAL (not byte) identity."""
    try:
        return json.dumps(obj, sort_keys=True, separators=(",", ":"),
                          default=str)
    except (TypeError, ValueError):
        return str(obj)


def as_schema(schema: Any) -> Optional[Dict[str, Any]]:
    """The schema as a dict, or None when it cannot be established.

    None is a first-class answer. Run 15's failure was treating an
    unusable schema as an empty one and carrying on.
    """
    if isinstance(schema, dict):
        return schema
    if isinstance(schema, str) and schema.strip():
        try:
            parsed = json.loads(schema)
        except ValueError:
            return None
        return parsed if isinstance(parsed, dict) else None
    return None


def required_fields_of(schema: Any) -> List[str]:
    """Required field names, read out of the schema that was SENT (R5).

    Returns [] both for 'no requirements' and for 'unusable' — which is
    why callers must use `as_schema` to tell those apart rather than
    inferring from an empty list. That conflation was the D216 defect.
    """
    obj = as_schema(schema)
    if obj is None:
        return []
    req = obj.get("required")
    if isinstance(req, list):
        return [str(r) for r in req]
    props = obj.get("properties")
    if isinstance(props, dict):
        return sorted(str(k) for k in props)
    return []


def _is_schema_shaped(obj: Dict[str, Any]) -> bool:
    """A JSON Schema *describing* an object, rather than being one."""
    keys = set(obj)
    if "properties" in keys and isinstance(obj.get("properties"), dict):
        return True
    if obj.get("type") == "object" and (keys & _SCHEMA_KEYS):
        return True
    return bool(keys & {"$schema", "$defs"})


def classify(raw: Optional[str], schema: Any) -> Tuple[str, str]:
    """(verdict, why). `schema` is the schema ACTUALLY SENT for this attempt.

    Never returns a success verdict when the schema's requirements could
    not be established — that is `CLASSIFIER_UNMEASURED`.
    """
    # 0 — is there a response at all?
    if raw is None or not str(raw).strip():
        return NO_RESPONSE, "the model returned nothing to classify"

    # 1 — is it JSON, and an object?
    try:
        obj = json.loads(raw)
    except ValueError as exc:
        return OTHER_INVALID, f"not JSON at all ({exc.__class__.__name__})"
    if not isinstance(obj, dict):
        return OTHER_INVALID, (f"JSON, but a {type(obj).__name__}, not an "
                               f"object")

    # 2 — CAN THE CONTRACT BE ESTABLISHED? Asked BEFORE any instance
    # test, so an unknown contract can never yield a success verdict.
    schema_obj = as_schema(schema)
    if schema_obj is None:
        return CLASSIFIER_UNMEASURED, (
            "SCHEMA REQUIREMENTS NOT ESTABLISHED — the schema sent for this "
            "attempt could not be read, so neither 'is it an instance' nor "
            "'is it the schema' can be answered. This is not a pass")
    required = required_fields_of(schema_obj)
    if not required:
        return CLASSIFIER_UNMEASURED, (
            "SCHEMA REQUIREMENTS NOT ESTABLISHED — the schema names no "
            "required fields and no properties, so 'valid instance' would "
            "be trivially true. This is not a pass")

    # 3 — is it the schema definition itself? Compared against the schema
    # ACTUALLY SENT, not against a generic shape alone.
    if _is_schema_shaped(obj):
        same = canonical(obj) == canonical(schema_obj)
        described = sorted(obj.get("properties", {}))
        return SCHEMA_ECHO, (
            f"this is the SCHEMA, not an instance of it — it describes "
            f"{described} instead of carrying {required}"
            + ("; BYTE-FOR-BYTE the schema that was sent, canonicalised"
               if same else "; schema-shaped but not identical to the one sent"))

    # 4 — is it a valid instance of THAT schema?
    missing = [f for f in required if f not in obj]
    if not missing:
        wrong = [f for f in required
                 if not isinstance(obj.get(f), (str, int, float, bool, list))]
        if not wrong:
            return VALID_INSTANCE, (f"carries every required field "
                                    f"({', '.join(required)})")
        return OTHER_INVALID, (f"has the required field(s) but with unusable "
                               f"value type(s): {', '.join(wrong)}")

    # 5 — anything else
    return OTHER_INVALID, (f"an object, but neither a valid instance nor a "
                           f"schema: missing {', '.join(missing)}")


def licenses_raw_response_claim(layer: Optional[str]) -> bool:
    """May evidence from `layer` support a claim about the MODEL's reply?

    D215. `INSTRUCTOR_RETURN = VALID INSTANCE` says nothing about the raw
    completion: instructor retried, parsed and validated in between.
    """
    return layer in LAYERS_LICENSING_RAW_RESPONSE_CLAIMS
