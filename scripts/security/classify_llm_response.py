"""KAI-GATE-048 C: what KIND of object did the model return?

Pure stdlib, no I/O — so it is calibratable on the host with no cognee,
no container and no model.

THE DISTINCTION THIS EXISTS TO MAKE
===================================

    a JSON **instance** conforming to a schema
    vs
    the **schema definition itself**

If this module cannot tell those apart it is not fit for the question,
because the observed failure is precisely the second wearing the shape of
the first: valid JSON, an object, parseable — and the wrong KIND of
thing.

    {"summary": "Ada Lovelace wrote ..."}                    <- INSTANCE
    {"title": "SummarizedContent", "type": "object",
     "properties": {"summary": {"type": "string"}},
     "required": ["summary"]}                                <- SCHEMA ECHO

ORDER IS LOAD-BEARING. Instance-validity is tested FIRST. A schema
definition happens to be a JSON object with string-valued keys, so a lax
instance check applied second would swallow it.

VALIDATOR FAILURES ARE NOT COLLAPSED
====================================

The operator's rule: do not collapse every validator failure into one
422. `OTHER INVALID STRUCTURE` is a distinct verdict from `SCHEMA ECHO`,
and `NO RESPONSE` from both — because those three have three different
owners (model compliance, prompt/mode construction, and the transport).
"""
from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, List, Optional, Tuple

VALID_INSTANCE = "VALID INSTANCE"
SCHEMA_ECHO = "SCHEMA ECHO"
OTHER_INVALID = "OTHER INVALID STRUCTURE"
NO_RESPONSE = "NO RESPONSE"
UNKNOWN = "UNKNOWN"

# Keys that only ever appear in a JSON Schema, never in an instance of
# the models cognee asks for. Derived from the JSON Schema vocabulary,
# not from the one response we happened to see.
_SCHEMA_KEYS = frozenset({
    "$schema", "$defs", "$ref", "properties", "required",
    "additionalProperties", "patternProperties", "definitions",
})


def sha256(text: Optional[str]) -> str:
    """Byte-level identity, so 'identical' is measured, not eyeballed."""
    if text is None:
        return ""
    return hashlib.sha256(text.encode("utf-8", "replace")).hexdigest()


def _looks_like_schema(obj: Dict[str, Any]) -> bool:
    """A JSON Schema *describing* an object, rather than being one."""
    keys = set(obj)
    if "properties" in keys and isinstance(obj.get("properties"), dict):
        return True
    if obj.get("type") == "object" and (keys & _SCHEMA_KEYS):
        return True
    # `$schema`/`$defs` alone are conclusive.
    return bool(keys & {"$schema", "$defs"})


def classify(raw: Optional[str],
             required_fields: List[str]) -> Tuple[str, str]:
    """(verdict, why). `required_fields` comes from the response model.

    Derived from the model under test, never a list maintained beside it
    (R5) — the caller reads it out of the schema actually sent.
    """
    if raw is None or not raw.strip():
        return NO_RESPONSE, "the model returned nothing to classify"

    try:
        obj = json.loads(raw)
    except ValueError as exc:
        return OTHER_INVALID, f"not JSON at all ({exc.__class__.__name__})"

    if not isinstance(obj, dict):
        return OTHER_INVALID, (f"JSON, but a {type(obj).__name__}, not an "
                               f"object")

    # INSTANCE FIRST. A schema is also a dict of strings, so testing
    # instance-shape second would let a schema satisfy a lax check.
    missing = [f for f in required_fields if f not in obj]
    if not missing:
        wrong = [f for f in required_fields
                 if not isinstance(obj.get(f), (str, int, float, bool, list))]
        if not wrong:
            return VALID_INSTANCE, (f"carries every required field "
                                    f"({', '.join(required_fields)})")
        return OTHER_INVALID, (f"has the required field(s) but with "
                               f"unusable value type(s): {', '.join(wrong)}")

    if _looks_like_schema(obj):
        return SCHEMA_ECHO, (
            f"this is the SCHEMA, not an instance of it — it describes "
            f"{sorted(obj.get('properties', {}))} instead of carrying "
            f"{required_fields}; missing {', '.join(missing)}")

    return OTHER_INVALID, (f"an object, but neither a valid instance nor a "
                           f"schema: missing {', '.join(missing)}")


def required_fields_of(schema: Any) -> List[str]:
    """The required field names, read out of the schema that was SENT.

    R5: the expected answer is derived from the payload under test rather
    than from a tuple kept in this file, so a schema change cannot leave
    the classifier quietly measuring the wrong thing.
    """
    if isinstance(schema, str):
        try:
            schema = json.loads(schema)
        except ValueError:
            return []
    if not isinstance(schema, dict):
        return []
    req = schema.get("required")
    if isinstance(req, list):
        return [str(r) for r in req]
    props = schema.get("properties")
    if isinstance(props, dict):
        return sorted(str(k) for k in props)
    return []
