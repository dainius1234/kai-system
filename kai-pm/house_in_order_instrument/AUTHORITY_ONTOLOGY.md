# Document Control Model — SCRATCHPAD SPEC v2 (multi-axis)
Subject: QUALIFICATION_SUBJECT 9d15bcd / tree 627104d6
Supersedes the single-role ontology of D329, which Kai disproved.

## Why the single role failed
One label was made to carry lifecycle, function, authority, generation,
validity and scope at once. Three oracle errors followed directly:
UH2_SENSOR_INGRESS_PLAN.md was called SUPERSEDED from the word
"supersedes" (it is rev 2 and supersedes rev 1); CHANGELOG.md and
SESSION_BACKLOG.md were called whole-document DERIVED from a
DIRECTIONALITY fact that said nothing about scope.

## Axis 1 — LIFECYCLE   ACTIVE | HISTORICAL | SUPERSEDED | UNKNOWN
SUPERSEDED requires a NAMED SUCCESSOR and requires the document to be
the PREDECESSOR. "X supersedes Y" makes Y superseded, not X.
Direction must be established, never inferred from the word.

## Axis 2 — FUNCTION
GOVERNANCE | STATUS | PLAN | EVIDENCE | REFERENCE | RUNTIME_INPUT
| TEMPLATE | MARKER | USER_GUIDE | OTHER | UNKNOWN
`.md` does NOT imply documentation. Function must be shown, not assumed
from extension or size.

## Axis 3 — AUTHORITY
AUTHORITATIVE | VERIFIED_DERIVED | ADVISORY | NON_AUTHORITY | UNKNOWN
Nothing may enter AUTHORITATIVE or VERIFIED_DERIVED before R4 claim
qualification. Default before R4 is UNKNOWN.

## Axis 4 — GENERATION   MANUAL | PARTIAL_DERIVED | FULL_DERIVED | UNKNOWN
Requires a PROVEN_WRITER with an EXACT resolved target AND a scope.
FULL_DERIVED requires the writer to own WHOLE_FILE. A writer owning a
region yields PARTIAL_DERIVED — never FULL.

## Axis 5 — VALIDITY_BINDING
CURRENT_TREE | EXACT_SNAPSHOT | RUN_ARTEFACT | TIME_BOUND | UNKNOWN

## Axis 6 — SCOPE  (stable selectors; NO byte or line offsets)
WHOLE_FILE | HEADING("## Project Status") | TABLE("Current State")
| MANAGED_REGION("<!-- sync:begin X -->")
File-level default + region overrides. Only mixed documents need
overrides; we do not explode 268 files into regions.

## UNKNOWN
First-class on EVERY axis, independently. A document may be
ACTIVE + UNKNOWN function + UNKNOWN authority. Abstention on one axis
never forces abstention on another.
