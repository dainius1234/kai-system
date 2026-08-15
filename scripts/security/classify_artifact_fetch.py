#!/usr/bin/env python3
"""Why did the artifact not arrive? Five answers, not one.

The operator's correction, which this file encodes:

> Fetch failure must not be collapsed into "not performed". "Run still in
> progress / no artifact yet" is different from GitHub permission
> failure, network failure, missing artifact after completion, or
> malformed artifact. Those states need to remain distinguishable.

They need to stay distinguishable because they demand different actions
and carry different meanings about the evidence:

  `SUBJECT_RUN_INCOMPLETE`   nothing is wrong. Wait.
  `ACCESS_DENIED`            a permissions problem here, not there.
  `NETWORK_FAILURE`          transport. Retry means something.
  `ARTIFACT_ABSENT`          the run finished and produced NOTHING. That
                             is a fact about the subject run and may be a
                             real finding.
  `ARTIFACT_EXPIRED`         it existed and is gone. The evidence is lost,
                             which is not the same as never produced.
  `ARTIFACT_MALFORMED`       it downloaded and does not contain what its
                             name promises.
  `ARTIFACT_PRESENT`         the only state that licenses measuring.

Collapsing these into "NOT PERFORMED" is the same defect as one abort
message for three selftest states (D218): the report names a condition
that may not have occurred, and a later reader cannot tell which did.

The classification is a **pure function** of facts the caller gathered,
so it can be calibrated without a network. Gathering those facts is the
caller's job; deciding what they mean is this file's.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

SUBJECT_RUN_INCOMPLETE = "SUBJECT_RUN_INCOMPLETE"
ACCESS_DENIED = "ACCESS_DENIED"
NETWORK_FAILURE = "NETWORK_FAILURE"
ARTIFACT_ABSENT = "ARTIFACT_ABSENT"
ARTIFACT_EXPIRED = "ARTIFACT_EXPIRED"
ARTIFACT_MALFORMED = "ARTIFACT_MALFORMED"
ARTIFACT_PRESENT = "ARTIFACT_PRESENT"

EXIT = {
    ARTIFACT_PRESENT: 0,
    SUBJECT_RUN_INCOMPLETE: 3,
    ACCESS_DENIED: 4,
    NETWORK_FAILURE: 5,
    ARTIFACT_ABSENT: 6,
    ARTIFACT_EXPIRED: 7,
    ARTIFACT_MALFORMED: 8,
}

MEANING = {
    ARTIFACT_PRESENT:
        "the artifact is here and holds the expected file. This is the "
        "ONLY state that licenses a measurement.",
    SUBJECT_RUN_INCOMPLETE:
        "the subject run has not finished, so there is nothing to fetch "
        "YET. Nothing is wrong; waiting is the whole remedy.",
    ACCESS_DENIED:
        "the API refused us. That is a fact about THIS job's permissions, "
        "not about the subject run, and it must never be reported as the "
        "subject having produced nothing.",
    NETWORK_FAILURE:
        "the API could not be reached at all. Nothing is known about the "
        "subject run either way.",
    ARTIFACT_ABSENT:
        "the subject run COMPLETED and no artifact of that name exists. "
        "This is a fact about the subject run and may be a real finding "
        "about it, not merely an availability problem here.",
    ARTIFACT_EXPIRED:
        "the artifact existed and has expired. The evidence is LOST, "
        "which is not the same as never produced — the subject run did "
        "its job and the retention window did not.",
    ARTIFACT_MALFORMED:
        "the artifact downloaded and does not contain the file its name "
        "promises. Something produced it wrongly; this is not absence.",
}


def classify_fetch(*, http_status: int | None, run_status: str | None,
                   artifact_names: list[str] | None, artifact_name: str,
                   artifact_expired: bool | None,
                   expected_file_present: bool) -> tuple[str, str]:
    """What happened, from facts the caller gathered. Order matters.

    Transport first, then authorisation, then the subject's own state,
    then the artifact's — because a later question cannot be answered
    when an earlier one failed, and answering it anyway is R11's defect:
    a dependent measurement taken after its prerequisite is unproven.
    """
    if expected_file_present:
        return ARTIFACT_PRESENT, MEANING[ARTIFACT_PRESENT]
    if http_status is None:
        return NETWORK_FAILURE, MEANING[NETWORK_FAILURE]
    if http_status in (401, 403):
        return ACCESS_DENIED, MEANING[ACCESS_DENIED]
    if http_status >= 500 or http_status == 0:
        return NETWORK_FAILURE, MEANING[NETWORK_FAILURE]
    if run_status is not None and run_status != "completed":
        return SUBJECT_RUN_INCOMPLETE, MEANING[SUBJECT_RUN_INCOMPLETE]
    if artifact_names is None:
        # We could not even list them, and the earlier branches did not
        # explain why. Absence of a list is not a list of absences.
        return NETWORK_FAILURE, MEANING[NETWORK_FAILURE]
    if artifact_name not in artifact_names:
        return ARTIFACT_ABSENT, MEANING[ARTIFACT_ABSENT]
    if artifact_expired:
        return ARTIFACT_EXPIRED, MEANING[ARTIFACT_EXPIRED]
    return ARTIFACT_MALFORMED, MEANING[ARTIFACT_MALFORMED]


def _load(path: str | None, missing: list[str]):
    """Read a JSON fact, RECORDING why it is absent when it is.

    An early `return None` on a missing file is the shape that lets
    absence read as an ordinary value. Every way this can fail to produce
    a fact is appended to `missing` and printed, so an unreadable input
    is visible in the output rather than inferred from a downstream
    state (I-1).
    """
    if not path:
        missing.append("run/artifact JSON not supplied")
        return None
    p = pathlib.Path(path)
    if not p.exists():
        missing.append(f"{path}: absent")
        return None
    text = p.read_text(errors="replace")
    if not text.strip():
        missing.append(f"{path}: present but empty")
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        missing.append(f"{path}: unparseable ({exc.msg}); {len(text)} bytes")
        return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--http-status", type=int, default=None,
                    help="HTTP status of the run query; omit if the request "
                         "could not be made at all")
    ap.add_argument("--run-json", help="the subject run's JSON, if fetched")
    ap.add_argument("--artifacts-json", help="the run's artifact list JSON")
    ap.add_argument("--artifact-name", required=True)
    ap.add_argument("--expected-file", required=True,
                    help="the file the artifact must contain")
    args = ap.parse_args()

    missing: list[str] = []
    run = _load(args.run_json, missing)
    arts = _load(args.artifacts_json, missing)
    names = None
    expired = None
    if isinstance(arts, dict) and isinstance(arts.get("artifacts"), list):
        names = [a.get("name") for a in arts["artifacts"]]
        for a in arts["artifacts"]:
            if a.get("name") == args.artifact_name:
                expired = bool(a.get("expired"))

    state, meaning = classify_fetch(
        http_status=args.http_status,
        run_status=(run or {}).get("status") if isinstance(run, dict) else None,
        artifact_names=names,
        artifact_name=args.artifact_name,
        artifact_expired=expired,
        expected_file_present=pathlib.Path(args.expected_file).is_file())

    for note in missing:
        print(f"  fact not available: {note}")
    print(f"  inspected: 1 artifact fetch across {len(EXIT)} "
          f"distinguishable state(s)")
    print(f"FETCH STATE: {state}")
    print(f"  {meaning}")
    if state != ARTIFACT_PRESENT:
        print("  No measurement follows. This is the fetch's state, NOT a "
              "verdict about what the artifact would have shown.")
    return EXIT[state]


if __name__ == "__main__":
    sys.exit(main())
