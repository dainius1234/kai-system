#!/usr/bin/env python3
"""Identity for an image named directly, not resolved through Compose.

WHY A SECOND MODULE INSTEAD OF A SECOND MODE
============================================

`collect_image_identity.py` resolves a service to an image by asking
Compose, which is right for the stack and wrong for Item 8, whose
subjects are experimental images built from derived Dockerfiles under
unique branch-specific tags. They have no Compose service.

The obvious move was to add an explicit-reference mode to that module.
Frozen design R2 forbids it, and the reasoning is better than the
convenience:

> *"This is better than modifying the old collector and then proving we
> supposedly didn't disturb it. We can simply say: old collector source
> bytes unchanged."*

That module is already wired into `stage1-replay.yml` — the frozen
Stage-2 execution path — at two points. **An unchanged instrument needs
no argument that it is unchanged.** So Item 8 gets this, and that file is
not touched.

HOW CONTRACT COMPATIBILITY IS GUARANTEED RATHER THAN ASSERTED
=============================================================

R2 requires this module to emit *the same identity JSONL contract*, so
that the existing collector's `--verify-executed --against` can read a
record written here without knowing which module wrote it.

The weak way to do that is to copy the field names across and hope they
stay in step — which is D272's failure exactly: two records of the same
thing, drifting, with nobody able to see it.

So the shared primitives are **imported** from
`collect_image_identity`: the inspect call, the ABSENT/NULL/VALUE
classification, the platform derivation, the state constants. Importing
reads that module; it does not modify it. If its contract ever moves,
this module moves with it by construction instead of by anyone
remembering to.

WHAT IT REFUSES
===============

The same boundaries as its sibling, because they were earned the same
way (R11): a failed inspect, an inspect exiting 0 with no payload, and a
payload carrying no `Id` all record **UNRECORDED** with the unmet
prerequisite named — never an empty string in an identity field. An
empty identity field reads as an answer; a missing one invites the
question.

One boundary is specific to this module: **it does not check that the
reference exists before inspecting it.** There is nothing to check
against — an explicit reference is the caller's claim, and `inspect`
failing IS the measurement of whether it was true.

Exit 0 = the image was found and its identity recorded.
Exit 3 = UNRECORDED. The row is still written, so the gap is visible in
         the artifact rather than inferred from a missing file.
Exit 1 = refused before collecting.
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

# Reading the sibling module. NOT modifying it -- R2 requires its bytes
# to stay exactly as D280 shipped them, and an import cannot change them.
from collect_image_identity import (  # noqa: E402
    ABSENT, IDENTITY_TYPE, RECORDED, UNRECORDED,
    git, inspect, platform_of, repo_digest,
)

REPO = pathlib.Path(__file__).resolve().parent.parent.parent


def collect(docker: str, image_ref: str, label: str,
            run_id: str | None) -> dict:
    """One record for one explicitly named image."""
    base = {
        "identity_type": IDENTITY_TYPE,
        "service": label,
        "image_ref": image_ref,
        "commit_sha": git("rev-parse", "HEAD"),
        "tree_sha": git("rev-parse", "HEAD^{tree}"),
        "run_id": run_id,
    }
    data, err = inspect(docker, image_ref)
    if err:
        return {**base, "identity_state": UNRECORDED,
                "docker_image_id": None, "repo_digest_state": ABSENT,
                "repo_digest": None, "platform": None,
                "unmet_prerequisite": err}
    image_id = data.get("Id")
    if not image_id:
        return {**base, "identity_state": UNRECORDED,
                "docker_image_id": None, "repo_digest_state": ABSENT,
                "repo_digest": None, "platform": None,
                "unmet_prerequisite":
                    f"`image inspect {image_ref}` returned an object with no "
                    f"Id. There is no identity to record"}
    state, digest = repo_digest(data)
    return {**base, "identity_state": RECORDED, "docker_image_id": image_id,
            "repo_digest_state": state, "repo_digest": digest,
            "platform": platform_of(data)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--image-ref", required=True,
                    help="the exact tag or id to inspect")
    ap.add_argument("--label", required=True,
                    help="what this image IS, carried in the record's "
                         "`service` field so the sibling module's "
                         "--verify-executed --service can select it")
    ap.add_argument("--out", help="write the JSONL record here")
    ap.add_argument("--run-id", default=os.environ.get("GITHUB_RUN_ID"))
    ap.add_argument("--docker", default="docker",
                    help="the docker binary; overridden by calibration so "
                         "the shipped code paths run without a daemon")
    args = ap.parse_args()

    if not args.image_ref.strip():
        print("REFUSED: --image-ref is empty. Inspecting nothing and "
              "recording the result would be an identity for no image.")
        return 1

    row = collect(args.docker, args.image_ref, args.label, args.run_id)

    print("EXPLICIT IMAGE IDENTITY")
    print("=" * 68)
    print(f"  label   : {row['service']}")
    print(f"  ref     : {row['image_ref']}")
    print(f"  state   : {row['identity_state']}")
    if row["identity_state"] == RECORDED:
        print(f"  image id: {row['docker_image_id']}   ({IDENTITY_TYPE})")
        print(f"  repo dig: {row['repo_digest_state']}"
              + (f" {row['repo_digest']}" if row["repo_digest"] else ""))
        print(f"  platform: {row['platform']}")
    else:
        print(f"  WHY     : {row['unmet_prerequisite']}")
    print(f"  tree    : {row['tree_sha']}   run: {row['run_id']}")
    print()
    print(f"  inspected: 1 explicit image reference, "
          f"{1 if row['identity_state'] == RECORDED else 0} recorded, "
          f"{0 if row['identity_state'] == RECORDED else 1} UNRECORDED")

    if args.out:
        p = pathlib.Path(args.out)
        p.write_text(json.dumps(row) + "\n")
        print(f"  written: {p} ({p.stat().st_size} bytes, 1 row)")

    print()
    print("  A local image ID proves WHICH IMAGE this reference named at")
    print("  inspection time. Binding it to what a container actually RAN")
    print("  is a separate measurement, and it is the sibling module's")
    print("  --verify-executed that makes it. (D280)")

    if row["identity_state"] != RECORDED:
        print()
        print("FAIL: no identity recorded for this reference. Any claim "
              "bound to it is unbound.")
        return 3
    print()
    print("PASS: the reference resolves to a recorded image identity.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
