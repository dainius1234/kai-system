#!/usr/bin/env python3
"""Bind a run to the image that executed it — in TWO steps, not one.

TWO CLAIMS, AND THE ONE THAT WAS OVERSTATED
===========================================

This module makes two separate, differently-sized claims, and the
distinction between them is the whole point:

  --collect  records the image Compose resolves for a service, read
             immediately after the build. That establishes **the image
             referenced immediately before execution.**

  --verify-executed  reads the `.Image` of the container that actually
             ran, and requires it to equal the collected id. That, and
             only that, establishes **the image that executed.**

**The first version of this file claimed the second while measuring the
first.** Its opening line asked *"Which image actually executed?"* while
running `docker image inspect` after the build and before any container
existed. Nothing was wrong with the mechanism; the sentence around it
was one step stronger than the measurement, which is doctrine rule 2 —
*present ≠ executed ≠ enforced* — inside the instrument written to bind
execution. A tag can be repointed between the build and the run, and a
pre-run lookup cannot see that happen.

Caught in review before any evidence depended on it (D280). The repair
is not a stronger adjective: it is a second measurement, taken from the
container, that can come back MISMATCH.

THE FAILURE THAT EARNED THIS
============================

D247 §6 item 10 — the tenth of the ten conditions KAI-GATE-048 C must
meet — requires *"every claim bound to the exact tested tree, image and
run id."*

The tree is bound. The run id is bound. **The image never was.** The
D277 closure review looked for it and found that no instrument in this
repository records an image identity: `stage1-replay.yml` builds
`memu-graph` and captures nothing about the result, and no image
identity appears anywhere in D267-D276. The programme's own bar demanded
an identity the programme did not collect, and nobody noticed for the
whole of Stage 1 — because a build that succeeds looks like a build that
was recorded.

That is rule 3's gap seen from the collection side: evidence identity is
immutable only if it was captured in the first place.

WHAT IT IS NAMED, AND WHY THAT MATTERS MORE THAN IT SOUNDS
==========================================================

These images are **built in the job and never pushed**. That has a
consequence most readers will get wrong unless the record says it out
loud:

* the **image ID** (`docker image inspect --format '{{.Id}}'`) is a
  sha256 over the image's config blob. It identifies *the image that
  executed on that runner*. It is a real identity and it is what we can
  have.
* a **repository digest** (`RepoDigests`) is a registry manifest digest.
  It is populated by a push or a pull, and it is what makes an image
  *retrievable by anyone else*. A never-pushed image has none.

So the field is `docker_image_id` and the record says
`identity_type: DOCKER_LOCAL_IMAGE_ID`. It is deliberately **not**
called `image_digest`, because someone reading `image_digest` a year
from now will take it for an OCI/registry manifest digest and believe
the image can be pulled. It cannot.

**The bound, stated in every record rather than in a comment nobody
reads:** a local image ID proves which image ran. It does **not** make
that image independently retrievable. That is sufficient for 048's
evidence identity and it is not sufficient for release provenance;
anything stronger (a published registry digest, build attestation) is a
different job and is not bolted on here.

`repo_digest_state` is kept as ABSENT / NULL / VALUE rather than as an
empty string, because doctrine rule 20 exists: *"ABSENT / NULL / VALUE
remain distinct wherever invocation identity depends on them."* "This
image was never pushed" and "this image was pushed and reported nothing"
are different facts about provenance and must not collapse.

HOW THE POPULATION IS DERIVED
=============================

R5: a check's scope comes from the data it traverses, not from a list
kept beside it. The caller names **services**; the image reference for
each is resolved by asking Compose — `docker compose config --images`,
which is client-side and needs no daemon — rather than by reconstructing
Compose's `<project>-<service>` naming convention here. Every service in
this repository's graph path is build-only with no `image:` key, so a
hand-written reference would be a guess about somebody else's naming
rule, and would keep working right up until it silently did not.

WHY IT FAILS CLOSED, AND WHAT IT REFUSES TO WRITE
=================================================

R11: no subject, no observation. If `inspect` fails, returns nothing, or
returns a payload with no `Id`, this records **UNRECORDED** with the
unmet prerequisite named — never an empty string in an identity field.

An empty identity field is worse than a missing one. A missing field
invites the question "where is it?"; an empty one reads as an answer,
and a table of blank identities looks like a table that was filled in.
That is the same shape as the collector which ran fifty probes against a
stack that did not exist and produced fifty correct-looking rows.

Exit 0 = every requested service has an identity (--collect), or the
         executed image MATCHES the collected one (--verify-executed).
Exit 3 = at least one is UNRECORDED, or the executed image is MISMATCH
         or UNRECORDED. The record is still written, so the gap is
         visible in the artifact rather than inferred from a missing
         file.
Exit 1 = refused before collecting: nothing to collect, or the caller's
         request could not be resolved at all.
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import subprocess
import sys

REPO = pathlib.Path(__file__).resolve().parent.parent.parent

IDENTITY_TYPE = "DOCKER_LOCAL_IMAGE_ID"

RECORDED = "RECORDED"
UNRECORDED = "UNRECORDED"

# ABSENT / NULL / VALUE, kept apart on purpose (rule 20).
ABSENT = "ABSENT"      # the key was not present at all
NULL = "NULL"          # the key was present and empty -- pushed, reported nothing
VALUE = "VALUE"        # a real repository digest exists


def run(argv: list[str], timeout: int = 120) -> tuple[int, str, str]:
    """Run a command, returning everything. Nothing goes to /dev/null (R10)."""
    try:
        p = subprocess.run(argv, capture_output=True, text=True, timeout=timeout)
        return p.returncode, p.stdout, p.stderr
    except FileNotFoundError as e:
        return 127, "", f"{e}"
    except subprocess.TimeoutExpired:
        return 124, "", f"timed out after {timeout}s"


def resolve_images(docker: str, compose_file: str, service: str
                   ) -> tuple[list[str], str]:
    """Ask Compose which image a service uses. Client-side, no daemon.

    Returns (refs, error). `config --images` prints one reference per
    line and resolves build-only services to the name Compose will
    actually use, which is exactly the thing this module must not guess.
    """
    code, out, err = run([docker, "compose", "-f", compose_file,
                          "config", "--images", service])
    if code != 0:
        return [], (f"`compose config --images {service}` exited {code}: "
                    f"{(err or out).strip()[:300]}")
    refs = [line.strip() for line in out.splitlines() if line.strip()]
    if not refs:
        return [], (f"`compose config --images {service}` exited 0 and named "
                    f"no image. Compose resolved the service to nothing, so "
                    f"there is no subject to inspect")
    return refs, ""


def inspect(docker: str, ref: str) -> tuple[dict | None, str]:
    """`docker image inspect` for one reference. Needs the daemon."""
    code, out, err = run([docker, "image", "inspect",
                          "--format", "{{json .}}", ref])
    if code != 0:
        return None, (f"`image inspect {ref}` exited {code}: "
                      f"{(err or out).strip()[:300]}")
    text = out.strip()
    if not text:
        return None, (f"`image inspect {ref}` exited 0 and printed nothing. "
                      f"An exit status is not a payload")
    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        return None, f"`image inspect {ref}` printed unparseable JSON: {e}"
    if isinstance(data, list):
        # Some daemons print a single-element array even with --format.
        if not data:
            return None, f"`image inspect {ref}` printed an empty array"
        data = data[0]
    if not isinstance(data, dict):
        return None, (f"`image inspect {ref}` printed a "
                      f"{type(data).__name__}, not an object")
    return data, ""


def repo_digest(data: dict) -> tuple[str, str | None]:
    """ABSENT / NULL / VALUE for the registry digest, kept distinct."""
    if "RepoDigests" not in data:
        return ABSENT, None
    digests = data.get("RepoDigests")
    if not digests:
        # Present and empty: the image exists locally and was never pushed
        # or pulled. That is the expected state for a built-in-job image,
        # and it is NOT the same fact as the key being absent.
        return NULL, None
    return VALUE, digests[0]


def platform_of(data: dict) -> str | None:
    os_ = data.get("Os")
    arch = data.get("Architecture")
    if not os_ or not arch:
        return None
    variant = data.get("Variant")
    return f"{os_}/{arch}" + (f"/{variant}" if variant else "")


def git(*args: str) -> str | None:
    code, out, _ = run(["git", "-C", str(REPO), *args])
    return out.strip() if code == 0 and out.strip() else None


def collect(docker: str, compose_file: str, services: list[str],
            run_id: str | None) -> list[dict]:
    """One record per service. Every service produces a row, always."""
    commit = git("rev-parse", "HEAD")
    tree = git("rev-parse", "HEAD^{tree}")
    rows: list[dict] = []
    for service in services:
        base = {
            "identity_type": IDENTITY_TYPE,
            "service": service,
            "commit_sha": commit,
            "tree_sha": tree,
            "run_id": run_id,
        }
        refs, err = resolve_images(docker, compose_file, service)
        if err:
            rows.append({**base, "identity_state": UNRECORDED,
                         "image_ref": None, "docker_image_id": None,
                         "repo_digest_state": ABSENT, "repo_digest": None,
                         "platform": None, "unmet_prerequisite": err})
            continue
        # A service resolving to more than one image is not a thing this
        # can silently pick from. Say so rather than taking refs[0].
        if len(refs) > 1:
            rows.append({**base, "identity_state": UNRECORDED,
                         "image_ref": None, "docker_image_id": None,
                         "repo_digest_state": ABSENT, "repo_digest": None,
                         "platform": None,
                         "unmet_prerequisite":
                             f"Compose resolved {service} to {len(refs)} "
                             f"images ({', '.join(refs)}); which one "
                             f"executed is not decidable here"})
            continue
        ref = refs[0]
        data, err = inspect(docker, ref)
        if err:
            rows.append({**base, "identity_state": UNRECORDED,
                         "image_ref": ref, "docker_image_id": None,
                         "repo_digest_state": ABSENT, "repo_digest": None,
                         "platform": None, "unmet_prerequisite": err})
            continue
        image_id = data.get("Id")
        if not image_id:
            # Exit 0 with a payload carrying no Id is the empty-identity
            # case this module exists to refuse.
            rows.append({**base, "identity_state": UNRECORDED,
                         "image_ref": ref, "docker_image_id": None,
                         "repo_digest_state": ABSENT, "repo_digest": None,
                         "platform": None,
                         "unmet_prerequisite":
                             f"`image inspect {ref}` returned an object with "
                             f"no Id. There is no identity to record"})
            continue
        state, digest = repo_digest(data)
        rows.append({**base, "identity_state": RECORDED, "image_ref": ref,
                     "docker_image_id": image_id,
                     "repo_digest_state": state, "repo_digest": digest,
                     "platform": platform_of(data)})
    return rows


# The execution binding, which is a different question from identity.
MATCH = "MATCH"
MISMATCH = "MISMATCH"


def container_image(docker: str, container: str) -> tuple[str | None, str]:
    """The image id the CONTAINER ran. Not the image a tag points at now."""
    code, out, err = run([docker, "container", "inspect",
                          "--format", "{{json .}}", container])
    if code != 0:
        return None, (f"`container inspect {container}` exited {code}: "
                      f"{(err or out).strip()[:300]}")
    text = out.strip()
    if not text:
        return None, (f"`container inspect {container}` exited 0 and printed "
                      f"nothing. An exit status is not a payload")
    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        return None, f"`container inspect {container}` printed bad JSON: {e}"
    if isinstance(data, list):
        if not data:
            return None, f"`container inspect {container}` printed an empty array"
        data = data[0]
    if not isinstance(data, dict):
        return None, (f"`container inspect {container}` printed a "
                      f"{type(data).__name__}, not an object")
    image = data.get("Image")
    if not image:
        return None, (f"`container inspect {container}` returned an object "
                      f"with no Image. What it ran is not knowable from this")
    return image, ""


def verify_executed(docker: str, container: str, collected: pathlib.Path,
                    service: str | None) -> list[dict]:
    """Did the container actually run the image we recorded?

    R11 at every boundary: if the collected record is missing, if it
    said UNRECORDED, if the container is gone, or if its inspect yields
    no Image — the answer is UNRECORDED, never an assumed MATCH.
    """
    base = {"identity_type": IDENTITY_TYPE, "container": container,
            "service": service}
    if not collected.is_file():
        return [{**base, "execution_binding": UNRECORDED,
                 "collected_image_id": None, "executed_image_id": None,
                 "unmet_prerequisite":
                     f"{collected} does not exist, so there is no recorded "
                     f"identity for the executed image to be compared with"}]
    rows = [json.loads(l) for l in collected.read_text().splitlines()
            if l.strip()]
    if service is not None:
        rows = [r for r in rows if r.get("service") == service]
    if len(rows) != 1:
        return [{**base, "execution_binding": UNRECORDED,
                 "collected_image_id": None, "executed_image_id": None,
                 "unmet_prerequisite":
                     f"{collected} holds {len(rows)} row(s) for "
                     f"service={service!r}; exactly one is required to make "
                     f"the comparison unambiguous"}]
    row = rows[0]
    base["service"] = row.get("service")
    base["run_id"] = row.get("run_id")
    base["tree_sha"] = row.get("tree_sha")
    collected_id = row.get("docker_image_id")
    if row.get("identity_state") != RECORDED or not collected_id:
        return [{**base, "execution_binding": UNRECORDED,
                 "collected_image_id": None, "executed_image_id": None,
                 "unmet_prerequisite":
                     f"the collected row is {row.get('identity_state')}, so "
                     f"there is nothing to compare the executed image with. "
                     f"An unbound run cannot be bound retroactively"}]
    executed, err = container_image(docker, container)
    if err:
        return [{**base, "execution_binding": UNRECORDED,
                 "collected_image_id": collected_id, "executed_image_id": None,
                 "unmet_prerequisite": err}]
    return [{**base,
             "execution_binding": MATCH if executed == collected_id else MISMATCH,
             "collected_image_id": collected_id,
             "executed_image_id": executed}]


def report_executed(rows: list[dict], out_path: pathlib.Path | None) -> int:
    r = rows[0]
    print("EXECUTED IMAGE — DID THE CONTAINER RUN WHAT WE RECORDED?")
    print("=" * 68)
    print(f"  container: {r['container']}   service: {r['service']}")
    print(f"  collected: {r['collected_image_id']}")
    print(f"  executed : {r['executed_image_id']}")
    print(f"  binding  : {r['execution_binding']}")
    if r.get("unmet_prerequisite"):
        print(f"  WHY      : {r['unmet_prerequisite']}")
    print()
    print(f"  inspected: 1 container(s) against 1 collected identity(s)")

    if out_path is not None:
        out_path.write_text("".join(json.dumps(x) + "\n" for x in rows))
        print(f"  written: {out_path} ({out_path.stat().st_size} bytes)")

    print()
    if r["execution_binding"] == MATCH:
        print("PASS: the container ran the image that was recorded. The claim")
        print("      'this image executed' is now a measurement, not an")
        print("      inference from a pre-run lookup.")
        return 0
    if r["execution_binding"] == MISMATCH:
        print("FAIL: the container ran a DIFFERENT image from the one")
        print("      recorded after the build. Every claim bound to the")
        print("      collected id is bound to the wrong image.")
        return 3
    print("FAIL: the executed image is UNRECORDED. What ran is UNKNOWN, and")
    print("      UNKNOWN is not a match (rule 27). D247 §6 item 10 is not")
    print("      met for this run.")
    return 3


def report(rows: list[dict], out_path: pathlib.Path | None) -> int:
    print("IMAGE IDENTITY — WHICH IMAGE ACTUALLY EXECUTED")
    print("=" * 68)
    for r in rows:
        print(f"  service : {r['service']}")
        print(f"  state   : {r['identity_state']}")
        if r["identity_state"] == RECORDED:
            print(f"  ref     : {r['image_ref']}")
            print(f"  image id: {r['docker_image_id']}   ({IDENTITY_TYPE})")
            print(f"  repo dig: {r['repo_digest_state']}"
                  + (f" {r['repo_digest']}" if r["repo_digest"] else ""))
            print(f"  platform: {r['platform']}")
        else:
            print(f"  WHY     : {r['unmet_prerequisite']}")
        print(f"  tree    : {r['tree_sha']}   run: {r['run_id']}")
        print()

    recorded = sum(1 for r in rows if r["identity_state"] == RECORDED)
    missing = len(rows) - recorded
    print(f"  inspected: {len(rows)} service(s), {recorded} recorded, "
          f"{missing} UNRECORDED")
    print()
    print("  A local image ID proves WHICH IMAGE RAN. It does NOT make that")
    print("  image independently retrievable — these images are built in the")
    print("  job and never pushed, so no registry digest exists. Sufficient")
    print("  for D247 §6 item 10; not sufficient for release provenance.")

    if out_path is not None:
        out_path.write_text("".join(json.dumps(r) + "\n" for r in rows))
        print()
        print(f"  written: {out_path} "
              f"({out_path.stat().st_size} bytes, {len(rows)} row(s))")

    if missing:
        print()
        print(f"FAIL: {missing} service(s) UNRECORDED. The run is not bound "
              f"to an image, and D247 §6 item 10 is not met for it.")
        return 3
    print()
    print("PASS: every requested service is bound to the image that ran.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--compose-file")
    ap.add_argument("--service", action="append", default=[],
                    help="repeatable; the population this run traverses")
    ap.add_argument("--out", help="write JSONL records here")
    ap.add_argument("--run-id", default=os.environ.get("GITHUB_RUN_ID"),
                    help="default: $GITHUB_RUN_ID")
    ap.add_argument("--docker", default="docker",
                    help="the docker binary; overridden by calibration so "
                         "the shipped code paths run without a daemon")
    ap.add_argument("--verify-executed", metavar="CONTAINER",
                    help="second measurement: read this container's .Image "
                         "and require it to equal the collected id")
    ap.add_argument("--against", metavar="JSONL",
                    help="the --collect record to compare against")
    args = ap.parse_args()

    if args.verify_executed:
        if not args.against:
            print("REFUSED: --verify-executed needs --against. Comparing the "
                  "executed image with nothing recorded would restate the "
                  "container's own answer and call it agreement.")
            return 1
        if len(args.service) > 1:
            print("REFUSED: --verify-executed compares ONE container with ONE "
                  "recorded identity; more than one --service makes the "
                  "comparison ambiguous.")
            return 1
        rows = verify_executed(args.docker, args.verify_executed,
                               pathlib.Path(args.against),
                               args.service[0] if args.service else None)
        return report_executed(rows, pathlib.Path(args.out) if args.out else None)

    if not args.compose_file:
        print("REFUSED: --compose-file is required to collect an identity.")
        return 1
    if not args.service:
        print("REFUSED: no --service given. A collector with an empty "
              "population reports success over nothing, which is the "
              "failure this file exists to prevent.")
        return 1
    if not pathlib.Path(args.compose_file).is_file():
        print(f"REFUSED: {args.compose_file} does not exist. Compose cannot "
              f"resolve an image from a file that is not there, and a guessed "
              f"reference is not an identity.")
        return 1

    rows = collect(args.docker, args.compose_file, args.service, args.run_id)
    return report(rows, pathlib.Path(args.out) if args.out else None)


if __name__ == "__main__":
    sys.exit(main())
