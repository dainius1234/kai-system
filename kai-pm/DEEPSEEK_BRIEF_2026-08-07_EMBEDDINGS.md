# Brief for DeepSeek — the embedding model and the impossible default

Sent 2026-08-07. Their answer drove D167; recorded here so the question
is readable alongside it.

## The failure

CI run 708 failed at step 49 of 67, the minimal bring-up:

    dependency failed to start: container sovereign-memu-core-minimal
    is unhealthy

after 109 seconds. Container logs:

    '[Errno -3] Temporary failure in name resolution' thrown while
    requesting HEAD https://huggingface.co/sentence-transformers/
    all-MiniLM-L6-v2/resolve/main/modules.json
    Retrying in 1s [Retry 1/5] … 2s … 4s … 8s … 8s [Retry 5/5]

repeated for `modules.json`, `adapter_config.json`, `config.json`.

## The code — `memu-core/app.py`

```python
_ALLOW_FAKE_EMBEDDINGS = os.getenv(
    "MEMU_ALLOW_FAKE_EMBEDDINGS", "false").lower() == "true"

try:
    from sentence_transformers import SentenceTransformer as _ST
    _st_model = _ST(EMBEDDING_MODEL_NAME)          # <-- always hits network
except Exception as _st_exc:
    if not _ALLOW_FAKE_EMBEDDINGS:
        raise RuntimeError("... Refusing to silently degrade ...") from _st_exc
    # hash-based SHA-256 pseudo-embedding fallback
```

Module import time, before uvicorn binds. **The flag is read in the
`except` branch — after the attempt.** So `=true` never meant "don't
try"; it meant "it is acceptable that trying failed".

Healthcheck: `start-period 10s, interval 30s, retries 3` → unhealthy at
~100s. The backoff burns 70–100s. The step died at 109s.

## The structural fact, verified across all three compose files

Every service that loads a model is attached **only** to networks
declared `internal: true`:

```
minimal.yml   internal: agent-net, control-net, data-net,
                        observability-net, sensor-net
  memu-core             nets=[agent-net, data-net]         egress=NONE
  memu-core-introspect  nets=[data-net, observability-net]  egress=NONE
  agentic               nets=[agent-net, control-net]       egress=NONE
full.yml      same + fusion-engine, memu-graph              egress=NONE
sovereign.yml same                                          egress=NONE
```

`memu-core/Dockerfile` installs `sentence-transformers>=2.7.0` and never
downloads a model. The container has nowhere to send the request.

## Measured evidence, not inferred

| run | commit | offline guard | result |
|---|---|---|---|
| 708/1 | `e0e9849` | no | fail, step 49, 109s |
| 708/2 | `e0e9849` | no | fail, step 49, 108s — re-run of the identical commit |
| 709 | `b5deaaa` | yes | success, 67/67 |

Attempt 2 was run on the unchanged commit specifically to test for a
flake. It is deterministic. The six preceding green runs had each won
the same race.

## The problem the shipped fix does *not* solve

`MEMU_ALLOW_FAKE_EMBEDDINGS=false` is the **documented production
default** ("real model or crash", by explicit design decision). In these
profiles it makes memu-core raise at import and die. Every CI bring-up
overrides the flag to `true`, so that configuration has never executed
anywhere.

**The production default of the main memory service cannot boot in any
profile the project ships.**

## Options put to them (critique, don't re-derive)

* **A.** Bake the model into the image at build time. Build has egress;
  runtime does not need it. Costs ~90MB and a build-time dependency on
  huggingface.co.
* **B.** Give memu-core egress. Contradicts the sovereignty goal, widens
  attack surface on the service holding all memory.
* **C.** Shared model-cache volume seeded by a one-shot init container.
* **D.** Change the documented default to fake embeddings.
* **E.** Sidecar embedding service on egress-net, called over HTTP.

## Questions asked

1. Is `HF_HUB_OFFLINE` the correct and complete knob for
   sentence-transformers ≥2.7.0, or does it still reach the network by
   another path?
2. Of A–E, which best serves "no runtime egress" without making the
   build fragile? Is pinning by revision SHA materially better than a
   mutable tag?
3. Is there a general rule beyond this instance — "capability flag
   consulted *after* the expensive attempt it was meant to govern"? Is
   that class statically decidable?
4. Two other sites do the same load lazily (`agentic/router.py:288`,
   `fusion-engine/app.py:166`). Should they share one mechanism?
5. Healthcheck sizing for a service that loads an ML model — defensible
   `start-period`, and is there an argument for separate readiness vs
   liveness probes?

## Outcome

Option A adopted; see D167. Their advice on Q4 (a shared
`common/embeddings.py`) was **declined on measurement** — only
`memu-core` declares `sentence-transformers`; the other two guard the
import and fall back instantly, so the real population is one. The brief
above had over-claimed that scope.
