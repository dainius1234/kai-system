# Graph-Memory Layer Design — `memu-graph`

> Design doc, not yet a decision. Follows D28 (`DECISIONS.md`) — Graphiti/Cognee
> spike findings. Nothing in this doc has been built. No code should be
> written against this design without a separate explicit go-ahead, per
> standing protocol (spike → design → confirm → implement).

## 1. Why this exists

The user has named memory architecture the highest-priority subsystem in the
whole project ("it's gonna be one of the most important pillars... it has to
have the most effort") and asked specifically to verify and integrate
graph-based memory (Cognee/Graphiti) so the system reflects 2025/2026 state of
the art, not just the flat vector store it has today.

`memu-core` today (verified by direct read this session) is a solid
vector-memory implementation: pluggable `VectorStore` backends (`postgres` /
`turbovec` / in-memory), real embeddings via `sentence-transformers` with a
hash fallback, importance scoring (`score_importance`), Ebbinghaus-style decay
(`_recency_weight`), and a nightly MARS consolidation cycle
(`/memory/consolidate`, `app.py:3241`) that prunes/fades/strengthens records
based on retention. It has **zero entity/relationship structure** — every
memory is a flat record (`MemoryRecord`, `app.py:148`) with a JSON `content`
blob and an embedding vector. There is no way today to ask "what's connected
to Grid B" or "who else was involved in that NCR" without re-deriving it from
free text on every query.

This doc is additive only: the graph layer sits *beside* the pgvector store,
per `SHOPPING_LIST_PLAN.md:108-112`'s own framing, not a replacement.

## 2. Domain shape this has to fit

`memu-core` is not a generic personal-assistant memory store — it has a
UK-construction domain classifier baked in (`CONSTRUCTION_CATEGORIES`,
`app.py:176`: setting-out, survey-data, RAMS, ITP, drawings, H&S briefings,
client NCRs, daily logs) alongside the personal/personality subsystems
(P17-P22: emotional timeline, autobiography, values, proactive agent, etc.,
all now Redis-native per D22-D27). A graph layer has two genuinely different
kinds of entities to extract, and the design needs to serve both without
treating them identically:

- **Domain entities** (construction): sites, grids/setting-out points,
  drawings, RAMS documents, NCRs, people, dates — these benefit hugely from
  graph structure ("show me everything on Grid B since the last RAMS
  revision" is a multi-hop graph query, not a similarity search).
- **Personality/identity entities** (P17-P22): people in the user's life,
  relationships, formed values, recurring emotional patterns — these are
  already Redis-native append-logs/hashes; the graph layer's job here is to
  surface *relationships between* personality events, not to replace their
  storage (e.g. "this autobiography entry and this conscience-log entry are
  about the same person/event").

## 3. Architecture: separate service, not an in-process library

**Recommendation: `memu-graph` as its own microservice**, not a module bolted
into `memu-core`. Rationale:

- Cognee's dependency footprint (~40 packages this session: `fastapi`,
  `sqlalchemy`, `alembic`, `lancedb`, `litellm`, `networkx`, etc.) risks
  version collisions with `memu-core`'s own FastAPI/pydantic stack if merged
  into the same process — confirmed in the D28 spike that Cognee already
  needed a clean venv to avoid a `setuptools`/`langdetect` conflict even on
  its own.
  - Graphiti's footprint is much lighter (~10 packages) and *could* fit
    in-process, but tying the design to whichever library is cheaper to
    embed this month is the wrong axis to optimize — the rest of this repo
    already splits concerns by service (`tool-gate`, `executor`, `agentic`,
    `wake-service`, `camera-service`), and graph-memory is a large enough,
    independently-failable concern to deserve the same treatment regardless
    of which library is chosen.
- A separate process means a graph-extraction outage or slowdown (LLM calls
  for entity extraction are not free, even against a local Ollama) cannot
  block `memu-core`'s synchronous write path. This matters a lot here:
  `/memory/memorize` (`app.py:1432`) is on the hot path for every event the
  rest of the system logs.
- Matches the same pattern already established for Ollama itself in Phase
  0.5: new service, own healthcheck, own `depends_on` ordering, own volume.

`memu-graph` would be a thin FastAPI wrapper around whichever library is
chosen (Cognee per D28's tentative lean, see §7), exposing:

- `POST /graph/ingest` — body: `{text, source_id, category, metadata}`.
  Wraps `cognee.add()` + `cognee.cognify()` (or Graphiti's `add_episode()`).
  Fire-and-forget from `memu-core`'s perspective — see §4.
- `GET /graph/query?q=...&top_k=...` — wraps `cognee.search()` (or Graphiti's
  hybrid search). Returns entities/relationships/source-episode references,
  not raw vector hits.
- `POST /graph/forget` — body: `{source_id}` or `{entity_id}`. Best-effort
  deletion, called from MARS pruning (§5).
- `GET /health` — standard healthcheck, same convention as every other
  service in `docker-compose.*.yml`.

## 4. Write path: best-effort fan-out, not a blocking write

`memu-core` keeps `store.insert(record)` as the source of truth (unchanged).
Each write endpoint that currently calls `store.insert()` — `/memory/memorize`
(`app.py:1432`), `/memory/note` (`2516`), `/memory/assert` (`2555`),
`/memory/autobiography/record` (`5217`), `/memory/relationship/milestone`
(`4897`), etc. — additionally fires an **async, non-blocking, fire-and-forget
POST** to `memu-graph`'s `/graph/ingest`, following the same `_safe(coro,
default)` pattern already used for the 10-way parallel fetch in `agentic`
(H1.3) and the `resilient_call` pattern already used in `dashboard`'s
`_proxy_get`/`_proxy_post` (H1.7) — both precedents already exist in this
codebase for "call an optional downstream service, never let its failure
propagate."

Not every write needs to go to the graph — only ones with extractable
entities/relationships are worth the LLM-extraction cost. Initial filter
(cheap, no LLM call): route to `/graph/ingest` only when `category` is a
construction domain category (not `general`) OR `event_type` matches a
P17-P22 personality-relationship shape (autobiography entries, relationship
milestones, formed values). Pure chatter/status-update memorize calls skip
the graph entirely. This is a tunable allowlist, not a hardcoded rule — keep
it as a small function (`_should_graph_ingest(record)`) so it can be adjusted
without touching every call site.

## 5. Read path and MARS interaction

- New `memu-core` endpoint `GET /memory/graph/query` proxies to `memu-graph`'s
  `/graph/query`, following the existing `_proxy_get`-style pattern. `agentic`
  can call this alongside the existing `/memory/retrieve` call in its
  parallel-fetch fan-out (same `_safe()` wrapper, same non-blocking
  tolerance for the service being down).
- MARS consolidation (`mars_consolidate`, `app.py:3241`) already deletes
  pruned records via `store.delete_record(record.id)`. When a record is
  deleted there, also fire a best-effort `POST /graph/forget` with that
  record's id as `source_id`, so the graph doesn't accumulate orphaned nodes
  for memories the vector store has already forgotten. This is the only
  change MARS needs — it does not need to know anything about graph
  *structure*, only that a source record went away.
- The conscience-filter save-from-pruning path (value-linked memories survive
  pruning, `app.py:3284`) needs no graph-side change — if the source record
  survives, its graph entities simply aren't forgotten either.

## 6. Storage and local-only operation

- `memu-graph` runs Cognee's `ladybug` (their owned Kuzu fork) as an embedded,
  file-backed graph DB inside its own container + volume — no separate
  graph-DB server process needed (this was the main point in Cognee's favor
  from D28: no Python-version gate, no upstream-deprecation exposure, unlike
  Graphiti's Kuzu-extra/FalkorDB-lite options).
- LLM + embeddings: Ollama via the OpenAI-compatible endpoint, reusing
  whatever model Phase 0.5 already wires up (`OLLAMA_URL`,
  `qwen2:0.5b`/whatever default is pulled) — no new model dependency.
- Explicitly set at startup: `ENABLE_BACKEND_ACCESS_CONTROL=false`,
  `CACHING=false` (this is a single-user personal system; Cognee's
  multi-tenant defaults don't apply and should be turned off, not left as
  surprise behavior), `HUGGINGFACE_TOKENIZER` pinned to match whatever local
  embedding model is actually used (this was an undocumented hard requirement
  found in the D28 spike).
- Known open risk carried over from D28: Cognee's `ladybug`/Kuzu backend
  downloads its `JSON` extension from `extension.kuzudb.com` on first use.
  This needs real internet access once, at first run, in whatever
  environment `memu-graph` actually deploys to (this sandbox's network
  policy blocked that host, which is why the D28 spike couldn't run a real
  `cognify()` call) — needs to be verified against the actual deployment
  target's network policy before relying on this, or the extension needs to
  be vendored/pre-downloaded into the image.

## 7. Library choice — still tentative, not finalized here

D28 found a tentative lean toward Cognee (owns its Kuzu fork, avoids the
upstream-deprecation risk found on Graphiti's side) at the cost of a heavier
dependency tree and defaults that need explicit overriding. This design is
written Cognee-first but the service boundary in §3 (`memu-graph` as an HTTP
microservice with `/graph/ingest`, `/graph/query`, `/graph/forget`) is
intentionally library-agnostic at the interface level — swapping the
implementation behind that boundary for Graphiti later would not require any
change to `memu-core`'s side of the integration. The actual pick should still
get its own explicit confirmation before implementation starts, not be
inherited silently from D28's "tentative" wording.

## 8. Phasing (for whenever implementation is greenlit)

1. **Phase A** — stand up `memu-graph` alone: FastAPI service, Cognee wired to
   local Ollama, `/graph/ingest`/`/graph/query`/`/graph/forget`/`/health`,
   new `docker-compose.*.yml` service block (own volume, healthcheck,
   `depends_on: ollama-pull` matching the Phase 0.5 pattern). No `memu-core`
   changes yet. Verifiable standalone via curl, same as Phase 0.5's spine
   verification steps.
2. **Phase B** — wire the write-side fan-out from `memu-core`'s existing
   write endpoints, gated by `_should_graph_ingest()`, fully best-effort/
   non-blocking.
3. **Phase C** — wire the read-side `/memory/graph/query` proxy and have
   `agentic` consume it alongside `/memory/retrieve`.
4. **Phase D** — wire MARS's delete path to `/graph/forget`.

Each phase is independently shippable and revertable, same discipline as the
P17-P22 conversion series (D22-D27) — no big-bang cutover.

## 9. Open questions before Phase A can start

- Confirm `memu-graph`'s actual deployment target has network access to
  `extension.kuzudb.com` (or pre-vendor the extension into the image) — this
  blocked the D28 spike entirely and would block Phase A's first real
  `cognify()` call the same way.
- Confirm which Ollama model is used for extraction quality — `qwen2:0.5b`
  (Phase 0.5's pulled default) is tiny; entity/relationship extraction
  quality at that size is unverified and may need a larger model gated
  behind the same GPU-availability checks the rest of the roadmap already
  uses.
- Confirm whether `memu-graph` needs its own Redis namespace or stays fully
  self-contained in its Kuzu file store — no Redis dependency is assumed in
  this design, but worth confirming once Phase A is actually built.
