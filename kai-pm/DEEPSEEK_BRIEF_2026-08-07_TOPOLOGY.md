# Brief — network topology decision (perception cannot reach memory)

2026-08-07. A decision I have deliberately not taken alone: it changes
the security topology of a system whose whole design goal is isolation.

## Context

Self-hosted multi-service AI system. Docker Compose, ~49 services, three
profiles. Network zones are **deliberate policy** with their own CI gate
(`check_network_zones`), so widening a service's network membership is a
security change, not a convenience.

Docker's embedded DNS only resolves names on networks a container has
joined. Services on disjoint networks cannot see each other at all — the
call fails at *name resolution*, before any connection attempt.

## The finding, narrowed three times

A new check reports only where intent is declared **twice**: the service
`depends_on` the target AND holds its URL in an environment value, yet
they share no network.

| rule | findings | of |
|---|---|---|
| any env URL naming a service on no shared network | 36 | 130 |
| any `depends_on` target on no shared network | 14 | 90 |
| **both** | **5** | **51** |

The broad forms are wrong. `dashboard` holds URLs for 14 optional
services it degrades without. `heartbeat depends_on memu-core` while
actually calling `memu-core-introspect`, which it *can* reach.

## The five, with what the code actually does

| caller | target | code | verdict |
|---|---|---|---|
| audio-service | memu-core | `perception/audio/app.py:200,512` POST `/memory/memorize` | real |
| screen-capture | memu-core | `app.py:148` POST `/memory/memorize` | real |
| supervisor | heartbeat | `app.py:46` health-checks it | real |
| executor | memu-core | **zero** occurrences of `MEMU` in `executor/app.py` | config with no consumer |

## The topology (measured, not assumed)

```
minimal.yml
  agent-net          internal  (18)  agentic, cortex, dashboard, memu-core, ...
  control-net        internal  (5)   agentic, dashboard, supervisor, tool-gate, vault-sync
  data-net           internal  (5)   memu-core, memu-core-introspect, postgres, redis, tool-gate
  observability-net  internal  (8)   cortex, dashboard, docker-watcher, heartbeat, ...
  sensor-net         internal  (6)   audio, clipboard, files, screen-watcher, vision, wake
  egress-net         external  (9)   broker-bridge, browser-agent, email-reader, ollama, ...
  edge-net           external  (1)   dashboard
```

**`sensor-net` is a sealed island.** Verified in both profiles: not one
of its members is attached to any other network. So no bridge exists,
and perception services cannot reach *anything* outside themselves —
while two of them are coded to POST memories to `memu-core`.

By contrast `execution-net` (executor, heartbeat, tool-gate) is bridged:
`tool-gate` is also on `control-net` and `data-net`.

## What I think, so you can attack it rather than re-derive it

1. **`sensor-net` being sealed looks deliberate**, and correct: perception
   is the least trusted surface (cameras, microphones, screen, clipboard,
   files). Giving it a route to the memory store would be the single most
   valuable pivot in the system.
2. If that is right, then the *code* is wrong, not the topology —
   perception should not write to memory directly, and the
   `/memory/memorize` calls are the defect.
3. `tool-gate` is the pattern the system already uses for a bridge from
   an isolated zone: `execution-net` reaches `data-net` only through it.
   The analogous fix for perception would be a mediated write path, not a
   new network edge.
4. `executor`'s `MEMU_URL`/`TOOL_GATE_URL`/`VAULT_ADDR` are declared and
   never read (522 lines, real endpoints, reads only `HEARTBEAT_URL`,
   `PORT`, `SCRIPTS_DIR`, `EXECUTION_TIMEOUT`, `MAX_OUTPUT_SIZE`,
   `AUDIT_REQUIRED`, `RECOVERY_ENABLED`, `LOG_PATH`). I have NOT deleted
   them: that is the stub-eraser's logic — removing the marker of an
   unfinished intent and calling it clean.

## Questions

**Q1.** Is (1) right — is a sealed perception zone the correct design for
a system holding cameras, microphones and clipboard access? Or is there a
standard pattern where perception writes to a store directly and the
isolation is enforced elsewhere?

**Q2.** If perception must persist observations, what is the least-privilege
shape? Candidates:
  a) a write-only mediator on both `sensor-net` and `data-net` that
     accepts append-only observations and cannot read memory back
  b) perception writes to a queue/spool on `sensor-net`; something on the
     memory side pulls
  c) `tool-gate` gains `sensor-net` and mediates, as it already does for
     `execution-net`
  d) perception simply does not persist; the consumer pulls from
     perception's own endpoints
Which, and what is the failure mode of each I have not named?

**Q3.** `supervisor` (agent-net, control-net) health-checks `heartbeat`
(observability-net). Should a supervisor sit on the observability plane,
or should health be exposed through a plane it already reaches? Is
"the monitor joins every zone it watches" an antipattern?

**Q4.** For `executor`'s three declared-but-unread variables — what is
the right disposition of configuration that names an intent the code has
not implemented? Deleting it loses the intent. Keeping it means the
compose file lies about what the service does. Is there a third option
that is honest in both directions?

**Q5.** Generalising: is "a service is given an address it cannot resolve"
a class worth gating in any compose-based system, and is the
declared-twice intersection the right decidable form — or is there a
sharper one?
