-- D116: Trust Ledger & Integrity Engine — PostgreSQL Schema
-- Schema: trust (isolated from sovereign app schema)

CREATE SCHEMA IF NOT EXISTS trust;

-- ── Trust events (immutable hash chain) ──────────────────────────────────────

CREATE TABLE IF NOT EXISTS trust.trust_events (
    id              BIGSERIAL PRIMARY KEY,
    event_id        UUID NOT NULL UNIQUE DEFAULT gen_random_uuid(),
    event_type      TEXT NOT NULL,
        -- 'GRANT'            : operator explicitly raises trust level
        -- 'REVOKE'           : operator withdraws a capability or level
        -- 'AUTONOMOUS_ACTION': KAI acted without real-time approval
        -- 'OVERRIDE'         : operator countermanded KAI's decision
        -- 'ALIGNMENT_AUDIT'  : periodic Ohana Core self-report
        -- 'QUEST_RESULT'     : outcome of a Trust Quest challenge
        -- 'MERKLE_PUBLISH'   : root published to external tamper-evident store
    timestamp       TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    -- Actor context
    initiator       TEXT NOT NULL,  -- 'operator' | 'kai' | 'system'

    -- Capability / tier context
    capability      TEXT,           -- e.g. 'real_money_trading', 'skill_hunting'
    trust_tier      TEXT,           -- tier name at time of event

    -- Rich payload (conviction score, world-state snapshot, reasoning, outcome)
    event_data      JSONB NOT NULL DEFAULT '{}',

    -- Integrity — hash chain
    -- signature     = HMAC-SHA512(event_id || timestamp || event_type || initiator || event_data::text)
    -- previous_hash = SHA256(prev.signature || prev.event_data::text)
    signature       TEXT NOT NULL,
    previous_hash   TEXT NOT NULL,  -- 'GENESIS' for the first entry

    -- Operator acknowledgement (set via PATCH /trust/events/{id}/ack)
    operator_ack    BOOLEAN NOT NULL DEFAULT FALSE,
    operator_note   TEXT
);

-- Indexes for common query patterns
CREATE INDEX IF NOT EXISTS idx_trust_events_type       ON trust.trust_events (event_type);
CREATE INDEX IF NOT EXISTS idx_trust_events_capability ON trust.trust_events (capability);
CREATE INDEX IF NOT EXISTS idx_trust_events_timestamp  ON trust.trust_events (timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_trust_events_initiator  ON trust.trust_events (initiator);

-- ── Merkle roots (tamper-evident checkpoints) ─────────────────────────────────

CREATE TABLE IF NOT EXISTS trust.merkle_roots (
    id              SERIAL PRIMARY KEY,
    event_id_start  BIGINT NOT NULL,
    event_id_end    BIGINT NOT NULL,
    event_count     INT NOT NULL,
    root_hash       TEXT NOT NULL,
    timestamp       TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    published       BOOLEAN NOT NULL DEFAULT FALSE,
    published_at    TIMESTAMPTZ,
    publish_target  TEXT    -- 'obsidian' | 'file' | 'blockchain' | null
);

-- ── Trust score snapshots (nightly recompute, kept for trend analysis) ────────

CREATE TABLE IF NOT EXISTS trust.score_snapshots (
    id                      SERIAL PRIMARY KEY,
    computed_at             TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    score                   NUMERIC(5,2) NOT NULL,  -- 0.00 – 100.00
    tier                    TEXT NOT NULL,
    approval_history_score  NUMERIC(5,2),
    conviction_score        NUMERIC(5,2),
    alignment_score         NUMERIC(5,2),
    empathy_score           NUMERIC(5,2),
    reliability_score       NUMERIC(5,2),
    challenge_score         NUMERIC(5,2),
    factor_breakdown        JSONB NOT NULL DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS idx_score_snapshots_at ON trust.score_snapshots (computed_at DESC);
