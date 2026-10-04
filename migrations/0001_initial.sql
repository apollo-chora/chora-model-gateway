-- 0001_initial.sql — chora-model-gateway schema.
--
-- Three tables, all owned by this service:
--
--   per_tenant_llm_budget   the spend ceiling, read under FOR UPDATE
--   token_usage_ledger      one row per completed dispatch
--   invoke_debit_claims     keyed idempotency: a redelivered dispatch bills once
--
-- Row-level security keys on a `chora.tenant_id` GUC that the application
-- sets with SET LOCAL at the start of every transaction. The application role
-- is deliberately NOT the table owner and is NOT granted BYPASSRLS, so a bug
-- that forgets the SET is a zero-row result rather than a cross-tenant read.

CREATE EXTENSION IF NOT EXISTS "pgcrypto";

-- ---------------------------------------------------------------------------
-- per_tenant_llm_budget
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS per_tenant_llm_budget (
    budget_id                   UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id                   UUID        NOT NULL,
    budget_usd_micros           BIGINT      NOT NULL CHECK (budget_usd_micros >= 0),
    spent_usd_micros            BIGINT      NOT NULL DEFAULT 0 CHECK (spent_usd_micros >= 0),
    -- allow | alert | downgrade | block. Decided by domain.BudgetState.Decide.
    policy                      TEXT        NOT NULL DEFAULT 'allow'
                                CHECK (policy IN ('allow', 'alert', 'downgrade', 'block')),
    -- On a downgrade, the registry id to dispatch instead. MUST name a
    -- registry entry; the gateway re-resolves it and fails loud if unknown.
    downgrade_to_logical_model_id TEXT,
    budget_period_start         TIMESTAMPTZ NOT NULL DEFAULT now(),
    budget_period_end           TIMESTAMPTZ NOT NULL DEFAULT (now() + INTERVAL '30 days'),
    created_at                  TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at                  TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- The hot path: "the active window for this tenant, newest first, locked".
CREATE INDEX IF NOT EXISTS per_tenant_llm_budget_active_idx
    ON per_tenant_llm_budget (tenant_id, budget_period_start DESC);

ALTER TABLE per_tenant_llm_budget ENABLE ROW LEVEL SECURITY;
ALTER TABLE per_tenant_llm_budget FORCE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS per_tenant_llm_budget_tenant_isolation ON per_tenant_llm_budget;
CREATE POLICY per_tenant_llm_budget_tenant_isolation ON per_tenant_llm_budget
    USING (
        tenant_id::text = current_setting('chora.tenant_id', true)
    )
    WITH CHECK (
        tenant_id::text = current_setting('chora.tenant_id', true)
    );

-- ---------------------------------------------------------------------------
-- token_usage_ledger
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS token_usage_ledger (
    -- The invocation id. Doubles as the idempotency key: a client retry of the
    -- same call collapses onto this row via ON CONFLICT DO NOTHING rather
    -- than failing the request on a unique violation.
    invocation_id              TEXT        PRIMARY KEY,
    tenant_id                  UUID        NOT NULL,
    gcid                       UUID        NOT NULL,
    model_id                   TEXT        NOT NULL,
    vendor                     TEXT        NOT NULL,
    input_tokens               BIGINT      NOT NULL DEFAULT 0,
    output_tokens              BIGINT      NOT NULL DEFAULT 0,
    cached_tokens              BIGINT      NOT NULL DEFAULT 0,
    cost_usd_micros            BIGINT      NOT NULL DEFAULT 0,
    agent_role                 TEXT,
    surface                    TEXT        NOT NULL DEFAULT 'unspecified',
    -- TEXT is a plain completion; IMAGE is a generated picture; GROUNDED is a
    -- completion produced with hosted web search, which is worth separating in
    -- the ledger because web search is usually priced PER SEARCH rather than
    -- per token, so a search-heavy tenant has to be identifiable.
    modality                   TEXT        NOT NULL DEFAULT 'TEXT'
                                CHECK (modality IN ('TEXT', 'IMAGE', 'GROUNDED')),
    -- JSON array of "vendor:model" for every target tried.
    fallback_chain             JSONB       NOT NULL DEFAULT '[]'::jsonb,
    -- True when a redelivered dispatch key suppressed the budget debit. The
    -- usage is real (the provider ran); only the money movement was skipped.
    debit_deduped              BOOLEAN     NOT NULL DEFAULT FALSE,
    traceparent                TEXT,
    tracestate                 TEXT,
    gateway_version            TEXT,
    recorded_at                TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS token_usage_ledger_tenant_recorded_idx
    ON token_usage_ledger (tenant_id, recorded_at DESC);
CREATE INDEX IF NOT EXISTS token_usage_ledger_model_idx
    ON token_usage_ledger (model_id, recorded_at DESC);

ALTER TABLE token_usage_ledger ENABLE ROW LEVEL SECURITY;
ALTER TABLE token_usage_ledger FORCE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS token_usage_ledger_tenant_isolation ON token_usage_ledger;
CREATE POLICY token_usage_ledger_tenant_isolation ON token_usage_ledger
    USING (
        tenant_id::text = current_setting('chora.tenant_id', true)
    )
    WITH CHECK (
        tenant_id::text = current_setting('chora.tenant_id', true)
    );

-- ---------------------------------------------------------------------------
-- invoke_debit_claims
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS invoke_debit_claims (
    claim_id                   UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
    gcid                       UUID        NOT NULL,
    dispatch_idempotency_key   TEXT        NOT NULL,
    action_code                TEXT        NOT NULL DEFAULT 'unspecified',
    -- How many redeliveries this key absorbed. A key with dedupe_count > 0
    -- fired the fallback in the delivering agent, which is the diagnostic an
    -- operator wants when spend looks wrong.
    dedupe_count               INTEGER     NOT NULL DEFAULT 0,
    last_deduped_at            TIMESTAMPTZ,
    created_at                 TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT invoke_debit_claims_key UNIQUE (gcid, dispatch_idempotency_key, action_code)
);

ALTER TABLE invoke_debit_claims ENABLE ROW LEVEL SECURITY;
ALTER TABLE invoke_debit_claims FORCE ROW LEVEL SECURITY;