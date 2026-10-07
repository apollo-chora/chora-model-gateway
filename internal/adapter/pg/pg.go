// Package pg implements domain.BudgetRepo and domain.OutboxWriter against
// the chora_observability Postgres database.
//
// Phase 0.2 migration 0008 created the per_tenant_llm_budget table with
// RLS keyed on current_setting('chora.tenant_id', true)::uuid. Every
// budget operation MUST SET LOCAL chora.tenant_id = $1 at the start of the
// transaction so the RLS policy enforces per-tenant isolation.
//
// The data-consistency outbox pattern requires the budget debit AND the
// token-usage outbox row to land in the SAME database transaction so they
// either both commit or both abort. The Repo.DebitAndEnqueue method
// (composing both ports into one call) makes this explicit when the
// service layer needs the atomic guarantee; individual port methods
// remain available for the Armor-PRE-block / Armor-POST-block flows
// where there is no budget debit (only an outbox emit covering the
// partial cost).
package pg

import (
	"context"
	"database/sql"
	"encoding/json"
	"errors"
	"fmt"
	"strconv"
	"time"

	commonv1 "github.com/apollo-chora/chora-contracts/gen/go/chora/common/v1"
	observabilityv1 "github.com/apollo-chora/chora-contracts/gen/go/chora/observability/v1"
	"google.golang.org/protobuf/proto"
	"google.golang.org/protobuf/types/known/timestamppb"

	"github.com/apollo-chora/chora-model-gateway/internal/domain"
)

// sourceService is the canonical producer identity stamped on every emitted
// envelope. chora-model-gateway shares chora_observability.outbox_events with
// chora-observability's own producer; source_service distinguishes them.
const sourceService = "chora-model-gateway"

// envelopeSchemaVersion is the v1 schema version for
// chora.observability.token_usage.recorded.v1.
const envelopeSchemaVersion = 1

// Repo holds the *sql.DB handle. Construct via New(); the DSN comes from
// Secret Manager (per secrets-and-env skill) — never hard-coded.
type Repo struct {
	db *sql.DB
}

// New constructs a Repo. The caller owns the *sql.DB lifecycle.
func New(db *sql.DB) (*Repo, error) {
	if db == nil {
		return nil, fmt.Errorf("pg: *sql.DB required")
	}
	return &Repo{db: db}, nil
}

// GetTenantBudget — implements domain.BudgetRepo.
//
// Loads the currently-active budget window for the tenant. Returns
// (nil, nil) when no window matches — the service layer treats this as
// "tenant has no configured budget; allow + billing reconciles via the
// ledger".
func (r *Repo) GetTenantBudget(ctx context.Context, tenantID string) (*domain.BudgetState, error) {
	if tenantID == "" {
		return nil, fmt.Errorf("pg: tenant_id required")
	}
	// Python parity (database.py budget()): the tenant_id is validated as a
	// UUID BEFORE a connection is acquired — a malformed id is a caller bug,
	// not a database error, and must fail fast without touching the pool.
	if !isValidUUID(tenantID) {
		return nil, fmt.Errorf("pg: invalid tenant_id %q (must be UUID)", tenantID)
	}
	tx, err := r.db.BeginTx(ctx, nil)
	if err != nil {
		return nil, fmt.Errorf("pg: begin GetTenantBudget txn: %w", err)
	}
	defer func() { _ = tx.Rollback() }()
	if err := setTenantContext(ctx, tx, tenantID); err != nil {
		return nil, err
	}
	state, err := readActiveBudget(ctx, tx, tenantID)
	if err != nil {
		return nil, err
	}
	return state, tx.Commit()
}

// DebitSpent — implements domain.BudgetRepo.
//
// Atomic increment of spent_usd_micros for the currently-active window.
// Should typically be called via DebitAndEnqueue (transactional with the
// outbox emit); the standalone version exists for the Armor-PRE-block
// path where there is no outbox emit + the debit is purely informational.
func (r *Repo) DebitSpent(ctx context.Context, tenantID string, usdMicrosDelta int64) error {
	if usdMicrosDelta <= 0 {
		return nil // no-op for partial-cost emits with zero output tokens
	}
	tx, err := r.db.BeginTx(ctx, nil)
	if err != nil {
		return fmt.Errorf("pg: begin DebitSpent txn: %w", err)
	}
	defer func() { _ = tx.Rollback() }()
	if err := setTenantContext(ctx, tx, tenantID); err != nil {
		return err
	}
	if err := debitActiveBudget(ctx, tx, tenantID, usdMicrosDelta); err != nil {
		return err
	}
	return tx.Commit()
}

// EnqueueTokenUsageRecorded — implements domain.OutboxWriter.
//
// Inserts a TokenUsageEvent row into the chora_observability outbox table
// (created in 0003_outbox.sql; producer-agnostic schema). Pub/Sub
// publisher process drains this table → canonical topic
// chora.observability.token_usage.recorded.v1.
//
// This standalone variant runs in its own transaction (NOT inside the
// budget debit txn). Use DebitAndEnqueue when the atomic-pair guarantee
// matters (happy-path + Armor POST short-circuit).
func (r *Repo) EnqueueTokenUsageRecorded(ctx context.Context, evt domain.TokenUsageEvent) error {
	tx, err := r.db.BeginTx(ctx, nil)
	if err != nil {
		return fmt.Errorf("pg: begin EnqueueTokenUsageRecorded txn: %w", err)
	}
	defer func() { _ = tx.Rollback() }()
	if err := setOutboxTenantContext(ctx, tx, evt.TenantID); err != nil {
		return err
	}
	if _, err := insertOutboxRow(ctx, tx, evt); err != nil {
		return err
	}
	return tx.Commit()
}

// DebitAndEnqueue runs the outbox insert + the budget debit in ONE
// transaction, with the debit GATED on the insert.
//
// This is the data-consistency outbox pattern (per data-consistency
// skill): two writes that MUST both commit or both abort. The service
// layer uses this on the happy path + Armor-POST-block path.
//
// Python parity (database.py debit_and_enqueue): the outbox insert runs
// FIRST and the debit is gated on it. ON CONFLICT (idempotency_key) DO
// NOTHING means a redelivery of the same usage_id inserts nothing
// (RowsAffected == 0) and must NOT debit the budget a second time — the
// insert is the redelivery guard, not just the debit.
func (r *Repo) DebitAndEnqueue(ctx context.Context, evt domain.TokenUsageEvent, usdMicrosDelta int64) error {
	tx, err := r.db.BeginTx(ctx, nil)
	if err != nil {
		return fmt.Errorf("pg: begin DebitAndEnqueue txn: %w", err)
	}
	defer func() { _ = tx.Rollback() }()
	if err := setOutboxTenantContext(ctx, tx, evt.TenantID); err != nil {
		return err
	}
	inserted, err := insertOutboxRow(ctx, tx, evt)
	if err != nil {
		return err
	}
	if inserted == 1 && usdMicrosDelta > 0 {
		if err := debitActiveBudget(ctx, tx, evt.TenantID, usdMicrosDelta); err != nil {
			return err
		}
	}
	return tx.Commit()
}

// Ping verifies the database is reachable + usable. Backs the /readyz HTTP
// probe (Python parity: database.ping() → SELECT 1 on a pooled connection).
// The gateway requires the DB for budget enforcement, debit idempotency and
// the usage outbox, so a model call that succeeds while settlement cannot be
// recorded is a billing-correctness failure — the probe must catch that.
func (r *Repo) Ping(ctx context.Context) error {
	return r.db.PingContext(ctx)
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

// setTenantContext SETs LOCAL chora.tenant_id within the transaction so
// the RLS policy from migration 0008 enforces per-tenant isolation. The
// existing chora-observability convention uses `chora.tenant_id` (not
// `app.current_tenant_id` as some other-domain skills propose); see
// services/chora-observability/migrations/0001_initial.sql for the
// pattern this matches.
func setTenantContext(ctx context.Context, tx *sql.Tx, tenantID string) error {
	// pgx + lib/pq both reject placeholder substitution in SET LOCAL;
	// caller must inline. tenant_id is a UUID — sanitised by the Postgres
	// driver since we cast to ::uuid at read time. Quote-injection prevented
	// by checking the caller-provided value is a syntactically-valid UUID
	// before stamping into SQL.
	if !isValidUUID(tenantID) {
		return fmt.Errorf("pg: invalid tenant_id %q (must be UUID)", tenantID)
	}
	_, err := tx.ExecContext(ctx, fmt.Sprintf("SET LOCAL chora.tenant_id = '%s'", tenantID))
	if err != nil {
		return fmt.Errorf("pg: SET LOCAL chora.tenant_id: %w", err)
	}
	return nil
}

// setOutboxTenantContext sets BOTH tenant GUCs the outbox write path needs:
// chora.tenant_id for the budget RLS policy (migration 0008) and
// app.current_tenant for the outbox_events RLS policy (migration 0006 —
// outbox RLS keys off app.current_tenant, NOT chora.tenant_id). The Python
// reference sets both, in this order, inside debit_and_enqueue's transaction.
func setOutboxTenantContext(ctx context.Context, tx *sql.Tx, tenantID string) error {
	if err := setTenantContext(ctx, tx, tenantID); err != nil {
		return err
	}
	_, err := tx.ExecContext(ctx, fmt.Sprintf("SET LOCAL app.current_tenant = '%s'", tenantID))
	if err != nil {
		return fmt.Errorf("pg: SET LOCAL app.current_tenant: %w", err)
	}
	return nil
}

func isValidUUID(s string) bool {
	// UUIDv7 format: 36 chars, 8-4-4-4-12 hex with hyphens.
	if len(s) != 36 {
		return false
	}
	for i, c := range s {
		switch i {
		case 8, 13, 18, 23:
			if c != '-' {
				return false
			}
		default:
			ok := (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f') || (c >= 'A' && c <= 'F')
			if !ok {
				return false
			}
		}
	}
	return true
}

// readActiveBudget runs the canonical lookup: most-recent window that
// currently encloses now(). Returns (nil, nil) on no-row-found.
func readActiveBudget(ctx context.Context, tx *sql.Tx, tenantID string) (*domain.BudgetState, error) {
	const q = `
		SELECT budget_id, budget_usd_micros, spent_usd_micros, policy, downgrade_to_logical_model_id
		  FROM per_tenant_llm_budget
		 WHERE tenant_id = $1::uuid
		   AND budget_period_start <= now()
		   AND budget_period_end   >  now()
		 ORDER BY budget_period_start DESC
		 LIMIT 1
		   FOR UPDATE
	`
	var (
		budgetID     string
		budgetMicros int64
		spentMicros  int64
		policy       string
		downgrade    sql.NullString
	)
	err := tx.QueryRowContext(ctx, q, tenantID).Scan(&budgetID, &budgetMicros, &spentMicros, &policy, &downgrade)
	if errors.Is(err, sql.ErrNoRows) {
		return nil, nil
	}
	if err != nil {
		return nil, fmt.Errorf("pg: scan budget row: %w", err)
	}
	return &domain.BudgetState{
		TenantID:         tenantID,
		BudgetUSDMicros:  budgetMicros,
		SpentUSDMicros:   spentMicros,
		Policy:           domain.BudgetPolicy(policy),
		DowngradeToModel: domain.LogicalModelID(downgrade.String),
	}, nil
}

// debitActiveBudget increments spent_usd_micros on the currently-active
// window. Uses an inline (subquery + UPDATE) so the row-lock from
// readActiveBudget's FOR UPDATE remains pinned within the transaction.
func debitActiveBudget(ctx context.Context, tx *sql.Tx, tenantID string, usdMicrosDelta int64) error {
	const q = `
		UPDATE per_tenant_llm_budget
		   SET spent_usd_micros = spent_usd_micros + $2,
		       updated_at       = now()
		 WHERE tenant_id = $1::uuid
		   AND budget_period_start <= now()
		   AND budget_period_end   >  now()
	`
	res, err := tx.ExecContext(ctx, q, tenantID, usdMicrosDelta)
	if err != nil {
		return fmt.Errorf("pg: debit update: %w", err)
	}
	// 0-row update is acceptable — caller may have decided to debit even
	// when no active budget window exists (no-op preserves caller's
	// happy-path semantics). The service layer never calls debit without
	// a prior GetTenantBudget return that proves a window exists.
	_, _ = res.RowsAffected()
	return nil
}

// outboxTableName is the canonical outbox table — created by
// 0003_outbox.sql. Schema (TEXT id PK; aggregate_type+aggregate_id;
// event_type+topic; payload BYTEA = protobuf bytes; envelope JSONB =
// event-envelope-shaped JSON metadata including tenant_id +
// traceparent; status = pending/published/failed/deadlettered).
//
// The chora-go-common/outbox PostgresRecorder + Relay drain this table
// → canonical Pub/Sub topic chora.observability.token_usage.recorded.v1.
const outboxTableName = "outbox_events"

const canonicalTopic = "chora.observability.token_usage.recorded.v1"

// insertOutboxRow inserts a TokenUsageEvent into outbox_events.
//
// Phase 2.4 (HANDOFF_OBSERVABILITY_OUTBOX_JAM_2026-05-29 Fix 2): the row is
// now SELF-SUFFICIENT for the chora-observability dispatcher that drains this
// shared table. Two halves:
//
//   - payload BYTEA = a marshalled observability.v1.TokenUsageRecorded proto
//     (with embedded common.v1.EventEnvelope). The dispatcher publishes these
//     bytes as the Pub/Sub message body; the consumer proto.Unmarshal's them.
//     (Previously []byte{} — the body was missing entirely.)
//   - envelope JSONB = a FLAT canonical snake_case map. The dispatcher's
//     reconstructEnvelope reads these as the Pub/Sub message attributes, and
//     the shared publisher VALIDATES them — a non-empty traceparent is
//     mandatory (per CLAUDE.md §6). (Previously a nested camelCase `event`
//     object that the strict reader could not parse and that lacked the
//     canonical key set.)
//
// Migration 0006 (D6.3 multi-tenant): the row also carries tenant_id, gcid
// + agid as TOP-LEVEL columns — the outbox RLS policy keys off
// app.current_tenant and the tenant isolation lookup reads these columns
// directly (the Python reference writes the same three values). agid is the
// AI Kernel agent identity (empty for system events).
//
// Returns the rows-affected count so DebitAndEnqueue can gate the budget
// debit on the insert actually landing (a conflict = redelivery = no debit).
func insertOutboxRow(ctx context.Context, tx *sql.Tx, evt domain.TokenUsageEvent) (int64, error) {
	occurredAt := evt.RecordedAt
	if occurredAt.IsZero() {
		occurredAt = time.Now().UTC()
	}

	payload, err := buildTokenUsagePayload(evt, occurredAt)
	if err != nil {
		return 0, fmt.Errorf("pg: marshal payload: %w", err)
	}
	envelopeBytes, err := json.Marshal(buildOutboxEnvelope(evt, occurredAt))
	if err != nil {
		return 0, fmt.Errorf("pg: marshal envelope: %w", err)
	}
	// idempotency_key MUST be set explicitly. Migration 0006_outbox_d6_canonical
	// declares the column NOT NULL DEFAULT '' with a UNIQUE index over ALL
	// rows (no WHERE idempotency_key IS NOT NULL clause), so omitting the
	// column collapses every gateway call onto the same empty-string row
	// and the second call fails with "duplicate key value violates unique
	// constraint outbox_events_idempotency_idx" (SQLSTATE 23505). Using
	// UsageID — which equals the per-call invocationID per service.go:383
	// — gives each LLM call its own idempotency row while still letting a
	// resume/retry of the same logical call collapse cleanly.
	//
	// ADR-254 D7: ON CONFLICT (idempotency_key) DO NOTHING, so a retry of the
	// same invocation AFTER the LLM ran (a redelivered dispatch, a client
	// retry) keeps its one ledger row instead of 23505-ing the whole Invoke
	// into UNAVAILABLE on its own bookkeeping.
	const q = `
		INSERT INTO ` + outboxTableName + ` (
			id, tenant_id, gcid, agid, aggregate_type, aggregate_id, event_type, topic,
			payload, envelope, occurred_at, status, retry_count,
			idempotency_key
		)
		VALUES (
			$1, $2, $3, $4, $5, $6, $7, $8,
			$9, $10::jsonb, $11, 'pending', 0,
			$12
		)
		ON CONFLICT (idempotency_key) DO NOTHING
	`
	res, err := tx.ExecContext(ctx, q,
		evt.UsageID,
		evt.TenantID,
		evt.GCID,
		evt.AgentID, // agid — AI Kernel agent identity (empty for system events)
		"token_usage",
		evt.UsageID, // aggregate_id == TokenUsageLedger entry id
		"chora.observability.token_usage.recorded.v1",
		canonicalTopic,
		payload, // marshalled TokenUsageRecorded protobuf bytes
		string(envelopeBytes),
		occurredAt,
		evt.UsageID, // idempotency_key == invocationID (per-call unique)
	)
	if err != nil {
		return 0, fmt.Errorf("pg: outbox insert: %w", err)
	}
	n, err := res.RowsAffected()
	if err != nil {
		return 0, fmt.Errorf("pg: outbox insert rows affected: %w", err)
	}
	return n, nil
}

// buildTokenUsagePayload marshals a TokenUsageEvent into the canonical
// observability.v1.TokenUsageRecorded protobuf (with its embedded
// common.v1.EventEnvelope as field 1). These bytes become the Pub/Sub message
// body that chora-observability's TokenUsageConsumer proto.Unmarshal's.
func buildTokenUsagePayload(evt domain.TokenUsageEvent, occurredAt time.Time) ([]byte, error) {
	rec := &observabilityv1.TokenUsageRecorded{
		Envelope: &commonv1.EventEnvelope{
			EventId:        evt.UsageID,
			IdempotencyKey: evt.UsageID,
			TenantId:       evt.TenantID,
			Gcid:           evt.GCID,
			OccurredAt:     timestamppb.New(occurredAt),
			Traceparent:    evt.Traceparent,
			Tracestate:     evt.Tracestate,
			SourceService:  sourceService,
			SchemaVersion:  envelopeSchemaVersion,
		},
		UsageId:               evt.UsageID,
		TenantId:              evt.TenantID,
		Gcid:                  evt.GCID,
		ModelId:               evt.ModelID,
		InputTokens:           evt.InputTokens,
		OutputTokens:          evt.OutputTokens,
		CachedTokens:          evt.CachedTokens,
		CostMicros:            evt.CostMicros,
		InvocationId:          evt.InvocationID,
		AgentRole:             evt.AgentRole,
		RecordedAt:            timestamppb.New(occurredAt),
		Vendor:                evt.Vendor,
		FallbackChain:         evt.FallbackChain,
		ModelArmorVerdictPre:  observabilityv1.ModelArmorVerdict(int32(evt.ArmorPre)),  // #nosec G115 -- ArmorVerdict enum is a small bounded domain, safe in int32
		ModelArmorVerdictPost: observabilityv1.ModelArmorVerdict(int32(evt.ArmorPost)), // #nosec G115 -- ArmorVerdict enum is a small bounded domain, safe in int32
		GatewayVersion:        evt.GatewayVersion,
		ManaUnits:             evt.ManaUnits,  // WS-1 umbrella-metered amount (0 when un-metered)
		ActionCode:            evt.ActionCode, // priced action label ("" when the caller stamped none)
	}
	return proto.Marshal(rec)
}

// buildOutboxEnvelope builds the FLAT canonical snake_case envelope map written
// to the envelope JSONB column. chora-observability's PostgresStore.FetchPending
// reads scalar string values from this map and reconstructEnvelope maps the
// canonical keys onto Pub/Sub message attributes; the shared CloudPublisher
// then validates them (traceparent is mandatory). Every value is a string so
// the strict reader parses each field cleanly — NO nested objects.
func buildOutboxEnvelope(evt domain.TokenUsageEvent, occurredAt time.Time) map[string]string {
	return map[string]string{
		"event_id":        evt.UsageID,
		"idempotency_key": evt.UsageID,
		"tenant_id":       evt.TenantID,
		"gcid":            evt.GCID,
		"traceparent":     evt.Traceparent,
		"tracestate":      evt.Tracestate,
		"occurred_at":     occurredAt.UTC().Format(time.RFC3339Nano),
		"source_service":  sourceService,
		"schema_version":  strconv.Itoa(envelopeSchemaVersion),
	}
}
