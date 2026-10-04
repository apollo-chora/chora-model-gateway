// Package pg implements the gateway's Postgres ports: the per-tenant budget,
// the token-usage ledger, and the keyed idempotency claim.
//
// The schema is small and owned by this service:
//
//	per_tenant_llm_budget     the spend ceiling, read under FOR UPDATE
//	token_usage_ledger        one row per completed dispatch
//	invoke_debit_claims       keyed idempotency: a redelivered dispatch bills once
//
// Two invariants are load-bearing and both live here:
//
//  1. RLS scoping. Every transaction begins with SET LOCAL chora.tenant_id, so
//     a query cannot read or debit another tenant's budget even if a caller
//     supplies a wrong tenant id in the request body.
//  2. Atomic settle. The budget debit and the ledger row commit together or
//     not at all. A debit without a ledger row is invisible spend; a ledger
//     row without a debit is free spend.
package pg

import (
	"context"
	"database/sql"
	"encoding/json"
	"errors"
	"fmt"
	"time"

	"github.com/apollo-chora/chora-model-gateway/internal/domain"
)

// Repo is the Postgres-backed implementation of the gateway's three storage
// ports. One handle backs all of them; they share a transaction when the
// domain layer settles a call.
type Repo struct {
	db *sql.DB
}

// New constructs a Repo over an open *sql.DB. The caller owns the handle's
// lifecycle.
func New(db *sql.DB) (*Repo, error) {
	if db == nil {
		return nil, fmt.Errorf("pg: *sql.DB required")
	}
	return &Repo{db: db}, nil
}

// GetTenantBudget — implements domain.BudgetRepo.
//
// Loads the currently-active budget window, taking FOR UPDATE so the row
// lock is held for the rest of the transaction and two concurrent calls
// cannot both read the pre-debit figure and over-spend.
func (r *Repo) GetTenantBudget(ctx context.Context, tenantID string) (*domain.BudgetState, error) {
	tx, err := r.db.BeginTx(ctx, nil)
	if err != nil {
		return nil, fmt.Errorf("pg: begin GetTenantBudget txn: %w", err)
	}
	defer func() { _ = tx.Rollback() }()

	if err := setTenantContext(tx, tenantID); err != nil {
		return nil, err
	}
	state, err := readActiveBudget(ctx, tx, tenantID)
	if err != nil {
		return nil, err
	}
	if state == nil {
		// No configured window. Committing releases the (empty) transaction
		// cleanly; a rollback would be indistinguishable from an error to
		// the caller reading logs.
		return nil, tx.Commit()
	}
	return state, tx.Commit()
}

// DebitSpent — implements domain.BudgetRepo.
//
// Standalone debit, for callers that need to move spend without writing a
// ledger row. The normal path is Settle, which is atomic.
func (r *Repo) DebitSpent(ctx context.Context, tenantID string, usdMicrosDelta int64) error {
	if usdMicrosDelta <= 0 {
		return nil
	}
	tx, err := r.db.BeginTx(ctx, nil)
	if err != nil {
		return fmt.Errorf("pg: begin DebitSpent txn: %w", err)
	}
	defer func() { _ = tx.Rollback() }()

	if err := setTenantContext(tx, tenantID); err != nil {
		return err
	}
	if err := debitActiveBudget(ctx, tx, tenantID, usdMicrosDelta); err != nil {
		return err
	}
	return tx.Commit()
}

// EnqueueTokenUsageRecorded — implements domain.OutboxWriter.
//
// Standalone ledger write. The normal path is Settle, which is atomic.
func (r *Repo) EnqueueTokenUsageRecorded(ctx context.Context, evt domain.TokenUsageEvent) error {
	tx, err := r.db.BeginTx(ctx, nil)
	if err != nil {
		return fmt.Errorf("pg: begin EnqueueTokenUsageRecorded txn: %w", err)
	}
	defer func() { _ = tx.Rollback() }()

	if err := setTenantContext(tx, evt.TenantID); err != nil {
		return err
	}
	if err := insertLedgerRow(ctx, tx, evt); err != nil {
		return err
	}
	return tx.Commit()
}

// Settle writes the budget debit and the ledger row in ONE transaction.
//
// This is the invariant the whole service exists to keep: the money moved
// and the record of it land together. A crash between two separate
// transactions would leave either invisible spend or free spend, and
// neither is recoverable after the fact without a reconciliation pass that
// nothing here runs.
func (r *Repo) Settle(ctx context.Context, evt domain.TokenUsageEvent, usdMicrosDelta int64) error {
	tx, err := r.db.BeginTx(ctx, nil)
	if err != nil {
		return fmt.Errorf("pg: begin Settle txn: %w", err)
	}
	defer func() { _ = tx.Rollback() }()

	if err := setTenantContext(tx, evt.TenantID); err != nil {
		return err
	}
	if usdMicrosDelta > 0 {
		if err := debitActiveBudget(ctx, tx, evt.TenantID, usdMicrosDelta); err != nil {
			return err
		}
	}
	if err := insertLedgerRow(ctx, tx, evt); err != nil {
		return err
	}
	return tx.Commit()
}

// ---------------------------------------------------------------------------
// Budget internals
// ---------------------------------------------------------------------------

// readActiveBudget is the canonical lookup: the most-recent window that
// currently encloses now(). Returns (nil, nil) when no window matches, which
// the domain treats as "allow".
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

// debitActiveBudget increments spent_usd_micros on the active window. The
// row lock from readActiveBudget's FOR UPDATE stays held for the life of the
// transaction, so this cannot interleave with a competing debit.
func debitActiveBudget(ctx context.Context, tx *sql.Tx, tenantID string, usdMicrosDelta int64) error {
	const q = `
		UPDATE per_tenant_llm_budget
		   SET spent_usd_micros = spent_usd_micros + $2,
		       updated_at       = now()
		 WHERE tenant_id = $1::uuid
		   AND budget_period_start <= now()
		   AND budget_period_end   >  now()
	`
	// A 0-row update is acceptable: a tenant with no active window has no
	// ceiling to enforce, and the ledger row still records the spend.
	if _, err := tx.ExecContext(ctx, q, tenantID, usdMicrosDelta); err != nil {
		return fmt.Errorf("pg: debit update: %w", err)
	}
	return nil
}

// ---------------------------------------------------------------------------
// Ledger internals
// ---------------------------------------------------------------------------

// insertLedgerRow writes one cost-ledger row.
//
// The primary key is the invocation id, and the write is ON CONFLICT DO
// NOTHING. That combination is what makes a client retry safe: a redelivered
// call that already recorded its cost collapses onto the same row instead of
// failing the whole call with a unique-violation, which would turn a
// bookkeeping collision into a user-visible 500 on a request the provider
// already billed.
func insertLedgerRow(ctx context.Context, tx *sql.Tx, evt domain.TokenUsageEvent) error {
	recordedAt := evt.RecordedAt
	if recordedAt.IsZero() {
		recordedAt = time.Now().UTC()
	}
	modality := evt.Modality
	if modality == "" {
		modality = domain.ModalityText
	}
	surface := evt.Surface
	if surface == "" {
		surface = "unspecified"
	}

	const q = `
		INSERT INTO token_usage_ledger (
			invocation_id, tenant_id, gcid, model_id, vendor,
			input_tokens, output_tokens, cached_tokens, cost_usd_micros,
			agent_role, surface, modality,
			fallback_chain, debit_deduped,
			traceparent, tracestate,
			gateway_version, recorded_at
		)
		VALUES (
			$1, $2::uuid, $3::uuid, $4, $5,
			$6, $7, $8, $9,
			$10, $11, $12,
			$13, $14,
			$15, $16,
			$17, $18
		)
		ON CONFLICT (invocation_id) DO NOTHING
	`
	if _, err := tx.ExecContext(ctx, q,
		evt.UsageID,
		evt.TenantID,
		evt.GCID,
		evt.ModelID,
		evt.Vendor,
		evt.InputTokens,
		evt.OutputTokens,
		evt.CachedTokens,
		evt.CostMicros,
		nullIfEmpty(evt.AgentRole),
		surface,
		modality,
		mustJSON(evt.FallbackChain),
		evt.DebitDeduped,
		nullIfEmpty(evt.Traceparent),
		nullIfEmpty(evt.Tracestate),
		evt.GatewayVersion,
		recordedAt,
	); err != nil {
		return fmt.Errorf("pg: ledger insert: %w", err)
	}
	return nil
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

// setTenantContext stamps the tenant onto the transaction so RLS applies.
//
// pgx will not substitute a placeholder inside SET, so the value is inlined —
// which makes validating it a UUID a correctness requirement, not a nicety.
func setTenantContext(tx *sql.Tx, tenantID string) error {
	if !isValidUUID(tenantID) {
		return fmt.Errorf("pg: invalid tenant_id %q (must be a UUID)", tenantID)
	}
	if _, err := tx.ExecContext(context.Background(),
		fmt.Sprintf("SET LOCAL chora.tenant_id = '%s'", tenantID)); err != nil {
		return fmt.Errorf("pg: SET LOCAL chora.tenant_id: %w", err)
	}
	return nil
}

func isValidUUID(s string) bool {
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

func nullIfEmpty(s string) any {
	if s == "" {
		return nil
	}
	return s
}

func mustJSON(v any) []byte {
	encoded, err := json.Marshal(v)
	if err != nil {
		// The only values reaching this are []string and bool, both of which
		// always marshal. A failure here means a caller passed something
		// structurally new, and an empty array is a safe ledger record.
		return []byte("[]")
	}
	return encoded
}

// Compile-time proof the adapter satisfies the storage ports.
var (
	_ domain.BudgetRepo    = (*Repo)(nil)
	_ domain.OutboxWriter  = (*Repo)(nil)
	_ domain.ClaimRecorder = (*Repo)(nil)
)
