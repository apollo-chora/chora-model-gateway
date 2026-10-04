// pg_debit_claim.go: implements domain.ClaimRecorder — keyed idempotency, so
// a redelivered dispatch bills once.
//
// The first claim on (gcid, dispatch_idempotency_key, action_code) inserts the
// row and wins: the caller debits. A later claim on the same key inserts
// nothing (ON CONFLICT DO NOTHING), and the dedupe is RECORDED on the claim
// row (dedupe_count, last_deduped_at) so an operator can read "this key
// billed once, deduped N times" straight off the ledger.
package pg

import (
	"context"
	"fmt"

	"github.com/apollo-chora/chora-model-gateway/internal/domain"
)

// ClaimDebit — implements domain.ClaimRecorder.
//
// A claimed-again key returns (false, nil) and the domain service skips the
// budget debit while still completing the call and still writing a ledger
// row. That is deliberate: the provider already ran and was already paid, so
// suppressing the whole response would leave the caller with a hole and the
// operator with a reconciliation problem.
func (r *Repo) ClaimDebit(ctx context.Context, gcid, dispatchKey, actionCode string) (bool, error) {
	switch {
	case gcid == "":
		return false, fmt.Errorf("pg: gcid required")
	case !isValidUUID(gcid):
		return false, fmt.Errorf("pg: invalid gcid %q (must be UUID)", gcid)
	case dispatchKey == "":
		return false, fmt.Errorf("pg: dispatch_idempotency_key required")
	case actionCode == "":
		// Falling back to a stable bucket keeps the claim working for a
		// caller that did not name an action, instead of refusing the call.
		actionCode = "unspecified"
	}

	tx, err := r.db.BeginTx(ctx, nil)
	if err != nil {
		return false, fmt.Errorf("pg: begin ClaimDebit txn: %w", err)
	}
	defer func() { _ = tx.Rollback() }()

	const claimQ = `
		INSERT INTO invoke_debit_claims (gcid, dispatch_idempotency_key, action_code)
		VALUES ($1::uuid, $2, $3)
		ON CONFLICT (gcid, dispatch_idempotency_key, action_code) DO NOTHING`
	res, err := tx.ExecContext(ctx, claimQ, gcid, dispatchKey, actionCode)
	if err != nil {
		return false, fmt.Errorf("pg: claim debit: %w", err)
	}
	n, err := res.RowsAffected()
	if err != nil {
		return false, fmt.Errorf("pg: claim debit rows affected: %w", err)
	}
	if n == 1 {
		return true, tx.Commit()
	}

	// Deduped: bump the counter on the winning row so the redelivery proof is
	// visible without a separate audit table.
	const dedupeQ = `
		UPDATE invoke_debit_claims
		   SET dedupe_count = dedupe_count + 1,
		       last_deduped_at = now()
		 WHERE gcid = $1::uuid AND dispatch_idempotency_key = $2 AND action_code = $3`
	if _, err := tx.ExecContext(ctx, dedupeQ, gcid, dispatchKey, actionCode); err != nil {
		return false, fmt.Errorf("pg: record debit dedupe: %w", err)
	}
	return false, tx.Commit()
}

var _ domain.ClaimRecorder = (*Repo)(nil)
