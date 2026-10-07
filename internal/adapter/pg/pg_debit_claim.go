// pg_debit_claim.go: implements middleware.DebitClaimer (ADR-254 D7, R22)
// against chora_observability migration 0019.
//
// The first claim on (gcid, dispatch_idempotency_key, action_code) inserts the
// row and wins: the caller debits. A later claim on the same key inserts
// nothing (ON CONFLICT DO NOTHING), and the dedupe is RECORDED on the claim
// row (dedupe_count, last_deduped_invocation_id, last_deduped_at) so the
// redelivery proof can read "this key billed once, deduped N times" straight
// off the gateway's ledger of dispatch-keyed debits.
package pg

import (
	"context"
	"fmt"

	"github.com/5007-Capstone/chora/services/chora-model-gateway/internal/adapter/middleware"
)

// ClaimDebit: implements middleware.DebitClaimer.
//
// Validation semantics mirror the Python reference (database.py claim()):
//   - empty dispatch key → the turn is not dispatch-keyed, so there is
//     nothing to dedupe: the claim PASSES (true) and the caller debits.
//   - empty / non-UUID gcid → error (the claim identity is unusable).
//   - empty invocation_id → error (the winning row records the invocation
//     that won the key; without it the ledger entry is meaningless).
//   - empty action → defaults to "unspecified" (the claim triple still
//     dedupes, under the explicit unspecified action_code).
func (r *Repo) ClaimDebit(ctx context.Context, gcid, dispatchKey, actionCode, invocationID string) (bool, error) {
	switch {
	case dispatchKey == "":
		return true, nil
	case gcid == "":
		return false, fmt.Errorf("pg: gcid required")
	case !isValidUUID(gcid):
		return false, fmt.Errorf("pg: invalid gcid %q (must be UUID)", gcid)
	case invocationID == "":
		return false, fmt.Errorf("pg: invocation_id required")
	}
	if actionCode == "" {
		actionCode = "unspecified"
	}
	tx, err := r.db.BeginTx(ctx, nil)
	if err != nil {
		return false, fmt.Errorf("pg: begin ClaimDebit txn: %w", err)
	}
	defer func() { _ = tx.Rollback() }()

	const claimQ = `
		INSERT INTO invoke_debit_claims (gcid, dispatch_idempotency_key, action_code, invocation_id)
		VALUES ($1::uuid, $2, $3, $4)
		ON CONFLICT (gcid, dispatch_idempotency_key, action_code) DO NOTHING`
	res, err := tx.ExecContext(ctx, claimQ, gcid, dispatchKey, actionCode, invocationID)
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

	// Deduped: record it on the winning row (the ledger entry of this debit).
	const dedupeQ = `
		UPDATE invoke_debit_claims
		   SET dedupe_count = dedupe_count + 1,
		       last_deduped_invocation_id = $4,
		       last_deduped_at = now()
		 WHERE gcid = $1::uuid AND dispatch_idempotency_key = $2 AND action_code = $3`
	if _, err := tx.ExecContext(ctx, dedupeQ, gcid, dispatchKey, actionCode, invocationID); err != nil {
		return false, fmt.Errorf("pg: record debit dedupe: %w", err)
	}
	return false, tx.Commit()
}

// Compile-time check.
var _ middleware.DebitClaimer = (*Repo)(nil)
