//go:build integration

// Integration tests for the R22 keyed debit claim (ADR-254 D7) against a real
// Postgres (observability migration 0019): a redelivered dispatch bills once.

package pg_test

import (
	"context"
	"testing"

	"github.com/apollo-chora/chora-model-gateway/internal/adapter/pg"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

const (
	learnerGCID = "01234567-89ab-7cde-8f01-2345678900dd"
	dispatchKey = "wf-0193a1b2-turn-7"
)

func TestIntegration_DebitClaim_FirstClaimWins_SecondDedupes(t *testing.T) {
	db, cleanup := setupPG(t)
	defer cleanup()
	repo, err := pg.New(db)
	require.NoError(t, err)
	ctx := context.Background()

	claimed, err := repo.ClaimDebit(ctx, learnerGCID, dispatchKey, actionBasic, "inv-first")
	require.NoError(t, err)
	assert.True(t, claimed, "the first claim on a key debits")

	claimed, err = repo.ClaimDebit(ctx, learnerGCID, dispatchKey, actionBasic, "inv-redelivered")
	require.NoError(t, err)
	assert.False(t, claimed, "a second claim on the same (gcid, key, action) debits nothing")

	// The dedupe is recorded on the claim row itself (the gateway's ledger of
	// debits), so the proof can read it back: debit_deduped=true in the ledger row.
	var dedupes int
	var lastInv string
	require.NoError(t, db.QueryRow(`SELECT dedupe_count, COALESCE(last_deduped_invocation_id, '') FROM invoke_debit_claims
	                                 WHERE gcid = $1::uuid AND dispatch_idempotency_key = $2 AND action_code = $3`,
		learnerGCID, dispatchKey, actionBasic).Scan(&dedupes, &lastInv))
	assert.Equal(t, 1, dedupes)
	assert.Equal(t, "inv-redelivered", lastInv)
}

func TestIntegration_DebitClaim_DifferentActionOrGCID_ClaimsAgain(t *testing.T) {
	db, cleanup := setupPG(t)
	defer cleanup()
	repo, err := pg.New(db)
	require.NoError(t, err)
	ctx := context.Background()

	claimed, err := repo.ClaimDebit(ctx, learnerGCID, dispatchKey, actionBasic, "inv-1")
	require.NoError(t, err)
	assert.True(t, claimed)

	// Same key, different action: a distinct debit (one turn may carry one
	// metered action; the triple is the claim identity per ADR-254 D7).
	claimed, err = repo.ClaimDebit(ctx, learnerGCID, dispatchKey, actionCite, "inv-2")
	require.NoError(t, err)
	assert.True(t, claimed)

	// Same key + action, a different learner: never cross-learner deduped.
	claimed, err = repo.ClaimDebit(ctx, operatorGCID, dispatchKey, actionBasic, "inv-3")
	require.NoError(t, err)
	assert.True(t, claimed)
}

// TestIntegration_DebitClaim_Validation pins the Python reference's claim
// validation (database.py claim()) against a real Postgres: an empty action
// defaults to "unspecified" and still dedupes; a missing gcid or
// invocation_id is an error. (The empty-dispatch-key path — claim passes —
// is covered by the pure unit test pg_claim_validation_test.go.)
func TestIntegration_DebitClaim_Validation(t *testing.T) {
	db, cleanup := setupPG(t)
	defer cleanup()
	repo, err := pg.New(db)
	require.NoError(t, err)
	ctx := context.Background()

	_, err = repo.ClaimDebit(ctx, "", dispatchKey, actionBasic, "inv")
	require.Error(t, err, "gcid required")
	_, err = repo.ClaimDebit(ctx, learnerGCID, dispatchKey, actionBasic, "")
	require.Error(t, err, "invocation_id required")

	// Empty action → defaults to "unspecified": the first claim on the
	// (gcid, key, unspecified) triple wins, a redelivery dedupes.
	claimed, err := repo.ClaimDebit(ctx, learnerGCID, dispatchKey+"-noaction", "", "inv")
	require.NoError(t, err)
	assert.True(t, claimed)
	claimed, err = repo.ClaimDebit(ctx, learnerGCID, dispatchKey+"-noaction", "", "inv-2")
	require.NoError(t, err)
	assert.False(t, claimed)
}

func TestIntegration_DebitClaim_MissingTable_ReturnsError(t *testing.T) {
	db, cleanup := setupPG(t)
	defer cleanup()
	repo, err := pg.New(db)
	require.NoError(t, err)

	_, err = db.Exec(`DROP TABLE invoke_debit_claims`)
	require.NoError(t, err)
	_, err = repo.ClaimDebit(context.Background(), learnerGCID, dispatchKey, actionBasic, "inv")
	require.Error(t, err, "an unreadable claim store must surface, never silently claim")
}
