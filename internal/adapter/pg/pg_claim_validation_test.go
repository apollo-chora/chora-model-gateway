// pg_claim_validation_test.go — pure unit tests (NO database) for the claim
// validation semantics ported from the Python reference (database.py claim()).
//
// Every validation path returns BEFORE the database is touched, so a zero
// Repo suffices — these run in the unit-test stage without docker/Postgres.
package pg

import (
	"context"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestClaimDebit_ValidationSemantics(t *testing.T) {
	repo := &Repo{}
	ctx := context.Background()

	const (
		gcid = "01234567-89ab-7cde-8f01-2345678900dd"
		key  = "wf-0193a1b2-turn-7"
	)

	// Empty dispatch key → the turn is not dispatch-keyed, so there is
	// nothing to dedupe: the claim PASSES (true) and the caller debits.
	// (Python: `if not key: return True`.)
	claimed, err := repo.ClaimDebit(ctx, gcid, "", "companion_chat_turn_basic", "inv")
	require.NoError(t, err)
	assert.True(t, claimed, "an un-keyed turn must not be dispatch-deduped")

	// Empty gcid → error (the claim identity is unusable).
	_, err = repo.ClaimDebit(ctx, "", key, "companion_chat_turn_basic", "inv")
	require.Error(t, err)
	assert.Contains(t, err.Error(), "gcid required")

	// Non-UUID gcid → error.
	_, err = repo.ClaimDebit(ctx, "not-a-uuid", key, "companion_chat_turn_basic", "inv")
	require.Error(t, err)
	assert.Contains(t, err.Error(), "invalid gcid")

	// Empty invocation_id → error (the winning row records the invocation
	// that won the key; without it the ledger entry is meaningless).
	_, err = repo.ClaimDebit(ctx, gcid, key, "companion_chat_turn_basic", "")
	require.Error(t, err)
	assert.Contains(t, err.Error(), "invocation_id required")
}
