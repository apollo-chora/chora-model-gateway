//go:build integration

// ADR-254 D7: the ledger outbox insert is keyed on invocation_id with
// ON CONFLICT DO NOTHING, so a retry after the LLM ran no longer 23505s into
// Unavailable (the second delivery of the same dispatch must not fail the whole
// Invoke on its own bookkeeping).

package pg_test

import (
	"context"
	"testing"
	"time"

	"github.com/apollo-chora/chora-model-gateway/internal/adapter/pg"
	"github.com/apollo-chora/chora-model-gateway/internal/domain"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestIntegration_Outbox_SameInvocationTwice_NoError_OneRow(t *testing.T) {
	db, cleanup := setupPG(t)
	defer cleanup()
	repo, err := pg.New(db)
	require.NoError(t, err)
	ctx := context.Background()

	evt := domain.TokenUsageEvent{
		UsageID:      "01234567-89ab-7cde-8f01-2345678900e1",
		InvocationID: "01234567-89ab-7cde-8f01-2345678900e1",
		TenantID:     tenantA,
		GCID:         learnerGCID,
		ModelID:      "gemini-2.5-flash",
		InputTokens:  10,
		OutputTokens: 5,
		CostMicros:   7,
		Traceparent:  "00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01",
		RecordedAt:   time.Now().UTC(),
	}
	require.NoError(t, repo.EnqueueTokenUsageRecorded(ctx, evt))
	require.NoError(t, repo.EnqueueTokenUsageRecorded(ctx, evt), "a retry with the same invocation id must not error")

	var n int
	require.NoError(t, db.QueryRow(`SELECT count(*) FROM outbox_events WHERE idempotency_key = $1`, evt.UsageID).Scan(&n))
	assert.Equal(t, 1, n, "one ledger row per invocation")
}
