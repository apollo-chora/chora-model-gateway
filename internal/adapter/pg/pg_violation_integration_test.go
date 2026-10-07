//go:build integration

// Integration test for the ADR-152 ViolationPublisher pg adapter (amendment
// 2026-08-07, requirement G2) against a real Postgres. Parity with
// TestIntegration_EnqueueExternalEgressAudited: proves the row actually lands
// on outbox_events under the tenant's RLS context with a non-empty BINARY proto
// payload, rather than trusting a no-error return.

package pg_test

import (
	"context"
	"testing"
	"time"

	"github.com/5007-Capstone/chora/services/chora-model-gateway/internal/adapter/pg"
	"github.com/5007-Capstone/chora/services/chora-model-gateway/internal/domain"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestIntegration_EnqueuePolicyViolationDetected(t *testing.T) {
	db, cleanup := setupPG(t)
	defer cleanup()
	repo, err := pg.New(db)
	require.NoError(t, err)

	evt := domain.PolicyViolationEvent{
		ViolationID:   "01234567-89ab-7cde-8f01-234567890ee1",
		TenantID:      tenantA,
		GCID:          "01234567-89ab-7cde-8f01-234567890a11",
		AgentID:       "familiar_companion",
		InvocationID:  "01234567-89ab-7cde-8f01-234567890b22",
		Leg:           domain.ArmorLegPre,
		ArmorTemplate: "projects/chora-489812/locations/asia-southeast1/templates/chora-guardrail-strict-dev",
		Verdict:       domain.ArmorVerdictBlock,
		DetectedAt:    time.Now().UTC(),
		Traceparent:   "00-trace-span-01",
	}
	require.NoError(t, repo.EnqueuePolicyViolationDetected(context.Background(), evt))

	// Assert the ROW, not the return value: a 200 is not persistence.
	var count int
	require.NoError(t, db.QueryRow(
		`SELECT count(*) FROM outbox_events WHERE topic = 'chora.governance.policy.violation_detected.v1' AND envelope->>'tenant_id' = $1`,
		tenantA,
	).Scan(&count))
	assert.Equal(t, 1, count)

	var payloadLen int
	var idem, dimension string
	require.NoError(t, db.QueryRow(
		`SELECT octet_length(payload), idempotency_key, envelope->>'chora_imda_dimension'
		   FROM outbox_events WHERE id = $1`,
		evt.ViolationID,
	).Scan(&payloadLen, &idem, &dimension))
	assert.Positive(t, payloadLen, "payload must be a non-empty BINARY proto, the topic schema rejects anything else")
	assert.Equal(t, evt.ViolationID, idem, "idempotency_key == violation_id: one event per blocked request")
	assert.Equal(t, "safety_and_robustness", dimension, "D3 routing label drives the policy_violation_log projection")

	// A redelivery of the same blocked request must not double-insert.
	err = repo.EnqueuePolicyViolationDetected(context.Background(), evt)
	assert.Error(t, err, "the outbox PK rejects a duplicate violation_id")
}

func TestIntegration_EnqueuePolicyViolationDetected_RejectsMissingIdentity(t *testing.T) {
	db, cleanup := setupPG(t)
	defer cleanup()
	repo, err := pg.New(db)
	require.NoError(t, err)

	// Fail-loud on an unusable event rather than writing a half-identified row.
	err = repo.EnqueuePolicyViolationDetected(context.Background(), domain.PolicyViolationEvent{
		ViolationID: "01234567-89ab-7cde-8f01-234567890ee2",
	})
	assert.Error(t, err, "tenant_id is required")

	err = repo.EnqueuePolicyViolationDetected(context.Background(), domain.PolicyViolationEvent{
		TenantID: tenantA,
	})
	assert.Error(t, err, "violation_id is required (it is the idempotency key)")
}
