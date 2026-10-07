//go:build integration

// Integration tests for the ExternalEgressGate + EgressAuditWriter pg adapter
// (ADR-231 D6) against a real Postgres (migration 0014). Prepare-smokes every
// grounded SQL const + validates fail-closed defaults + the daily counter.

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

func TestIntegration_Egress_NoRow_DeniesDisabled(t *testing.T) {
	db, cleanup := setupPG(t)
	defer cleanup()
	repo, err := pg.New(db)
	require.NoError(t, err)

	// No external_egress_policy row ⇒ OFF (ADR-220 D4 franchise default).
	auth, err := repo.Authorize(context.Background(), tenantA)
	require.NoError(t, err)
	assert.False(t, auth.Allowed)
	assert.Equal(t, domain.EgressDenyDisabled, auth.Reason)
}

func TestIntegration_Egress_Enabled_Allows(t *testing.T) {
	db, cleanup := setupPG(t)
	defer cleanup()
	repo, err := pg.New(db)
	require.NoError(t, err)

	_, err = db.Exec(`INSERT INTO external_egress_policy (tenant_id, egress_enabled, daily_call_ceiling) VALUES ($1::uuid, TRUE, 50)`, tenantA)
	require.NoError(t, err)

	auth, err := repo.Authorize(context.Background(), tenantA)
	require.NoError(t, err)
	assert.True(t, auth.Allowed)
	assert.Empty(t, auth.Reason)
}

func TestIntegration_Egress_KillSwitch_Denies(t *testing.T) {
	db, cleanup := setupPG(t)
	defer cleanup()
	repo, err := pg.New(db)
	require.NoError(t, err)

	// Even an entitled tenant is denied when the platform kill-switch is on.
	_, err = db.Exec(`INSERT INTO external_egress_policy (tenant_id, egress_enabled) VALUES ($1::uuid, TRUE)`, tenantA)
	require.NoError(t, err)
	_, err = db.Exec(`UPDATE platform_egress_killswitch SET engaged = TRUE WHERE singleton`)
	require.NoError(t, err)

	auth, err := repo.Authorize(context.Background(), tenantA)
	require.NoError(t, err)
	assert.False(t, auth.Allowed)
	assert.Equal(t, domain.EgressDenyKillSwitch, auth.Reason)
}

func TestIntegration_Egress_DailyCeiling_Denies(t *testing.T) {
	db, cleanup := setupPG(t)
	defer cleanup()
	repo, err := pg.New(db)
	require.NoError(t, err)

	_, err = db.Exec(`INSERT INTO external_egress_policy (tenant_id, egress_enabled, daily_call_ceiling) VALUES ($1::uuid, TRUE, 2)`, tenantA)
	require.NoError(t, err)

	// Two recorded calls hit the ceiling of 2.
	require.NoError(t, repo.RecordEgress(context.Background(), tenantA))
	require.NoError(t, repo.RecordEgress(context.Background(), tenantA))

	auth, err := repo.Authorize(context.Background(), tenantA)
	require.NoError(t, err)
	assert.False(t, auth.Allowed)
	assert.Equal(t, domain.EgressDenyDailyCeil, auth.Reason)

	// The counter is per-day and per-tenant.
	var count int
	require.NoError(t, db.QueryRow(`SELECT call_count FROM grounded_search_daily_usage WHERE tenant_id=$1::uuid AND usage_date=current_date`, tenantA).Scan(&count))
	assert.Equal(t, 2, count)
}

func TestIntegration_Egress_UnderCeiling_Allows(t *testing.T) {
	db, cleanup := setupPG(t)
	defer cleanup()
	repo, err := pg.New(db)
	require.NoError(t, err)

	_, err = db.Exec(`INSERT INTO external_egress_policy (tenant_id, egress_enabled, daily_call_ceiling) VALUES ($1::uuid, TRUE, 5)`, tenantA)
	require.NoError(t, err)
	require.NoError(t, repo.RecordEgress(context.Background(), tenantA))

	auth, err := repo.Authorize(context.Background(), tenantA)
	require.NoError(t, err)
	assert.True(t, auth.Allowed)
}

func TestIntegration_EnqueueExternalEgressAudited(t *testing.T) {
	db, cleanup := setupPG(t)
	defer cleanup()
	repo, err := pg.New(db)
	require.NoError(t, err)

	evt := domain.ExternalEgressAuditEvent{
		AuditID:          "01234567-89ab-7cde-8f01-234567890fff",
		TenantID:         tenantA,
		ActorGCID:        "01234567-89ab-7cde-8f01-234567890a11",
		AgentID:          "familiar_seeker",
		ActionCode:       "familiar_far_sight_grounded_search",
		DirectiveHash:    "abc123hash",
		WebSearchQueries: []string{"q1", "q2"},
		CitationCount:    2,
		Result:           domain.EgressAuditAllowed,
		ArmorVerdictPre:  "allow",
		ArmorVerdictPost: "allow",
		Vendor:           "vertex_ai_gemini",
		ModelVersion:     "gemini-2.5-flash",
		OccurredAt:       time.Now().UTC(),
		Traceparent:      "00-trace-span-01",
	}
	require.NoError(t, repo.EnqueueExternalEgressAudited(context.Background(), evt))

	var count int
	require.NoError(t, db.QueryRow(
		`SELECT count(*) FROM outbox_events WHERE topic = 'chora.governance.audit.external_egress.v1' AND envelope->>'tenant_id' = $1`,
		tenantA,
	).Scan(&count))
	assert.Equal(t, 1, count)

	// Payload is a non-empty protobuf; idempotency_key == audit_id.
	var payloadLen int
	var idem string
	require.NoError(t, db.QueryRow(
		`SELECT octet_length(payload), idempotency_key FROM outbox_events WHERE topic = 'chora.governance.audit.external_egress.v1'`,
	).Scan(&payloadLen, &idem))
	assert.Greater(t, payloadLen, 0)
	assert.Equal(t, evt.AuditID, idem)
}
