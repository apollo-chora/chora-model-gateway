//go:build integration

// Integration tests for the ADR-252 D1 companion-suspension read (ADR-254 D7)
// against a real Postgres (observability migration 0018). Prepare-smokes the
// SQL + validates the D6 asymmetry: an EMPTY store allows, an UNREADABLE store
// errors (the decorator turns that error into a deny).

package pg_test

import (
	"context"
	"testing"

	"github.com/5007-Capstone/chora/services/chora-model-gateway/internal/adapter/pg"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

const (
	operatorGCID = "01234567-89ab-7cde-8f01-2345678900aa"
	actionBasic  = "companion_chat_turn_basic"
	actionCite   = "companion_skill_cite_atom"
)

func TestIntegration_Suspension_EmptyStore_NotSuspended(t *testing.T) {
	db, cleanup := setupPG(t)
	defer cleanup()
	repo, err := pg.New(db)
	require.NoError(t, err)

	v, err := repo.CheckSuspension(context.Background(), tenantA, actionBasic)
	require.NoError(t, err)
	assert.False(t, v.Suspended, "no row means not suspended (ADR-252 D6)")
}

func TestIntegration_Suspension_PlatformAllSkills_SuspendsEveryTenantAndSkill(t *testing.T) {
	db, cleanup := setupPG(t)
	defer cleanup()
	repo, err := pg.New(db)
	require.NoError(t, err)

	_, err = db.Exec(`INSERT INTO platform_companion_suspension (id, skill_key, engaged, reason, engaged_by, engaged_at)
	                  VALUES ('01234567-89ab-7cde-8f01-2345678900b1'::uuid, NULL, TRUE, 'incident 42', $1::uuid, now())`, operatorGCID)
	require.NoError(t, err)

	for _, tenant := range []string{tenantA, tenantB} {
		for _, action := range []string{actionBasic, actionCite, ""} {
			v, err := repo.CheckSuspension(context.Background(), tenant, action)
			require.NoError(t, err)
			assert.True(t, v.Suspended, "tenant=%s action=%q", tenant, action)
			assert.Equal(t, "platform", v.Scope)
			assert.Equal(t, "incident 42", v.Reason)
		}
	}
}

func TestIntegration_Suspension_PlatformOneSkill_SuspendsOnlyThatSkill(t *testing.T) {
	db, cleanup := setupPG(t)
	defer cleanup()
	repo, err := pg.New(db)
	require.NoError(t, err)

	_, err = db.Exec(`INSERT INTO platform_companion_suspension (id, skill_key, engaged, reason, engaged_by, engaged_at)
	                  VALUES ('01234567-89ab-7cde-8f01-2345678900b2'::uuid, $1, TRUE, 'cite_atom leaks', $2::uuid, now())`, actionCite, operatorGCID)
	require.NoError(t, err)

	v, err := repo.CheckSuspension(context.Background(), tenantA, actionCite)
	require.NoError(t, err)
	assert.True(t, v.Suspended)

	v, err = repo.CheckSuspension(context.Background(), tenantA, actionBasic)
	require.NoError(t, err)
	assert.False(t, v.Suspended, "a per-skill suspension must not catch other skills")
}

func TestIntegration_Suspension_ReleasedRow_DoesNotSuspend(t *testing.T) {
	db, cleanup := setupPG(t)
	defer cleanup()
	repo, err := pg.New(db)
	require.NoError(t, err)

	// Release is a state change on the row, never a delete (soft-delete doctrine):
	// the history survives and the gate reads it as not engaged.
	_, err = db.Exec(`INSERT INTO platform_companion_suspension (id, skill_key, engaged, reason, engaged_by, engaged_at, released_by, released_at)
	                  VALUES ('01234567-89ab-7cde-8f01-2345678900b3'::uuid, NULL, FALSE, 'resolved', $1::uuid, now() - interval '1 hour', $1::uuid, now())`, operatorGCID)
	require.NoError(t, err)

	v, err := repo.CheckSuspension(context.Background(), tenantA, actionBasic)
	require.NoError(t, err)
	assert.False(t, v.Suspended)
}

func TestIntegration_Suspension_TenantScope_RLS_OnlyThatTenant(t *testing.T) {
	db, cleanup := setupPG(t)
	defer cleanup()
	repo, err := pg.New(db)
	require.NoError(t, err)

	// Seeded as the owning (migrate) role, which bypasses RLS; the gateway's
	// read runs under SET LOCAL chora.tenant_id and must see ONLY its tenant.
	_, err = db.Exec(`INSERT INTO companion_suspension_policy (id, tenant_id, skill_key, engaged, reason, engaged_by, engaged_at)
	                  VALUES ('01234567-89ab-7cde-8f01-2345678900c1'::uuid, $1::uuid, NULL, TRUE, 'tenant A paused', $2::uuid, now())`, tenantA, operatorGCID)
	require.NoError(t, err)

	v, err := repo.CheckSuspension(context.Background(), tenantA, actionBasic)
	require.NoError(t, err)
	assert.True(t, v.Suspended)
	assert.Equal(t, "tenant", v.Scope)

	v, err = repo.CheckSuspension(context.Background(), tenantB, actionBasic)
	require.NoError(t, err)
	assert.False(t, v.Suspended, "tenant B is not contained by tenant A's row")
}

func TestIntegration_Suspension_TenantScope_OneSkill(t *testing.T) {
	db, cleanup := setupPG(t)
	defer cleanup()
	repo, err := pg.New(db)
	require.NoError(t, err)

	_, err = db.Exec(`INSERT INTO companion_suspension_policy (id, tenant_id, skill_key, engaged, reason, engaged_by, engaged_at)
	                  VALUES ('01234567-89ab-7cde-8f01-2345678900c2'::uuid, $1::uuid, $2, TRUE, 'skill paused', $3::uuid, now())`, tenantA, actionCite, operatorGCID)
	require.NoError(t, err)

	v, err := repo.CheckSuspension(context.Background(), tenantA, actionCite)
	require.NoError(t, err)
	assert.True(t, v.Suspended)

	v, err = repo.CheckSuspension(context.Background(), tenantA, actionBasic)
	require.NoError(t, err)
	assert.False(t, v.Suspended)
}

func TestIntegration_Suspension_MissingTable_ReturnsError(t *testing.T) {
	db, cleanup := setupPG(t)
	defer cleanup()
	repo, err := pg.New(db)
	require.NoError(t, err)

	// Simulate the migration not having run: the read must ERROR (the decorator
	// denies), never read as "nobody is suspended".
	_, err = db.Exec(`DROP TABLE platform_companion_suspension`)
	require.NoError(t, err)

	_, err = repo.CheckSuspension(context.Background(), tenantA, actionBasic)
	require.Error(t, err)
}

func TestIntegration_Suspension_BadTenant_Rejected(t *testing.T) {
	db, cleanup := setupPG(t)
	defer cleanup()
	repo, err := pg.New(db)
	require.NoError(t, err)

	_, err = repo.CheckSuspension(context.Background(), "not-a-uuid", actionBasic)
	require.Error(t, err)
	_, err = repo.CheckSuspension(context.Background(), "", actionBasic)
	require.Error(t, err)
}
