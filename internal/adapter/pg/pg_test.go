//go:build integration

// Integration tests for the pg adapter — exercise real Postgres via auto-
// started postgres:18-alpine container. Tagged `integration` so the unit-
// test stage (no -tags) skips this file, and the integration-test stage
// (-tags=integration) runs it. Cloud Build's integration-test stage installs
// docker.io (per cloudbuild.yaml comment block) so the docker.LookPath +
// `docker run` invocations below succeed against the host docker daemon
// mounted at /var/run/docker.sock.
//
// Per [[feedback-no-stubs-real-wiring]] — these are REAL postgres tests,
// not sqlmock. Coverage boost from ~2% (constructor only) to 70-90% (every
// public method + RLS path exercised). Achieves the 85% total coverage gate.

package pg_test

import (
	"context"
	"database/sql"
	"fmt"
	"net"
	"os"
	"os/exec"
	"strings"
	"testing"
	"time"

	_ "github.com/jackc/pgx/v5/stdlib"

	"github.com/apollo-chora/chora-model-gateway/internal/adapter/pg"
	"github.com/apollo-chora/chora-model-gateway/internal/domain"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// Test tenants — UUIDv7-shaped.
const (
	tenantA = "01234567-89ab-7cde-8f01-234567890aaa"
	tenantB = "01234567-89ab-7cde-8f01-234567890bbb"
)

// ----------------------------------------------------------------------------
// Pure-function tests (no DB)
// ----------------------------------------------------------------------------

func TestNew_NilDB(t *testing.T) {
	_, err := pg.New(nil)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "*sql.DB required")
}

// ----------------------------------------------------------------------------
// Integration tests against a real Postgres testcontainer.
//
// Auto-starts postgres:18-alpine via docker if available. Skips when
// docker is missing OR TEST_PG_SKIP is set (so unit-test-only CI runs
// stay fast). Same pattern as the Phase 0.2 migration validation.
// ----------------------------------------------------------------------------

func setupPG(t *testing.T) (*sql.DB, func()) {
	t.Helper()
	if os.Getenv("TEST_PG_SKIP") == "1" {
		t.Skip("TEST_PG_SKIP=1 — skipping integration tests")
	}
	// External-server mode: TEST_PG_DSN names an admin DSN on an already
	// running Postgres (a local cluster, a CI service). Each test gets its
	// own freshly created database, migrated from the real .sql files and
	// dropped on cleanup, so the isolation matches the per-test container.
	// Needed where docker is absent (developer laptops without a daemon);
	// the docker path below stays the Cloud Build shape.
	if adminDSN := os.Getenv("TEST_PG_DSN"); adminDSN != "" {
		return setupExternalPG(t, adminDSN)
	}
	if _, err := exec.LookPath("docker"); err != nil {
		t.Skip("docker not available — skipping integration tests")
	}

	// Docker-network mode (Cloud Build): if TEST_PG_DOCKER_NETWORK is set,
	// start postgres on the named docker network + connect by container name
	// (port 5432, no -p host mapping needed). Required for Cloud Build's
	// docker-out-of-docker — the test container's `127.0.0.1` isn't the host
	// loopback, so the legacy `-p random:5432 + 127.0.0.1` pattern fails
	// closed in CB even though postgres starts fine.
	//
	// Local mode (default): existing `-p freePort:5432 + 127.0.0.1` pattern.
	network := os.Getenv("TEST_PG_DOCKER_NETWORK")
	containerName := fmt.Sprintf("pg-test-%d", time.Now().UnixNano())

	var (
		runArgs      []string
		dsn          string
		readyTimeout = 30 * time.Second
	)
	if network != "" {
		runArgs = []string{
			"run", "-d", "--rm",
			"--name", containerName,
			"--network", network,
			"-e", "POSTGRES_PASSWORD=test",
			"-e", "POSTGRES_DB=chora_observability",
			"postgres:18-alpine",
		}
		dsn = fmt.Sprintf("postgres://postgres:test@%s:5432/chora_observability?sslmode=disable", containerName)
		// Cloud Build sometimes pulls the image fresh; allow extra time.
		readyTimeout = 90 * time.Second
	} else {
		port := freePort(t)
		runArgs = []string{
			"run", "-d", "--rm",
			"--name", containerName,
			"-e", "POSTGRES_PASSWORD=test",
			"-e", "POSTGRES_DB=chora_observability",
			"-p", fmt.Sprintf("%d:5432", port),
			"postgres:18-alpine",
		}
		dsn = fmt.Sprintf("postgres://postgres:test@127.0.0.1:%d/chora_observability?sslmode=disable", port)
	}

	containerID := strings.TrimSpace(runDocker(t, runArgs...))
	cleanup := func() {
		_ = exec.Command("docker", "stop", containerID).Run()
	}

	// Wait for Postgres readiness.
	deadline := time.Now().Add(readyTimeout)
	var db *sql.DB
	for time.Now().Before(deadline) {
		d, err := sql.Open("pgx", dsn)
		if err == nil {
			if perr := d.Ping(); perr == nil {
				db = d
				break
			}
			_ = d.Close()
		}
		time.Sleep(500 * time.Millisecond)
	}
	if db == nil {
		cleanup()
		t.Fatalf("postgres testcontainer did not become ready within %s (network=%q, dsn=%s)", readyTimeout, network, dsn)
	}
	if err := applyMigrations(db); err != nil {
		_ = db.Close()
		cleanup()
		t.Fatalf("apply migrations: %v", err)
	}
	return db, func() { _ = db.Close(); cleanup() }
}

// setupExternalPG creates a throwaway database on an external server, applies
// the migrations, and returns a handle whose cleanup drops the database.
func setupExternalPG(t *testing.T, adminDSN string) (*sql.DB, func()) {
	t.Helper()
	admin, err := sql.Open("pgx", adminDSN)
	require.NoError(t, err)
	require.NoError(t, admin.Ping(), "TEST_PG_DSN must point at a running Postgres")

	dbName := fmt.Sprintf("mg_test_%d", time.Now().UnixNano())
	// Roles are CLUSTER-global: a per-test docker cluster never sees the
	// chora_observability_app_rw role another test created, but a shared
	// server does. Drop a stale one (best-effort: its only privileges lived
	// in databases already dropped) so the RLS test's CREATE ROLE still works.
	_, _ = admin.Exec(`DROP ROLE IF EXISTS chora_observability_app_rw`)
	_, err = admin.Exec(`CREATE DATABASE ` + dbName)
	require.NoError(t, err)

	// Re-point the DSN at the new database (replace the trailing path segment).
	slash := strings.LastIndex(adminDSN, "/")
	q := strings.Index(adminDSN[slash:], "?")
	var dsn string
	if q >= 0 {
		dsn = adminDSN[:slash+1] + dbName + adminDSN[slash+q:]
	} else {
		dsn = adminDSN[:slash+1] + dbName
	}
	db, err := sql.Open("pgx", dsn)
	require.NoError(t, err)
	require.NoError(t, db.Ping())

	cleanup := func() {
		_ = db.Close()
		_, _ = admin.Exec(`DROP DATABASE IF EXISTS ` + dbName + ` WITH (FORCE)`)
		_, _ = admin.Exec(`DROP ROLE IF EXISTS chora_observability_app_rw`)
		_ = admin.Close()
	}
	if err := applyMigrations(db); err != nil {
		cleanup()
		t.Fatalf("apply migrations: %v", err)
	}
	return db, cleanup
}

func freePort(t *testing.T) int {
	t.Helper()
	l, err := net.Listen("tcp", "127.0.0.1:0")
	require.NoError(t, err)
	defer l.Close()
	return l.Addr().(*net.TCPAddr).Port
}

func runDocker(t *testing.T, args ...string) string {
	t.Helper()
	cmd := exec.Command("docker", args...)
	out, err := cmd.CombinedOutput()
	if err != nil {
		t.Fatalf("docker %s failed: %s\noutput: %s", strings.Join(args, " "), err, out)
	}
	return string(out)
}

// applyMigrations runs 0001_initial.sql + 0003_outbox.sql + 0008.up.sql.
// We can't shell out to a `migrate` CLI because that adds a heavy
// dependency; instead we read the .sql files directly and exec them.
func applyMigrations(db *sql.DB) error {
	repoRoot := "../../../.." // pg → adapter → internal → chora-model-gateway → repo root
	files := []string{
		repoRoot + "/chora-observability/migrations/0001_initial.sql",
		repoRoot + "/chora-observability/migrations/0003_outbox.sql",
		// 0006 adds outbox_events.idempotency_key NOT NULL UNIQUE column +
		// producer-side dedupe index. pg.go insertOutboxRow writes this
		// column on every TokenUsageRecorded insert per [[event-driven]] +
		// canonical event envelope mandatory fields (per CLAUDE.md §6).
		// Missing this migration causes `column "idempotency_key" of relation
		// "outbox_events" does not exist (SQLSTATE 42703)` — surfaced 2026-05-26.
		repoRoot + "/chora-observability/migrations/0006_outbox_d6_canonical.up.sql",
		repoRoot + "/chora-observability/migrations/0008_per_tenant_llm_budget.up.sql",
		// 0014 adds the external_egress governance tables (kill-switch +
		// per-tenant entitlement + daily counter) the GroundedSearch chain
		// reads/writes (ADR-231 D6). Applying it here prepare-smokes the
		// migration SQL AND lets pg_grounded_test.go exercise the adapter.
		repoRoot + "/chora-observability/migrations/0014_external_egress_policy.up.sql",
		// 0018 adds the two ADR-252 companion-suspension tables the Invoke
		// chain reads uncached, deny-before-debit (ADR-254 D7); 0019 adds the
		// R22 keyed debit-claim table. A test that forgets one here skips
		// silently green, so both are listed the moment they exist.
		repoRoot + "/chora-observability/migrations/0018_companion_suspension.up.sql",
		repoRoot + "/chora-observability/migrations/0019_invoke_debit_claims.up.sql",
	}
	for _, f := range files {
		raw, err := os.ReadFile(f)
		if err != nil {
			return fmt.Errorf("read %s: %w", f, err)
		}
		if _, err := db.Exec(string(raw)); err != nil {
			return fmt.Errorf("apply %s: %w", f, err)
		}
	}
	return nil
}

// ----------------------------------------------------------------------------
// Integration: GetTenantBudget
// ----------------------------------------------------------------------------

func TestIntegration_GetTenantBudget_ReturnsActiveWindow(t *testing.T) {
	db, cleanup := setupPG(t)
	defer cleanup()
	repo, err := pg.New(db)
	require.NoError(t, err)

	// Seed an active window for tenant A.
	_, err = db.Exec(`
		INSERT INTO per_tenant_llm_budget
		(budget_id, tenant_id, budget_period_start, budget_period_end, budget_usd_micros, spent_usd_micros, policy)
		VALUES (gen_random_uuid(), $1::uuid, now() - interval '1 hour', now() + interval '23 hours', 1000000, 12500, 'block')
	`, tenantA)
	require.NoError(t, err)

	state, err := repo.GetTenantBudget(context.Background(), tenantA)
	require.NoError(t, err)
	require.NotNil(t, state)
	assert.Equal(t, int64(1_000_000), state.BudgetUSDMicros)
	assert.Equal(t, int64(12_500), state.SpentUSDMicros)
	assert.Equal(t, domain.BudgetPolicyBlock, state.Policy)
}

func TestIntegration_GetTenantBudget_NoWindow_ReturnsNil(t *testing.T) {
	db, cleanup := setupPG(t)
	defer cleanup()
	repo, err := pg.New(db)
	require.NoError(t, err)

	state, err := repo.GetTenantBudget(context.Background(), tenantA)
	require.NoError(t, err)
	assert.Nil(t, state)
}

// ----------------------------------------------------------------------------
// Integration: DebitAndEnqueue — atomic
// ----------------------------------------------------------------------------

func TestIntegration_DebitAndEnqueue_AtomicSuccess(t *testing.T) {
	db, cleanup := setupPG(t)
	defer cleanup()
	repo, err := pg.New(db)
	require.NoError(t, err)

	_, err = db.Exec(`
		INSERT INTO per_tenant_llm_budget
		(tenant_id, budget_period_start, budget_period_end, budget_usd_micros, spent_usd_micros, policy)
		VALUES ($1::uuid, now() - interval '1 hour', now() + interval '23 hours', 1000000, 0, 'block')
	`, tenantA)
	require.NoError(t, err)

	evt := domain.TokenUsageEvent{
		UsageID:        "01234567-89ab-7cde-8f01-234567890ee1",
		TenantID:       tenantA,
		GCID:           "01234567-89ab-7cde-8f01-234567890e11",
		ModelID:        "gemini-2.5-pro",
		InputTokens:    100,
		OutputTokens:   50,
		CostMicros:     362,
		InvocationID:   "01234567-89ab-7cde-8f01-234567890ee1",
		AgentRole:      "qgen_question",
		RecordedAt:     time.Now().UTC(),
		Vendor:         "vertex_ai_gemini",
		FallbackChain:  []string{"vertex_ai_gemini:gemini-2.5-pro"},
		ArmorPre:       domain.ArmorVerdictAllow,
		ArmorPost:      domain.ArmorVerdictAllow,
		GatewayVersion: "chora-model-gateway:test",
		Traceparent:    "00-trace-span-01",
	}

	err = repo.DebitAndEnqueue(context.Background(), evt, 362)
	require.NoError(t, err)

	// Verify budget debited.
	var spent int64
	err = db.QueryRow(`SELECT spent_usd_micros FROM per_tenant_llm_budget WHERE tenant_id = $1::uuid`, tenantA).Scan(&spent)
	require.NoError(t, err)
	assert.Equal(t, int64(362), spent)

	// Verify outbox row inserted. tenant_id lives inside envelope JSONB
	// per the chora-go-common/outbox schema.
	var outboxCount int
	err = db.QueryRow(`SELECT count(*) FROM outbox_events WHERE envelope->>'tenant_id' = $1 AND event_type = 'chora.observability.token_usage.recorded.v1'`, tenantA).Scan(&outboxCount)
	require.NoError(t, err)
	assert.Equal(t, 1, outboxCount)
}

// ----------------------------------------------------------------------------
// Integration: RLS isolation — tenant A cannot see tenant B's budget
// ----------------------------------------------------------------------------

func TestIntegration_RLS_TenantIsolation(t *testing.T) {
	db, cleanup := setupPG(t)
	defer cleanup()

	// Create a non-bypass-RLS role (the migrate user owns + bypasses; the
	// chora_app_user equivalent in production is the one the gateway pod's
	// SA assumes). We simulate that by manually toggling row_security off
	// for the test session before checking RLS — wait, that's the wrong
	// direction. Better: create a non-superuser, GRANT it, SET ROLE to it.
	_, err := db.Exec(`CREATE ROLE chora_observability_app_rw NOLOGIN`)
	require.NoError(t, err)
	_, err = db.Exec(`GRANT USAGE ON SCHEMA public TO chora_observability_app_rw`)
	require.NoError(t, err)
	_, err = db.Exec(`GRANT SELECT, INSERT, UPDATE ON ALL TABLES IN SCHEMA public TO chora_observability_app_rw`)
	require.NoError(t, err)
	_, err = db.Exec(`GRANT USAGE, SELECT, UPDATE ON ALL SEQUENCES IN SCHEMA public TO chora_observability_app_rw`)
	require.NoError(t, err)

	// Seed both tenants from the migrate role (which bypasses RLS).
	_, err = db.Exec(`
		INSERT INTO per_tenant_llm_budget (tenant_id, budget_period_start, budget_period_end, budget_usd_micros, policy)
		VALUES ($1::uuid, now() - interval '1 hour', now() + interval '23 hours', 1000000, 'block'),
		       ($2::uuid, now() - interval '1 hour', now() + interval '23 hours', 2000000, 'alert')
	`, tenantA, tenantB)
	require.NoError(t, err)

	// Switch to the app-rw role + try to read tenant A's budget with
	// the RLS tenant context set. Should succeed.
	ctx := context.Background()
	tx, err := db.BeginTx(ctx, nil)
	require.NoError(t, err)
	defer tx.Rollback()
	_, err = tx.ExecContext(ctx, `SET LOCAL ROLE chora_observability_app_rw`)
	require.NoError(t, err)
	_, err = tx.ExecContext(ctx, fmt.Sprintf("SET LOCAL chora.tenant_id = '%s'", tenantA))
	require.NoError(t, err)

	var seen int
	err = tx.QueryRowContext(ctx, `SELECT count(*) FROM per_tenant_llm_budget`).Scan(&seen)
	require.NoError(t, err)
	// RLS enforces the policy: app_rw sees ONLY tenant A's row.
	assert.Equal(t, 1, seen)
}

// ----------------------------------------------------------------------------
// SET LOCAL UUID validation — reject SQL-injection-style tenant_id
// ----------------------------------------------------------------------------

func TestIntegration_GetTenantBudget_RejectsInvalidUUID(t *testing.T) {
	db, cleanup := setupPG(t)
	defer cleanup()
	repo, err := pg.New(db)
	require.NoError(t, err)
	_, err = repo.GetTenantBudget(context.Background(), "'; DROP TABLE per_tenant_llm_budget; --")
	require.Error(t, err)
	assert.Contains(t, err.Error(), "invalid tenant_id")
}

func TestIntegration_DebitSpent_NoOpOnZeroDelta(t *testing.T) {
	db, cleanup := setupPG(t)
	defer cleanup()
	repo, err := pg.New(db)
	require.NoError(t, err)
	err = repo.DebitSpent(context.Background(), tenantA, 0)
	require.NoError(t, err)
	err = repo.DebitSpent(context.Background(), tenantA, -5)
	require.NoError(t, err) // negative deltas no-op too
}

func TestIntegration_GetTenantBudget_EmptyTenantID(t *testing.T) {
	db, cleanup := setupPG(t)
	defer cleanup()
	repo, err := pg.New(db)
	require.NoError(t, err)
	_, err = repo.GetTenantBudget(context.Background(), "")
	require.Error(t, err)
	assert.Contains(t, err.Error(), "tenant_id required")
}

// ----------------------------------------------------------------------------
// EnqueueTokenUsageRecorded standalone path
// ----------------------------------------------------------------------------

func TestIntegration_EnqueueTokenUsageRecorded_Standalone(t *testing.T) {
	db, cleanup := setupPG(t)
	defer cleanup()
	repo, err := pg.New(db)
	require.NoError(t, err)

	evt := domain.TokenUsageEvent{
		UsageID:        "01234567-89ab-7cde-8f01-234567890ee2",
		TenantID:       tenantA,
		GCID:           "01234567-89ab-7cde-8f01-234567890e22",
		ModelID:        "gemini-2.5-pro",
		CostMicros:     100,
		InvocationID:   "01234567-89ab-7cde-8f01-234567890ee2",
		AgentRole:      "qgen_question",
		Vendor:         "vertex_ai_gemini",
		FallbackChain:  []string{"vertex_ai_gemini:gemini-2.5-pro"},
		ArmorPre:       domain.ArmorVerdictAllow,
		ArmorPost:      domain.ArmorVerdictAllow,
		GatewayVersion: "test",
	}
	err = repo.EnqueueTokenUsageRecorded(context.Background(), evt)
	require.NoError(t, err)
	var count int
	err = db.QueryRow(`SELECT count(*) FROM outbox_events WHERE envelope->>'tenant_id' = $1`, tenantA).Scan(&count)
	require.NoError(t, err)
	assert.Equal(t, 1, count)
}

// ----------------------------------------------------------------------------
// DebitAndEnqueue — debit GATED on the outbox insert (Python parity:
// database.py debit_and_enqueue runs the insert first and debits only when
// it actually landed, so a redelivery cannot debit the budget twice).
// ----------------------------------------------------------------------------

func TestIntegration_DebitAndEnqueue_Redelivery_DoesNotDoubleDebit(t *testing.T) {
	db, cleanup := setupPG(t)
	defer cleanup()
	repo, err := pg.New(db)
	require.NoError(t, err)

	_, err = db.Exec(`
		INSERT INTO per_tenant_llm_budget
		(tenant_id, budget_period_start, budget_period_end, budget_usd_micros, spent_usd_micros, policy)
		VALUES ($1::uuid, now() - interval '1 hour', now() + interval '23 hours', 1000000, 0, 'block')
	`, tenantA)
	require.NoError(t, err)

	evt := domain.TokenUsageEvent{
		UsageID:        "01234567-89ab-7cde-8f01-234567890ee3",
		TenantID:       tenantA,
		GCID:           "01234567-89ab-7cde-8f01-234567890e33",
		ModelID:        "gemini-2.5-pro",
		InputTokens:    100,
		OutputTokens:   50,
		CostMicros:     362,
		InvocationID:   "01234567-89ab-7cde-8f01-234567890ee3",
		AgentRole:      "qgen_question",
		RecordedAt:     time.Now().UTC(),
		Vendor:         "vertex_ai_gemini",
		FallbackChain:  []string{"vertex_ai_gemini:gemini-2.5-pro"},
		ArmorPre:       domain.ArmorVerdictAllow,
		ArmorPost:      domain.ArmorVerdictAllow,
		GatewayVersion: "chora-model-gateway:test",
		Traceparent:    "00-trace-span-03",
	}

	// First delivery: insert lands (RowsAffected == 1) → debits.
	require.NoError(t, repo.DebitAndEnqueue(context.Background(), evt, 362))
	// Redelivery of the same usage_id: ON CONFLICT DO NOTHING inserts nothing
	// → the debit is gated off.
	require.NoError(t, repo.DebitAndEnqueue(context.Background(), evt, 362))

	var spent int64
	require.NoError(t, db.QueryRow(`SELECT spent_usd_micros FROM per_tenant_llm_budget WHERE tenant_id = $1::uuid`, tenantA).Scan(&spent))
	assert.Equal(t, int64(362), spent, "a redelivery must not debit the budget twice")

	var rows int
	require.NoError(t, db.QueryRow(`SELECT count(*) FROM outbox_events WHERE idempotency_key = $1`, evt.UsageID).Scan(&rows))
	assert.Equal(t, 1, rows, "one outbox row per usage_id")
}

// ----------------------------------------------------------------------------
// Outbox row carries tenant_id / gcid / agid top-level columns (migration
// 0006 D6.3) — the outbox RLS policy keys off app.current_tenant and the
// tenant isolation lookup reads these columns directly.
// ----------------------------------------------------------------------------

func TestIntegration_OutboxRow_PopulatesTenantGcidAgidColumns(t *testing.T) {
	db, cleanup := setupPG(t)
	defer cleanup()
	repo, err := pg.New(db)
	require.NoError(t, err)

	evt := domain.TokenUsageEvent{
		UsageID:        "01234567-89ab-7cde-8f01-234567890ee4",
		TenantID:       tenantA,
		GCID:           "01234567-89ab-7cde-8f01-234567890e44",
		ModelID:        "gemini-2.5-pro",
		CostMicros:     100,
		InvocationID:   "01234567-89ab-7cde-8f01-234567890ee4",
		AgentRole:      "qgen_question",
		AgentID:        "qgen_question",
		ActionCode:     "question_generation",
		RecordedAt:     time.Now().UTC(),
		Vendor:         "vertex_ai_gemini",
		FallbackChain:  []string{"vertex_ai_gemini:gemini-2.5-pro"},
		ArmorPre:       domain.ArmorVerdictAllow,
		ArmorPost:      domain.ArmorVerdictAllow,
		GatewayVersion: "test",
		Traceparent:    "00-trace-span-04",
	}
	require.NoError(t, repo.EnqueueTokenUsageRecorded(context.Background(), evt))

	var tenantID, gcid, agid string
	require.NoError(t, db.QueryRow(
		`SELECT tenant_id, gcid, agid FROM outbox_events WHERE idempotency_key = $1`,
		evt.UsageID).Scan(&tenantID, &gcid, &agid))
	assert.Equal(t, tenantA, tenantID)
	assert.Equal(t, "01234567-89ab-7cde-8f01-234567890e44", gcid)
	assert.Equal(t, "qgen_question", agid)
}
