// pg_suspension.go: implements middleware.SuspensionGate (ADR-252 D1 via
// ADR-254 D7) against chora_observability migration 0018.
//
// Evaluation order copies Repo.Authorize exactly: the platform table FIRST
// (no tenant column, no RLS), then the tenant table under SET LOCAL
// chora.tenant_id. Any engaged row matching (all skills | this action_code)
// denies. Read-only transaction, uncached, one round of reads per companion
// turn. Absent rows mean NOT suspended; an unreadable store ERRORS (the
// decorator denies): the two are never confused (ADR-252 D6).
package pg

import (
	"context"
	"database/sql"
	"errors"
	"fmt"

	"github.com/apollo-chora/chora-model-gateway/internal/adapter/middleware"
)

// CheckSuspension: implements middleware.SuspensionGate.
func (r *Repo) CheckSuspension(ctx context.Context, tenantID, actionCode string) (middleware.SuspensionVerdict, error) {
	if tenantID == "" {
		return middleware.SuspensionVerdict{}, fmt.Errorf("pg: tenant_id required")
	}
	if !isValidUUID(tenantID) {
		return middleware.SuspensionVerdict{}, fmt.Errorf("pg: invalid tenant_id %q (must be UUID)", tenantID)
	}
	tx, err := r.db.BeginTx(ctx, &sql.TxOptions{ReadOnly: true})
	if err != nil {
		return middleware.SuspensionVerdict{}, fmt.Errorf("pg: begin CheckSuspension txn: %w", err)
	}
	defer func() { _ = tx.Rollback() }()

	// 1. Platform scope (no RLS). skill_key IS NULL = every companion turn.
	const platformQ = `
		SELECT COALESCE(skill_key, ''), reason
		  FROM platform_companion_suspension
		 WHERE engaged = TRUE
		   AND (skill_key IS NULL OR skill_key = $1)
		 ORDER BY (skill_key IS NULL) DESC, engaged_at DESC
		 LIMIT 1`
	var skill, reason string
	err = tx.QueryRowContext(ctx, platformQ, actionCode).Scan(&skill, &reason)
	switch {
	case err == nil:
		return middleware.SuspensionVerdict{Suspended: true, Scope: "platform", SkillKey: skill, Reason: reason}, nil
	case errors.Is(err, sql.ErrNoRows):
		// fall through to the tenant scope
	default:
		return middleware.SuspensionVerdict{}, fmt.Errorf("pg: read platform_companion_suspension: %w", err)
	}

	// 2. Tenant scope, under RLS (chora.tenant_id).
	if err := setTenantContext(ctx, tx, tenantID); err != nil {
		return middleware.SuspensionVerdict{}, err
	}
	const tenantQ = `
		SELECT COALESCE(skill_key, ''), reason
		  FROM companion_suspension_policy
		 WHERE tenant_id = $1::uuid
		   AND engaged = TRUE
		   AND (skill_key IS NULL OR skill_key = $2)
		 ORDER BY (skill_key IS NULL) DESC, engaged_at DESC
		 LIMIT 1`
	err = tx.QueryRowContext(ctx, tenantQ, tenantID, actionCode).Scan(&skill, &reason)
	switch {
	case err == nil:
		return middleware.SuspensionVerdict{Suspended: true, Scope: "tenant", SkillKey: skill, Reason: reason}, nil
	case errors.Is(err, sql.ErrNoRows):
		return middleware.SuspensionVerdict{Suspended: false}, tx.Commit()
	default:
		return middleware.SuspensionVerdict{}, fmt.Errorf("pg: read companion_suspension_policy: %w", err)
	}
}

// Compile-time check.
var _ middleware.SuspensionGate = (*Repo)(nil)
