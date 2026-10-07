// pg_grounded.go — implements domain.ExternalEgressGate + domain.EgressAuditWriter
// (ADR-231 D6) against chora_observability. The gateway reads the fail-closed
// external-egress governance (kill-switch + per-tenant entitlement + daily
// ceiling) here, and emits the external_egress audit event through the SAME
// outbox_events table the token-usage ledger uses (the dispatcher publishes by
// row.Topic — chora.governance.audit.external_egress.v1).
//
// Tables come from migration 0014_external_egress_policy. Until that migration
// is applied the reads fail (missing relation) ⇒ the gateway fails CLOSED (all
// grounded egress denied) — the safe default while the Seekers are DARK.
package pg

import (
	"context"
	"database/sql"
	"encoding/json"
	"errors"
	"fmt"
	"strconv"
	"time"

	commonv1 "github.com/apollo-chora/chora-contracts/gen/go/chora/common/v1"
	governancev1 "github.com/apollo-chora/chora-contracts/gen/go/chora/governance/v1"
	"google.golang.org/protobuf/proto"
	"google.golang.org/protobuf/types/known/timestamppb"

	"github.com/apollo-chora/chora-model-gateway/internal/domain"
)

// externalEgressTopic is the canonical topic the egress audit event publishes to.
const externalEgressTopic = "chora.governance.audit.external_egress.v1"

// Authorize — implements domain.ExternalEgressGate. Fail-closed: any read error
// propagates so the service denies (never grants egress on an unreadable store).
func (r *Repo) Authorize(ctx context.Context, tenantID string) (domain.EgressAuthorization, error) {
	if tenantID == "" {
		return domain.EgressAuthorization{}, fmt.Errorf("pg: tenant_id required")
	}
	tx, err := r.db.BeginTx(ctx, &sql.TxOptions{ReadOnly: true})
	if err != nil {
		return domain.EgressAuthorization{}, fmt.Errorf("pg: begin Authorize txn: %w", err)
	}
	defer func() { _ = tx.Rollback() }()
	if err := setTenantContext(ctx, tx, tenantID); err != nil {
		return domain.EgressAuthorization{}, err
	}

	// 1. Platform-wide O+ kill-switch (no RLS — platform-level).
	var engaged bool
	err = tx.QueryRowContext(ctx, `SELECT engaged FROM platform_egress_killswitch LIMIT 1`).Scan(&engaged)
	if err != nil && !errors.Is(err, sql.ErrNoRows) {
		return domain.EgressAuthorization{}, fmt.Errorf("pg: read kill-switch: %w", err)
	}
	if engaged {
		return domain.EgressAuthorization{Allowed: false, Reason: domain.EgressDenyKillSwitch}, nil
	}

	// 2. Per-tenant external_egress entitlement + daily ceiling. NO row ⇒ OFF
	//    (ADR-220 D4 franchise default).
	var enabled bool
	var ceiling int
	err = tx.QueryRowContext(ctx,
		`SELECT egress_enabled, daily_call_ceiling FROM external_egress_policy WHERE tenant_id = $1::uuid`,
		tenantID,
	).Scan(&enabled, &ceiling)
	if errors.Is(err, sql.ErrNoRows) {
		return domain.EgressAuthorization{Allowed: false, Reason: domain.EgressDenyDisabled}, nil
	}
	if err != nil {
		return domain.EgressAuthorization{}, fmt.Errorf("pg: read external_egress_policy: %w", err)
	}
	if !enabled {
		return domain.EgressAuthorization{Allowed: false, Reason: domain.EgressDenyDisabled}, nil
	}

	// 3. Daily grounded-call ceiling.
	var count int
	err = tx.QueryRowContext(ctx,
		`SELECT call_count FROM grounded_search_daily_usage WHERE tenant_id = $1::uuid AND usage_date = current_date`,
		tenantID,
	).Scan(&count)
	if err != nil && !errors.Is(err, sql.ErrNoRows) {
		return domain.EgressAuthorization{}, fmt.Errorf("pg: read daily usage: %w", err)
	}
	if count >= ceiling {
		return domain.EgressAuthorization{Allowed: false, Reason: domain.EgressDenyDailyCeil}, nil
	}

	return domain.EgressAuthorization{Allowed: true}, tx.Commit()
}

// RecordEgress — implements domain.ExternalEgressGate. Increments the tenant's
// grounded-call counter for today (post-success), upserting the per-day row.
func (r *Repo) RecordEgress(ctx context.Context, tenantID string) error {
	if tenantID == "" {
		return fmt.Errorf("pg: tenant_id required")
	}
	tx, err := r.db.BeginTx(ctx, nil)
	if err != nil {
		return fmt.Errorf("pg: begin RecordEgress txn: %w", err)
	}
	defer func() { _ = tx.Rollback() }()
	if err := setTenantContext(ctx, tx, tenantID); err != nil {
		return err
	}
	const q = `
		INSERT INTO grounded_search_daily_usage (tenant_id, usage_date, call_count)
		VALUES ($1::uuid, current_date, 1)
		ON CONFLICT (tenant_id, usage_date)
		DO UPDATE SET call_count = grounded_search_daily_usage.call_count + 1,
		              updated_at = now()
	`
	if _, err := tx.ExecContext(ctx, q, tenantID); err != nil {
		return fmt.Errorf("pg: increment daily usage: %w", err)
	}
	return tx.Commit()
}

// EnqueueExternalEgressAudited — implements domain.EgressAuditWriter. Inserts the
// ExternalEgressAudited event onto outbox_events with the governance topic; the
// shared chora-observability dispatcher publishes it by row.Topic (ADR-231 D6).
func (r *Repo) EnqueueExternalEgressAudited(ctx context.Context, evt domain.ExternalEgressAuditEvent) error {
	if evt.TenantID == "" {
		return fmt.Errorf("pg: tenant_id required")
	}
	occurredAt := evt.OccurredAt
	if occurredAt.IsZero() {
		occurredAt = time.Now().UTC()
	}
	auditID := evt.AuditID
	if auditID == "" {
		return fmt.Errorf("pg: external_egress audit_id required (idempotency key)")
	}

	payload, err := buildExternalEgressPayload(evt, occurredAt)
	if err != nil {
		return fmt.Errorf("pg: marshal external_egress payload: %w", err)
	}
	envelopeBytes, err := json.Marshal(buildEgressAuditEnvelope(evt, occurredAt))
	if err != nil {
		return fmt.Errorf("pg: marshal external_egress envelope: %w", err)
	}

	tx, err := r.db.BeginTx(ctx, nil)
	if err != nil {
		return fmt.Errorf("pg: begin EnqueueExternalEgressAudited txn: %w", err)
	}
	defer func() { _ = tx.Rollback() }()
	// Both tenant GUCs: the outbox RLS policy keys off app.current_tenant
	// (migration 0006); chora.tenant_id rides along for the shared-tenant
	// convention the token-usage path sets.
	if err := setOutboxTenantContext(ctx, tx, evt.TenantID); err != nil {
		return err
	}
	const q = `
		INSERT INTO ` + outboxTableName + ` (
			id, tenant_id, gcid, agid, aggregate_type, aggregate_id, event_type, topic,
			payload, envelope, occurred_at, status, retry_count,
			idempotency_key
		)
		VALUES (
			$1, $2, $3, $4, $5, $6, $7, $8,
			$9, $10::jsonb, $11, 'pending', 0,
			$12
		)
	`
	if _, err := tx.ExecContext(ctx, q,
		auditID,
		evt.TenantID,
		evt.ActorGCID,
		evt.AgentID,
		"external_egress_audit",
		auditID,
		externalEgressTopic,
		externalEgressTopic,
		payload,
		string(envelopeBytes),
		occurredAt,
		auditID, // idempotency_key == audit_id (per-call unique)
	); err != nil {
		return fmt.Errorf("pg: external_egress outbox insert: %w", err)
	}
	return tx.Commit()
}

// buildExternalEgressPayload marshals the domain audit event into the canonical
// governance.v1.ExternalEgressAudited proto (with its embedded EventEnvelope).
func buildExternalEgressPayload(evt domain.ExternalEgressAuditEvent, occurredAt time.Time) ([]byte, error) {
	rec := &governancev1.ExternalEgressAudited{
		Envelope: &commonv1.EventEnvelope{
			EventId:            evt.AuditID,
			IdempotencyKey:     evt.AuditID,
			TenantId:           evt.TenantID,
			Gcid:               evt.ActorGCID,
			OccurredAt:         timestamppb.New(occurredAt),
			Traceparent:        evt.Traceparent,
			Tracestate:         evt.Tracestate,
			SourceService:      sourceService,
			SchemaVersion:      envelopeSchemaVersion,
			ChoraImdaDimension: "transparency", // IMDA D2 — external-egress evidence
		},
		AuditId:               evt.AuditID,
		ActorGcid:             evt.ActorGCID,
		AgentId:               evt.AgentID,
		ActionCode:            evt.ActionCode,
		DirectiveHash:         evt.DirectiveHash,
		WebSearchQueries:      evt.WebSearchQueries,
		CitationCount:         evt.CitationCount,
		Result:                governancev1.AuditResult(int32(evt.Result)), // #nosec G115 -- small bounded enum
		DenialReason:          evt.DenialReason,
		ModelArmorVerdictPre:  evt.ArmorVerdictPre,
		ModelArmorVerdictPost: evt.ArmorVerdictPost,
		Vendor:                evt.Vendor,
		ModelVersion:          evt.ModelVersion,
		OccurredAt:            timestamppb.New(occurredAt),
	}
	return proto.Marshal(rec)
}

// buildEgressAuditEnvelope builds the FLAT canonical snake_case envelope map for
// the outbox envelope JSONB column (same reader contract as the token-usage
// envelope). Every value is a string so the strict dispatcher reader parses it.
func buildEgressAuditEnvelope(evt domain.ExternalEgressAuditEvent, occurredAt time.Time) map[string]string {
	return map[string]string{
		"event_id":             evt.AuditID,
		"idempotency_key":      evt.AuditID,
		"tenant_id":            evt.TenantID,
		"gcid":                 evt.ActorGCID,
		"traceparent":          evt.Traceparent,
		"tracestate":           evt.Tracestate,
		"occurred_at":          occurredAt.UTC().Format(time.RFC3339Nano),
		"source_service":       sourceService,
		"schema_version":       strconv.Itoa(envelopeSchemaVersion),
		"chora_imda_dimension": "transparency",
	}
}
