// pg_violation.go: implements domain.ViolationPublisher (ADR-152 amendment
// 2026-08-07, requirement G2) against chora_observability.
//
// A Cloud Model Armor BLOCK at the gateway is enqueued onto the SAME
// outbox_events table the token-usage ledger and the ADR-231 egress audit use;
// the shared chora-observability dispatcher publishes it by row.Topic onto
// chora.governance.policy.violation_detected.v1.
//
// WIRE CONSTRAINT. That topic is bound to a BINARY PROTOCOL_BUFFER Pub/Sub
// schema (chora-governance-policy-violation_detected-v1). The payload is
// therefore a proto.Marshal of governance.v1.PolicyViolationDetected. A JSON
// body 400s AT PUBLISH and never reaches a DLQ. This is the same constraint
// that refutes the originally ratified sink design: a Cloud Logging sink
// publishes a LogEntry, which the schema rejects and which carries none of the
// mandatory envelope fields. See
// docs/familiar/evidence/2026-08-07/g2-armor-sink.txt §2.
//
// FIELD MAPPING NOTE. The ratified proto has no agent, leg or verdict field,
// and adding one requires a Pub/Sub schema revision (an additive proto field
// 400s at publish until the revision lands), so those three facts ride in the
// description built by domain.PolicyViolationEvent.Description().
package pg

import (
	"context"
	"encoding/json"
	"fmt"
	"strconv"
	"time"

	commonv1 "github.com/apollo-chora/chora-contracts/gen/go/chora/common/v1"
	governancev1 "github.com/apollo-chora/chora-contracts/gen/go/chora/governance/v1"
	"google.golang.org/protobuf/proto"
	"google.golang.org/protobuf/types/known/timestamppb"

	"github.com/5007-Capstone/chora/services/chora-model-gateway/internal/domain"
)

// policyViolationTopic is the canonical topic the violation event publishes to.
const policyViolationTopic = "chora.governance.policy.violation_detected.v1"

// violationDetector is the UPPER_SNAKE_CASE detector identity the governance
// evidence row carries, per the proto's documented detector taxonomy.
const violationDetector = "MODEL_ARMOR"

// violationImdaDimension routes the event to IMDA D3. The governance projector
// sends safety_and_robustness evidence whose event type contains
// "policy_violation" into policy_violation_log (ADR-141 canonical label).
const violationImdaDimension = "safety_and_robustness"

// violationLifecycleStage marks this as runtime-stage evidence (the block
// happened while serving traffic, not in CI or pre-deploy).
const violationLifecycleStage = "runtime"

// EnqueuePolicyViolationDetected implements domain.ViolationPublisher.
func (r *Repo) EnqueuePolicyViolationDetected(ctx context.Context, evt domain.PolicyViolationEvent) error {
	if evt.TenantID == "" {
		return fmt.Errorf("pg: tenant_id required")
	}
	if evt.ViolationID == "" {
		return fmt.Errorf("pg: policy_violation violation_id required (idempotency key)")
	}
	detectedAt := evt.DetectedAt
	if detectedAt.IsZero() {
		detectedAt = time.Now().UTC()
	}

	payload, err := buildPolicyViolationPayload(evt, detectedAt)
	if err != nil {
		return fmt.Errorf("pg: marshal policy_violation payload: %w", err)
	}
	envelopeBytes, err := json.Marshal(buildViolationEnvelope(evt, detectedAt))
	if err != nil {
		return fmt.Errorf("pg: marshal policy_violation envelope: %w", err)
	}

	tx, err := r.db.BeginTx(ctx, nil)
	if err != nil {
		return fmt.Errorf("pg: begin EnqueuePolicyViolationDetected txn: %w", err)
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
		evt.ViolationID,
		evt.TenantID,
		evt.GCID,
		evt.AgentID,
		"policy_violation",
		evt.ViolationID,
		policyViolationTopic,
		policyViolationTopic,
		payload,
		string(envelopeBytes),
		detectedAt,
		evt.ViolationID, // idempotency_key == violation_id (one per blocked request)
	); err != nil {
		return fmt.Errorf("pg: policy_violation outbox insert: %w", err)
	}
	return tx.Commit()
}

// buildPolicyViolationPayload marshals the domain violation into the canonical
// governance.v1.PolicyViolationDetected proto (with its embedded EventEnvelope).
func buildPolicyViolationPayload(evt domain.PolicyViolationEvent, detectedAt time.Time) ([]byte, error) {
	rec := &governancev1.PolicyViolationDetected{
		Envelope: &commonv1.EventEnvelope{
			EventId:            evt.ViolationID,
			IdempotencyKey:     evt.ViolationID,
			TenantId:           evt.TenantID,
			Gcid:               evt.GCID,
			OccurredAt:         timestamppb.New(detectedAt),
			Traceparent:        evt.Traceparent,
			Tracestate:         evt.Tracestate,
			SourceService:      sourceService,
			SchemaVersion:      envelopeSchemaVersion,
			ChoraImdaDimension: violationImdaDimension,
			ImdaLifecycleStage: violationLifecycleStage,
		},
		ViolationId: evt.ViolationID,
		// The Armor template IS the policy that fired.
		PolicyId:   evt.ArmorTemplate,
		PolicyKind: governancev1.PolicyKind_POLICY_KIND_AI_USAGE,
		// A BLOCK already refused the call, so the row is for immediate review
		// rather than an auto-suspend pager (which is what CRITICAL means).
		ViolationSeverity: governancev1.ViolationSeverity_VIOLATION_SEVERITY_HIGH,
		SubjectGcid:       evt.GCID,
		ResourceUri:       evt.ResourceURI(),
		Description:       evt.Description(),
		Detector:          violationDetector,
		DetectedAt:        timestamppb.New(detectedAt),
	}
	return proto.Marshal(rec)
}

// buildViolationEnvelope builds the FLAT canonical snake_case envelope map for
// the outbox envelope JSONB column (same reader contract as the token-usage and
// egress-audit envelopes). Every value is a string so the strict dispatcher
// reader parses it.
func buildViolationEnvelope(evt domain.PolicyViolationEvent, detectedAt time.Time) map[string]string {
	return map[string]string{
		"event_id":             evt.ViolationID,
		"idempotency_key":      evt.ViolationID,
		"tenant_id":            evt.TenantID,
		"gcid":                 evt.GCID,
		"traceparent":          evt.Traceparent,
		"tracestate":           evt.Tracestate,
		"occurred_at":          detectedAt.UTC().Format(time.RFC3339Nano),
		"source_service":       sourceService,
		"schema_version":       strconv.Itoa(envelopeSchemaVersion),
		"chora_imda_dimension": violationImdaDimension,
		"imda_lifecycle_stage": violationLifecycleStage,
	}
}
