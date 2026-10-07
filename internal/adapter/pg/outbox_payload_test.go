// outbox_payload_test.go — pure-function unit tests (NO database) for the
// Phase-2.4 protobuf payload + canonical flat envelope builders.
//
// Regression context (HANDOFF_OBSERVABILITY_OUTBOX_JAM_2026-05-29 Fix 2):
// model-gateway previously wrote payload=[]byte{} and stuffed the event body
// into a nested camelCase `event` object in the envelope JSONB. The
// chora-observability dispatcher publishes row.Payload as the Pub/Sub message
// body + proto.Unmarshal's it on the consumer, and validates the envelope
// (reconstructed from the JSONB) — which REQUIRES a non-empty traceparent.
// The old shape therefore (a) carried no decodable body and (b) failed publish
// on `traceparent is required`. These tests pin the corrected shape.
package pg

import (
	"testing"
	"time"

	commonv1 "github.com/apollo-chora/chora-contracts/gen/go/chora/common/v1"
	observabilityv1 "github.com/apollo-chora/chora-contracts/gen/go/chora/observability/v1"
	"google.golang.org/protobuf/proto"

	"github.com/5007-Capstone/chora/services/chora-model-gateway/internal/domain"
)

func sampleEvent() domain.TokenUsageEvent {
	return domain.TokenUsageEvent{
		UsageID:        "019e72bd-d27b-735c-9302-64b99c30a242",
		TenantID:       "11111111-1111-1111-1111-111111111111",
		GCID:           "22222222-2222-2222-2222-222222222222",
		ModelID:        "gemini-2.5-flash",
		InputTokens:    120,
		OutputTokens:   48,
		CachedTokens:   16,
		CostMicros:     2_500,
		InvocationID:   "019e72bd-d27b-735c-9302-64b99c30a242",
		AgentRole:      "qgen_question",
		AgentID:        "qgen_question",
		ActionCode:     "question_generation",
		ManaUnits:      50,
		RecordedAt:     time.Date(2026, 5, 29, 7, 59, 21, 0, time.UTC),
		Vendor:         "vertex_ai_gemini",
		FallbackChain:  []string{"vertex_ai_gemini:gemini-2.5-flash"},
		ArmorPre:       domain.ArmorVerdictAllow,
		ArmorPost:      domain.ArmorVerdictSanitise,
		GatewayVersion: "chora-model-gateway:a1e5eabf",
		Traceparent:    "00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01",
		Tracestate:     "chora=1",
	}
}

func TestBuildTokenUsagePayload_RoundTripsProto(t *testing.T) {
	evt := sampleEvent()
	occurred := evt.RecordedAt

	raw, err := buildTokenUsagePayload(evt, occurred)
	if err != nil {
		t.Fatalf("buildTokenUsagePayload: %v", err)
	}
	if len(raw) == 0 {
		t.Fatal("payload bytes empty — the body must be a marshalled TokenUsageRecorded, not []byte{}")
	}

	got := &observabilityv1.TokenUsageRecorded{}
	if err := proto.Unmarshal(raw, got); err != nil {
		t.Fatalf("payload is not a valid TokenUsageRecorded: %v", err)
	}

	// Scalar fields.
	if got.GetUsageId() != evt.UsageID {
		t.Errorf("usage_id = %q; want %q", got.GetUsageId(), evt.UsageID)
	}
	if got.GetTenantId() != evt.TenantID {
		t.Errorf("tenant_id = %q; want %q", got.GetTenantId(), evt.TenantID)
	}
	if got.GetGcid() != evt.GCID {
		t.Errorf("gcid = %q; want %q", got.GetGcid(), evt.GCID)
	}
	if got.GetModelId() != evt.ModelID {
		t.Errorf("model_id = %q; want %q", got.GetModelId(), evt.ModelID)
	}
	if got.GetInputTokens() != evt.InputTokens || got.GetOutputTokens() != evt.OutputTokens ||
		got.GetCachedTokens() != evt.CachedTokens || got.GetCostMicros() != evt.CostMicros {
		t.Errorf("token/cost mismatch: in=%d out=%d cached=%d cost=%d",
			got.GetInputTokens(), got.GetOutputTokens(), got.GetCachedTokens(), got.GetCostMicros())
	}
	if got.GetAgentRole() != evt.AgentRole {
		t.Errorf("agent_role = %q; want %q", got.GetAgentRole(), evt.AgentRole)
	}
	if got.GetManaUnits() != evt.ManaUnits {
		t.Errorf("mana_units = %d; want %d", got.GetManaUnits(), evt.ManaUnits)
	}
	if got.GetActionCode() != evt.ActionCode {
		t.Errorf("action_code = %q; want %q", got.GetActionCode(), evt.ActionCode)
	}
	if got.GetModelArmorVerdictPre() != observabilityv1.ModelArmorVerdict_MODEL_ARMOR_VERDICT_ALLOW {
		t.Errorf("armor_pre = %v; want ALLOW", got.GetModelArmorVerdictPre())
	}
	if got.GetModelArmorVerdictPost() != observabilityv1.ModelArmorVerdict_MODEL_ARMOR_VERDICT_SANITISE {
		t.Errorf("armor_post = %v; want SANITISE", got.GetModelArmorVerdictPost())
	}
	if got.GetRecordedAt() == nil || !got.GetRecordedAt().AsTime().Equal(occurred) {
		t.Errorf("recorded_at = %v; want %v", got.GetRecordedAt(), occurred)
	}

	// Embedded EventEnvelope (field 1) — the consumer reads this as authoritative.
	env := got.GetEnvelope()
	if env == nil {
		t.Fatal("embedded EventEnvelope is nil; must be populated")
	}
	if env.GetEventId() != evt.UsageID || env.GetIdempotencyKey() != evt.UsageID {
		t.Errorf("envelope event_id/idempotency_key = %q/%q; want %q",
			env.GetEventId(), env.GetIdempotencyKey(), evt.UsageID)
	}
	if env.GetTenantId() != evt.TenantID || env.GetGcid() != evt.GCID {
		t.Errorf("envelope tenant/gcid mismatch: %+v", env)
	}
	if env.GetTraceparent() != evt.Traceparent || env.GetTracestate() != evt.Tracestate {
		t.Errorf("envelope trace ctx mismatch: tp=%q ts=%q", env.GetTraceparent(), env.GetTracestate())
	}
	if env.GetSourceService() != "chora-model-gateway" {
		t.Errorf("envelope source_service = %q; want chora-model-gateway", env.GetSourceService())
	}
	if env.GetSchemaVersion() != 1 {
		t.Errorf("envelope schema_version = %d; want 1", env.GetSchemaVersion())
	}
	var _ = commonv1.EventEnvelope{} // assert the type import is the canonical common envelope
}

func TestBuildOutboxEnvelope_FlatCanonicalSnakeCase(t *testing.T) {
	evt := sampleEvent()
	occurred := evt.RecordedAt

	env := buildOutboxEnvelope(evt, occurred)

	// MUST be flat canonical snake_case (the keys observability's
	// reconstructEnvelope reads + the publisher validates).
	wantScalar := map[string]string{
		"event_id":        evt.UsageID,
		"idempotency_key": evt.UsageID,
		"tenant_id":       evt.TenantID,
		"gcid":            evt.GCID,
		"traceparent":     evt.Traceparent,
		"tracestate":      evt.Tracestate,
		"source_service":  "chora-model-gateway",
		"schema_version":  "1",
	}
	for k, want := range wantScalar {
		if env[k] != want {
			t.Errorf("envelope[%q] = %q; want %q", k, env[k], want)
		}
	}
	if env["occurred_at"] == "" {
		t.Error("envelope missing occurred_at")
	}

	// MUST NOT carry the old nested object or camelCase keys.
	if _, ok := env["event"]; ok {
		t.Error("envelope must not carry the nested 'event' object")
	}
	if _, ok := env["tenantId"]; ok {
		t.Error("envelope must not carry camelCase 'tenantId'")
	}
}

// TestBuildOutboxEnvelope_AlwaysHasTraceparent guards the publish-reject
// regression: the publisher rejects an empty traceparent. Even if the event
// somehow arrives without trace context, the envelope must surface whatever
// was set (server.go guarantees a valid one upstream via EnsureTraceparent).
func TestBuildOutboxEnvelope_TraceparentSurfaced(t *testing.T) {
	evt := sampleEvent()
	env := buildOutboxEnvelope(evt, evt.RecordedAt)
	if env["traceparent"] == "" {
		t.Error("traceparent must be surfaced in the envelope (publisher rejects empty)")
	}
}
