// violation_payload_test.go: pure-function unit tests (NO database) for the
// ADR-152 PolicyViolationDetected payload + flat envelope builders.
//
// The destination topic chora.governance.policy.violation_detected.v1 is bound
// to a BINARY PROTOCOL_BUFFER schema, so the payload MUST be a proto-marshalled
// governance.v1.PolicyViolationDetected: a JSON body is rejected AT PUBLISH and
// never reaches a DLQ. These tests pin the wire shape.
package pg

import (
	"testing"
	"time"

	governancev1 "github.com/apollo-chora/chora-contracts/gen/go/chora/governance/v1"
	"google.golang.org/protobuf/proto"

	"github.com/apollo-chora/chora-model-gateway/internal/domain"
)

func sampleViolation() domain.PolicyViolationEvent {
	return domain.PolicyViolationEvent{
		ViolationID:   "019e72bd-d27b-735c-9302-64b99c30a242",
		TenantID:      "11111111-1111-1111-1111-111111111111",
		GCID:          "22222222-2222-2222-2222-222222222222",
		AgentID:       "familiar_companion",
		InvocationID:  "019e72c0-0000-7000-8000-000000000001",
		Leg:           domain.ArmorLegPre,
		ArmorTemplate: "projects/chora-489812/locations/asia-southeast1/templates/chora-guardrail-strict-dev",
		Verdict:       domain.ArmorVerdictBlock,
		DetectedAt:    time.Date(2026, 8, 7, 9, 15, 0, 0, time.UTC),
		Traceparent:   "00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01",
		Tracestate:    "chora=1",
	}
}

func TestBuildPolicyViolationPayload_RoundTripsProto(t *testing.T) {
	evt := sampleViolation()

	raw, err := buildPolicyViolationPayload(evt, evt.DetectedAt)
	if err != nil {
		t.Fatalf("buildPolicyViolationPayload: %v", err)
	}
	if len(raw) == 0 {
		t.Fatal("payload is empty, the dispatcher would publish an undecodable body")
	}

	var got governancev1.PolicyViolationDetected
	if err := proto.Unmarshal(raw, &got); err != nil {
		t.Fatalf("proto.Unmarshal: %v", err)
	}

	if got.GetViolationId() != evt.ViolationID {
		t.Errorf("violation_id = %q, want %q", got.GetViolationId(), evt.ViolationID)
	}
	if got.GetPolicyId() != evt.ArmorTemplate {
		t.Errorf("policy_id = %q, want the Armor template %q", got.GetPolicyId(), evt.ArmorTemplate)
	}
	if got.GetPolicyKind() != governancev1.PolicyKind_POLICY_KIND_AI_USAGE {
		t.Errorf("policy_kind = %v, want POLICY_KIND_AI_USAGE (an Armor template IS the per-agent guardrail config)", got.GetPolicyKind())
	}
	if got.GetViolationSeverity() != governancev1.ViolationSeverity_VIOLATION_SEVERITY_HIGH {
		t.Errorf("violation_severity = %v, want HIGH (immediate human review)", got.GetViolationSeverity())
	}
	if got.GetSubjectGcid() != evt.GCID {
		t.Errorf("subject_gcid = %q, want %q", got.GetSubjectGcid(), evt.GCID)
	}
	if got.GetDetector() != violationDetector {
		t.Errorf("detector = %q, want %q", got.GetDetector(), violationDetector)
	}
	if got.GetResourceUri() != "chora.ai_kernel/invocation:"+evt.InvocationID {
		t.Errorf("resource_uri = %q, want the refused invocation", got.GetResourceUri())
	}
	if got.GetDetectedAt().AsTime() != evt.DetectedAt {
		t.Errorf("detected_at = %v, want %v", got.GetDetectedAt().AsTime(), evt.DetectedAt)
	}

	// The ratified proto carries no agent, leg or verdict field and the topic
	// schema is frozen, so those three ride in the description.
	desc := got.GetDescription()
	for _, want := range []string{"PRE", "familiar_companion", evt.InvocationID, "block"} {
		if !contains(desc, want) {
			t.Errorf("description %q is missing %q", desc, want)
		}
	}
}

func TestBuildPolicyViolationPayload_EnvelopeIsComplete(t *testing.T) {
	evt := sampleViolation()

	raw, err := buildPolicyViolationPayload(evt, evt.DetectedAt)
	if err != nil {
		t.Fatalf("buildPolicyViolationPayload: %v", err)
	}
	var got governancev1.PolicyViolationDetected
	if err := proto.Unmarshal(raw, &got); err != nil {
		t.Fatalf("proto.Unmarshal: %v", err)
	}

	env := got.GetEnvelope()
	if env == nil {
		t.Fatal("envelope is nil, the dispatcher rejects an envelope-less event")
	}
	if env.GetEventId() != evt.ViolationID {
		t.Errorf("event_id = %q, want %q", env.GetEventId(), evt.ViolationID)
	}
	if env.GetIdempotencyKey() != evt.ViolationID {
		t.Errorf("idempotency_key = %q, want %q", env.GetIdempotencyKey(), evt.ViolationID)
	}
	if env.GetTenantId() != evt.TenantID {
		t.Errorf("tenant_id = %q, want %q", env.GetTenantId(), evt.TenantID)
	}
	if env.GetGcid() != evt.GCID {
		t.Errorf("gcid = %q, want %q", env.GetGcid(), evt.GCID)
	}
	if env.GetTraceparent() != evt.Traceparent {
		t.Errorf("traceparent = %q, want %q, the dispatcher REQUIRES a non-empty traceparent", env.GetTraceparent(), evt.Traceparent)
	}
	if env.GetTracestate() != evt.Tracestate {
		t.Errorf("tracestate = %q, want %q", env.GetTracestate(), evt.Tracestate)
	}
	if env.GetSourceService() != sourceService {
		t.Errorf("source_service = %q, want %q", env.GetSourceService(), sourceService)
	}
	if env.GetSchemaVersion() != envelopeSchemaVersion {
		t.Errorf("schema_version = %d, want %d", env.GetSchemaVersion(), envelopeSchemaVersion)
	}
	// The governance projector routes D3 evidence into policy_violation_log on
	// chora_imda_dimension=safety_and_robustness (ADR-141 canonical label).
	if env.GetChoraImdaDimension() != violationImdaDimension {
		t.Errorf("chora_imda_dimension = %q, want %q", env.GetChoraImdaDimension(), violationImdaDimension)
	}
}

func TestBuildViolationEnvelope_FlatMapIsAllStrings(t *testing.T) {
	evt := sampleViolation()
	env := buildViolationEnvelope(evt, evt.DetectedAt)

	want := map[string]string{
		"event_id":             evt.ViolationID,
		"idempotency_key":      evt.ViolationID,
		"tenant_id":            evt.TenantID,
		"gcid":                 evt.GCID,
		"traceparent":          evt.Traceparent,
		"tracestate":           evt.Tracestate,
		"source_service":       sourceService,
		"chora_imda_dimension": violationImdaDimension,
	}
	for k, v := range want {
		if env[k] != v {
			t.Errorf("envelope[%q] = %q, want %q", k, env[k], v)
		}
	}
	if env["occurred_at"] == "" {
		t.Error("envelope occurred_at is empty")
	}
	if env["schema_version"] != "1" {
		t.Errorf("envelope schema_version = %q, want \"1\"", env["schema_version"])
	}
}

func TestBuildPolicyViolationPayload_PostLegIsDistinguishable(t *testing.T) {
	evt := sampleViolation()
	evt.Leg = domain.ArmorLegPost

	raw, err := buildPolicyViolationPayload(evt, evt.DetectedAt)
	if err != nil {
		t.Fatalf("buildPolicyViolationPayload: %v", err)
	}
	var got governancev1.PolicyViolationDetected
	if err := proto.Unmarshal(raw, &got); err != nil {
		t.Fatalf("proto.Unmarshal: %v", err)
	}
	if !contains(got.GetDescription(), "POST") {
		t.Errorf("description %q does not identify the POST leg", got.GetDescription())
	}
}

// contains is a tiny substring helper so the assertions read as intent.
func contains(haystack, needle string) bool {
	return len(needle) == 0 || len(haystack) >= len(needle) && indexOf(haystack, needle) >= 0
}

func indexOf(haystack, needle string) int {
	for i := 0; i+len(needle) <= len(haystack); i++ {
		if haystack[i:i+len(needle)] == needle {
			return i
		}
	}
	return -1
}
