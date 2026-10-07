package domain_test

import (
	"context"
	"errors"
	"testing"

	"github.com/apollo-chora/chora-model-gateway/internal/domain"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// ----------------------------------------------------------------------------
// ADR-152 amendment 2026-08-07 (G2 producer): a Model Armor BLOCK at the
// gateway becomes a governance violation event.
//
// The gateway is the single un-bypassable LLM chokepoint (ADR-163), so it is
// the only place that holds the verdict, the tenant context and the agent
// identity at the same instant. These tests pin the producer contract:
//
//	BLOCK on PRE  → exactly one event, leg PRE
//	BLOCK on POST → exactly one event, leg POST
//	ALLOW         → no event at all
//	publish error → the caller's refusal is unchanged (non-blocking)
//	ERROR verdict → no event (a guardrail fault is not a detected violation)
// ----------------------------------------------------------------------------

// fakeViolationPublisher captures PolicyViolationDetected emissions so the
// tests can assert on the typed fields the governance projector needs.
type fakeViolationPublisher struct {
	events []domain.PolicyViolationEvent
	err    error
	calls  int
}

func (f *fakeViolationPublisher) EnqueuePolicyViolationDetected(_ context.Context, evt domain.PolicyViolationEvent) error {
	f.calls++
	if f.err != nil {
		return f.err
	}
	f.events = append(f.events, evt)
	return nil
}

// withViolations returns a ServiceConfig mod wiring the violation publisher.
func withViolations(p *fakeViolationPublisher) func(cfg *domain.ServiceConfig) {
	return func(cfg *domain.ServiceConfig) { cfg.Violations = p }
}

func TestInvoke_ArmorPreBlock_PublishesPolicyViolation(t *testing.T) {
	violations := &fakeViolationPublisher{}
	svc, _, _, _, _, _ := newTestService(t,
		func(cfg *domain.ServiceConfig) {
			cfg.Armor = &fakeArmor{preVerdict: domain.ArmorVerdictBlock}
		},
		withViolations(violations),
	)

	resp, err := svc.Invoke(context.Background(), happyRequest())
	require.NoError(t, err)
	require.Equal(t, domain.FinishReasonModelArmorBlock, resp.FinishReason)

	require.Len(t, violations.events, 1, "exactly one violation per blocked request")
	evt := violations.events[0]
	assert.Equal(t, domain.ArmorLegPre, evt.Leg)
	assert.Equal(t, domain.ArmorVerdictBlock, evt.Verdict)
	assert.Equal(t, "tenant-1", evt.TenantID)
	assert.Equal(t, "gcid-1", evt.GCID)
	assert.Equal(t, "qgen_question", evt.AgentID)
	assert.Equal(t, "test-invocation-id", evt.InvocationID)
	assert.Equal(t,
		"projects/chora-489812/locations/asia-southeast1/templates/chora-guardrail-balanced-dev",
		evt.ArmorTemplate,
		"the template that fired is the policy identity")
	assert.Equal(t, fixedTime(), evt.DetectedAt)
	assert.Equal(t,
		"00-0123456789abcdef0123456789abcdef-0123456789abcdef-01",
		evt.Traceparent,
		"W3C trace context propagates onto the governance event")
	assert.NotEmpty(t, evt.ViolationID, "UUIDv7 violation id doubles as event_id + idempotency_key")
}

func TestInvoke_ArmorPostBlock_PublishesPolicyViolation(t *testing.T) {
	violations := &fakeViolationPublisher{}
	svc, _, _, _, _, _ := newTestService(t,
		func(cfg *domain.ServiceConfig) {
			cfg.Armor = &fakeArmor{
				preVerdict:  domain.ArmorVerdictAllow,
				postVerdict: domain.ArmorVerdictBlock,
			}
		},
		withViolations(violations),
	)

	resp, err := svc.Invoke(context.Background(), happyRequest())
	require.NoError(t, err)
	require.Equal(t, domain.FinishReasonModelArmorBlock, resp.FinishReason)

	require.Len(t, violations.events, 1, "the POST leg publishes once, not once per leg evaluated")
	evt := violations.events[0]
	assert.Equal(t, domain.ArmorLegPost, evt.Leg)
	assert.Equal(t, domain.ArmorVerdictBlock, evt.Verdict)
	assert.Equal(t, "tenant-1", evt.TenantID)
	assert.Equal(t, "gcid-1", evt.GCID)
	assert.Equal(t, "qgen_question", evt.AgentID)
	assert.Equal(t, "test-invocation-id", evt.InvocationID)
}

func TestInvoke_ArmorAllow_PublishesNoViolation(t *testing.T) {
	violations := &fakeViolationPublisher{}
	svc, _, _, _, _, _ := newTestService(t, withViolations(violations))

	resp, err := svc.Invoke(context.Background(), happyRequest())
	require.NoError(t, err)
	require.Equal(t, domain.FinishReasonComplete, resp.FinishReason)

	assert.Zero(t, violations.calls, "an allowed call is not a policy violation")
	assert.Empty(t, violations.events)
}

func TestInvoke_ViolationPublishError_DoesNotChangeRefusal(t *testing.T) {
	violations := &fakeViolationPublisher{err: errors.New("outbox insert failed")}
	svc, _, _, _, _, _ := newTestService(t,
		func(cfg *domain.ServiceConfig) {
			cfg.Armor = &fakeArmor{preVerdict: domain.ArmorVerdictBlock}
		},
		withViolations(violations),
	)

	resp, err := svc.Invoke(context.Background(), happyRequest())

	// Fail-loud but NON-BLOCKING: the publisher error is logged at ERROR by the
	// service, never folded into the caller's response. The refusal is the
	// user-visible outcome and must survive a governance-lane outage.
	require.NoError(t, err, "a publisher error must not turn a clean refusal into an RPC error")
	assert.Equal(t, domain.FinishReasonModelArmorBlock, resp.FinishReason)
	assert.Equal(t, domain.ArmorVerdictBlock, resp.ArmorPre)
	assert.Equal(t, "test-invocation-id", resp.InvocationID)
	assert.Equal(t, 1, violations.calls, "the publish was attempted")
}

func TestInvoke_ArmorErrorVerdict_PublishesNoViolation(t *testing.T) {
	violations := &fakeViolationPublisher{}
	svc, _, _, _, _, _ := newTestService(t,
		func(cfg *domain.ServiceConfig) {
			cfg.Armor = &fakeArmor{preVerdict: domain.ArmorVerdictError}
		},
		withViolations(violations),
	)

	resp, err := svc.Invoke(context.Background(), happyRequest())
	require.NoError(t, err)
	require.Equal(t, domain.FinishReasonModelArmorBlock, resp.FinishReason,
		"an ERROR verdict still refuses the call (degrade-safe)")

	// ArmorVerdictError means Armor could not classify the text, NOT that a
	// filter matched. Recording it as a detected violation would fabricate
	// governance evidence, so the producer stays silent here. The refusal is
	// still observable: the token-usage ledger carries the ERROR enum verbatim.
	assert.Zero(t, violations.calls, "a guardrail fault is not a detected violation")
}

func TestPolicyViolationEvent_DescriptionCarriesTheUnmappableFacts(t *testing.T) {
	// The ratified proto has no agent, leg or verdict field and the topic
	// schema is frozen, so Description is the only place those three survive
	// onto the wire. Pin the wording: a consumer parses it.
	evt := domain.PolicyViolationEvent{
		AgentID:      "familiar_companion",
		InvocationID: "019e72c0-0000-7000-8000-000000000001",
		Leg:          domain.ArmorLegPre,
		Verdict:      domain.ArmorVerdictBlock,
	}
	assert.Equal(t,
		"Cloud Model Armor refused the call at the PRE leg: "+
			"agent_id=familiar_companion "+
			"invocation_id=019e72c0-0000-7000-8000-000000000001 "+
			"verdict=block",
		evt.Description())

	evt.Leg = domain.ArmorLegPost
	assert.Contains(t, evt.Description(), "POST leg")
}

func TestPolicyViolationEvent_ResourceURINamesTheInvocation(t *testing.T) {
	evt := domain.PolicyViolationEvent{InvocationID: "019e72c0-0000-7000-8000-000000000001"}
	assert.Equal(t,
		"chora.ai_kernel/invocation:019e72c0-0000-7000-8000-000000000001",
		evt.ResourceURI())
}

func TestInvoke_BlockWithoutViolationPublisher_StillRefuses(t *testing.T) {
	// The port is optional at construction (same precedent as the ADR-231
	// grounded ports) so Invoke-only wiring and the existing test suite need
	// not provide it. An unwired publisher must never break the refusal.
	svc, _, _, _, _, _ := newTestService(t, func(cfg *domain.ServiceConfig) {
		cfg.Armor = &fakeArmor{preVerdict: domain.ArmorVerdictBlock}
	})

	resp, err := svc.Invoke(context.Background(), happyRequest())
	require.NoError(t, err)
	assert.Equal(t, domain.FinishReasonModelArmorBlock, resp.FinishReason)
}
