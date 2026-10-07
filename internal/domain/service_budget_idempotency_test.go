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
// Test fakes for the new ports (ModelResolver, DebitClaimer, AtomicBudgetOutbox)
// ----------------------------------------------------------------------------

type fakeModelResolver struct {
	info domain.ModelInfo
	err  error
}

func (f *fakeModelResolver) Resolve(_ context.Context, _ domain.LogicalModelID) (domain.ModelInfo, error) {
	return f.info, f.err
}

type fakeClaimer struct {
	claimed bool
	err     error
	calls   int
}

func (f *fakeClaimer) ClaimDebit(_ context.Context, _, _, _, _ string) (bool, error) {
	f.calls++
	return f.claimed, f.err
}

type fakeAtomicOutbox struct {
	events []domain.TokenUsageEvent
	debits int64
	err    error
}

func (f *fakeAtomicOutbox) DebitAndEnqueue(_ context.Context, evt domain.TokenUsageEvent, debit int64) error {
	if f.err != nil {
		return f.err
	}
	f.events = append(f.events, evt)
	f.debits += debit
	return nil
}

// ----------------------------------------------------------------------------
// Idempotency claim suppresses the tenant-budget debit
// ----------------------------------------------------------------------------

func TestInvoke_DispatchKey_Deduped_SkipsBudgetDebit(t *testing.T) {
	claims := &fakeClaimer{claimed: false} // redelivery: key already claimed
	svc, vendor, _, budget, outbox, _ := newTestService(t, func(cfg *domain.ServiceConfig) {
		cfg.Claims = claims
	})

	req := happyRequest()
	req.DispatchIdempotencyKey = "wf-0193-turn-7"
	resp, err := svc.Invoke(context.Background(), req)
	require.NoError(t, err)
	assert.Equal(t, 1, claims.calls, "the domain service takes the claim")
	assert.True(t, resp.Deduped, "the response reports the dedup")
	assert.Equal(t, 1, vendor.calls, "the call still proceeds")
	assert.Equal(t, 0, budget.debitCalls, "a deduped dispatch does NOT debit the budget")
	require.Len(t, outbox.events, 1, "but still enqueues the real cost")
	assert.Equal(t, int64(12_500), outbox.events[0].CostMicros)
}

func TestInvoke_DispatchKey_FirstDelivery_Debits(t *testing.T) {
	claims := &fakeClaimer{claimed: true} // first delivery: won the claim
	svc, _, _, budget, _, _ := newTestService(t, func(cfg *domain.ServiceConfig) {
		cfg.Claims = claims
	})

	req := happyRequest()
	req.DispatchIdempotencyKey = "wf-0193-turn-7"
	resp, err := svc.Invoke(context.Background(), req)
	require.NoError(t, err)
	assert.Equal(t, 1, claims.calls)
	assert.False(t, resp.Deduped)
	assert.Equal(t, 1, budget.debitCalls, "first delivery debits")
	assert.Equal(t, int64(12_500), budget.debits)
}

func TestInvoke_ClaimError_HardError(t *testing.T) {
	claims := &fakeClaimer{err: errors.New("claim store down")}
	svc, vendor, _, _, _, _ := newTestService(t, func(cfg *domain.ServiceConfig) {
		cfg.Claims = claims
	})

	req := happyRequest()
	req.DispatchIdempotencyKey = "wf-0193-turn-7"
	_, err := svc.Invoke(context.Background(), req)
	require.Error(t, err, "a claim failure is a hard error, not a served-but-uncharged")
	var ierr *domain.InvokeError
	require.True(t, errors.As(err, &ierr))
	assert.Contains(t, ierr.Detail, "idempotency claim failed")
	assert.Equal(t, 0, vendor.calls, "no dispatch on claim failure")
}

// ----------------------------------------------------------------------------
// Budget downgrade re-resolves the target model
// ----------------------------------------------------------------------------

func TestInvoke_BudgetDowngrade_ReResolvesModel(t *testing.T) {
	models := &fakeModelResolver{
		info: domain.ModelInfo{
			ID:           "gemini-2.5-flash-lite",
			Vendor:       domain.VendorFamilyVertexGemini,
			Capabilities: []string{"chat"},
		},
	}
	svc, vendor, _, _, _, _ := newTestService(t, func(cfg *domain.ServiceConfig) {
		cfg.Budget = &fakeBudget{
			state: &domain.BudgetState{
				TenantID:         "tenant-1",
				BudgetUSDMicros:  100,
				SpentUSDMicros:   100,
				Policy:           domain.BudgetPolicyDowngrade,
				DowngradeToModel: domain.LogicalModelID("gemini-2.5-flash-lite"),
			},
		}
		cfg.Models = models
	})

	resp, err := svc.Invoke(context.Background(), happyRequest())
	require.NoError(t, err)
	assert.Equal(t, domain.FinishReasonComplete, resp.FinishReason)
	// The vendor was called with the downgrade model (re-resolved through
	// the registry, not just the ID swapped).
	assert.Equal(t, domain.LogicalModelID("gemini-2.5-flash-lite"), vendor.lastReq.LogicalModelID)
}

// ----------------------------------------------------------------------------
// Atomic debit+outbox transaction
// ----------------------------------------------------------------------------

func TestInvoke_AtomicOutbox_UsedWhenWired(t *testing.T) {
	atomic := &fakeAtomicOutbox{}
	svc, _, _, budget, outbox, _ := newTestService(t, func(cfg *domain.ServiceConfig) {
		cfg.AtomicOutbox = atomic
	})

	_, err := svc.Invoke(context.Background(), happyRequest())
	require.NoError(t, err)
	// The atomic port was used: the separate budget debit + outbox were NOT called.
	assert.Equal(t, 0, budget.debitCalls, "the atomic port handles the debit")
	assert.Empty(t, outbox.events, "the atomic port handles the outbox")
	require.Len(t, atomic.events, 1, "the atomic port enqueued the event")
	assert.Equal(t, int64(12_500), atomic.debits, "the atomic port debited the cost")
}

// ----------------------------------------------------------------------------
// IMAGE action_code defaulting
// ----------------------------------------------------------------------------

func TestInvoke_ImageModality_DefaultsActionCode(t *testing.T) {
	svc, _, _, _, outbox, _ := newTestService(t, func(cfg *domain.ServiceConfig) {
		cfg.DefaultAgentID = "openai_compat"
	})

	req := happyRequest()
	req.ResponseModality = "IMAGE"
	req.ActionCode = "" // no explicit action_code
	_, err := svc.Invoke(context.Background(), req)
	require.NoError(t, err)
	require.Len(t, outbox.events, 1)
	assert.Equal(t, "openai_compat_image", outbox.events[0].ActionCode,
		"IMAGE modality defaults the action_code to {defaultAgentID}_image")
}

func TestInvoke_ImageModality_ExplicitActionCodeWins(t *testing.T) {
	svc, _, _, _, outbox, _ := newTestService(t, func(cfg *domain.ServiceConfig) {
		cfg.DefaultAgentID = "openai_compat"
	})

	req := happyRequest()
	req.ResponseModality = "IMAGE"
	req.ActionCode = "my_custom_image_action"
	_, err := svc.Invoke(context.Background(), req)
	require.NoError(t, err)
	require.Len(t, outbox.events, 1)
	assert.Equal(t, "my_custom_image_action", outbox.events[0].ActionCode,
		"an explicit action_code wins over the IMAGE default")
}

// ----------------------------------------------------------------------------
// Output ceiling clamping per fallback target
// ----------------------------------------------------------------------------

func TestInvoke_OutputCeilingClamped(t *testing.T) {
	models := &fakeModelResolver{
		info: domain.ModelInfo{
			ID:              "gemini-2.5-pro",
			Vendor:          domain.VendorFamilyVertexGemini,
			Capabilities:    []string{"chat"},
			MaxOutputTokens: 100,
		},
	}
	svc, vendor, _, _, _, _ := newTestService(t, func(cfg *domain.ServiceConfig) {
		cfg.Models = models
	})

	req := happyRequest()
	req.GenerationConfig = map[string]any{"max_tokens": float64(500)}
	_, err := svc.Invoke(context.Background(), req)
	require.NoError(t, err)
	// The vendor request carries the clamped max_tokens.
	assert.Equal(t, 100, vendor.lastReq.GenerationConfig["max_tokens"],
		"max_tokens is clamped to the model's output ceiling")
}

// ----------------------------------------------------------------------------
// Capability checks per dispatch target
// ----------------------------------------------------------------------------

func TestInvoke_CapabilityCheck_FailsForMissingCapability(t *testing.T) {
	models := &fakeModelResolver{
		info: domain.ModelInfo{
			ID:           "text-model",
			Vendor:       domain.VendorFamilyVertexGemini,
			Capabilities: []string{"chat"}, // no "image"
		},
	}
	svc, vendor, _, _, _, _ := newTestService(t, func(cfg *domain.ServiceConfig) {
		cfg.Models = models
	})

	req := happyRequest()
	req.ResponseModality = "IMAGE" // requires "image" capability
	_, err := svc.Invoke(context.Background(), req)
	require.Error(t, err, "a model without the required capability fails")
	var ierr *domain.InvokeError
	require.True(t, errors.As(err, &ierr))
	assert.Contains(t, ierr.Detail, "does not advertise")
	assert.Equal(t, 0, vendor.calls, "no dispatch to a model without the capability")
}

// ----------------------------------------------------------------------------
// Credential pre-validation
// ----------------------------------------------------------------------------

func TestInvoke_CredentialPreValidation_FailsForEmptyKey(t *testing.T) {
	models := &fakeModelResolver{
		info: domain.ModelInfo{
			ID:           "gemini-2.5-pro",
			Vendor:       domain.VendorFamilyVertexGemini,
			Capabilities: []string{"chat"},
			APIKeyEnv:    "MY_API_KEY",
			APIKey:       "", // resolves to empty
		},
	}
	svc, vendor, _, _, _, _ := newTestService(t, func(cfg *domain.ServiceConfig) {
		cfg.Models = models
	})

	_, err := svc.Invoke(context.Background(), happyRequest())
	require.Error(t, err, "a credential reference that resolves to empty fails")
	var ierr *domain.InvokeError
	require.True(t, errors.As(err, &ierr))
	assert.Contains(t, ierr.Detail, "credential reference")
	assert.Equal(t, 0, vendor.calls, "no dispatch with an empty credential")
}

// ----------------------------------------------------------------------------
// Modality validation
// ----------------------------------------------------------------------------

func TestInvoke_ModalityValidation_RejectsUnknown(t *testing.T) {
	svc, vendor, _, _, _, _ := newTestService(t)

	req := happyRequest()
	req.ResponseModality = "AUDIO" // not one of TEXT, IMAGE, GROUNDED
	_, err := svc.Invoke(context.Background(), req)
	require.Error(t, err, "an unknown modality is refused")
	var ierr *domain.InvokeError
	require.True(t, errors.As(err, &ierr))
	assert.Contains(t, ierr.Detail, "response_modality")
	assert.Equal(t, 0, vendor.calls, "no dispatch on an invalid modality")
}

// ----------------------------------------------------------------------------
// prompt OR contents validation
// ----------------------------------------------------------------------------

func TestInvoke_PromptOrContents_AllowsContentsOnly(t *testing.T) {
	svc, vendor, _, _, _, _ := newTestService(t)

	req := happyRequest()
	req.Prompt = "" // no prompt
	req.ContentsJSON = `[{"role":"user","parts":[{"text":"hello"}]}]`
	_, err := svc.Invoke(context.Background(), req)
	require.NoError(t, err, "contents_json alone is a valid request")
	assert.Equal(t, 1, vendor.calls)
}

func TestInvoke_PromptOrContents_RejectsEmpty(t *testing.T) {
	svc, vendor, _, _, _, _ := newTestService(t)

	req := happyRequest()
	req.Prompt = ""
	req.ContentsJSON = ""
	_, err := svc.Invoke(context.Background(), req)
	require.Error(t, err, "an empty request (no prompt AND no contents) is refused")
	var ierr *domain.InvokeError
	require.True(t, errors.As(err, &ierr))
	assert.Contains(t, ierr.Detail, "prompt or contents_json required")
	assert.Equal(t, 0, vendor.calls, "no dispatch on an empty request")
}

// ----------------------------------------------------------------------------
// Action code on the usage event
// ----------------------------------------------------------------------------

func TestInvoke_ActionCode_OnUsageEvent(t *testing.T) {
	svc, _, _, _, outbox, _ := newTestService(t)

	req := happyRequest()
	req.ActionCode = "my_action"
	_, err := svc.Invoke(context.Background(), req)
	require.NoError(t, err)
	require.Len(t, outbox.events, 1)
	assert.Equal(t, "my_action", outbox.events[0].ActionCode,
		"the usage event carries the caller's action_code")
}

func TestInvoke_ActionCode_DefaultsToAgentID(t *testing.T) {
	svc, _, _, _, outbox, _ := newTestService(t)

	req := happyRequest()
	req.ActionCode = "" // no explicit action_code
	_, err := svc.Invoke(context.Background(), req)
	require.NoError(t, err)
	require.Len(t, outbox.events, 1)
	assert.Equal(t, "qgen_question", outbox.events[0].ActionCode,
		"an empty action_code defaults to the agent id")
}
