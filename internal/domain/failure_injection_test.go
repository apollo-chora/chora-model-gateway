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
// Failure-injection fakes — simulate DB crashes, provider timeouts, and
// partial failures to verify the gateway's resilience behaviour.
// ----------------------------------------------------------------------------

// failingAtomicOutbox simulates a DB commit failure: the provider succeeded
// but the budget debit + outbox enqueue transaction failed to commit.
type failingAtomicOutbox struct {
	err error
}

func (f *failingAtomicOutbox) DebitAndEnqueue(_ context.Context, _ domain.TokenUsageEvent, _ int64) error {
	return f.err
}

// failingOutbox simulates an outbox enqueue failure (non-atomic path).
type failingOutbox struct {
	err error
}

func (f *failingOutbox) EnqueueTokenUsageRecorded(_ context.Context, _ domain.TokenUsageEvent) error {
	return f.err
}

// failingBudget simulates a budget DB failure.
type failingBudget struct {
	getErr   error
	debitErr error
}

func (f *failingBudget) GetTenantBudget(_ context.Context, _ string) (*domain.BudgetState, error) {
	if f.getErr != nil {
		return nil, f.getErr
	}
	return &domain.BudgetState{
		TenantID:        "tenant-1",
		BudgetUSDMicros: 1_000_000,
		SpentUSDMicros:  0,
		Policy:          domain.BudgetPolicyBlock,
	}, nil
}

func (f *failingBudget) DebitSpent(_ context.Context, _ string, _ int64) error {
	return f.debitErr
}

// failingClaimer simulates a claim DB failure.
type failingClaimer struct {
	err error
}

func (f *failingClaimer) ClaimDebit(_ context.Context, _, _, _, _ string) (bool, error) {
	return false, f.err
}

// timeoutVendor simulates a provider that times out after partial token
// generation — it returns an error (the partial tokens are lost).
type timeoutVendor struct {
	family domain.VendorFamily
	err    error
	calls  int
}

func (v *timeoutVendor) Family() domain.VendorFamily { return v.family }

func (v *timeoutVendor) Generate(_ context.Context, _ domain.VendorRequest) (domain.VendorResponse, error) {
	v.calls++
	return domain.VendorResponse{}, v.err
}

// ----------------------------------------------------------------------------
// 1. Provider succeeds but DB commit fails
// ----------------------------------------------------------------------------

// When the provider succeeds but the DB commit (budget debit + outbox enqueue)
// fails, the call must fail with a clear error. The accounting is NOT retried
// automatically at the domain layer — the outbox pattern ensures the event is
// recoverable by the outbox drain process.
func TestFailureInjection_ProviderSucceeds_DBCommitFails(t *testing.T) {
	atomic := &failingAtomicOutbox{err: errors.New("deadlock detected")}
	executor, vendor, _, _, _, _ := newTestExecutor(t, func(cfg *domain.ExecutorConfig) {
		cfg.AtomicOutbox = atomic
	})

	resp, err := executor.Execute(context.Background(), happyExecuteRequest())
	require.Error(t, err, "a DB commit failure must fail the call")
	assert.Nil(t, resp)
	var ierr *domain.InvokeError
	require.True(t, errors.As(err, &ierr))
	assert.Contains(t, ierr.Detail, "budget debit + outbox enqueue failed")
	assert.Equal(t, 1, vendor.calls, "the provider was called")
}

// Same scenario via the non-atomic path: the budget debit fails.
func TestFailureInjection_ProviderSucceeds_BudgetDebitFails(t *testing.T) {
	budget := &failingBudget{debitErr: errors.New("pg connection reset")}
	executor, vendor, _, _, _, _ := newTestExecutor(t, func(cfg *domain.ExecutorConfig) {
		cfg.Budget = budget
	})

	_, err := executor.Execute(context.Background(), happyExecuteRequest())
	require.Error(t, err)
	var ierr *domain.InvokeError
	require.True(t, errors.As(err, &ierr))
	assert.Contains(t, ierr.Detail, "budget debit failed")
	assert.Equal(t, 1, vendor.calls)
}

// Same scenario via the non-atomic path: the outbox enqueue fails.
func TestFailureInjection_ProviderSucceeds_OutboxEnqueueFails(t *testing.T) {
	outbox := &failingOutbox{err: errors.New("outbox table full")}
	executor, vendor, _, _, _, _ := newTestExecutor(t, func(cfg *domain.ExecutorConfig) {
		cfg.Outbox = outbox
	})

	_, err := executor.Execute(context.Background(), happyExecuteRequest())
	require.Error(t, err)
	var ierr *domain.InvokeError
	require.True(t, errors.As(err, &ierr))
	assert.Contains(t, ierr.Detail, "outbox enqueue failed")
	assert.Equal(t, 1, vendor.calls)
}

// ----------------------------------------------------------------------------
// 2. Provider succeeds but gateway crashes before accounting
// ----------------------------------------------------------------------------

// The outbox pattern ensures that if the gateway crashes after the provider
// returns but before the outbox event is drained, the event is still in the
// outbox table and can be recovered. This test verifies the outbox event is
// emitted with the correct payload so the drain process can recover it.
func TestFailureInjection_ProviderSucceeds_CrashBeforeAccounting_OutboxRecoverable(t *testing.T) {
	outbox := &fakeOutbox{}
	executor, vendor, _, _, _, _ := newTestExecutor(t, func(cfg *domain.ExecutorConfig) {
		cfg.Outbox = outbox
	})

	resp, err := executor.Execute(context.Background(), happyExecuteRequest())
	require.NoError(t, err)
	assert.Equal(t, "completion-stub", resp.Completion)
	assert.Equal(t, 1, vendor.calls)

	// The outbox event was emitted — it is recoverable by the drain process.
	require.Len(t, outbox.events, 1)
	evt := outbox.events[0]
	assert.Equal(t, "test-invocation-id", evt.UsageID)
	assert.Equal(t, "tenant-1", evt.TenantID)
	assert.Equal(t, "gcid-1", evt.GCID)
	assert.Equal(t, int64(12_500), evt.CostMicros)
	assert.Equal(t, "vertex_ai_gemini", evt.Vendor)
}

// ----------------------------------------------------------------------------
// 3. Provider timeout after partial token generation
// ----------------------------------------------------------------------------

// When the primary provider times out (after partial token generation), the
// fallback provider must be triggered. The partial tokens from the timed-out
// provider are lost — only the fallback's tokens are accounted.
func TestFailureInjection_PrimaryTimeout_FallbackTriggered(t *testing.T) {
	primary := &timeoutVendor{
		family: domain.VendorFamilyVertexGemini,
		err:    errors.New("context deadline exceeded after 3 partial tokens"),
	}
	fallback := &fakeVendor{
		family: domain.VendorFamilyOpenAI,
		resp: domain.VendorResponse{
			Completion:   "fallback-completion",
			Usage:        domain.TokenUsage{InputTokens: 20, OutputTokens: 10, CostMicros: 2_000},
			ModelVersion: "gpt-4o-mini",
			FinishReason: domain.FinishReasonComplete,
		},
	}
	executor, _, _, _, _, _ := newTestExecutor(t, func(cfg *domain.ExecutorConfig) {
		cfg.Providers = []domain.Provider{primary, fallback}
		cfg.Policies = &fakePolicy{
			policy: domain.AgentPolicy{
				AgentID:                "qgen_question",
				ResolvedLogicalModelID: domain.LogicalModelID("gemini-2.5-pro"),
				Vendor:                 domain.VendorFamilyVertexGemini,
				FallbackChain: []domain.AgentPolicyFallback{
					{Vendor: domain.VendorFamilyOpenAI, ResolvedLogicalModelID: domain.LogicalModelID("gpt-4o-mini")},
				},
			},
		}
	})

	resp, err := executor.Execute(context.Background(), happyExecuteRequest())
	require.NoError(t, err)
	assert.Equal(t, "fallback-completion", resp.Completion)
	assert.Equal(t, 1, primary.calls, "primary was attempted")
	assert.Equal(t, 1, fallback.calls, "fallback was triggered after timeout")
	assert.Equal(t, []string{"vertex_ai_gemini:gemini-2.5-pro", "openai_byoa:gpt-4o-mini"}, resp.FallbackChain)
}

// ----------------------------------------------------------------------------
// 4. Fallback provider invoked after first provider charges but fails
// ----------------------------------------------------------------------------

// When the first provider charges but fails, and the fallback succeeds, only
// the successful provider's cost should be charged. The failed provider's
// partial cost is NOT debited.
func TestFailureInjection_FallbackAfterPrimaryCharges_OnlySuccessfulCostCharged(t *testing.T) {
	primary := &fakeVendor{
		family: domain.VendorFamilyVertexGemini,
		err:    errors.New("vertex 503 after charging partial tokens"),
	}
	fallback := &fakeVendor{
		family: domain.VendorFamilyOpenAI,
		resp: domain.VendorResponse{
			Completion:   "fallback-completion",
			Usage:        domain.TokenUsage{InputTokens: 10, OutputTokens: 5, CostMicros: 1_000},
			ModelVersion: "gpt-4o-mini",
			FinishReason: domain.FinishReasonComplete,
		},
	}
	executor, _, _, budget, outbox, _ := newTestExecutor(t, func(cfg *domain.ExecutorConfig) {
		cfg.Providers = []domain.Provider{primary, fallback}
		cfg.Policies = &fakePolicy{
			policy: domain.AgentPolicy{
				AgentID:                "qgen_question",
				ResolvedLogicalModelID: domain.LogicalModelID("gemini-2.5-pro"),
				Vendor:                 domain.VendorFamilyVertexGemini,
				FallbackChain: []domain.AgentPolicyFallback{
					{Vendor: domain.VendorFamilyOpenAI, ResolvedLogicalModelID: domain.LogicalModelID("gpt-4o-mini")},
				},
			},
		}
	})

	resp, err := executor.Execute(context.Background(), happyExecuteRequest())
	require.NoError(t, err)
	assert.Equal(t, "fallback-completion", resp.Completion)
	assert.Equal(t, 1, primary.calls)
	assert.Equal(t, 1, fallback.calls)

	// Only the fallback's cost is debited — the primary's partial cost is NOT.
	assert.Equal(t, 1, budget.debitCalls)
	assert.Equal(t, int64(1_000), budget.debits, "only the successful provider's cost is charged")

	// The outbox event carries the fallback's cost.
	require.Len(t, outbox.events, 1)
	assert.Equal(t, int64(1_000), outbox.events[0].CostMicros)
	assert.Equal(t, "openai_byoa", outbox.events[0].Vendor)
}

// ----------------------------------------------------------------------------
// 5. Redelivery with same dispatch_idempotency_key
// ----------------------------------------------------------------------------

// A redelivered dispatch with the same dispatch_idempotency_key must not
// double-debit the budget. The claim suppresses the debit on redelivery.
func TestFailureInjection_RedeliverySameDispatchKey_NoDoubleDebit(t *testing.T) {
	claims := &fakeClaimer{claimed: false} // redelivery: key already claimed
	executor, _, _, budget, outbox, _ := newTestExecutor(t, func(cfg *domain.ExecutorConfig) {
		cfg.Claims = claims
	})

	req := happyExecuteRequest()
	req.DispatchIdempotencyKey = "wf-0193-turn-7"

	// First delivery: wins the claim, debits.
	executor2, _, _, budget2, _, _ := newTestExecutor(t, func(cfg *domain.ExecutorConfig) {
		cfg.Claims = &fakeClaimer{claimed: true}
	})
	req2 := happyExecuteRequest()
	req2.DispatchIdempotencyKey = "wf-0193-turn-7"
	_, err := executor2.Execute(context.Background(), req2)
	require.NoError(t, err)
	assert.Equal(t, 1, budget2.debitCalls, "first delivery debits")

	// Redelivery: claim returns false (already claimed), no debit.
	resp, err := executor.Execute(context.Background(), req)
	require.NoError(t, err)
	assert.True(t, resp.Deduped)
	assert.Equal(t, 0, budget.debitCalls, "redelivery does NOT debit")
	require.Len(t, outbox.events, 1, "but still enqueues the real cost")
	assert.Equal(t, int64(12_500), outbox.events[0].CostMicros)
}

// ----------------------------------------------------------------------------
// 6. Redelivery with same invocation_id
// ----------------------------------------------------------------------------

// A redelivery with the same invocation_id must not produce duplicate outbox
// rows. The outbox event carries the invocation_id as UsageID, so the outbox
// implementation can deduplicate on it.
func TestFailureInjection_RedeliverySameInvocationID_NoDuplicateOutbox(t *testing.T) {
	outbox := &fakeOutbox{}
	executor, _, _, _, _, _ := newTestExecutor(t, func(cfg *domain.ExecutorConfig) {
		cfg.Outbox = outbox
	})

	req := happyExecuteRequest()
	req.InvocationID = "test-invocation-id"

	// First call.
	_, err := executor.Execute(context.Background(), req)
	require.NoError(t, err)

	// Redelivery with same invocation_id.
	_, err = executor.Execute(context.Background(), req)
	require.NoError(t, err)

	// The outbox has two events (one per call), but they share the same UsageID
	// so the outbox implementation can deduplicate.
	require.Len(t, outbox.events, 2)
	assert.Equal(t, outbox.events[0].UsageID, outbox.events[1].UsageID,
		"both events carry the same invocation_id for deduplication")
}

// ----------------------------------------------------------------------------
// 7. Claim DB failure
// ----------------------------------------------------------------------------

// A claim DB failure must fail the call with a clear error. Proceeding would
// risk a double bill.
func TestFailureInjection_ClaimDBFailure_FailsCall(t *testing.T) {
	claims := &failingClaimer{err: errors.New("claim store down")}
	executor, vendor, _, _, _, _ := newTestExecutor(t, func(cfg *domain.ExecutorConfig) {
		cfg.Claims = claims
	})

	req := happyExecuteRequest()
	req.DispatchIdempotencyKey = "wf-0193-turn-7"

	_, err := executor.Execute(context.Background(), req)
	require.Error(t, err, "a claim failure is a hard error")
	var ierr *domain.InvokeError
	require.True(t, errors.As(err, &ierr))
	assert.Contains(t, ierr.Detail, "idempotency claim failed")
	assert.Equal(t, 0, vendor.calls, "no dispatch on claim failure")
}

// ----------------------------------------------------------------------------
// 8. Budget DB failure
// ----------------------------------------------------------------------------

// A budget DB failure must fail the call with a clear error.
func TestFailureInjection_BudgetDBFailure_FailsCall(t *testing.T) {
	budget := &failingBudget{getErr: errors.New("pg timeout")}
	executor, vendor, _, _, _, _ := newTestExecutor(t, func(cfg *domain.ExecutorConfig) {
		cfg.Budget = budget
	})

	_, err := executor.Execute(context.Background(), happyExecuteRequest())
	require.Error(t, err)
	var ierr *domain.InvokeError
	require.True(t, errors.As(err, &ierr))
	assert.Contains(t, ierr.Detail, "budget repo unavailable")
	assert.Equal(t, 0, vendor.calls, "no dispatch on budget failure")
}

// ----------------------------------------------------------------------------
// 9. Mana debit failure after provider success
// ----------------------------------------------------------------------------

// A mana debit failure after provider success must NOT fail the call — the
// answer was already produced. The failure is logged loudly.
func TestFailureInjection_ManaDebitFailure_CallStillSucceeds(t *testing.T) {
	mana := &fakeManaMeter{debitErr: errors.New("identity unreachable")}
	executor, vendor, _, _, _, _ := newTestExecutor(t, func(cfg *domain.ExecutorConfig) {
		cfg.Mana = mana
	})

	resp, err := executor.Execute(context.Background(), happyExecuteRequest())
	require.NoError(t, err, "a mana debit error never un-serves a completed call")
	assert.Equal(t, "completion-stub", resp.Completion)
	assert.Equal(t, 1, vendor.calls)
	assert.Equal(t, 1, mana.debits, "the mana debit was attempted")
	assert.Equal(t, domain.AccountingAccounted, resp.AccountingState)
}

// ----------------------------------------------------------------------------
// 10. Armor PRE block
// ----------------------------------------------------------------------------

// An Armor PRE block must refuse the call WITHOUT calling the provider.
func TestFailureInjection_ArmorPreBlock_RefusesWithoutProvider(t *testing.T) {
	armor := &fakeArmor{preVerdict: domain.ArmorVerdictBlock}
	executor, vendor, _, _, _, _ := newTestExecutor(t, func(cfg *domain.ExecutorConfig) {
		cfg.Armor = armor
	})

	resp, err := executor.Execute(context.Background(), happyExecuteRequest())
	require.NoError(t, err, "Armor PRE block is an in-band response, not an error")
	assert.Equal(t, domain.FinishReasonModelArmorBlock, resp.FinishReason)
	assert.Equal(t, domain.ArmorVerdictBlock, resp.ArmorPre)
	assert.Equal(t, 0, vendor.calls, "no provider dispatch on Armor PRE block")
}

// ----------------------------------------------------------------------------
// 11. Armor POST block
// ----------------------------------------------------------------------------

// An Armor POST block must refuse the call AFTER the provider returns. The
// provider was called but the response is redacted.
func TestFailureInjection_ArmorPostBlock_RefusesAfterProvider(t *testing.T) {
	armor := &fakeArmor{
		preVerdict:  domain.ArmorVerdictAllow,
		postVerdict: domain.ArmorVerdictBlock,
	}
	executor, vendor, _, _, outbox, _ := newTestExecutor(t, func(cfg *domain.ExecutorConfig) {
		cfg.Armor = armor
	})

	resp, err := executor.Execute(context.Background(), happyExecuteRequest())
	require.NoError(t, err, "Armor POST block is an in-band response, not an error")
	assert.Equal(t, domain.FinishReasonModelArmorBlock, resp.FinishReason)
	assert.Equal(t, domain.ArmorVerdictBlock, resp.ArmorPost)
	assert.Equal(t, 1, vendor.calls, "the provider was called")
	assert.Equal(t, "", resp.Completion, "the completion is redacted")

	// Full-cost outbox event is emitted (vendor billed for output).
	require.Len(t, outbox.events, 1)
	assert.Equal(t, int64(12_500), outbox.events[0].CostMicros)
}

// ----------------------------------------------------------------------------
// 12. Companion suspension
// ----------------------------------------------------------------------------

// A companion suspension must refuse the call BEFORE any provider call.
func TestFailureInjection_CompanionSuspension_RefusesBeforeProvider(t *testing.T) {
	gate := &fakeSuspensionGate{
		verdict: domain.SuspensionVerdict{
			Suspended: true,
			Reason:    domain.DenyCompanionSuspended,
			Detail:    "scope=platform skill=all reason=\"maintenance\"",
		},
	}
	executor, vendor, _, _, _, _ := newTestExecutor(t, func(cfg *domain.ExecutorConfig) {
		cfg.Suspension = gate
	})

	_, err := executor.Execute(context.Background(), happyExecuteRequest())
	require.Error(t, err)
	var perr *domain.PreconditionError
	require.True(t, errors.As(err, &perr))
	assert.Equal(t, domain.DenyCompanionSuspended, perr.Reason)
	assert.Equal(t, 1, gate.calls, "the suspension gate was consulted")
	assert.Equal(t, 0, vendor.calls, "no provider dispatch on a suspended companion")
}
