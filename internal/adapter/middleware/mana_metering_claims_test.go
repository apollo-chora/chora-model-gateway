package middleware_test

// R22 / ADR-254 D7: a redelivered dispatch bills once. The domain service
// takes the keyed claim (gcid, dispatch_idempotency_key, action_code) BEFORE
// any spend and reports the result on InvokeResponse.Deduped; the metering
// seam skips the mana debit for a deduped turn — one claim suppresses BOTH
// the tenant-budget debit and the mana debit.

import (
	"context"
	"testing"

	"github.com/5007-Capstone/chora/services/chora-model-gateway/internal/adapter/middleware"
	"github.com/5007-Capstone/chora/services/chora-model-gateway/internal/domain"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func dispatchedReq() domain.InvokeRequest {
	return domain.InvokeRequest{
		InvocationID:           "inv-1",
		TenantID:               "tenant-1",
		GCID:                   "gcid-1",
		AgentID:                "companion_chat",
		Surface:                "companion_chat",
		ActionCode:             "companion_chat_turn_basic",
		DispatchIdempotencyKey: "wf-0193-turn-7",
		Prompt:                 "hi",
	}
}

func servedResp() domain.InvokeResponse {
	return domain.InvokeResponse{InvocationID: "inv-1", FinishReason: domain.FinishReasonComplete}
}

// First delivery: the domain service won the claim (Deduped=false) → the
// mana debit is taken.
func TestManaMetering_DispatchKey_FirstDelivery_Debits(t *testing.T) {
	next := &fakeNext{resp: servedResp()}
	mp := &fakeMana{quote: middleware.ManaQuote{Affordable: true}, debit: middleware.ManaDebit{Success: true}}
	m := middleware.NewManaMetering(next, mp, testConfig())

	_, err := m.Invoke(context.Background(), dispatchedReq())
	require.NoError(t, err)
	assert.Equal(t, 1, mp.debitCalls, "first delivery debits once")
}

// Redelivery: the domain service's claim lost (Deduped=true) → the call
// proceeds but the mana debit is skipped.
func TestManaMetering_DispatchKey_Redelivery_DedupesDebit(t *testing.T) {
	resp := servedResp()
	resp.Deduped = true
	next := &fakeNext{resp: resp}
	mp := &fakeMana{quote: middleware.ManaQuote{Affordable: true}, debit: middleware.ManaDebit{Success: true}}
	m := middleware.NewManaMetering(next, mp, testConfig())

	got, err := m.Invoke(context.Background(), dispatchedReq())
	require.NoError(t, err, "the call proceeds; only the debit is skipped")
	assert.Equal(t, domain.FinishReasonComplete, got.FinishReason)
	assert.Equal(t, 0, mp.debitCalls, "a redelivered dispatch bills once")
}

// No dispatch key: nothing to dedupe — the debit is taken.
func TestManaMetering_NoDispatchKey_DebitsWithoutClaim(t *testing.T) {
	next := &fakeNext{resp: servedResp()}
	mp := &fakeMana{quote: middleware.ManaQuote{Affordable: true}, debit: middleware.ManaDebit{Success: true}}
	m := middleware.NewManaMetering(next, mp, testConfig())

	req := dispatchedReq()
	req.DispatchIdempotencyKey = ""
	_, err := m.Invoke(context.Background(), req)
	require.NoError(t, err)
	assert.Equal(t, 1, mp.debitCalls)
}

// Non-billable response: no debit even when not deduped.
func TestManaMetering_DispatchKey_NoDebitWhenNotServed(t *testing.T) {
	next := &fakeNext{resp: domain.InvokeResponse{InvocationID: "inv-1", FinishReason: domain.FinishReasonModelArmorBlock}}
	mp := &fakeMana{quote: middleware.ManaQuote{Affordable: true}}
	m := middleware.NewManaMetering(next, mp, testConfig())

	_, err := m.Invoke(context.Background(), dispatchedReq())
	require.NoError(t, err)
	assert.Equal(t, 0, mp.debitCalls, "nothing billable, nothing debited")
}
