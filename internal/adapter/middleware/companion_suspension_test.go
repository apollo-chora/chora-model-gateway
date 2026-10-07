package middleware_test

import (
	"context"
	"errors"
	"testing"

	"github.com/5007-Capstone/chora/services/chora-model-gateway/internal/adapter/middleware"
	"github.com/5007-Capstone/chora/services/chora-model-gateway/internal/domain"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// fakeGate records the suspension reads the decorator performs.
type fakeGate struct {
	verdict middleware.SuspensionVerdict
	err     error
	calls   int
	last    struct{ tenant, action string }
}

func (f *fakeGate) CheckSuspension(_ context.Context, tenantID, actionCode string) (middleware.SuspensionVerdict, error) {
	f.calls++
	f.last.tenant = tenantID
	f.last.action = actionCode
	return f.verdict, f.err
}

func suspensionReq(surface string) domain.InvokeRequest {
	return domain.InvokeRequest{
		InvocationID: "inv-1",
		TenantID:     "01234567-89ab-7cde-8f01-234567890aaa",
		GCID:         "gcid-1",
		AgentID:      "companion_chat",
		Surface:      surface,
		ActionCode:   "companion_chat_turn_basic",
		Prompt:       "hello",
	}
}

// ADR-254 D7: an ABSENT surface is a loud condition, denied before any debit.
func TestCompanionSuspension_AbsentSurface_DeniesBeforeNext(t *testing.T) {
	next := &fakeNext{}
	gate := &fakeGate{}
	dec := middleware.NewCompanionSuspension(next, gate, middleware.SuspensionConfig{})

	_, err := dec.Invoke(context.Background(), suspensionReq(""))
	require.Error(t, err)
	var perr *domain.PreconditionError
	require.True(t, errors.As(err, &perr), "must be a PreconditionError, got %T", err)
	assert.Equal(t, domain.DenySurfaceUnstamped, perr.Reason)
	assert.Equal(t, 0, next.called, "the LLM must not be called")
	assert.Equal(t, 0, gate.calls, "no read is needed to refuse an unstamped call")
}

// A non-companion surface is not governed by the suspension: no read, straight through.
func TestCompanionSuspension_NonCompanionSurface_PassesWithoutRead(t *testing.T) {
	next := &fakeNext{resp: domain.InvokeResponse{InvocationID: "inv-1", FinishReason: domain.FinishReasonComplete}}
	gate := &fakeGate{}
	dec := middleware.NewCompanionSuspension(next, gate, middleware.SuspensionConfig{})

	resp, err := dec.Invoke(context.Background(), suspensionReq("qgen"))
	require.NoError(t, err)
	assert.Equal(t, "inv-1", resp.InvocationID)
	assert.Equal(t, 1, next.called)
	assert.Equal(t, 0, gate.calls)
}

func TestCompanionSuspension_CompanionSurface_NotSuspended_Passes(t *testing.T) {
	next := &fakeNext{resp: domain.InvokeResponse{InvocationID: "inv-1", FinishReason: domain.FinishReasonComplete}}
	gate := &fakeGate{verdict: middleware.SuspensionVerdict{Suspended: false}}
	dec := middleware.NewCompanionSuspension(next, gate, middleware.SuspensionConfig{})

	for _, surface := range []string{"companion_chat", "companion_diagnosis"} {
		_, err := dec.Invoke(context.Background(), suspensionReq(surface))
		require.NoError(t, err, surface)
	}
	assert.Equal(t, 2, next.called)
	assert.Equal(t, 2, gate.calls, "every companion call reads the suspension, uncached")
	assert.Equal(t, "01234567-89ab-7cde-8f01-234567890aaa", gate.last.tenant)
	assert.Equal(t, "companion_chat_turn_basic", gate.last.action)
}

// ADR-252 D5: a contained turn refuses honestly with companion_suspended and costs nothing.
func TestCompanionSuspension_Suspended_DeniesBeforeNext(t *testing.T) {
	next := &fakeNext{}
	gate := &fakeGate{verdict: middleware.SuspensionVerdict{Suspended: true, Scope: "tenant", Reason: "unsafe persona drift"}}
	dec := middleware.NewCompanionSuspension(next, gate, middleware.SuspensionConfig{})

	_, err := dec.Invoke(context.Background(), suspensionReq("companion_chat"))
	require.Error(t, err)
	var perr *domain.PreconditionError
	require.True(t, errors.As(err, &perr))
	assert.Equal(t, domain.DenyCompanionSuspended, perr.Reason)
	assert.Contains(t, perr.Detail, "tenant")
	assert.Equal(t, 0, next.called, "deny-before-debit: nothing downstream runs")
}

// ADR-252 D6: an UNREADABLE store denies (fail closed); an EMPTY store allows.
// The two are told apart by error, never by an empty result.
func TestCompanionSuspension_ReadError_FailsClosed(t *testing.T) {
	next := &fakeNext{}
	gate := &fakeGate{err: errors.New("relation platform_companion_suspension does not exist")}
	dec := middleware.NewCompanionSuspension(next, gate, middleware.SuspensionConfig{})

	_, err := dec.Invoke(context.Background(), suspensionReq("companion_diagnosis"))
	require.Error(t, err)
	var perr *domain.PreconditionError
	require.True(t, errors.As(err, &perr))
	assert.Equal(t, domain.DenySuspensionUnreadable, perr.Reason)
	assert.Equal(t, 0, next.called)
}

func TestNewCompanionSuspension_RequiresRealWiring(t *testing.T) {
	assert.Panics(t, func() { middleware.NewCompanionSuspension(nil, &fakeGate{}, middleware.SuspensionConfig{}) })
	assert.Panics(t, func() { middleware.NewCompanionSuspension(&fakeNext{}, nil, middleware.SuspensionConfig{}) })
}
