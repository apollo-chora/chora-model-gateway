package domain_test

import (
	"context"
	"errors"
	"sync/atomic"
	"testing"

	"github.com/apollo-chora/chora-model-gateway/internal/domain"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// ----------------------------------------------------------------------------
// Per-tenant concurrency ceiling (CHORA_LLM_MAX_CONCURRENT_PER_TENANT) and the
// demo embedding-route pin (CHORA_EMBEDDING_PIN_*).
// ----------------------------------------------------------------------------

// blockingVendor holds the FIRST Generate call inside the provider until the
// test releases it, so a second invocation can be issued while the first is in
// flight. Later calls return immediately (the provider is no longer blocked).
type blockingVendor struct {
	family  domain.VendorFamily
	entered chan struct{}
	release chan struct{}
	calls   int32
}

func newBlockingVendor(family domain.VendorFamily) *blockingVendor {
	return &blockingVendor{family: family, entered: make(chan struct{}, 1), release: make(chan struct{})}
}

func (b *blockingVendor) Family() domain.VendorFamily { return b.family }

func (b *blockingVendor) Generate(ctx context.Context, req domain.VendorRequest) (domain.VendorResponse, error) {
	if atomic.AddInt32(&b.calls, 1) == 1 {
		b.entered <- struct{}{}
		<-b.release
	}
	return domain.VendorResponse{
		Completion:   "completion-stub",
		Usage:        domain.TokenUsage{InputTokens: 10, OutputTokens: 5, CostMicros: 100},
		ModelVersion: "gemini-2.5-pro",
		FinishReason: domain.FinishReasonComplete,
	}, nil
}

// blockingEmbedVendor is blockingVendor for the Embed path.
type blockingEmbedVendor struct {
	entered chan struct{}
	release chan struct{}
	calls   int32
}

func newBlockingEmbedVendor() *blockingEmbedVendor {
	return &blockingEmbedVendor{entered: make(chan struct{}, 1), release: make(chan struct{})}
}

func (b *blockingEmbedVendor) Family() domain.VendorFamily { return domain.VendorFamilyVertexGemini }

func (b *blockingEmbedVendor) EmbedText(ctx context.Context, req domain.EmbedVendorRequest) (domain.EmbedVendorResponse, error) {
	if atomic.AddInt32(&b.calls, 1) == 1 {
		b.entered <- struct{}{}
		<-b.release
	}
	return domain.EmbedVendorResponse{Values: []float32{0.1}, ModelVersion: "text-embedding-004", InputTokens: 3}, nil
}

// blockingGroundedVendor is blockingVendor for the GroundedSearch path.
type blockingGroundedVendor struct {
	entered chan struct{}
	release chan struct{}
	calls   int32
}

func newBlockingGroundedVendor() *blockingGroundedVendor {
	return &blockingGroundedVendor{entered: make(chan struct{}, 1), release: make(chan struct{})}
}

func (b *blockingGroundedVendor) Family() domain.VendorFamily { return domain.VendorFamilyVertexGemini }

func (b *blockingGroundedVendor) GroundedGenerate(ctx context.Context, req domain.GroundedVendorRequest) (domain.GroundedVendorResponse, error) {
	if atomic.AddInt32(&b.calls, 1) == 1 {
		b.entered <- struct{}{}
		<-b.release
	}
	return domain.GroundedVendorResponse{
		Answer:       "grounded answer",
		Citations:    []domain.GroundedCitation{{URI: "https://redirect/1", Title: "t", Domain: "nasa.gov"}},
		Usage:        domain.TokenUsage{InputTokens: 10, OutputTokens: 5, CostMicros: 100},
		ModelVersion: "gemini-2.5-flash",
		FinishReason: domain.FinishReasonComplete,
	}, nil
}

// ----------------------------------------------------------------------------
// Limiter semantics.
// ----------------------------------------------------------------------------

func TestTenantConcurrencyLimiter_UnlimitedWhenOff(t *testing.T) {
	l := domain.NewTenantConcurrencyLimiter(0)
	assert.Nil(t, l, "0 = unlimited/off must be the nil limiter")
	release, ok := l.Acquire("tenant-1")
	require.True(t, ok)
	require.NotNil(t, release)
	release() // must not panic
	assert.Equal(t, 0, l.Limit())
}

func TestTenantConcurrencyLimiter_PerTenantIsolationAndRelease(t *testing.T) {
	l := domain.NewTenantConcurrencyLimiter(1)
	require.NotNil(t, l)
	assert.Equal(t, 1, l.Limit())

	rel1, ok := l.Acquire("tenant-1")
	require.True(t, ok)
	// A different tenant is unaffected.
	rel2, ok := l.Acquire("tenant-2")
	require.True(t, ok)
	// The same tenant is refused while its slot is held.
	_, ok = l.Acquire("tenant-1")
	assert.False(t, ok)

	rel1()
	rel1() // double release must be a no-op (sync.Once), not a second free
	rel3, ok := l.Acquire("tenant-1")
	assert.True(t, ok)
	rel2()
	rel3()
}

// A double release must not free a slot that a LATER Acquire took.
func TestTenantConcurrencyLimiter_DoubleReleaseDoesNotFreeAnotherSlot(t *testing.T) {
	l := domain.NewTenantConcurrencyLimiter(1)
	rel1, ok := l.Acquire("tenant-1")
	require.True(t, ok)
	rel1()
	rel1()

	rel2, ok := l.Acquire("tenant-1")
	require.True(t, ok)
	_, ok = l.Acquire("tenant-1")
	assert.False(t, ok, "the second holder's slot must still be held")
	rel2()
}

// ----------------------------------------------------------------------------
// Invoke / Embed / GroundedSearch: enforced before billable dispatch.
// ----------------------------------------------------------------------------

func TestInvoke_ConcurrencyLimitRefusesSecondInFlight(t *testing.T) {
	vendor := newBlockingVendor(domain.VendorFamilyVertexGemini)
	svc, _, _, _, _, _ := newTestService(t, func(cfg *domain.ServiceConfig) {
		cfg.Vendors = []domain.VendorClient{vendor}
		cfg.MaxConcurrentPerTenant = 1
	})

	firstDone := make(chan error, 1)
	go func() {
		_, err := svc.Invoke(context.Background(), happyRequest())
		firstDone <- err
	}()
	<-vendor.entered

	_, err := svc.Invoke(context.Background(), happyRequest())
	var overloaded *domain.OverloadedError
	require.ErrorAs(t, err, &overloaded)
	assert.Equal(t, 1, overloaded.Limit)
	assert.Equal(t, int32(1), atomic.LoadInt32(&vendor.calls), "the refused invocation never reached the vendor")

	close(vendor.release)
	require.NoError(t, <-firstDone)

	// The slot was released on the success path: a later call proceeds.
	_, err = svc.Invoke(context.Background(), happyRequest())
	require.NoError(t, err)
	assert.Equal(t, int32(2), atomic.LoadInt32(&vendor.calls))
}

func TestEmbed_ConcurrencyLimitRefusesSecondInFlight(t *testing.T) {
	vendor := newBlockingEmbedVendor()
	svc := newEmbedServiceWithConfig(t, false, nil, vendor, &fakeEmbedOutbox{}, func(cfg *domain.ServiceConfig) {
		cfg.MaxConcurrentPerTenant = 1
	})

	firstDone := make(chan error, 1)
	go func() {
		_, err := svc.Embed(context.Background(), validEmbedRequest())
		firstDone <- err
	}()
	<-vendor.entered

	_, err := svc.Embed(context.Background(), validEmbedRequest())
	var overloaded *domain.OverloadedError
	require.ErrorAs(t, err, &overloaded)
	assert.Equal(t, int32(1), atomic.LoadInt32(&vendor.calls), "the refused embed never reached the vendor")

	close(vendor.release)
	require.NoError(t, <-firstDone)
}

func TestGroundedSearch_ConcurrencyLimitRefusesSecondInFlight(t *testing.T) {
	vendor := newBlockingGroundedVendor()
	svc, audit := newGroundedServiceWithLimit(t, vendor, 1)

	firstDone := make(chan error, 1)
	go func() {
		_, err := svc.GroundedSearch(context.Background(), groundedReq())
		firstDone <- err
	}()
	<-vendor.entered

	_, err := svc.GroundedSearch(context.Background(), groundedReq())
	var gerr *domain.GroundedSearchError
	require.ErrorAs(t, err, &gerr)
	assert.Equal(t, domain.EgressDenyOverloaded, gerr.Reason)
	var overloaded *domain.OverloadedError
	assert.ErrorAs(t, err, &overloaded)
	assert.Equal(t, int32(1), atomic.LoadInt32(&vendor.calls), "the refused search never egressed")

	// The refusal is audited like every other pre-egress deny (the invariant is
	// exactly one egress audit event per call).
	require.Len(t, audit.events, 1)
	assert.Equal(t, domain.EgressAuditDenied, audit.events[0].Result)
	assert.Equal(t, domain.EgressDenyOverloaded, audit.events[0].DenialReason)

	close(vendor.release)
	require.NoError(t, <-firstDone)
}

// newGroundedServiceWithLimit builds a grounded-capable Service with the
// per-tenant ceiling set and the supplied grounded vendor. It returns the
// audit fake so a test can assert the egress-audit invariant.
func newGroundedServiceWithLimit(t *testing.T, vendor domain.GroundedVendorClient, limit int) (*domain.Service, *fakeEgressAudit) {
	t.Helper()
	audit := &fakeEgressAudit{}
	svc, err := domain.NewService(domain.ServiceConfig{
		Vendors:                []domain.VendorClient{&fakeVendor{family: domain.VendorFamilyVertexGemini}},
		Armor:                  &fakeArmor{preVerdict: domain.ArmorVerdictAllow, postVerdict: domain.ArmorVerdictAllow},
		Budget:                 &fakeBudget{state: &domain.BudgetState{BudgetUSDMicros: 1_000_000, Policy: domain.BudgetPolicyBlock}},
		Outbox:                 &fakeOutbox{},
		Policies:               &fakePolicy{policy: domain.AgentPolicy{ResolvedLogicalModelID: "gemini-2.5-flash", Vendor: domain.VendorFamilyVertexGemini, ArmorTemplate: "tpl"}},
		GatewayVersion:         "chora-model-gateway:test",
		EgressGate:             &fakeEgressGate{auth: domain.EgressAuthorization{Allowed: true}},
		Grounded:               vendor,
		Mana:                   &fakeMana{quote: domain.ManaQuote{Affordable: true, RequiredUnits: 1, AvailableUnits: 10}, debit: domain.ManaDebit{Success: true}},
		EgressAudit:            audit,
		MaxConcurrentPerTenant: limit,
		Now:                    fixedTime,
		NewID:                  func() string { return "gs-invocation-id" },
	})
	require.NoError(t, err)
	return svc, audit
}

// ----------------------------------------------------------------------------
// Embedding-route pin: a runtime override is refused before dispatch.
// ----------------------------------------------------------------------------

func TestEmbed_EmbeddingRoutePinRefusesOverride(t *testing.T) {
	vendor := &fakeEmbedVendor{family: domain.VendorFamilyVertexGemini, response: domain.EmbedVendorResponse{Values: []float32{0.1}, ModelVersion: "text-embedding-004"}}
	svc := newEmbedServiceWithConfig(t, false, nil, vendor, &fakeEmbedOutbox{}, func(cfg *domain.ServiceConfig) {
		cfg.EmbeddingPin = &domain.EmbeddingRoutePin{LogicalID: domain.DefaultEmbeddingModelID}
	})

	// The pinned id is accepted.
	req := validEmbedRequest()
	req.LogicalModelID = domain.DefaultEmbeddingModelID
	_, err := svc.Embed(context.Background(), req)
	require.NoError(t, err)

	// An allowlisted but DIFFERENT embedding model is refused, not re-routed.
	req = validEmbedRequest()
	req.LogicalModelID = "text-embedding-005"
	_, err = svc.Embed(context.Background(), req)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "pinned")
	assert.Equal(t, 1, vendor.calls, "the refused override never reached the vendor")
}

// ----------------------------------------------------------------------------
// Safety property (ChatGPT round 8): NO fallback to allow when the budget
// repository errors. Every dispatch path refuses.
// ----------------------------------------------------------------------------

func TestBudgetRepoError_FailsClosedOnEveryPath(t *testing.T) {
	repoErr := errors.New("postgres: connection refused")

	// Invoke.
	svc, _, _, _, _, _ := newTestService(t, func(cfg *domain.ServiceConfig) {
		cfg.Budget = &fakeBudget{getErr: repoErr}
	})
	_, err := svc.Invoke(context.Background(), happyRequest())
	var ierr *domain.InvokeError
	require.ErrorAs(t, err, &ierr)
	assert.Equal(t, "budget repo unavailable", ierr.Detail)

	// Embed (budget is read only in fail-closed "budget required" mode).
	embedSvc := newEmbedServiceWithConfig(t, true, nil, &fakeEmbedVendor{}, &fakeEmbedOutbox{}, func(cfg *domain.ServiceConfig) {
		cfg.Budget = &fakeBudget{getErr: repoErr}
	})
	_, err = embedSvc.Embed(context.Background(), validEmbedRequest())
	require.Error(t, err)
	assert.Contains(t, err.Error(), "budget repo unavailable")

	// GroundedSearch (the budget is read on every grounded call).
	gSvc, _ := newGroundedTestService(t, func(f *groundedFakes) {
		f.budget = &fakeBudget{getErr: repoErr}
	})
	_, err = gSvc.GroundedSearch(context.Background(), groundedReq())
	var gerr *domain.GroundedSearchError
	require.ErrorAs(t, err, &gerr)
	assert.Equal(t, domain.EgressDenyBudget, gerr.Reason)
}
