package domain_test

import (
	"context"
	"errors"
	"strings"
	"testing"
	"time"

	"github.com/apollo-chora/chora-model-gateway/internal/domain"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// G1'-1 (owner-ruled 2026-08-07): embeddings ride the chokepoint. The Embed
// flow must ledger every call (that is the point of the re-route), attach no
// price, run no Armor leg (permissive entry, Bypassed markers), and fail
// loud when unwired rather than silently falling back to an unverified
// adapter.
//
// Release B: the flow resolves the logical model through the model registry
// and dispatches through the adapter for the RESOLVED provider family, so the
// registry — not the logical id's name shape — decides the route. The fakes
// below are therefore OpenAI-family: that is what the deployment's embedding
// entry declares.

type fakeEmbedVendor struct {
	family   domain.VendorFamily
	lastReq  domain.EmbedVendorRequest
	response domain.EmbedVendorResponse
	err      error
	calls    int
}

func (f *fakeEmbedVendor) Family() domain.VendorFamily { return f.family }
func (f *fakeEmbedVendor) EmbedText(_ context.Context, req domain.EmbedVendorRequest) (domain.EmbedVendorResponse, error) {
	f.calls++
	f.lastReq = req
	return f.response, f.err
}

type fakeEmbedOutbox struct {
	events []domain.TokenUsageEvent
	err    error
}

func (f *fakeEmbedOutbox) EnqueueTokenUsageRecorded(_ context.Context, evt domain.TokenUsageEvent) error {
	f.events = append(f.events, evt)
	return f.err
}

// embedClients adapts the single-embedder test helper to the per-family
// table: a nil embedder means "nothing wired", which is itself a case under
// test.
func embedClients(embedder domain.EmbeddingClient) []domain.EmbeddingClient {
	if embedder == nil {
		return nil
	}
	return []domain.EmbeddingClient{embedder}
}

// embedModelResolver is the test registry: the deployment's embedding route.
// The logical id `text-embedding-004` is an ALIAS — the entry behind it is an
// OpenAI-provider upstream, not a Google publisher model.
func embedModelResolver() *mapModelResolver {
	return &mapModelResolver{infos: map[domain.LogicalModelID]domain.ModelInfo{
		"text-embedding-004": {
			ID:             "text-embedding-004",
			Vendor:         domain.VendorFamilyOpenAI,
			UpstreamModel:  "text-embedding-004",
			BaseURL:        "https://api.openai.com/v1",
			Capabilities:   []string{"embeddings"},
			APIKeyEnv:      "EMBEDDING_LLM_API_KEY",
			APIKey:         "test-key",
			FallbackIDs:    nil,
		},
	}}
}

func newEmbedService(t *testing.T, embedder domain.EmbeddingClient, outbox domain.OutboxWriter) *domain.Service {
	t.Helper()
	return newEmbedServiceWithConfig(t, false, nil, embedder, outbox)
}

// newEmbedServiceWithConfig is newEmbedService with the fail-closed "budget
// required" mode (CHORA_LLM_BUDGET_REQUIRED) and its tenant-scoped override
// (CHORA_LLM_BUDGET_REQUIRED_TENANTS) settable. The budget fake carries a nil
// state (no active window) so the flags' effect is observable; a mod can
// install a budget with an active window.
func newEmbedServiceWithConfig(t *testing.T, budgetRequired bool, tenants domain.BudgetRequiredTenants, embedder domain.EmbeddingClient, outbox domain.OutboxWriter, mods ...func(cfg *domain.ServiceConfig)) *domain.Service {
	t.Helper()
	cfg := domain.ServiceConfig{
		Vendors:        []domain.VendorClient{&fakeVendor{family: domain.VendorFamilyOpenAI}},
		Armor:          &fakeArmor{},
		Budget:         &fakeBudget{},
		Outbox:         outbox,
		Policies:       &fakePolicy{},
		GatewayVersion: "chora-model-gateway:test",
		Embedders:      embedClients(embedder),
		Models:         embedModelResolver(),
		BudgetRequired: budgetRequired,
		BudgetRequiredTenants: tenants,
		Now:            func() time.Time { return time.Unix(1754500000, 0) },
		NewID:          func() string { return "01988888-8888-7888-8888-888888888888" },
	}
	for _, m := range mods {
		m(&cfg)
	}
	svc, err := domain.NewService(cfg)
	if err != nil {
		t.Fatalf("NewService: %v", err)
	}
	return svc
}

func validEmbedRequest() domain.EmbedFlowRequest {
	return domain.EmbedFlowRequest{
		TenantID:    "11111111-1111-7111-8111-111111111111",
		GCID:        "00000000-0000-7000-8000-000000001999",
		AgentID:     "consumption_memory_embedder",
		CrewKind:    "familiar",
		Text:        "adding fractions with unlike denominators",
		TaskType:    "RETRIEVAL_DOCUMENT",
		Traceparent: "00-abc-def-01",
	}
}

func TestEmbed_happyPathLedgersWithZeroCostAndBypassedMarkers(t *testing.T) {
	vendor := &fakeEmbedVendor{
		family: domain.VendorFamilyOpenAI,
		response: domain.EmbedVendorResponse{
			Values:       []float32{0.1, 0.2, 0.3},
			ModelVersion: "text-embedding-004",
			InputTokens:  7,
		},
	}
	outbox := &fakeEmbedOutbox{}
	svc := newEmbedService(t, vendor, outbox)

	resp, err := svc.Embed(context.Background(), validEmbedRequest())
	if err != nil {
		t.Fatalf("Embed: %v", err)
	}
	if len(resp.Values) != 3 {
		t.Fatalf("want the vendor vector back; got %v", resp.Values)
	}
	if resp.ModelVersion != "text-embedding-004" {
		t.Errorf("model_version = %q", resp.ModelVersion)
	}
	if resp.Usage.CostMicros != 0 {
		t.Errorf("no price may be attached (owner ruling); got cost_micros=%d", resp.Usage.CostMicros)
	}
	if resp.Usage.InputTokens != 7 {
		t.Errorf("vendor-reported token count must surface; got %d", resp.Usage.InputTokens)
	}
	// The ledger row is the acceptance criterion of G1'-1.
	if len(outbox.events) != 1 {
		t.Fatalf("exactly one token-usage event must be enqueued; got %d", len(outbox.events))
	}
	evt := outbox.events[0]
	if evt.CostMicros != 0 || evt.InputTokens != 7 {
		t.Errorf("ledger event tokens/cost wrong: %+v", evt)
	}
	if evt.AgentRole != "consumption_memory_embedder" {
		t.Errorf("ledger attribution must carry the embed agent id; got %q", evt.AgentRole)
	}
	if evt.ArmorPre != domain.ArmorVerdictBypassed || evt.ArmorPost != domain.ArmorVerdictBypassed {
		t.Errorf("permissive entry records Bypassed markers; got pre=%v post=%v", evt.ArmorPre, evt.ArmorPost)
	}
	if evt.TenantID != "11111111-1111-7111-8111-111111111111" {
		t.Errorf("tenant attribution lost: %+v", evt)
	}
}

func TestEmbed_refusesMissingRequiredFields(t *testing.T) {
	svc := newEmbedService(t, &fakeEmbedVendor{family: domain.VendorFamilyOpenAI}, &fakeEmbedOutbox{})
	cases := map[string]func(*domain.EmbedFlowRequest){
		"tenant": func(r *domain.EmbedFlowRequest) { r.TenantID = "" },
		"gcid":   func(r *domain.EmbedFlowRequest) { r.GCID = "" },
		"agent":  func(r *domain.EmbedFlowRequest) { r.AgentID = "" },
		"text":   func(r *domain.EmbedFlowRequest) { r.Text = "   " },
	}
	for name, mutate := range cases {
		req := validEmbedRequest()
		mutate(&req)
		if _, err := svc.Embed(context.Background(), req); err == nil {
			t.Errorf("missing %s must be refused", name)
		}
	}
}

func TestEmbed_refusesNonEmbeddingModel(t *testing.T) {
	svc := newEmbedService(t, &fakeEmbedVendor{family: domain.VendorFamilyOpenAI}, &fakeEmbedOutbox{})
	req := validEmbedRequest()
	req.LogicalModelID = "gemini-2.5-flash"
	_, err := svc.Embed(context.Background(), req)
	if err == nil {
		t.Fatal("a generation model on the embed path must be refused, never silently re-routed")
	}
	if !strings.Contains(err.Error(), "embed") {
		t.Errorf("error should say what was refused; got %v", err)
	}
}

// No embedding adapter wired at all: the flow must fail loud rather than
// dispatch through an unverified adapter.
func TestEmbed_unwiredEmbedderFailsLoud(t *testing.T) {
	svc := newEmbedServiceWithConfig(t, false, nil, nil, &fakeEmbedOutbox{})
	_, err := svc.Embed(context.Background(), validEmbedRequest())
	if err == nil {
		t.Fatal("an unwired embedding adapter must fail loud, not dispatch through an unverified route")
	}
	var noProvider *domain.NoProviderError
	if !errors.As(err, &noProvider) {
		t.Fatalf("the refusal must name the missing adapter; got %v", err)
	}
}

// No model registry resolver wired: the flow must fail loud rather than fall
// back to a hard-wired adapter.
func TestEmbed_unwiredRegistryFailsLoud(t *testing.T) {
	vendor := &fakeEmbedVendor{family: domain.VendorFamilyOpenAI, response: domain.EmbedVendorResponse{Values: []float32{0.1}}}
	svc := newEmbedServiceWithConfig(t, false, nil, vendor, &fakeEmbedOutbox{}, func(cfg *domain.ServiceConfig) {
		cfg.Models = nil
	})
	_, err := svc.Embed(context.Background(), validEmbedRequest())
	if err == nil {
		t.Fatal("an unwired registry resolver must fail loud, not dispatch through a hard-wired adapter")
	}
	assert.Contains(t, err.Error(), "no model registry resolver")
	assert.Equal(t, 0, vendor.calls)
}

func TestEmbed_outboxFailureFailsLoud(t *testing.T) {
	vendor := &fakeEmbedVendor{
		family:   domain.VendorFamilyOpenAI,
		response: domain.EmbedVendorResponse{Values: []float32{0.1}, ModelVersion: "text-embedding-004"},
	}
	svc := newEmbedService(t, vendor, &fakeEmbedOutbox{err: errors.New("pg down")})
	if _, err := svc.Embed(context.Background(), validEmbedRequest()); err == nil {
		t.Fatal("an un-ledgered embed must not succeed silently (the ledger row IS the requirement)")
	}
}

func TestEmbed_vendorErrorSurfaces(t *testing.T) {
	vendor := &fakeEmbedVendor{family: domain.VendorFamilyOpenAI, err: errors.New("predict 503")}
	svc := newEmbedService(t, vendor, &fakeEmbedOutbox{})
	if _, err := svc.Embed(context.Background(), validEmbedRequest()); err == nil {
		t.Fatal("vendor error must surface")
	}
}

// The model defaults; the DIMENSION does not. A caller that omits `dimensions`
// must not be given a gateway-invented one: the old 768 default asked every
// model for 768 values, which the pinned production route (a 1024-dimension
// model that rejects the parameter) answers with a 400. An omitted dimension
// means "send no override" — the resolved model's own declared length decides
// what the response must contain.
func TestEmbed_defaultsModelAndSendsNoDimensionsOverride(t *testing.T) {
	vendor := &fakeEmbedVendor{
		family:   domain.VendorFamilyOpenAI,
		response: domain.EmbedVendorResponse{Values: []float32{0.1}, ModelVersion: "text-embedding-004"},
	}
	svc := newEmbedService(t, vendor, &fakeEmbedOutbox{})
	req := validEmbedRequest()
	req.LogicalModelID = ""
	req.OutputDimensions = 0
	if _, err := svc.Embed(context.Background(), req); err != nil {
		t.Fatalf("empty model/dims must default, not refuse: %v", err)
	}
	if vendor.lastReq.LogicalModelID != "text-embedding-004" {
		t.Errorf("default model must be text-embedding-004; got %q", vendor.lastReq.LogicalModelID)
	}
	if vendor.lastReq.OutputDimensions != 0 {
		t.Errorf("an omitted dimension must send NO override; got %d", vendor.lastReq.OutputDimensions)
	}
}

// ----------------------------------------------------------------------------
// Fail-closed "budget required" mode (CHORA_LLM_BUDGET_REQUIRED)
// ----------------------------------------------------------------------------

// Flag ON + no active budget window ⇒ the embed is refused BEFORE the
// vendor call; no vector, no ledger row.
func TestEmbed_BudgetRequired_MissingWindow_RefusesBeforeVendor(t *testing.T) {
	vendor := &fakeEmbedVendor{
		family:   domain.VendorFamilyOpenAI,
		response: domain.EmbedVendorResponse{Values: []float32{0.1}, ModelVersion: "text-embedding-004"},
	}
	outbox := &fakeEmbedOutbox{}
	svc := newEmbedServiceWithConfig(t, true, nil, vendor, outbox)

	_, err := svc.Embed(context.Background(), validEmbedRequest())
	require.Error(t, err)
	assert.Contains(t, err.Error(), "budget required")
	assert.Equal(t, 0, vendor.calls, "a missing budget window must not reach the provider")
	assert.Len(t, outbox.events, 0, "a refused embed must not ledger")
}

// Flag OFF (default) + no active budget window ⇒ the historical allow is
// preserved: the vendor is called and the ledger row is written.
func TestEmbed_BudgetRequired_Off_MissingWindow_Allows(t *testing.T) {
	vendor := &fakeEmbedVendor{
		family:   domain.VendorFamilyOpenAI,
		response: domain.EmbedVendorResponse{Values: []float32{0.1}, ModelVersion: "text-embedding-004"},
	}
	outbox := &fakeEmbedOutbox{}
	svc := newEmbedServiceWithConfig(t, false, nil, vendor, outbox)

	resp, err := svc.Embed(context.Background(), validEmbedRequest())
	require.NoError(t, err)
	assert.Len(t, resp.Values, 1)
	assert.Equal(t, 1, vendor.calls)
	assert.Len(t, outbox.events, 1)
}

// ----------------------------------------------------------------------------
// Tenant-scoped override (CHORA_LLM_BUDGET_REQUIRED_TENANTS)
// ----------------------------------------------------------------------------

// Global flag OFF + listed tenant + no active budget window ⇒ the embed is
// refused BEFORE the vendor call; no vector, no ledger row.
func TestEmbed_BudgetRequired_TenantListed_RefusesDespiteGlobalOff(t *testing.T) {
	vendor := &fakeEmbedVendor{
		family:   domain.VendorFamilyOpenAI,
		response: domain.EmbedVendorResponse{Values: []float32{0.1}, ModelVersion: "text-embedding-004"},
	}
	outbox := &fakeEmbedOutbox{}
	svc := newEmbedServiceWithConfig(t, false, domain.BudgetRequiredTenants{"11111111-1111-7111-8111-111111111111": {}}, vendor, outbox)

	_, err := svc.Embed(context.Background(), validEmbedRequest())
	require.Error(t, err)
	assert.Contains(t, err.Error(), "budget required")
	assert.Equal(t, 0, vendor.calls, "a missing budget window must not reach the provider")
	assert.Len(t, outbox.events, 0, "a refused embed must not ledger")
}

// Global flag OFF + unlisted tenant + no active budget window ⇒ the
// historical allow is unchanged: the vendor is called and the ledger row is
// written.
func TestEmbed_BudgetRequired_TenantNotListed_Allows(t *testing.T) {
	vendor := &fakeEmbedVendor{
		family:   domain.VendorFamilyOpenAI,
		response: domain.EmbedVendorResponse{Values: []float32{0.1}, ModelVersion: "text-embedding-004"},
	}
	outbox := &fakeEmbedOutbox{}
	svc := newEmbedServiceWithConfig(t, false, domain.BudgetRequiredTenants{"99999999-9999-7999-8999-999999999999": {}}, vendor, outbox)

	resp, err := svc.Embed(context.Background(), validEmbedRequest())
	require.NoError(t, err)
	assert.Len(t, resp.Values, 1)
	assert.Equal(t, 1, vendor.calls)
	assert.Len(t, outbox.events, 1)
}

// Budget row present + listed tenant ⇒ unchanged allow: the override only
// fails closed on a MISSING window.
func TestEmbed_BudgetRequired_TenantListed_ActiveWindow_Allows(t *testing.T) {
	vendor := &fakeEmbedVendor{
		family:   domain.VendorFamilyOpenAI,
		response: domain.EmbedVendorResponse{Values: []float32{0.1}, ModelVersion: "text-embedding-004"},
	}
	outbox := &fakeEmbedOutbox{}
	svc := newEmbedServiceWithConfig(t, false, domain.BudgetRequiredTenants{"11111111-1111-7111-8111-111111111111": {}}, vendor, outbox, func(cfg *domain.ServiceConfig) {
		cfg.Budget = &fakeBudget{state: &domain.BudgetState{
			TenantID:        "11111111-1111-7111-8111-111111111111",
			BudgetUSDMicros: 1_000_000,
			SpentUSDMicros:  0,
			Policy:          domain.BudgetPolicyBlock,
		}}
	})

	resp, err := svc.Embed(context.Background(), validEmbedRequest())
	require.NoError(t, err)
	assert.Len(t, resp.Values, 1)
	assert.Equal(t, 1, vendor.calls)
	assert.Len(t, outbox.events, 1)
}
