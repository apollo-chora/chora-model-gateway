package domain_test

import (
	"context"
	"errors"
	"strings"
	"testing"
	"time"

	"github.com/apollo-chora/chora-model-gateway/internal/domain"
)

// G1'-1 (owner-ruled 2026-08-07): embeddings ride the chokepoint. The Embed
// flow must ledger every call (that is the point of the re-route), attach no
// price, run no Armor leg (permissive entry, Bypassed markers), and fail
// loud when unwired rather than silently falling back to direct Vertex.

type fakeEmbedVendor struct {
	family   domain.VendorFamily
	lastReq  domain.EmbedVendorRequest
	response domain.EmbedVendorResponse
	err      error
}

func (f *fakeEmbedVendor) Family() domain.VendorFamily { return f.family }
func (f *fakeEmbedVendor) EmbedText(_ context.Context, req domain.EmbedVendorRequest) (domain.EmbedVendorResponse, error) {
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

func newEmbedService(t *testing.T, embedder domain.EmbeddingClient, outbox domain.OutboxWriter) *domain.Service {
	t.Helper()
	svc, err := domain.NewService(domain.ServiceConfig{
		Vendors:        []domain.VendorClient{&fakeVendor{family: domain.VendorFamilyVertexGemini}},
		Armor:          &fakeArmor{},
		Budget:         &fakeBudget{},
		Outbox:         outbox,
		Policies:       &fakePolicy{},
		GatewayVersion: "chora-model-gateway:test",
		Embedder:       embedder,
		Now:            func() time.Time { return time.Unix(1754500000, 0) },
		NewID:          func() string { return "01988888-8888-7888-8888-888888888888" },
	})
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
		family: domain.VendorFamilyVertexGemini,
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
	svc := newEmbedService(t, &fakeEmbedVendor{family: domain.VendorFamilyVertexGemini}, &fakeEmbedOutbox{})
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
	svc := newEmbedService(t, &fakeEmbedVendor{family: domain.VendorFamilyVertexGemini}, &fakeEmbedOutbox{})
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

func TestEmbed_unwiredEmbedderFailsLoud(t *testing.T) {
	svc := newEmbedService(t, nil, &fakeEmbedOutbox{})
	if _, err := svc.Embed(context.Background(), validEmbedRequest()); err == nil {
		t.Fatal("nil embedder must fail loud, not fall back to direct Vertex")
	}
}

func TestEmbed_outboxFailureFailsLoud(t *testing.T) {
	vendor := &fakeEmbedVendor{
		family:   domain.VendorFamilyVertexGemini,
		response: domain.EmbedVendorResponse{Values: []float32{0.1}, ModelVersion: "text-embedding-004"},
	}
	svc := newEmbedService(t, vendor, &fakeEmbedOutbox{err: errors.New("pg down")})
	if _, err := svc.Embed(context.Background(), validEmbedRequest()); err == nil {
		t.Fatal("an un-ledgered embed must not succeed silently (the ledger row IS the requirement)")
	}
}

func TestEmbed_vendorErrorSurfaces(t *testing.T) {
	vendor := &fakeEmbedVendor{family: domain.VendorFamilyVertexGemini, err: errors.New("predict 503")}
	svc := newEmbedService(t, vendor, &fakeEmbedOutbox{})
	if _, err := svc.Embed(context.Background(), validEmbedRequest()); err == nil {
		t.Fatal("vendor error must surface")
	}
}

func TestEmbed_defaultsModelAndDimensions(t *testing.T) {
	vendor := &fakeEmbedVendor{
		family:   domain.VendorFamilyVertexGemini,
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
	if vendor.lastReq.OutputDimensions != 768 {
		t.Errorf("default dimensionality must be 768; got %d", vendor.lastReq.OutputDimensions)
	}
}
