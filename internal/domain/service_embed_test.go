package domain

import (
	"context"
	"errors"
	"testing"
	"time"
)

type embedStub struct {
	response EmbedVendorResponse
	err      error
	requests []EmbedVendorRequest
	calls    int
}

func (s *embedStub) Family() VendorFamily { return VendorFamilyOpenAI }

func (s *embedStub) EmbedText(_ context.Context, req EmbedVendorRequest) (EmbedVendorResponse, error) {
	s.calls++
	s.requests = append(s.requests, req)
	if s.err != nil {
		return EmbedVendorResponse{}, s.err
	}
	return s.response, nil
}

type embedSecrets struct{ values map[string]string }

func (s embedSecrets) ResolveCredential(_ context.Context, ref string) (string, error) {
	return s.values[ref], nil
}

func embedTarget(caps ...string) TargetModel {
	return TargetModel{
		Vendor:         VendorFamilyOpenAI,
		LogicalModelID: "emb",
		UpstreamModel:  "text-embedding-3-small",
		BaseURL:        "https://api.example.test/v1",
		APIKeyRef:      "EMB_KEY",
		Capabilities:   caps,
	}
}

func embedPolicies(target TargetModel, allow map[LogicalModelID]bool) PolicyLoader {
	return allowingLoader{allowed: allow, target: target}
}

type allowingLoader struct {
	allowed map[LogicalModelID]bool
	target  TargetModel
}

func (l allowingLoader) ResolveAgentPolicy(
	_ context.Context, _, _ string, requested LogicalModelID, _ []LogicalModelID,
) (AgentPolicy, error) {
	if len(l.allowed) > 0 && !l.allowed[requested] {
		return AgentPolicy{}, errors.New("model \"" + string(requested) + "\" is not in the registry")
	}
	t := l.target
	t.LogicalModelID = requested
	return AgentPolicy{ResolvedLogicalModelID: requested, Target: t}, nil
}

func newEmbedService(t *testing.T, embedder EmbeddingClient, policies PolicyLoader, secrets SecretClient) *Service {
	t.Helper()
	svc, err := NewService(ServiceConfig{
		Vendors:        []VendorClient{&stubVendor{family: VendorFamilyOpenAI}},
		Budget:         &atomicBudget{},
		Outbox:         &stubOutbox{},
		Secrets:        secrets,
		Policies:       policies,
		Embedder:       embedder,
		GatewayVersion: "test",
		Now:            func() time.Time { return time.Unix(1_700_000_000, 0) },
		NewID:          func() string { return "emb-generated" },
	})
	if err != nil {
		t.Fatalf("NewService: %v", err)
	}
	return svc
}

func validEmbedRequest() EmbedRequest {
	return EmbedRequest{
		TenantID:       "tenant-1",
		GCID:           "gcid-1",
		AgentID:        "agent-1",
		LogicalModelID: "emb",
		Text:           "hello",
	}
}

func TestEmbed_HappyPath(t *testing.T) {
	embedder := &embedStub{response: EmbedVendorResponse{
		Values:       []float32{0.1, 0.2, 0.3},
		ModelVersion: "text-embedding-3-small",
		InputTokens:  8,
	}}
	outbox := &stubOutbox{}
	svc, err := NewService(ServiceConfig{
		Vendors:        []VendorClient{&stubVendor{family: VendorFamilyOpenAI}},
		Budget:         &atomicBudget{},
		Outbox:         outbox,
		Secrets:        embedSecrets{values: map[string]string{"EMB_KEY": "sk-emb"}},
		Policies:       embedPolicies(embedTarget(CapabilityEmbeddings), nil),
		Embedder:       embedder,
		GatewayVersion: "test",
		Now:            time.Now,
		NewID:          func() string { return "emb-generated" },
	})
	if err != nil {
		t.Fatalf("NewService: %v", err)
	}

	resp, err := svc.Embed(context.Background(), validEmbedRequest())
	if err != nil {
		t.Fatalf("Embed: %v", err)
	}
	if len(resp.Values) != 3 {
		t.Errorf("values = %v", resp.Values)
	}
	if resp.InvocationID != "emb-generated" {
		t.Errorf("invocation id = %q, want the generated one", resp.InvocationID)
	}
	if resp.ModelVersion != "text-embedding-3-small" {
		t.Errorf("model = %q", resp.ModelVersion)
	}

	// The ledger row is the reason this RPC is routed through the gateway at
	// all, so an un-ledgered embed must be impossible. Embeddings are unpriced
	// so there is no debit to make atomic — the standalone ledger write is the
	// whole settlement here.
	if len(outbox.events) != 1 {
		t.Fatalf("ledger rows = %d, want 1", len(outbox.events))
	}
	if outbox.events[0].InputTokens != 8 {
		t.Errorf("ledger input tokens = %d, want 8", outbox.events[0].InputTokens)
	}
	// Embeddings carry no price: a priced embedding is user-visible.
	if outbox.events[0].CostMicros != 0 {
		t.Errorf("ledger cost = %d, want 0 for an embedding", outbox.events[0].CostMicros)
	}

	// The credential is resolved and travels on the request.
	if got := embedder.requests[0].Credential; got != "sk-emb" {
		t.Errorf("credential = %q", got)
	}
}

func TestEmbed_RequiredFields(t *testing.T) {
	cases := []struct {
		name    string
		mutate  func(*EmbedRequest)
		wantErr string
	}{
		{"no tenant", func(r *EmbedRequest) { r.TenantID = "" }, "tenant_id required"},
		{"no gcid", func(r *EmbedRequest) { r.GCID = "" }, "gcid required"},
		{"no agent", func(r *EmbedRequest) { r.AgentID = "" }, "agent_id required"},
		{"no text", func(r *EmbedRequest) { r.Text = "" }, "text required"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			svc := newEmbedService(t, &embedStub{}, embedPolicies(embedTarget(CapabilityEmbeddings), nil), embedSecrets{})
			req := validEmbedRequest()
			tc.mutate(&req)
			if _, err := svc.Embed(context.Background(), req); err == nil {
				t.Fatalf("expected %q", tc.wantErr)
			} else if !contains(err.Error(), tc.wantErr) {
				t.Fatalf("error = %q, want it to contain %q", err, tc.wantErr)
			}
		})
	}
}

func TestEmbed_NoEmbedderFailsLoud(t *testing.T) {
	// The direct-to-provider bypass this port exists to close must not have a
	// silent fallback.
	svc := newEmbedService(t, nil, embedPolicies(embedTarget(CapabilityEmbeddings), nil), embedSecrets{})
	if _, err := svc.Embed(context.Background(), validEmbedRequest()); err == nil {
		t.Fatal("expected an error when no EmbeddingClient is wired")
	}
}

func TestEmbed_DefaultModelWhenCallerOmitsIt(t *testing.T) {
	embedder := &embedStub{response: EmbedVendorResponse{Values: []float32{0.5}}}
	svc := newEmbedService(t, embedder,
		embedPolicies(embedTarget(CapabilityEmbeddings), map[LogicalModelID]bool{DefaultEmbeddingModelID: true}),
		embedSecrets{values: map[string]string{"EMB_KEY": "sk"}})

	req := validEmbedRequest()
	req.LogicalModelID = ""
	if _, err := svc.Embed(context.Background(), req); err != nil {
		t.Fatalf("Embed: %v", err)
	}
	if got := embedder.requests[0].Target.LogicalModelID; got != DefaultEmbeddingModelID {
		t.Errorf("resolved model = %q, want the default", got)
	}
}

func TestEmbed_NonEmbeddingModelIsRefused(t *testing.T) {
	// A generation model on the embed path is refused, never silently
	// re-routed: a caller that asked for a vector must not get a chat answer.
	svc := newEmbedService(t, &embedStub{},
		embedPolicies(embedTarget(CapabilityChat), map[LogicalModelID]bool{"emb": true}),
		embedSecrets{values: map[string]string{"EMB_KEY": "sk"}})

	_, err := svc.Embed(context.Background(), validEmbedRequest())
	if err == nil {
		t.Fatal("expected a refusal")
	}
	var cfgErr *ConfigError
	if !errors.As(err, &cfgErr) {
		t.Fatalf("error = %v, want a *ConfigError so the facade reports 500", err)
	}
	if !contains(cfgErr.Detail, CapabilityEmbeddings) {
		t.Errorf("detail = %q, want it to name the embeddings capability", cfgErr.Detail)
	}
}

func TestEmbed_UnknownModelFailsLoud(t *testing.T) {
	svc := newEmbedService(t, &embedStub{},
		embedPolicies(embedTarget(CapabilityEmbeddings), map[LogicalModelID]bool{"other": true}),
		embedSecrets{values: map[string]string{"EMB_KEY": "sk"}})

	if _, err := svc.Embed(context.Background(), validEmbedRequest()); err == nil {
		t.Fatal("expected an error for a model that is not in the registry")
	}
}

func TestEmbed_EmptyCredentialIsAMisconfiguration(t *testing.T) {
	svc := newEmbedService(t, &embedStub{},
		embedPolicies(embedTarget(CapabilityEmbeddings), nil),
		embedSecrets{values: map[string]string{}}) // EMBED_KEY unset

	_, err := svc.Embed(context.Background(), validEmbedRequest())
	var cfgErr *ConfigError
	if !errors.As(err, &cfgErr) {
		t.Fatalf("error = %v, want a *ConfigError", err)
	}
}

func TestEmbed_TargetWithNoCredentialRefNeedsNoKey(t *testing.T) {
	// A local embedding server needs no key.
	embedder := &embedStub{response: EmbedVendorResponse{Values: []float32{0.5}}}
	tgt := embedTarget(CapabilityEmbeddings)
	tgt.APIKeyRef = ""
	svc := newEmbedService(t, embedder, embedPolicies(tgt, nil), embedSecrets{})

	if _, err := svc.Embed(context.Background(), validEmbedRequest()); err != nil {
		t.Fatalf("Embed: %v", err)
	}
	if embedder.requests[0].Credential != "" {
		t.Errorf("credential = %q, want empty", embedder.requests[0].Credential)
	}
}

func TestEmbed_DefaultDimensionsApplied(t *testing.T) {
	embedder := &embedStub{response: EmbedVendorResponse{Values: []float32{0.5}}}
	svc := newEmbedService(t, embedder,
		embedPolicies(embedTarget(CapabilityEmbeddings), nil),
		embedSecrets{values: map[string]string{"EMB_KEY": "sk"}})

	if _, err := svc.Embed(context.Background(), validEmbedRequest()); err != nil {
		t.Fatalf("Embed: %v", err)
	}
	if got := embedder.requests[0].OutputDimensions; got != defaultEmbeddingDimensions {
		t.Errorf("dimensions = %d, want the default %d", got, defaultEmbeddingDimensions)
	}
}

func TestEmbed_ExplicitDimensionsWin(t *testing.T) {
	embedder := &embedStub{response: EmbedVendorResponse{Values: []float32{0.5}}}
	svc := newEmbedService(t, embedder,
		embedPolicies(embedTarget(CapabilityEmbeddings), nil),
		embedSecrets{values: map[string]string{"EMB_KEY": "sk"}})

	req := validEmbedRequest()
	req.OutputDimensions = 256
	if _, err := svc.Embed(context.Background(), req); err != nil {
		t.Fatalf("Embed: %v", err)
	}
	if got := embedder.requests[0].OutputDimensions; got != 256 {
		t.Errorf("dimensions = %d, want 256", got)
	}
}

func TestEmbed_VendorFailurePropagates(t *testing.T) {
	svc := newEmbedService(t, &embedStub{err: errors.New("provider down")},
		embedPolicies(embedTarget(CapabilityEmbeddings), nil),
		embedSecrets{values: map[string]string{"EMB_KEY": "sk"}})

	if _, err := svc.Embed(context.Background(), validEmbedRequest()); err == nil {
		t.Fatal("expected an error")
	}
}

func TestEmbed_UnledgeredEmbedMustNotSucceed(t *testing.T) {
	// If the ledger write fails the call fails, even though the vector was
	// already produced and the provider already billed for it. The alternative
	// is usage nobody can account for.
	failingOutbox := &failingOutbox{}
	svc, err := NewService(ServiceConfig{
		Vendors:        []VendorClient{&stubVendor{family: VendorFamilyOpenAI}},
		Budget:         &atomicBudget{},
		Outbox:         failingOutbox,
		Secrets:        embedSecrets{values: map[string]string{"EMB_KEY": "sk"}},
		Policies:       embedPolicies(embedTarget(CapabilityEmbeddings), nil),
		Embedder:       &embedStub{response: EmbedVendorResponse{Values: []float32{0.5}}},
		GatewayVersion: "test",
		Now:            time.Now,
		NewID:          func() string { return "id" },
	})
	if err != nil {
		t.Fatalf("NewService: %v", err)
	}

	if _, err := svc.Embed(context.Background(), validEmbedRequest()); err == nil {
		t.Fatal("expected an un-ledgered embed to fail")
	}
}

type failingOutbox struct{}

func (failingOutbox) EnqueueTokenUsageRecorded(context.Context, TokenUsageEvent) error {
	return errors.New("ledger write failed")
}
