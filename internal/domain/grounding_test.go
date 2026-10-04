package domain

import (
	"context"
	"errors"
	"testing"
	"time"
)

// ---------------------------------------------------------------------------
// Grounding config defaults
// ---------------------------------------------------------------------------

func TestGrounding_EffectiveSurfaceDefaultsByVendor(t *testing.T) {
	g := &Grounding{}
	// An openai entry with no surface grounds on the Responses API, because
	// that is where OpenAI puts its hosted search tool.
	if got := g.EffectiveSurface(VendorFamilyOpenAI); got != SurfaceResponses {
		t.Errorf("openai surface = %q, want %q", got, SurfaceResponses)
	}
	// Anthropic puts its search on the Messages API.
	if got := g.EffectiveSurface(VendorFamilyAnthropic); got != SurfaceMessages {
		t.Errorf("anthropic surface = %q, want %q", got, SurfaceMessages)
	}
	// An explicit surface always wins.
	explicit := &Grounding{Surface: SurfaceChatCompletions}
	if got := explicit.EffectiveSurface(VendorFamilyOpenAI); got != SurfaceChatCompletions {
		t.Errorf("explicit surface = %q, want it honoured", got)
	}
}

func TestGrounding_EffectiveToolTypeDefaultsBySurface(t *testing.T) {
	cases := []struct {
		surface GroundingSurface
		want    string
	}{
		{SurfaceResponses, "web_search"},
		{SurfaceChatCompletions, "web_search_preview"},
		{SurfaceMessages, "web_search_20250305"},
	}
	for _, tc := range cases {
		g := &Grounding{}
		if got := g.EffectiveToolType(tc.surface); got != tc.want {
			t.Errorf("tool type for %q = %q, want %q", tc.surface, got, tc.want)
		}
	}
	// An override wins over the default, for a provider that renamed the tool.
	g := &Grounding{ToolType: "web_search_2025_08_26"}
	if got := g.EffectiveToolType(SurfaceResponses); got != "web_search_2025_08_26" {
		t.Errorf("override = %q, want it honoured", got)
	}
	if got := (&Grounding{}).EffectiveToolName(); got != "web_search" {
		t.Errorf("default tool name = %q", got)
	}
}

func TestTargetModel_GroundingURL(t *testing.T) {
	cases := []struct {
		name      string
		target    TargetModel
		want      string
		wantEmpty bool
	}{
		{
			name: "no grounding block",
			// A missing block means "this provider has no hosted search", so
			// the URL is empty and the service refuses rather than guessing.
			target:    TargetModel{Vendor: VendorFamilyOpenAI, BaseURL: "https://x.test/v1"},
			wantEmpty: true,
		},
		{
			name:   "responses default path",
			target: TargetModel{Vendor: VendorFamilyOpenAI, BaseURL: "https://x.test/v1", Grounding: &Grounding{}},
			want:   "https://x.test/v1/responses",
		},
		{
			name: "custom responses path",
			target: TargetModel{Vendor: VendorFamilyOpenAI, BaseURL: "https://x.test/v1",
				Grounding: &Grounding{ResponsesPath: "/gen/responses"}},
			want: "https://x.test/v1/gen/responses",
		},
		{
			name: "anthropic surfaces on messages",
			target: TargetModel{Vendor: VendorFamilyAnthropic, BaseURL: "https://api.anthropic.com",
				Grounding: &Grounding{}},
			want: "https://api.anthropic.com/v1/messages",
		},
		{
			name: "chat completions surface",
			target: TargetModel{Vendor: VendorFamilyOpenAI, BaseURL: "https://x.test/v1",
				Grounding: &Grounding{Surface: SurfaceChatCompletions}},
			want: "https://x.test/v1/chat/completions",
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got := tc.target.GroundingURL()
			if tc.wantEmpty {
				if got != "" {
					t.Errorf("GroundingURL = %q, want empty", got)
				}
				return
			}
			if got != tc.want {
				t.Errorf("GroundingURL = %q, want %q", got, tc.want)
			}
		})
	}
}

// ---------------------------------------------------------------------------
// Modality → capability gate
// ---------------------------------------------------------------------------

func TestRequiredCapability(t *testing.T) {
	cases := map[string]string{
		"":                 CapabilityChat,
		ModalityText:       CapabilityChat,
		ModalityImage:      CapabilityImage,
		ModalityGrounded:   CapabilityWebSearch,
		"UNKNOWN-MODALITY": CapabilityChat,
	}
	for modality, want := range cases {
		if got := RequiredCapability(modality); got != want {
			t.Errorf("RequiredCapability(%q) = %q, want %q", modality, got, want)
		}
	}
}

func TestInvokeRequest_ValidateAcceptsGrounded(t *testing.T) {
	req := validRequest()
	req.ResponseModality = ModalityGrounded
	if err := req.Validate(); err != nil {
		t.Errorf("a grounded request must validate: %v", err)
	}
	req.ResponseModality = "AUDIO"
	if err := req.Validate(); err == nil {
		t.Error("an unknown modality must still be refused")
	}
}

// ---------------------------------------------------------------------------
// Service behaviour
// ---------------------------------------------------------------------------

func groundedTargetModel(caps ...string) TargetModel {
	t := target("model-a", caps...)
	t.Grounding = &Grounding{}
	return t
}

func TestInvoke_GroundedHappyPath(t *testing.T) {
	vendor := &stubVendor{family: VendorFamilyOpenAI, response: VendorResponse{
		Completion:    "Max Verstappen won.",
		Usage:         TokenUsage{InputTokens: 20, OutputTokens: 8, CostMicros: 900},
		Citations:     []GroundingCitation{{URL: "https://f1.example/a", Title: "Race report"}},
		SearchQueries: []string{"most recent f1 race"},
	}}
	svc := newTestService(t, &atomicBudget{}, &stubOutbox{}, vendor,
		&stubPolicies{policy: AgentPolicy{Target: groundedTargetModel(CapabilityChat, CapabilityWebSearch)}})

	req := validRequest()
	req.ResponseModality = ModalityGrounded
	resp, err := svc.Invoke(context.Background(), req)
	if err != nil {
		t.Fatalf("Invoke: %v", err)
	}
	if resp.Completion != "Max Verstappen won." {
		t.Errorf("completion = %q", resp.Completion)
	}
	if len(resp.Citations) != 1 || resp.Citations[0].URL != "https://f1.example/a" {
		t.Errorf("citations = %+v", resp.Citations)
	}
	if len(resp.SearchQueries) != 1 {
		t.Errorf("search queries = %v", resp.SearchQueries)
	}
	if resp.GroundingSurface != string(SurfaceResponses) {
		t.Errorf("grounding surface = %q, want responses", resp.GroundingSurface)
	}
	// A grounded call is a normal dispatch: it is budgeted and ledgered.
	if vendor.gotReq[0].ResponseModality != ModalityGrounded {
		t.Errorf("the modality did not reach the vendor: %+v", vendor.gotReq[0])
	}
}

func TestInvoke_GroundedOnANonGroundedModelIsRefused(t *testing.T) {
	// The failure this prevents: dispatching a search tool to a provider that
	// ignores it, and returning an answer that LOOKS researched.
	vendor := &stubVendor{family: VendorFamilyOpenAI, response: VendorResponse{}}
	svc := newTestService(t, &atomicBudget{}, &stubOutbox{}, vendor,
		&stubPolicies{policy: AgentPolicy{Target: target("model-a", CapabilityChat)}})

	req := validRequest()
	req.ResponseModality = ModalityGrounded
	_, err := svc.Invoke(context.Background(), req)
	if err == nil {
		t.Fatal("expected a refusal")
	}
	var cfgErr *ConfigError
	if !errors.As(err, &cfgErr) {
		t.Fatalf("error = %v, want a *ConfigError", err)
	}
	if !contains(cfgErr.Detail, CapabilityWebSearch) {
		t.Errorf("detail = %q, want it to name the missing capability", cfgErr.Detail)
	}
	if vendor.callsMade != 0 {
		t.Errorf("the provider was called %d times despite the refusal", vendor.callsMade)
	}
}

func TestInvoke_GroundedWithoutAConfiguredEndpointIsRefused(t *testing.T) {
	// The capability can be advertised without a grounding block; that is a
	// configuration gap, and it must refuse rather than guess an endpoint.
	tgt := target("model-a", CapabilityChat, CapabilityWebSearch)
	tgt.Grounding = nil // capability advertised, nothing configured
	vendor := &stubVendor{family: VendorFamilyOpenAI}
	svc := newTestService(t, &atomicBudget{}, &stubOutbox{}, vendor,
		&stubPolicies{policy: AgentPolicy{Target: tgt}})

	req := validRequest()
	req.ResponseModality = ModalityGrounded
	_, err := svc.Invoke(context.Background(), req)
	var cfgErr *ConfigError
	if !errors.As(err, &cfgErr) {
		t.Fatalf("error = %v, want a *ConfigError", err)
	}
	if !contains(cfgErr.Detail, "grounding") {
		t.Errorf("detail = %q, want it to mention the missing grounding block", cfgErr.Detail)
	}
}

func TestInvoke_GroundedCallIsBudgetedAndLedgered(t *testing.T) {
	budget := &atomicBudget{}
	vendor := &stubVendor{family: VendorFamilyOpenAI, response: VendorResponse{
		Usage:      TokenUsage{CostMicros: 1234},
		Citations:  []GroundingCitation{{URL: "https://x"}},
		Completion: "answer",
	}}
	svc := newTestService(t, budget, &stubOutbox{}, vendor,
		&stubPolicies{policy: AgentPolicy{Target: groundedTargetModel(CapabilityChat, CapabilityWebSearch)}})

	req := validRequest()
	req.ResponseModality = ModalityGrounded
	if _, err := svc.Invoke(context.Background(), req); err != nil {
		t.Fatalf("Invoke: %v", err)
	}
	// Web search is priced per request by most providers, so a token-only
	// budget under-charges it — but the ledger must still record the call, or
	// the search is invisible.
	if budget.settleCalls != 1 {
		t.Errorf("settle calls = %d, want 1", budget.settleCalls)
	}
	if len(budget.settled) != 1 {
		t.Fatalf("ledger rows = %d, want 1", len(budget.settled))
	}
	if budget.settled[0].Modality != ModalityGrounded {
		t.Errorf("ledger modality = %q, want %q so a search-heavy tenant is identifiable",
			budget.settled[0].Modality, ModalityGrounded)
	}
}

func TestInvoke_TextCallReportsNoGroundingSurface(t *testing.T) {
	vendor := &stubVendor{family: VendorFamilyOpenAI, response: VendorResponse{Completion: "plain"}}
	svc := newTestService(t, &atomicBudget{}, &stubOutbox{}, vendor,
		&stubPolicies{policy: AgentPolicy{Target: groundedTargetModel(CapabilityChat, CapabilityWebSearch)}})

	// A model CAN be grounded without every call being grounded, so a plain
	// text call must not claim a surface.
	resp, err := svc.Invoke(context.Background(), validRequest())
	if err != nil {
		t.Fatalf("Invoke: %v", err)
	}
	if resp.GroundingSurface != "" {
		t.Errorf("grounding surface = %q on a text call, want empty", resp.GroundingSurface)
	}
	if resp.Citations != nil {
		t.Errorf("citations = %+v on a text call, want none", resp.Citations)
	}
}

// flakyVendor fails the first dispatch and succeeds the second, which is how a
// real adapter behaves when one model is down and its fallback is not. Two
// adapters for the same family is not a thing: the registry is keyed by
// VendorFamily precisely so one instance serves every entry sharing a provider.
type flakyVendor struct {
	family   VendorFamily
	failOnce bool
	calls    int
	response VendorResponse
}

func (f *flakyVendor) Family() VendorFamily { return f.family }

func (f *flakyVendor) Generate(_ context.Context, _ VendorRequest) (VendorResponse, error) {
	f.calls++
	if f.failOnce && f.calls == 1 {
		return VendorResponse{}, errors.New("search quota exhausted")
	}
	return f.response, nil
}

func TestInvoke_GroundedWalksTheFallbackChain(t *testing.T) {
	// Grounding must not bypass the fallback machinery: a model with search
	// that goes down has to fall through like anything else.
	vendor := &flakyVendor{family: VendorFamilyOpenAI, failOnce: true, response: VendorResponse{
		Completion: "answered by the fallback",
		Citations:  []GroundingCitation{{URL: "https://fallback.example"}},
	}}

	fb := groundedTargetModel(CapabilityChat, CapabilityWebSearch)
	fb.LogicalModelID = "model-b"
	fb.UpstreamModel = "model-b"
	fb.APIKeyRef = "model-b_KEY"

	svc, err := NewService(ServiceConfig{
		Vendors: []VendorClient{vendor},
		Budget:  &atomicBudget{},
		Outbox:  &stubOutbox{},
		Secrets: &stubSecrets{values: map[string]string{"model-a_KEY": "sk-a", "model-b_KEY": "sk-b"}},
		Policies: &stubPolicies{policy: AgentPolicy{
			Target:        groundedTargetModel(CapabilityChat, CapabilityWebSearch),
			FallbackChain: []AgentPolicyFallback{{Vendor: VendorFamilyOpenAI, Target: fb}},
		}},
		GatewayVersion: "test",
		Now:            time.Now,
		NewID:          func() string { return "id" },
	})
	if err != nil {
		t.Fatalf("NewService: %v", err)
	}

	req := validRequest()
	req.ResponseModality = ModalityGrounded
	resp, err := svc.Invoke(context.Background(), req)
	if err != nil {
		t.Fatalf("Invoke: %v", err)
	}
	if resp.Completion != "answered by the fallback" {
		t.Errorf("completion = %q", resp.Completion)
	}
	if vendor.calls != 2 {
		t.Errorf("vendor calls = %d, want 2 (primary failed, fallback answered)", vendor.calls)
	}
	if len(resp.FallbackChain) != 2 {
		t.Errorf("fallback chain = %v, want both attempts", resp.FallbackChain)
	}
	if len(resp.Citations) != 1 {
		t.Errorf("citations lost across the fallback: %+v", resp.Citations)
	}
}

// TestInvoke_GroundedMessagesSurviveTheDomainLayer is the propagation guard.
// The adapter produces an ordered message list, and an earlier version of the
// service dropped it while copying the rest of the response — which showed up
// live as `chora_gateway.messages: 0` beside a correct answer. Every hop
// between the adapter and the wire must carry it.
func TestInvoke_GroundedMessagesSurviveTheDomainLayer(t *testing.T) {
	messages := []GroundingMessage{
		{Text: "I'll look that up."},
		{Text: "6.21 million.", Citations: []GroundingCitation{{URL: "https://population.gov.sg"}}},
	}
	vendor := &stubVendor{family: VendorFamilyOpenAI, response: VendorResponse{
		Completion: "6.21 million.",
		Messages:   messages,
		Citations:  messages[1].Citations,
	}}
	svc := newTestService(t, &atomicBudget{}, &stubOutbox{}, vendor,
		&stubPolicies{policy: AgentPolicy{Target: groundedTargetModel(CapabilityChat, CapabilityWebSearch)}})

	req := validRequest()
	req.ResponseModality = ModalityGrounded
	resp, err := svc.Invoke(context.Background(), req)
	if err != nil {
		t.Fatalf("Invoke: %v", err)
	}
	if len(resp.Messages) != 2 {
		t.Fatalf("messages = %d, want 2 carried through the domain layer", len(resp.Messages))
	}
	if resp.Messages[0].Text != "I'll look that up." {
		t.Errorf("messages[0] = %q; narration must survive, not just the answer", resp.Messages[0].Text)
	}
	if len(resp.Messages[1].Citations) != 1 {
		t.Errorf("the answer's citations were lost: %+v", resp.Messages[1])
	}
}
