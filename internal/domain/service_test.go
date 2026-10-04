package domain

import (
	"context"
	"errors"
	"fmt"
	"testing"
	"time"
)

// ---------------------------------------------------------------------------
// Test doubles
// ---------------------------------------------------------------------------

type stubVendor struct {
	family    VendorFamily
	response  VendorResponse
	err       error
	gotReq    []VendorRequest
	callsMade int
}

func (s *stubVendor) Family() VendorFamily { return s.family }

func (s *stubVendor) Generate(_ context.Context, req VendorRequest) (VendorResponse, error) {
	s.callsMade++
	s.gotReq = append(s.gotReq, req)
	if s.err != nil {
		return VendorResponse{}, s.err
	}
	return s.response, nil
}

type stubEmbedder struct {
	response EmbedVendorResponse
	err      error
	gotReq   []EmbedVendorRequest
}

func (s *stubEmbedder) Family() VendorFamily { return VendorFamilyOpenAI }

func (s *stubEmbedder) EmbedText(_ context.Context, req EmbedVendorRequest) (EmbedVendorResponse, error) {
	s.gotReq = append(s.gotReq, req)
	if s.err != nil {
		return EmbedVendorResponse{}, s.err
	}
	return s.response, nil
}

type stubBudget struct {
	state    *BudgetState
	stateErr error
	debited  int64
	debitErr error
	getCalls int
}

func (s *stubBudget) GetTenantBudget(context.Context, string) (*BudgetState, error) {
	s.getCalls++
	return s.state, s.stateErr
}

func (s *stubBudget) DebitSpent(_ context.Context, _ string, delta int64) error {
	s.debited += delta
	return s.debitErr
}

// atomicBudget also implements Settler, which is the pg adapter's shape: ONE
// transaction writes BOTH the budget debit and the ledger row. Modelling that
// faithfully matters — on this path the standalone OutboxWriter is deliberately
// not called, because Settle is the ledger writer.
type atomicBudget struct {
	stubBudget
	settleCalls int
	settleErr   error
	settled     []TokenUsageEvent
}

func (s *atomicBudget) Settle(_ context.Context, evt TokenUsageEvent, delta int64) error {
	s.settleCalls++
	if s.settleErr != nil {
		return s.settleErr
	}
	s.debited += delta
	s.settled = append(s.settled, evt)
	return nil
}

type stubOutbox struct {
	events []TokenUsageEvent
	err    error
}

func (s *stubOutbox) EnqueueTokenUsageRecorded(_ context.Context, evt TokenUsageEvent) error {
	s.events = append(s.events, evt)
	return s.err
}

type stubSecrets struct {
	values map[string]string
}

func (s *stubSecrets) ResolveCredential(_ context.Context, ref string) (string, error) {
	return s.values[ref], nil
}

type stubClaims struct {
	claimed bool
	err     error
	keys    []string
}

func (s *stubClaims) ClaimDebit(_ context.Context, _ string, key, _ string) (bool, error) {
	s.keys = append(s.keys, key)
	return s.claimed, s.err
}

type stubPolicies struct {
	policy   AgentPolicy
	err      error
	resolved []LogicalModelID
}

func (s *stubPolicies) ResolveAgentPolicy(
	_ context.Context, _, _ string, requested LogicalModelID, _ []LogicalModelID,
) (AgentPolicy, error) {
	s.resolved = append(s.resolved, requested)
	if s.err != nil {
		return AgentPolicy{}, s.err
	}
	return s.policy, nil
}

func target(id string, caps ...string) TargetModel {
	return TargetModel{
		Vendor:          VendorFamilyOpenAI,
		LogicalModelID:  LogicalModelID(id),
		UpstreamModel:   id,
		BaseURL:         "https://example.test/v1",
		APIKeyRef:       id + "_KEY",
		MaxOutputTokens: 256,
		Capabilities:    caps,
		ContextWindow:   4096,
	}
}

func validRequest() InvokeRequest {
	return InvokeRequest{
		TenantID:       "tenant-1",
		GCID:           "gcid-1",
		AgentID:        "agent-1",
		LogicalModelID: "model-a",
		Prompt:         "hello",
	}
}

// newTestService assembles a Service from the doubles with the minimum wiring
// Invoke needs.
func newTestService(t *testing.T, budget BudgetRepo, outbox OutboxWriter, vendor *stubVendor, policies PolicyLoader) *Service {
	t.Helper()
	svc, err := NewService(ServiceConfig{
		Vendors:        []VendorClient{vendor},
		Budget:         budget,
		Outbox:         outbox,
		Secrets:        &stubSecrets{values: map[string]string{"model-a_KEY": "sk-a", "model-b_KEY": "sk-b"}},
		Policies:       policies,
		GatewayVersion: "test",
		Now:            func() time.Time { return time.Unix(1_700_000_000, 0) },
		NewID:          func() string { return "inv-generated" },
	})
	if err != nil {
		t.Fatalf("NewService: %v", err)
	}
	return svc
}

// ---------------------------------------------------------------------------
// NewService validation
// ---------------------------------------------------------------------------

func TestNewService_RequiredPorts(t *testing.T) {
	cases := []struct {
		name    string
		cfg     ServiceConfig
		wantErr string
	}{
		{"no budget", ServiceConfig{Outbox: &stubOutbox{}, Secrets: &stubSecrets{}, Policies: &stubPolicies{}, GatewayVersion: "v", Vendors: []VendorClient{&stubVendor{family: VendorFamilyOpenAI}}}, "BudgetRepo required"},
		{"no outbox", ServiceConfig{Budget: &stubBudget{}, Secrets: &stubSecrets{}, Policies: &stubPolicies{}, GatewayVersion: "v", Vendors: []VendorClient{&stubVendor{family: VendorFamilyOpenAI}}}, "OutboxWriter required"},
		{"no secrets", ServiceConfig{Budget: &stubBudget{}, Outbox: &stubOutbox{}, Policies: &stubPolicies{}, GatewayVersion: "v", Vendors: []VendorClient{&stubVendor{family: VendorFamilyOpenAI}}}, "SecretClient required"},
		{"no policies", ServiceConfig{Budget: &stubBudget{}, Outbox: &stubOutbox{}, Secrets: &stubSecrets{}, GatewayVersion: "v", Vendors: []VendorClient{&stubVendor{family: VendorFamilyOpenAI}}}, "PolicyLoader required"},
		{"no vendors", ServiceConfig{Budget: &stubBudget{}, Outbox: &stubOutbox{}, Secrets: &stubSecrets{}, Policies: &stubPolicies{}, GatewayVersion: "v"}, "at least one VendorClient"},
		{"no version", ServiceConfig{Budget: &stubBudget{}, Outbox: &stubOutbox{}, Secrets: &stubSecrets{}, Policies: &stubPolicies{}, Vendors: []VendorClient{&stubVendor{family: VendorFamilyOpenAI}}}, "GatewayVersion required"},
		{"duplicate family", ServiceConfig{Budget: &stubBudget{}, Outbox: &stubOutbox{}, Secrets: &stubSecrets{}, Policies: &stubPolicies{}, GatewayVersion: "v", Vendors: []VendorClient{&stubVendor{family: VendorFamilyOpenAI}, &stubVendor{family: VendorFamilyOpenAI}}}, "duplicate VendorClient"},
		{"empty family", ServiceConfig{Budget: &stubBudget{}, Outbox: &stubOutbox{}, Secrets: &stubSecrets{}, Policies: &stubPolicies{}, GatewayVersion: "v", Vendors: []VendorClient{&stubVendor{}}}, "Family() returned empty"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			_, err := NewService(tc.cfg)
			if err == nil {
				t.Fatalf("expected an error containing %q, got nil", tc.wantErr)
			}
			if got := err.Error(); !contains(got, tc.wantErr) {
				t.Fatalf("error = %q, want it to contain %q", got, tc.wantErr)
			}
		})
	}
}

func contains(haystack, needle string) bool {
	return len(needle) == 0 || (len(haystack) >= len(needle) && indexOf(haystack, needle) >= 0)
}

func indexOf(h, n string) int {
	for i := 0; i+len(n) <= len(h); i++ {
		if h[i:i+len(n)] == n {
			return i
		}
	}
	return -1
}

// ---------------------------------------------------------------------------
// Request validation
// ---------------------------------------------------------------------------

func TestInvokeRequest_Validate(t *testing.T) {
	cases := []struct {
		name    string
		mutate  func(*InvokeRequest)
		wantErr string
	}{
		{"valid text", func(r *InvokeRequest) {}, ""},
		{"valid via contents", func(r *InvokeRequest) { r.Prompt = ""; r.ContentsJSON = `[]` }, ""},
		{"no tenant", func(r *InvokeRequest) { r.TenantID = "" }, "tenant_id required"},
		{"no gcid", func(r *InvokeRequest) { r.GCID = "" }, "gcid required"},
		{"no agent", func(r *InvokeRequest) { r.AgentID = "" }, "agent_id required"},
		{"no model", func(r *InvokeRequest) { r.LogicalModelID = "" }, "logical_model_id required"},
		{"empty request", func(r *InvokeRequest) { r.Prompt = ""; r.ContentsJSON = "" }, "prompt or contents_json required"},
		{"bad modality", func(r *InvokeRequest) { r.ResponseModality = "AUDIO" }, "not one of TEXT, IMAGE"},
		{"IMAGE is accepted", func(r *InvokeRequest) { r.ResponseModality = ModalityImage }, ""},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			req := validRequest()
			tc.mutate(&req)
			err := req.Validate()
			if tc.wantErr == "" {
				if err != nil {
					t.Fatalf("unexpected error: %v", err)
				}
				return
			}
			if err == nil {
				t.Fatalf("expected %q, got nil", tc.wantErr)
			}
			if !contains(err.Error(), tc.wantErr) {
				t.Fatalf("error = %q, want it to contain %q", err, tc.wantErr)
			}
		})
	}
}

// ---------------------------------------------------------------------------
// Happy path
// ---------------------------------------------------------------------------

func TestInvoke_HappyPath(t *testing.T) {
	budget := &atomicBudget{}
	outbox := &stubOutbox{}
	vendor := &stubVendor{
		family: VendorFamilyOpenAI,
		response: VendorResponse{
			Completion:   "the answer",
			ModelVersion: "upstream-1",
			Usage:        TokenUsage{InputTokens: 10, OutputTokens: 5, CostMicros: 777},
		},
	}
	svc := newTestService(t, budget, outbox, vendor, &stubPolicies{policy: AgentPolicy{Target: target("model-a")}})

	resp, err := svc.Invoke(context.Background(), validRequest())
	if err != nil {
		t.Fatalf("Invoke: %v", err)
	}
	if resp.Completion != "the answer" {
		t.Errorf("completion = %q, want %q", resp.Completion, "the answer")
	}
	if resp.InvocationID != "inv-generated" {
		t.Errorf("invocation id = %q, want the generated one", resp.InvocationID)
	}
	if resp.ModelVersion != "upstream-1" {
		t.Errorf("model version = %q", resp.ModelVersion)
	}
	if resp.GatewayVersion != "test" {
		t.Errorf("gateway version = %q", resp.GatewayVersion)
	}

	// The atomic path must be used, not two separate writes.
	if budget.settleCalls != 1 {
		t.Errorf("Settle called %d times, want exactly 1", budget.settleCalls)
	}
	if budget.debited != 777 {
		t.Errorf("debited %d, want 777", budget.debited)
	}
	// On the atomic path Settle is the ledger writer, so the ledger row is
	// observed on the budget double, not the outbox double.
	if len(outbox.events) != 0 {
		t.Errorf("standalone outbox was written %d times on the atomic path, want 0", len(outbox.events))
	}
	if len(budget.settled) != 1 {
		t.Fatalf("ledger rows = %d, want 1", len(budget.settled))
	}
	evt := budget.settled[0]
	if evt.CostMicros != 777 || evt.InputTokens != 10 || evt.OutputTokens != 5 {
		t.Errorf("ledger usage = %+v, want the vendor's numbers", evt.Usage2())
	}
	if evt.Vendor != string(VendorFamilyOpenAI) {
		t.Errorf("ledger vendor = %q", evt.Vendor)
	}
	// The chain records the DISPATCHED target (the registry's upstream model),
	// not what the provider echoed back in its response. Those differ when an
	// alias maps one name onto another, and the alias is the useful answer to
	// "which model actually ran".
	if len(evt.FallbackChain) != 1 || evt.FallbackChain[0] != "openai:model-a" {
		t.Errorf("fallback chain = %v, want [openai:model-a]", evt.FallbackChain)
	}

	// The credential travels on the request, resolved from the registry ref.
	if got := vendor.gotReq[0].Credential; got != "sk-a" {
		t.Errorf("credential = %q, want the resolved %q", got, "sk-a")
	}
}

func TestInvoke_FallsBackToDebitPlusLedgerWhenNotAtomic(t *testing.T) {
	budget := &stubBudget{}
	outbox := &stubOutbox{}
	vendor := &stubVendor{family: VendorFamilyOpenAI, response: VendorResponse{Usage: TokenUsage{CostMicros: 42}}}
	svc := newTestService(t, budget, outbox, vendor, &stubPolicies{policy: AgentPolicy{Target: target("model-a")}})

	if _, err := svc.Invoke(context.Background(), validRequest()); err != nil {
		t.Fatalf("Invoke: %v", err)
	}
	if budget.debited != 42 {
		t.Errorf("debited %d, want 42", budget.debited)
	}
	if len(outbox.events) != 1 {
		t.Errorf("ledger rows = %d, want 1", len(outbox.events))
	}
}

// ---------------------------------------------------------------------------
// Budget
// ---------------------------------------------------------------------------

func TestInvoke_BudgetBlockShortCircuits(t *testing.T) {
	budget := &atomicBudget{stubBudget: stubBudget{state: &BudgetState{
		BudgetUSDMicros: 100, SpentUSDMicros: 200, Policy: BudgetPolicyBlock,
	}}}
	outbox := &stubOutbox{}
	vendor := &stubVendor{family: VendorFamilyOpenAI}
	svc := newTestService(t, budget, outbox, vendor, &stubPolicies{policy: AgentPolicy{Target: target("model-a")}})

	resp, err := svc.Invoke(context.Background(), validRequest())
	if err != nil {
		t.Fatalf("Invoke: %v", err)
	}
	if resp.FinishReason != FinishReasonBudgetBlock {
		t.Errorf("finish reason = %v, want budget_block", resp.FinishReason)
	}
	if vendor.callsMade != 0 {
		t.Errorf("vendor was called %d times under a budget block, want 0", vendor.callsMade)
	}
	if budget.settleCalls != 0 {
		t.Errorf("settled under a budget block; want no spend recorded")
	}
	if len(outbox.events) != 0 {
		t.Errorf("a blocked call wrote %d ledger rows, want 0", len(outbox.events))
	}
}

func TestInvoke_BudgetDowngradeReresolvesTarget(t *testing.T) {
	// A downgrade only fires once the budget is actually exhausted; below the
	// ceiling the decision is BudgetAllow regardless of policy.
	budget := &atomicBudget{stubBudget: stubBudget{state: &BudgetState{
		BudgetUSDMicros: 1000, SpentUSDMicros: 1000,
		Policy: BudgetPolicyDowngrade, DowngradeToModel: "model-b",
	}}}
	outbox := &stubOutbox{}
	vendor := &stubVendor{family: VendorFamilyOpenAI, response: VendorResponse{}}
	policies := &stubPolicies{policy: AgentPolicy{Target: target("model-a")}}
	svc := newTestService(t, budget, outbox, vendor, policies)

	if _, err := svc.Invoke(context.Background(), validRequest()); err != nil {
		t.Fatalf("Invoke: %v", err)
	}
	// The downgrade must be re-resolved through the policy loader, otherwise
	// the expensive model the budget exists to avoid is the one dispatched.
	if len(policies.resolved) != 2 {
		t.Fatalf("policy resolved %d times (%v), want 2", len(policies.resolved), policies.resolved)
	}
	if policies.resolved[1] != "model-b" {
		t.Errorf("second resolve = %q, want model-b", policies.resolved[1])
	}
}

func TestInvoke_BudgetDowngradeToUnknownModelFailsLoud(t *testing.T) {
	// A loader that knows model-a but not the downgrade target.
	known := &stubPolicies{policy: AgentPolicy{Target: target("model-a")}}
	onlyA := &policyLoaderAllowing{allowed: map[LogicalModelID]bool{"model-a": true}}
	_ = known

	budget := &atomicBudget{stubBudget: stubBudget{state: &BudgetState{
		BudgetUSDMicros: 1000, SpentUSDMicros: 1000,
		Policy: BudgetPolicyDowngrade, DowngradeToModel: "model-typo",
	}}}
	svc := newTestService(t, budget, &stubOutbox{},
		&stubVendor{family: VendorFamilyOpenAI}, onlyA)

	_, err := svc.Invoke(context.Background(), validRequest())
	if err == nil {
		t.Fatal("expected an error for a downgrade to an unregistered model")
	}
	var cfgErr *ConfigError
	if !errors.As(err, &cfgErr) {
		t.Fatalf("error = %v, want a *ConfigError", err)
	}
}

type policyLoaderAllowing struct {
	allowed map[LogicalModelID]bool
}

func (p *policyLoaderAllowing) ResolveAgentPolicy(
	_ context.Context, _, _ string, requested LogicalModelID, _ []LogicalModelID,
) (AgentPolicy, error) {
	if !p.allowed[requested] {
		return AgentPolicy{}, fmt.Errorf("model %q is not in the registry", requested)
	}
	return AgentPolicy{Target: target(string(requested))}, nil
}

func TestInvoke_BudgetRepoErrorPropagates(t *testing.T) {
	budget := &stubBudget{stateErr: errors.New("db down")}
	svc := newTestService(t, budget, &stubOutbox{},
		&stubVendor{family: VendorFamilyOpenAI}, &stubPolicies{policy: AgentPolicy{Target: target("model-a")}})

	if _, err := svc.Invoke(context.Background(), validRequest()); err == nil {
		t.Fatal("expected an error when the budget repo is unavailable")
	}
}

// ---------------------------------------------------------------------------
// Credentials
// ---------------------------------------------------------------------------

func TestInvoke_ConfiguredButEmptyCredentialIsAMisconfiguration(t *testing.T) {
	svc, err := NewService(ServiceConfig{
		Vendors:        []VendorClient{&stubVendor{family: VendorFamilyOpenAI}},
		Budget:         &atomicBudget{},
		Outbox:         &stubOutbox{},
		Secrets:        &stubSecrets{values: map[string]string{}}, // nothing set
		Policies:       &stubPolicies{policy: AgentPolicy{Target: target("model-a")}},
		GatewayVersion: "test",
		Now:            time.Now,
		NewID:          func() string { return "id" },
	})
	if err != nil {
		t.Fatalf("NewService: %v", err)
	}

	_, err = svc.Invoke(context.Background(), validRequest())
	if err == nil {
		t.Fatal("expected a refusal when the configured credential is empty")
	}
	var cfgErr *ConfigError
	if !errors.As(err, &cfgErr) {
		t.Fatalf("error = %v (%T), want a *ConfigError so the facade reports 500 not 502", err, err)
	}
}

func TestInvoke_TargetWithNoCredentialRefNeedsNoKey(t *testing.T) {
	noKey := target("model-a")
	noKey.APIKeyRef = "" // the local-server case: no api_key_env at all
	vendor := &stubVendor{family: VendorFamilyOpenAI, response: VendorResponse{Completion: "ok"}}
	svc := newTestService(t, &atomicBudget{}, &stubOutbox{}, vendor,
		&stubPolicies{policy: AgentPolicy{Target: noKey}})

	resp, err := svc.Invoke(context.Background(), validRequest())
	if err != nil {
		t.Fatalf("Invoke: %v", err)
	}
	if resp.Completion != "ok" {
		t.Errorf("completion = %q", resp.Completion)
	}
	if vendor.gotReq[0].Credential != "" {
		t.Errorf("credential = %q, want empty", vendor.gotReq[0].Credential)
	}
}

// ---------------------------------------------------------------------------
// Capabilities
// ---------------------------------------------------------------------------

func TestInvoke_ImageRequestToNonImageModelIsRefused(t *testing.T) {
	req := validRequest()
	req.ResponseModality = ModalityImage
	svc := newTestService(t, &atomicBudget{}, &stubOutbox{},
		&stubVendor{family: VendorFamilyOpenAI},
		&stubPolicies{policy: AgentPolicy{Target: target("model-a", CapabilityChat)}})

	_, err := svc.Invoke(context.Background(), req)
	if err == nil {
		t.Fatal("expected a refusal for an image request to a chat-only model")
	}
	var cfgErr *ConfigError
	if !errors.As(err, &cfgErr) {
		t.Fatalf("error = %v, want a *ConfigError", err)
	}
	if !contains(cfgErr.Detail, "image") {
		t.Errorf("detail = %q, want it to name the image capability", cfgErr.Detail)
	}
}

// ---------------------------------------------------------------------------
// Fallback chain
// ---------------------------------------------------------------------------

func TestInvoke_WalksFallbackChainOnVendorFailure(t *testing.T) {
	primary := &stubVendor{family: VendorFamilyOpenAI, err: errors.New("503 from provider")}
	fallback := &stubVendor{family: VendorFamilyAnthropic, response: VendorResponse{Completion: "from the fallback"}}

	svc, err := NewService(ServiceConfig{
		Vendors: []VendorClient{primary, fallback},
		Budget:  &atomicBudget{},
		Outbox:  &stubOutbox{},
		Secrets: &stubSecrets{values: map[string]string{"model-a_KEY": "sk-a", "model-b_KEY": "sk-b"}},
		Policies: &stubPolicies{policy: AgentPolicy{
			Target: target("model-a"),
			FallbackChain: []AgentPolicyFallback{{
				Vendor: VendorFamilyAnthropic,
				Target: func() TargetModel {
					t := target("model-b")
					t.Vendor = VendorFamilyAnthropic
					return t
				}(),
			}},
		}},
		GatewayVersion: "test",
		Now:            time.Now,
		NewID:          func() string { return "id" },
	})
	if err != nil {
		t.Fatalf("NewService: %v", err)
	}

	resp, err := svc.Invoke(context.Background(), validRequest())
	if err != nil {
		t.Fatalf("Invoke: %v", err)
	}
	if resp.Completion != "from the fallback" {
		t.Errorf("completion = %q, want the fallback's", resp.Completion)
	}
	if len(resp.FallbackChain) != 2 {
		t.Errorf("fallback chain = %v, want two entries", resp.FallbackChain)
	}
	if resp.Vendor != string(VendorFamilyAnthropic) {
		t.Errorf("vendor = %q, want anthropic (the one that answered)", resp.Vendor)
	}
}

func TestInvoke_ExhaustedChainReportsWhatItTried(t *testing.T) {
	only := &stubVendor{family: VendorFamilyOpenAI, err: errors.New("provider unreachable")}
	svc, err := NewService(ServiceConfig{
		Vendors: []VendorClient{only},
		Budget:  &atomicBudget{},
		Outbox:  &stubOutbox{},
		Secrets: &stubSecrets{values: map[string]string{"model-a_KEY": "sk-a", "model-b_KEY": "sk-b"}},
		Policies: &stubPolicies{policy: AgentPolicy{
			Target: target("model-a"),
			FallbackChain: []AgentPolicyFallback{
				{Vendor: VendorFamilyAnthropic, Target: func() TargetModel {
					t := target("model-b")
					t.Vendor = VendorFamilyAnthropic
					return t
				}()},
			},
		}},
		GatewayVersion: "test",
		Now:            time.Now,
		NewID:          func() string { return "id" },
	})
	if err != nil {
		t.Fatalf("NewService: %v", err)
	}

	_, err = svc.Invoke(context.Background(), validRequest())
	if err == nil {
		t.Fatal("expected an error when every target fails")
	}
	var invokeErr *InvokeError
	if !errors.As(err, &invokeErr) {
		t.Fatalf("error = %v, want an *InvokeError", err)
	}
	if len(invokeErr.FallbackLog) != 1 {
		// Only the openai target was reached; the anthropic one has no
		// registered adapter in this wiring.
		t.Errorf("fallback log = %v, want the attempted targets", invokeErr.FallbackLog)
	}
}

func TestInvoke_UnknownVendorFamilyIsSkipped(t *testing.T) {
	svc := newTestService(t, &atomicBudget{}, &stubOutbox{},
		&stubVendor{family: VendorFamilyOpenAI},
		&stubPolicies{policy: AgentPolicy{Target: TargetModel{
			Vendor:         "no-such-family",
			LogicalModelID: "model-a",
			UpstreamModel:  "x",
		}}})

	if _, err := svc.Invoke(context.Background(), validRequest()); err == nil {
		t.Fatal("expected an error when no adapter is registered for the family")
	}
}

// ---------------------------------------------------------------------------
// Idempotency
// ---------------------------------------------------------------------------

func TestInvoke_DuplicateDispatchKeySkipsTheDebitButStillLedgers(t *testing.T) {
	budget := &atomicBudget{}
	outbox := &stubOutbox{}
	claims := &stubClaims{claimed: false} // already claimed
	vendor := &stubVendor{family: VendorFamilyOpenAI, response: VendorResponse{
		Completion: "ran anyway",
		Usage:      TokenUsage{CostMicros: 500},
	}}

	svc, err := NewService(ServiceConfig{
		Vendors:        []VendorClient{vendor},
		Budget:         budget,
		Outbox:         outbox,
		Secrets:        &stubSecrets{values: map[string]string{"model-a_KEY": "sk-a"}},
		Policies:       &stubPolicies{policy: AgentPolicy{Target: target("model-a")}},
		Claims:         claims,
		GatewayVersion: "test",
		Now:            time.Now,
		NewID:          func() string { return "id" },
	})
	if err != nil {
		t.Fatalf("NewService: %v", err)
	}

	req := validRequest()
	req.DispatchIdempotencyKey = "dispatch-1"
	resp, err := svc.Invoke(context.Background(), req)
	if err != nil {
		t.Fatalf("Invoke: %v", err)
	}
	// The provider already ran and was paid, so the call completes...
	if resp.Completion != "ran anyway" {
		t.Errorf("completion = %q, want the call to still complete", resp.Completion)
	}
	// ...the debit is suppressed, while the ledger row still lands (Settle is
	// how both happen, so the assertion is on the delta, not on the call).
	if budget.settleCalls != 1 {
		t.Errorf("Settle called %d times, want 1 (the ledger row must still be written)", budget.settleCalls)
	}
	if budget.debited != 0 {
		t.Errorf("debited %d on a duplicate dispatch key, want 0", budget.debited)
	}
	// ...and the usage is still recorded, flagged as deduped.
	if len(budget.settled) != 1 || !budget.settled[0].DebitDeduped {
		t.Errorf("ledger = %+v, want one row with debit_deduped set", budget.settled)
	}
	_ = outbox
}

func TestInvoke_NoDispatchKeyMeansNoClaim(t *testing.T) {
	claims := &stubClaims{claimed: true}
	svc, err := NewService(ServiceConfig{
		Vendors:        []VendorClient{&stubVendor{family: VendorFamilyOpenAI, response: VendorResponse{Usage: TokenUsage{CostMicros: 9}}}},
		Budget:         &atomicBudget{},
		Outbox:         &stubOutbox{},
		Secrets:        &stubSecrets{values: map[string]string{"model-a_KEY": "sk-a"}},
		Policies:       &stubPolicies{policy: AgentPolicy{Target: target("model-a")}},
		Claims:         claims,
		GatewayVersion: "test",
		Now:            time.Now,
		NewID:          func() string { return "id" },
	})
	if err != nil {
		t.Fatalf("NewService: %v", err)
	}
	if _, err := svc.Invoke(context.Background(), validRequest()); err != nil {
		t.Fatalf("Invoke: %v", err)
	}
	if len(claims.keys) != 0 {
		t.Errorf("claimed %v, want no claim without a dispatch key", claims.keys)
	}
}

// ---------------------------------------------------------------------------
// Settle failure
// ---------------------------------------------------------------------------

func TestInvoke_SettleFailureFailsTheCall(t *testing.T) {
	budget := &atomicBudget{settleErr: errors.New("deadlock detected")}
	svc := newTestService(t, budget, &stubOutbox{},
		&stubVendor{family: VendorFamilyOpenAI, response: VendorResponse{Completion: "generated"}},
		&stubPolicies{policy: AgentPolicy{Target: target("model-a")}})

	// The completion already cost money. Reporting success would hand the
	// caller output nobody was billed for, so the failure is surfaced.
	if _, err := svc.Invoke(context.Background(), validRequest()); err == nil {
		t.Fatal("expected the call to fail when the settle transaction fails")
	}
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

// Usage2 exposes the ledger row's usage fields for assertion messages.
func (e TokenUsageEvent) Usage2() TokenUsage {
	return TokenUsage{
		InputTokens:  e.InputTokens,
		OutputTokens: e.OutputTokens,
		CostMicros:   e.CostMicros,
	}
}
