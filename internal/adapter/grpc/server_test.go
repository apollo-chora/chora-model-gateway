package modelgatewaygrpc_test

import (
	"context"
	"errors"
	"net"
	"testing"
	"time"

	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/credentials/insecure"
	"google.golang.org/grpc/metadata"
	"google.golang.org/grpc/status"
	"google.golang.org/grpc/test/bufconn"
	"google.golang.org/protobuf/types/known/structpb"

	mgv1 "github.com/apollo-chora/chora-contracts/gen/go/chora/services/model_gateway/v1"

	modelgatewaygrpc "github.com/5007-Capstone/chora/services/chora-model-gateway/internal/adapter/grpc"
	"github.com/5007-Capstone/chora/services/chora-model-gateway/internal/domain"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// ----------------------------------------------------------------------------
// Test fakes — minimum to drive the domain.Service through the gRPC adapter.
// ----------------------------------------------------------------------------

type fakeVendor struct {
	resp domain.VendorResponse
	err  error
	calls int
	lastReq domain.VendorRequest
}

func (f *fakeVendor) Family() domain.VendorFamily { return domain.VendorFamilyVertexGemini }
func (f *fakeVendor) Generate(ctx context.Context, req domain.VendorRequest) (domain.VendorResponse, error) {
	f.calls++
	f.lastReq = req
	if f.err != nil {
		return domain.VendorResponse{}, f.err
	}
	return f.resp, nil
}

type fakeArmor struct {
	pre  domain.ArmorVerdict
	post domain.ArmorVerdict
}

func (f *fakeArmor) SanitizeUserPrompt(ctx context.Context, t, p string) (domain.ArmorVerdict, string, error) {
	if f.pre == 0 {
		return domain.ArmorVerdictAllow, p, nil
	}
	return f.pre, p, nil
}
func (f *fakeArmor) SanitizeModelResponse(ctx context.Context, t, r string) (domain.ArmorVerdict, string, error) {
	if f.post == 0 {
		return domain.ArmorVerdictAllow, r, nil
	}
	return f.post, r, nil
}

type fakeBudget struct{ state *domain.BudgetState }

func (f *fakeBudget) GetTenantBudget(ctx context.Context, tenantID string) (*domain.BudgetState, error) {
	return f.state, nil
}
func (f *fakeBudget) DebitSpent(ctx context.Context, tenantID string, delta int64) error {
	return nil
}

type fakeOutbox struct{ events []domain.TokenUsageEvent }

func (f *fakeOutbox) EnqueueTokenUsageRecorded(ctx context.Context, e domain.TokenUsageEvent) error {
	f.events = append(f.events, e)
	return nil
}

type fakePolicy struct {
	p                 domain.AgentPolicy
	gotFallbackModels []domain.LogicalModelID
}

func (f *fakePolicy) ResolveAgentPolicy(ctx context.Context, agentID, crewKind string, requested domain.LogicalModelID, fallbackModels []domain.LogicalModelID) (domain.AgentPolicy, error) {
	f.gotFallbackModels = fallbackModels
	return f.p, nil
}

// ----------------------------------------------------------------------------
// Helpers
// ----------------------------------------------------------------------------

func newServerWithFakes(t *testing.T, vendor *fakeVendor) (mgv1.ModelGatewayServiceClient, func()) {
	t.Helper()
	armor := &fakeArmor{}
	budget := &fakeBudget{state: &domain.BudgetState{TenantID: "t1", BudgetUSDMicros: 1_000_000, Policy: domain.BudgetPolicyBlock}}
	outbox := &fakeOutbox{}
	policy := &fakePolicy{p: domain.AgentPolicy{
		AgentID:                "qgen_question",
		ResolvedLogicalModelID: domain.LogicalModelID("gemini-2.5-pro"),
		Vendor:                 domain.VendorFamilyVertexGemini,
		ArmorTemplate:          "projects/x/locations/asia-southeast1/templates/balanced",
	}}
	svc, err := domain.NewService(domain.ServiceConfig{
		Vendors:        []domain.VendorClient{vendor},
		Armor:          armor,
		Budget:         budget,
		Outbox:         outbox,
		Policies:       policy,
		GatewayVersion: "test:0",
		Now:            func() time.Time { return time.Date(2026, 5, 24, 10, 0, 0, 0, time.UTC) },
		NewID:          func() string { return "test-id" },
	})
	require.NoError(t, err)
	srvAdapter, err := modelgatewaygrpc.NewServer(svc, svc, svc)
	require.NoError(t, err)

	lis := bufconn.Listen(1024 * 1024)
	gs := grpc.NewServer()
	mgv1.RegisterModelGatewayServiceServer(gs, srvAdapter)
	go func() { _ = gs.Serve(lis) }()

	dialer := func(ctx context.Context, addr string) (net.Conn, error) {
		return lis.DialContext(ctx)
	}
	conn, err := grpc.NewClient(
		"passthrough://bufnet",
		grpc.WithContextDialer(dialer),
		grpc.WithTransportCredentials(insecure.NewCredentials()),
	)
	require.NoError(t, err)
	client := mgv1.NewModelGatewayServiceClient(conn)
	cleanup := func() {
		_ = conn.Close()
		gs.Stop()
	}
	return client, cleanup
}

func happyVendorResponse() *fakeVendor {
	return &fakeVendor{
		resp: domain.VendorResponse{
			Completion:   "completion-stub",
			Usage:        domain.TokenUsage{InputTokens: 100, OutputTokens: 50, CostMicros: 362},
			ModelVersion: "gemini-2.5-pro@001",
			FinishReason: domain.FinishReasonComplete,
		},
	}
}

// ----------------------------------------------------------------------------
// Happy path — full proto round-trip
// ----------------------------------------------------------------------------

func TestInvoke_HappyPath_ProtoRoundtrip(t *testing.T) {
	vendor := happyVendorResponse()
	client, cleanup := newServerWithFakes(t, vendor)
	defer cleanup()

	cfg, err := structpb.NewStruct(map[string]any{"temperature": 0.7})
	require.NoError(t, err)

	resp, err := client.Invoke(context.Background(), &mgv1.InvokeRequest{
		TenantId:         "t1",
		Gcid:             "g1",
		AgentId:          "qgen_question",
		LogicalModelId:   "gemini-2.5-pro",
		Prompt:           "what's scrum cadence?",
		GenerationConfig: cfg,
		Traceparent:      "00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01",
	})
	require.NoError(t, err)

	assert.Equal(t, "test-id", resp.InvocationId)
	assert.Equal(t, "completion-stub", resp.Completion)
	assert.Equal(t, mgv1.FinishReason_FINISH_REASON_COMPLETE, resp.FinishReason)
	assert.Equal(t, mgv1.ModelArmorVerdict_MODEL_ARMOR_VERDICT_ALLOW, resp.ModelArmorVerdictPre)
	assert.Equal(t, mgv1.ModelArmorVerdict_MODEL_ARMOR_VERDICT_ALLOW, resp.ModelArmorVerdictPost)
	assert.Equal(t, "vertex_ai_gemini", resp.Vendor)
	assert.Equal(t, []string{"vertex_ai_gemini:gemini-2.5-pro"}, resp.FallbackChain)
	require.NotNil(t, resp.Usage)
	assert.Equal(t, int64(362), resp.Usage.CostMicros)
	assert.Equal(t, "test:0", resp.GatewayVersion)

	// Generation config + traceparent propagated to the vendor.
	assert.Equal(t, "00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01", vendor.lastReq.Traceparent)
	require.NotNil(t, vendor.lastReq.GenerationConfig)
	assert.Equal(t, 0.7, vendor.lastReq.GenerationConfig["temperature"])
}

// ----------------------------------------------------------------------------
// Image modality (W8, CR 2026-06-01) — proto⇄domain mapping both directions
// ----------------------------------------------------------------------------

// TestInvoke_ImageModality_ProtoRoundtrip asserts the adapter maps
// proto InvokeRequest.ResponseModality → domain (inbound) and domain
// InvokeResponse.ImageBytes/ImageMIMEType → proto ImageBytes/ImageMimeType
// (outbound) across the full bufconn round-trip.
func TestInvoke_ImageModality_ProtoRoundtrip(t *testing.T) {
	imgBytes := []byte{0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a}
	vendor := &fakeVendor{
		resp: domain.VendorResponse{
			Completion:    "", // image-only
			ImageBytes:    imgBytes,
			ImageMIMEType: "image/png",
			Usage:         domain.TokenUsage{InputTokens: 12, OutputTokens: 0, CostMicros: 15},
			ModelVersion:  "gemini-3-pro-image@001",
			FinishReason:  domain.FinishReasonComplete,
		},
	}
	client, cleanup := newServerWithFakes(t, vendor)
	defer cleanup()

	resp, err := client.Invoke(context.Background(), &mgv1.InvokeRequest{
		TenantId:         "t1",
		Gcid:             "g1",
		AgentId:          "qgen_question",
		LogicalModelId:   "gemini-3-pro-image",
		Prompt:           "draw a red square",
		ResponseModality: "IMAGE",
	})
	require.NoError(t, err)

	// Inbound: proto response_modality reached the vendor request.
	assert.Equal(t, "IMAGE", vendor.lastReq.ResponseModality)

	// Outbound: domain image fields surfaced on the proto response.
	assert.Equal(t, imgBytes, resp.ImageBytes)
	assert.Equal(t, "image/png", resp.ImageMimeType)
	assert.Equal(t, "", resp.Completion)
}

// ----------------------------------------------------------------------------
// Traceparent fallback via gRPC metadata
// ----------------------------------------------------------------------------

func TestInvoke_TraceparentFromMetadata(t *testing.T) {
	vendor := happyVendorResponse()
	client, cleanup := newServerWithFakes(t, vendor)
	defer cleanup()

	ctx := metadata.AppendToOutgoingContext(context.Background(),
		"traceparent", "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01",
		"tracestate", "vendor=foo",
	)
	_, err := client.Invoke(ctx, &mgv1.InvokeRequest{
		TenantId:       "t1",
		Gcid:           "g1",
		AgentId:        "qgen_question",
		LogicalModelId: "gemini-2.5-pro",
		Prompt:         "x",
	})
	require.NoError(t, err)
	assert.Equal(t, "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01", vendor.lastReq.Traceparent)
	assert.Equal(t, "vendor=foo", vendor.lastReq.Tracestate)
}

// ----------------------------------------------------------------------------
// Required-field validation
// ----------------------------------------------------------------------------

func TestInvoke_MissingFields_InvalidArgument(t *testing.T) {
	client, cleanup := newServerWithFakes(t, happyVendorResponse())
	defer cleanup()

	cases := []struct {
		name string
		mut  func(r *mgv1.InvokeRequest)
		want string
	}{
		{"no tenant", func(r *mgv1.InvokeRequest) { r.TenantId = "" }, "tenant_id"},
		{"no gcid", func(r *mgv1.InvokeRequest) { r.Gcid = "" }, "gcid"},
		{"no agent", func(r *mgv1.InvokeRequest) { r.AgentId = "" }, "agent_id"},
		{"no model", func(r *mgv1.InvokeRequest) { r.LogicalModelId = "" }, "logical_model_id"},
		{"no prompt", func(r *mgv1.InvokeRequest) { r.Prompt = "" }, "prompt"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			req := &mgv1.InvokeRequest{
				TenantId: "t1", Gcid: "g1", AgentId: "qgen", LogicalModelId: "gemini-2.5-pro", Prompt: "x",
			}
			tc.mut(req)
			_, err := client.Invoke(context.Background(), req)
			require.Error(t, err)
			st, ok := status.FromError(err)
			require.True(t, ok)
			assert.Equal(t, codes.InvalidArgument, st.Code())
			assert.Contains(t, st.Message(), tc.want)
		})
	}
}

// ----------------------------------------------------------------------------
// Vendor exhaustion → UNAVAILABLE
// ----------------------------------------------------------------------------

func TestInvoke_VendorExhausted_Unavailable(t *testing.T) {
	client, cleanup := newServerWithFakes(t, &fakeVendor{err: errors.New("vertex 503")})
	defer cleanup()
	_, err := client.Invoke(context.Background(), &mgv1.InvokeRequest{
		TenantId: "t1", Gcid: "g1", AgentId: "qgen_question", LogicalModelId: "gemini-2.5-pro", Prompt: "x",
	})
	require.Error(t, err)
	st, ok := status.FromError(err)
	require.True(t, ok)
	assert.Equal(t, codes.Unavailable, st.Code())
}

// upstreamError is a vendor error that carries the provider's own HTTP
// status, satisfying domain.UpstreamStatusProvider so the relay path is
// exercised end-to-end through the domain walk + adapter mapping.
type upstreamError struct {
	status int
	msg    string
}

func (e *upstreamError) Error() string     { return e.msg }
func (e *upstreamError) UpstreamStatus() int { return e.status }

// TestInvoke_UpstreamStatusRelay pins the upstream HTTP status relay: a
// vendor failure carrying the provider's own status is relayed to the
// closest gRPC code rather than flattened into one.
func TestInvoke_UpstreamStatusRelay(t *testing.T) {
	cases := []struct {
		name string
		status int
		want codes.Code
	}{
		{"401 → Unauthenticated", 401, codes.Unauthenticated},
		{"403 → Unauthenticated", 403, codes.Unauthenticated},
		{"404 → NotFound", 404, codes.NotFound},
		{"429 → ResourceExhausted", 429, codes.ResourceExhausted},
		{"500 → Unavailable", 500, codes.Unavailable},
		{"503 → Unavailable", 503, codes.Unavailable},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			client, cleanup := newServerWithFakes(t, &fakeVendor{err: &upstreamError{status: tc.status, msg: "upstream blew up"}})
			defer cleanup()
			_, err := client.Invoke(context.Background(), &mgv1.InvokeRequest{
				TenantId: "t1", Gcid: "g1", AgentId: "qgen_question", LogicalModelId: "gemini-2.5-pro", Prompt: "x",
			})
			require.Error(t, err)
			st, ok := status.FromError(err)
			require.True(t, ok)
			assert.Equal(t, tc.want, st.Code())
		})
	}
}

// TestInvoke_SystemPromptPrependedToContents verifies that when a structured
// conversation (contents_json) is present, the system prompt is carried as
// the leading system message inside contents (and the vendor-native system
// instruction is cleared so the instruction is not stated twice).
func TestInvoke_SystemPromptPrependedToContents(t *testing.T) {
	vendor := happyVendorResponse()
	client, cleanup := newServerWithFakes(t, vendor)
	defer cleanup()

	_, err := client.Invoke(context.Background(), &mgv1.InvokeRequest{
		TenantId: "t1", Gcid: "g1", AgentId: "qgen_question", LogicalModelId: "gemini-2.5-pro",
		Prompt:       "trigger turn",
		SystemPrompt: "you are a helpful agent",
		ContentsJson: `[{"role":"user","content":"hello"}]`,
	})
	require.NoError(t, err)

	// The system message is prepended to the contents the vendor receives.
	assert.JSONEq(t,
		`[{"role":"system","content":"you are a helpful agent"},{"role":"user","content":"hello"}]`,
		vendor.lastReq.ContentsJSON)
	// The vendor-native system instruction is cleared (the instruction now
	// rides in contents).
	assert.Empty(t, vendor.lastReq.SystemPrompt)
}

// TestInvoke_SystemPromptNotPrependedWhenContentsEmpty verifies the guard:
// with contents_json empty the user turn is the flat prompt, so the system
// prompt is NOT prepended (prepending to an empty list would drop the user
// turn) and instead rides as the vendor-native system instruction.
func TestInvoke_SystemPromptNotPrependedWhenContentsEmpty(t *testing.T) {
	vendor := happyVendorResponse()
	client, cleanup := newServerWithFakes(t, vendor)
	defer cleanup()

	_, err := client.Invoke(context.Background(), &mgv1.InvokeRequest{
		TenantId: "t1", Gcid: "g1", AgentId: "qgen_question", LogicalModelId: "gemini-2.5-pro",
		Prompt:       "trigger turn",
		SystemPrompt: "you are a helpful agent",
	})
	require.NoError(t, err)

	assert.Empty(t, vendor.lastReq.ContentsJSON)
	assert.Equal(t, "you are a helpful agent", vendor.lastReq.SystemPrompt)
}

// TestInvoke_MalformedContentsJSON_DegradesGracefully verifies that a
// malformed contents_json never raises past the status mapper: the system
// turn is not prepended and the raw string is passed to the vendor, which
// falls back to the flat prompt.
func TestInvoke_MalformedContentsJSON_DegradesGracefully(t *testing.T) {
	vendor := happyVendorResponse()
	client, cleanup := newServerWithFakes(t, vendor)
	defer cleanup()

	_, err := client.Invoke(context.Background(), &mgv1.InvokeRequest{
		TenantId: "t1", Gcid: "g1", AgentId: "qgen_question", LogicalModelId: "gemini-2.5-pro",
		Prompt:       "trigger turn",
		SystemPrompt: "you are a helpful agent",
		ContentsJson: `{not valid json`,
	})
	require.NoError(t, err)

	// The raw malformed string is passed through; the vendor re-parses it and
	// falls back to the flat prompt. No system turn is prepended.
	assert.Equal(t, `{not valid json`, vendor.lastReq.ContentsJSON)
	assert.Equal(t, "you are a helpful agent", vendor.lastReq.SystemPrompt)
}

// TestInvoke_MalformedToolsJSON_DegradesGracefully verifies that a malformed
// tools_json is dropped (empty) rather than passed to the vendor.
func TestInvoke_MalformedToolsJSON_DegradesGracefully(t *testing.T) {
	vendor := happyVendorResponse()
	client, cleanup := newServerWithFakes(t, vendor)
	defer cleanup()

	_, err := client.Invoke(context.Background(), &mgv1.InvokeRequest{
		TenantId: "t1", Gcid: "g1", AgentId: "qgen_question", LogicalModelId: "gemini-2.5-pro",
		Prompt:      "trigger turn",
		ToolsJson:   `{not valid json`,
	})
	require.NoError(t, err)
	assert.Empty(t, vendor.lastReq.ToolsJSON)
}

// TestInvoke_ModalityDefaultsToText verifies an absent response_modality is
// mapped to TEXT (mirroring the Python adapter's `or "TEXT"`).
func TestInvoke_ModalityDefaultsToText(t *testing.T) {
	vendor := happyVendorResponse()
	client, cleanup := newServerWithFakes(t, vendor)
	defer cleanup()

	_, err := client.Invoke(context.Background(), &mgv1.InvokeRequest{
		TenantId: "t1", Gcid: "g1", AgentId: "qgen_question", LogicalModelId: "gemini-2.5-pro",
		Prompt: "x",
	})
	require.NoError(t, err)
	assert.Equal(t, "TEXT", vendor.lastReq.ResponseModality)
}

// TestInvoke_ConfigError_FailedPrecondition verifies the error taxonomy:
// a gateway misconfiguration (the policy references a vendor family that is
// not registered) maps to FailedPrecondition, not Unavailable — the caller
// cannot retry their way out of a misconfiguration.
func TestInvoke_ConfigError_FailedPrecondition(t *testing.T) {
	vendor := happyVendorResponse()
	armor := &fakeArmor{}
	budget := &fakeBudget{state: &domain.BudgetState{TenantID: "t1", BudgetUSDMicros: 1_000_000, Policy: domain.BudgetPolicyBlock}}
	outbox := &fakeOutbox{}
	// The policy references anthropic_byoa, which is NOT registered in this
	// wiring (only vertex_ai_gemini is) → a config refusal.
	policy := &fakePolicy{p: domain.AgentPolicy{
		AgentID:                "qgen_question",
		ResolvedLogicalModelID: domain.LogicalModelID("claude-opus-4-7"),
		Vendor:                 domain.VendorFamilyAnthropic,
	}}
	svc, err := domain.NewService(domain.ServiceConfig{
		Vendors:        []domain.VendorClient{vendor},
		Armor:          armor,
		Budget:         budget,
		Outbox:         outbox,
		Policies:       policy,
		GatewayVersion: "test:0",
		Now:            time.Now,
		NewID:          func() string { return "id" },
	})
	require.NoError(t, err)
	srvAdapter, err := modelgatewaygrpc.NewServer(svc, svc, svc)
	require.NoError(t, err)
	lis := bufconn.Listen(1024 * 1024)
	gs := grpc.NewServer()
	mgv1.RegisterModelGatewayServiceServer(gs, srvAdapter)
	go func() { _ = gs.Serve(lis) }()
	defer gs.Stop()
	conn, err := grpc.NewClient("passthrough://bufnet",
		grpc.WithContextDialer(func(ctx context.Context, _ string) (net.Conn, error) { return lis.DialContext(ctx) }),
		grpc.WithTransportCredentials(insecure.NewCredentials()),
	)
	require.NoError(t, err)
	defer func() { _ = conn.Close() }()
	client := mgv1.NewModelGatewayServiceClient(conn)

	_, err = client.Invoke(context.Background(), &mgv1.InvokeRequest{
		TenantId: "t1", Gcid: "g1", AgentId: "qgen_question", LogicalModelId: "claude-opus-4-7", Prompt: "x",
	})
	require.Error(t, err)
	st, ok := status.FromError(err)
	require.True(t, ok)
	assert.Equal(t, codes.FailedPrecondition, st.Code())
}

// ----------------------------------------------------------------------------
// In-band budget block — success-with-FinishReasonBudgetBlock (not error)
// ----------------------------------------------------------------------------

func TestInvoke_BudgetBlock_InBand(t *testing.T) {
	// Stub adapter direct rather than via the helper because we need to
	// flip Budget to BLOCK + Spent=Budget.
	vendor := happyVendorResponse()
	armor := &fakeArmor{}
	budget := &fakeBudget{state: &domain.BudgetState{
		TenantID: "t1", BudgetUSDMicros: 100, SpentUSDMicros: 100, Policy: domain.BudgetPolicyBlock,
	}}
	outbox := &fakeOutbox{}
	policy := &fakePolicy{p: domain.AgentPolicy{
		AgentID: "qgen_question", ResolvedLogicalModelID: "gemini-2.5-pro",
		Vendor: domain.VendorFamilyVertexGemini, ArmorTemplate: "projects/x/locations/asia-southeast1/templates/b",
	}}
	svc, err := domain.NewService(domain.ServiceConfig{
		Vendors:        []domain.VendorClient{vendor},
		Armor:          armor,
		Budget:         budget,
		Outbox:         outbox,
		Policies:       policy,
		GatewayVersion: "test:0",
		Now:            time.Now,
		NewID:          func() string { return "id" },
	})
	require.NoError(t, err)
	srvAdapter, err := modelgatewaygrpc.NewServer(svc, svc, svc)
	require.NoError(t, err)
	lis := bufconn.Listen(1024 * 1024)
	gs := grpc.NewServer()
	mgv1.RegisterModelGatewayServiceServer(gs, srvAdapter)
	go func() { _ = gs.Serve(lis) }()
	defer gs.Stop()
	conn, err := grpc.NewClient("passthrough://bufnet",
		grpc.WithContextDialer(func(ctx context.Context, _ string) (net.Conn, error) { return lis.DialContext(ctx) }),
		grpc.WithTransportCredentials(insecure.NewCredentials()),
	)
	require.NoError(t, err)
	defer func() { _ = conn.Close() }()
	client := mgv1.NewModelGatewayServiceClient(conn)

	resp, err := client.Invoke(context.Background(), &mgv1.InvokeRequest{
		TenantId: "t1", Gcid: "g1", AgentId: "qgen_question",
		LogicalModelId: "gemini-2.5-pro", Prompt: "x",
	})
	require.NoError(t, err) // in-band, no gRPC error
	assert.Equal(t, mgv1.FinishReason_FINISH_REASON_BUDGET_BLOCK, resp.FinishReason)
	assert.Equal(t, 0, vendor.calls)
	assert.Empty(t, outbox.events)
}

// ----------------------------------------------------------------------------
// NewServer validation
// ----------------------------------------------------------------------------

func TestNewServer_NilService(t *testing.T) {
	_, err := modelgatewaygrpc.NewServer(nil, nil, nil)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "required")
}

// ----------------------------------------------------------------------------
// PullTraceparent helper
// ----------------------------------------------------------------------------

func TestPullTraceparent(t *testing.T) {
	// No metadata → empty.
	assert.Equal(t, "", modelgatewaygrpc.PullTraceparent(context.Background()))
	// With metadata → trimmed.
	md := metadata.New(map[string]string{"traceparent": "  00-abc-def-01  "})
	ctx := metadata.NewIncomingContext(context.Background(), md)
	assert.Equal(t, "00-abc-def-01", modelgatewaygrpc.PullTraceparent(ctx))
}
