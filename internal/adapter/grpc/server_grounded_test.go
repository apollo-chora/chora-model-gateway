package modelgatewaygrpc_test

import (
	"context"
	"net"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/credentials/insecure"
	"google.golang.org/grpc/status"
	"google.golang.org/grpc/test/bufconn"

	mgv1 "github.com/apollo-chora/chora-contracts/gen/go/chora/services/model_gateway/v1"

	modelgatewaygrpc "github.com/5007-Capstone/chora/services/chora-model-gateway/internal/adapter/grpc"
	"github.com/5007-Capstone/chora/services/chora-model-gateway/internal/domain"
)

// ---- grounded-port fakes ----

type fakeEgressGate struct {
	auth domain.EgressAuthorization
	err  error
}

func (f *fakeEgressGate) Authorize(ctx context.Context, tenantID string) (domain.EgressAuthorization, error) {
	return f.auth, f.err
}
func (f *fakeEgressGate) RecordEgress(ctx context.Context, tenantID string) error { return nil }

type fakeGrounded struct {
	resp domain.GroundedVendorResponse
	err  error
}

func (f *fakeGrounded) Family() domain.VendorFamily { return domain.VendorFamilyVertexGemini }
func (f *fakeGrounded) GroundedGenerate(ctx context.Context, req domain.GroundedVendorRequest) (domain.GroundedVendorResponse, error) {
	if f.err != nil {
		return domain.GroundedVendorResponse{}, f.err
	}
	return f.resp, nil
}

type fakeMana struct{ quote domain.ManaQuote }

func (f *fakeMana) Quote(ctx context.Context, gcid, tenantID, ac string) (domain.ManaQuote, error) {
	return f.quote, nil
}
func (f *fakeMana) Debit(ctx context.Context, gcid, tenantID, ac, idem string) (domain.ManaDebit, error) {
	return domain.ManaDebit{Success: true}, nil
}

type fakeEgressAudit struct{ n int }

func (f *fakeEgressAudit) EnqueueExternalEgressAudited(ctx context.Context, e domain.ExternalEgressAuditEvent) error {
	f.n++
	return nil
}

// newGroundedServer builds a bufconn server whose domain.Service has the
// grounded ports wired. mods tweak the fakes for a specific case.
func newGroundedServer(t *testing.T, mods ...func(eg *fakeEgressGate, gv *fakeGrounded, mn *fakeMana, ar *fakeArmor)) (mgv1.ModelGatewayServiceClient, func()) {
	t.Helper()
	eg := &fakeEgressGate{auth: domain.EgressAuthorization{Allowed: true}}
	gv := &fakeGrounded{resp: domain.GroundedVendorResponse{
		Answer: "cited answer",
		Citations: []domain.GroundedCitation{
			{URI: "https://redirect/1", Title: "NASA", Domain: "nasa.gov", Snippet: "s", Confidence: 0.9},
		},
		SearchEntryPointHTML: "<div>chip</div>",
		WebSearchQueries:     []string{"q"},
		Usage:                domain.TokenUsage{InputTokens: 10, OutputTokens: 20, CostMicros: 44_000},
		ModelVersion:         "gemini-2.5-flash",
		FinishReason:         domain.FinishReasonComplete,
	}}
	mn := &fakeMana{quote: domain.ManaQuote{Affordable: true, RequiredUnits: 80, AvailableUnits: 500}}
	ar := &fakeArmor{}
	for _, m := range mods {
		m(eg, gv, mn, ar)
	}

	svc, err := domain.NewService(domain.ServiceConfig{
		Vendors:        []domain.VendorClient{&fakeVendor{}},
		Armor:          ar,
		Budget:         &fakeBudget{state: &domain.BudgetState{TenantID: "t", BudgetUSDMicros: 1_000_000, Policy: domain.BudgetPolicyBlock}},
		Outbox:         &fakeOutbox{},
		Policies:       &fakePolicy{p: domain.AgentPolicy{AgentID: "familiar_seeker", ResolvedLogicalModelID: "gemini-2.5-flash", Vendor: domain.VendorFamilyVertexGemini, ArmorTemplate: "projects/x/locations/asia-southeast1/templates/balanced"}},
		GatewayVersion: "test:0",
		EgressGate:     eg,
		Grounded:       gv,
		Mana:           mn,
		EgressAudit:    &fakeEgressAudit{},
		Now:            func() time.Time { return time.Date(2026, 7, 11, 10, 0, 0, 0, time.UTC) },
		NewID:          func() string { return "gs-id" },
	})
	require.NoError(t, err)
	srvAdapter, err := modelgatewaygrpc.NewServer(svc, svc, svc)
	require.NoError(t, err)

	lis := bufconn.Listen(1024 * 1024)
	gs := grpc.NewServer()
	mgv1.RegisterModelGatewayServiceServer(gs, srvAdapter)
	go func() { _ = gs.Serve(lis) }()
	conn, err := grpc.NewClient("passthrough://bufnet",
		grpc.WithContextDialer(func(ctx context.Context, _ string) (net.Conn, error) { return lis.DialContext(ctx) }),
		grpc.WithTransportCredentials(insecure.NewCredentials()))
	require.NoError(t, err)
	return mgv1.NewModelGatewayServiceClient(conn), func() { _ = conn.Close(); gs.Stop() }
}

func groundedProtoReq() *mgv1.GroundedSearchRequest {
	return &mgv1.GroundedSearchRequest{
		TenantId:   "11111111-1111-7111-8111-111111111111",
		Gcid:       "22222222-2222-7222-8222-222222222222",
		AgentId:    "familiar_seeker",
		Directive:  "Is the Great Wall visible from space?",
		ActionCode: "familiar_far_sight_grounded_search",
	}
}

func TestGroundedSearch_ProtoRoundtrip(t *testing.T) {
	client, cleanup := newGroundedServer(t)
	defer cleanup()

	resp, err := client.GroundedSearch(context.Background(), groundedProtoReq())
	require.NoError(t, err)
	require.Len(t, resp.Citations, 1)
	assert.Equal(t, "nasa.gov", resp.Citations[0].Domain)
	assert.Equal(t, "NASA", resp.Citations[0].Title)
	assert.InDelta(t, 0.9, resp.Citations[0].Confidence, 0.001)
	assert.Equal(t, "cited answer", resp.GroundedAnswer)
	assert.Equal(t, "<div>chip</div>", resp.SearchEntryPointHtml)
	assert.Equal(t, []string{"q"}, resp.WebSearchQueries)
	assert.Equal(t, mgv1.FinishReason_FINISH_REASON_COMPLETE, resp.FinishReason)
	assert.Equal(t, "gs-id", resp.InvocationId)
}

func TestGroundedSearch_EgressDisabled_FailedPrecondition(t *testing.T) {
	client, cleanup := newGroundedServer(t, func(eg *fakeEgressGate, _ *fakeGrounded, _ *fakeMana, _ *fakeArmor) {
		eg.auth = domain.EgressAuthorization{Allowed: false, Reason: domain.EgressDenyDisabled}
	})
	defer cleanup()
	_, err := client.GroundedSearch(context.Background(), groundedProtoReq())
	require.Error(t, err)
	assert.Equal(t, codes.FailedPrecondition, status.Code(err))
	assert.Contains(t, status.Convert(err).Message(), domain.EgressDenyDisabled)
}

func TestGroundedSearch_ArmorPreBlock_PermissionDenied(t *testing.T) {
	client, cleanup := newGroundedServer(t, func(_ *fakeEgressGate, _ *fakeGrounded, _ *fakeMana, ar *fakeArmor) {
		ar.pre = domain.ArmorVerdictBlock
	})
	defer cleanup()
	_, err := client.GroundedSearch(context.Background(), groundedProtoReq())
	require.Error(t, err)
	assert.Equal(t, codes.PermissionDenied, status.Code(err))
}

func TestGroundedSearch_ArmorPostBlock_ResourceExhausted(t *testing.T) {
	client, cleanup := newGroundedServer(t, func(_ *fakeEgressGate, _ *fakeGrounded, _ *fakeMana, ar *fakeArmor) {
		ar.post = domain.ArmorVerdictBlock
	})
	defer cleanup()
	_, err := client.GroundedSearch(context.Background(), groundedProtoReq())
	require.Error(t, err)
	assert.Equal(t, codes.ResourceExhausted, status.Code(err))
}

func TestGroundedSearch_ManaBlock_InBand(t *testing.T) {
	client, cleanup := newGroundedServer(t, func(_ *fakeEgressGate, _ *fakeGrounded, mn *fakeMana, _ *fakeArmor) {
		mn.quote = domain.ManaQuote{Affordable: false, RequiredUnits: 80, AvailableUnits: 5}
	})
	defer cleanup()
	resp, err := client.GroundedSearch(context.Background(), groundedProtoReq())
	require.NoError(t, err, "mana block is in-band, not a gRPC error")
	assert.Equal(t, mgv1.FinishReason_FINISH_REASON_MANA_BLOCK, resp.FinishReason)
	assert.Contains(t, resp.FinishDetail, "insufficient_mana")
}

func TestGroundedSearch_VendorError_Unavailable(t *testing.T) {
	client, cleanup := newGroundedServer(t, func(_ *fakeEgressGate, gv *fakeGrounded, _ *fakeMana, _ *fakeArmor) {
		gv.err = assertErr("vertex 503")
	})
	defer cleanup()
	_, err := client.GroundedSearch(context.Background(), groundedProtoReq())
	require.Error(t, err)
	assert.Equal(t, codes.Unavailable, status.Code(err))
}

func TestGroundedSearch_MissingFields_InvalidArgument(t *testing.T) {
	client, cleanup := newGroundedServer(t)
	defer cleanup()
	for _, tc := range []struct {
		name string
		mut  func(r *mgv1.GroundedSearchRequest)
	}{
		{"no tenant", func(r *mgv1.GroundedSearchRequest) { r.TenantId = "" }},
		{"no gcid", func(r *mgv1.GroundedSearchRequest) { r.Gcid = "" }},
		{"no agent", func(r *mgv1.GroundedSearchRequest) { r.AgentId = "" }},
		{"no directive", func(r *mgv1.GroundedSearchRequest) { r.Directive = "" }},
		{"no action_code", func(r *mgv1.GroundedSearchRequest) { r.ActionCode = "" }},
	} {
		t.Run(tc.name, func(t *testing.T) {
			r := groundedProtoReq()
			tc.mut(r)
			_, err := client.GroundedSearch(context.Background(), r)
			require.Error(t, err)
			assert.Equal(t, codes.InvalidArgument, status.Code(err))
		})
	}
}

type assertErr string

func (e assertErr) Error() string { return string(e) }
