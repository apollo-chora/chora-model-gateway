package modelgatewaygrpc_test

// ADR-254 D7 wire contract tests: the two new InvokeRequest fields (surface,
// dispatch_idempotency_key) reach the domain request verbatim, and a
// domain.PreconditionError maps to FAILED_PRECONDITION with its machine token
// first in the status message (companion_suspended / surface_unstamped), except
// an unreadable suspension store which maps to UNAVAILABLE (retryable, not a
// policy verdict).

import (
	"context"
	"net"
	"testing"

	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/credentials/insecure"
	"google.golang.org/grpc/status"
	"google.golang.org/grpc/test/bufconn"

	mgv1 "github.com/apollo-chora/chora-contracts/gen/go/chora/services/model_gateway/v1"

	modelgatewaygrpc "github.com/apollo-chora/chora-model-gateway/internal/adapter/grpc"
	"github.com/apollo-chora/chora-model-gateway/internal/domain"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// capturingInvoker stands in for the Executor (the governance chain).
type capturingInvoker struct {
	got  domain.InvokeRequest
	resp domain.InvokeResponse
	err  error
}

func (c *capturingInvoker) Execute(_ context.Context, req domain.InvokeRequest) (domain.InvokeResponse, error) {
	c.got = req
	return c.resp, c.err
}

func newServerWithInvoker(t *testing.T, inv *capturingInvoker) (mgv1.ModelGatewayServiceClient, func()) {
	t.Helper()
	// GroundedSearch + Embed are not exercised here; the real service is still
	// required by the constructor (no nil wiring), so build a minimal one.
	vendor := happyVendorResponse()
	svc, err := domain.NewService(domain.ServiceConfig{
		Vendors:        []domain.VendorClient{vendor},
		Armor:          &fakeArmor{},
		Budget:         &fakeBudget{},
		Outbox:         &fakeOutbox{},
		Policies:       &fakePolicy{p: domain.AgentPolicy{AgentID: "companion_chat", ResolvedLogicalModelID: "gemini-2.5-flash", Vendor: domain.VendorFamilyVertexGemini}},
		GatewayVersion: "test:0",
		NewID:          func() string { return "test-id" },
	})
	require.NoError(t, err)
	srvAdapter, err := modelgatewaygrpc.NewServer(inv, svc, svc)
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

func d7Request() *mgv1.InvokeRequest {
	return &mgv1.InvokeRequest{
		InvocationId:           "inv-d7",
		TenantId:               "01234567-89ab-7cde-8f01-234567890aaa",
		Gcid:                   "gcid-1",
		AgentId:                "companion_chat",
		CrewKind:               "companion_chat",
		LogicalModelId:         "gemini-2.5-flash",
		Prompt:                 "explain photosynthesis",
		ActionCode:             "companion_chat_turn_basic",
		Surface:                "companion_chat",
		DispatchIdempotencyKey: "wf-0193-turn-7",
	}
}

func TestInvoke_D7Fields_ReachDomainVerbatim(t *testing.T) {
	inv := &capturingInvoker{resp: domain.InvokeResponse{InvocationID: "inv-d7", FinishReason: domain.FinishReasonComplete}}
	client, cleanup := newServerWithInvoker(t, inv)
	defer cleanup()

	_, err := client.Invoke(context.Background(), d7Request())
	require.NoError(t, err)
	assert.Equal(t, "companion_chat", inv.got.Surface)
	assert.Equal(t, "wf-0193-turn-7", inv.got.DispatchIdempotencyKey)
	assert.Equal(t, "companion_chat_turn_basic", inv.got.ActionCode)
}

func TestInvoke_PreconditionError_MapsToFailedPrecondition_TokenFirst(t *testing.T) {
	cases := []struct {
		reason string
		want   codes.Code
	}{
		{domain.DenyCompanionSuspended, codes.FailedPrecondition},
		{domain.DenySurfaceUnstamped, codes.FailedPrecondition},
		{domain.DenySuspensionUnreadable, codes.Unavailable},
	}
	for _, tc := range cases {
		t.Run(tc.reason, func(t *testing.T) {
			inv := &capturingInvoker{err: &domain.PreconditionError{Reason: tc.reason, Detail: "detail"}}
			client, cleanup := newServerWithInvoker(t, inv)
			defer cleanup()

			_, err := client.Invoke(context.Background(), d7Request())
			require.Error(t, err)
			st, ok := status.FromError(err)
			require.True(t, ok)
			assert.Equal(t, tc.want, st.Code())
			assert.True(t, len(st.Message()) >= len(tc.reason) && st.Message()[:len(tc.reason)] == tc.reason,
				"token must lead the message, got %q", st.Message())
		})
	}
}
