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

	mgv1 "github.com/apollo-chora/chora-contracts/gen/go/chora/services/model_gateway/v1"

	modelgatewaygrpc "github.com/apollo-chora/chora-model-gateway/internal/adapter/grpc"
	"github.com/apollo-chora/chora-model-gateway/internal/domain"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// The Embed RPC (G1'-1) shipped and is serving in production, and the gRPC
// adapter half of it had NO test coverage at all: `Embed` read 0.0 percent on
// the lane's own profile. It is the seam that turns proto into domain and vendor
// errors into status codes, which is precisely where a silent mismapping hides.
// Two of its behaviours are load-bearing beyond the happy path.
//
// The ERROR MAPPING is string-matched, so it is fragile by construction: a
// reworded domain error silently changes the gRPC code a caller sees, turning a
// retryable Unavailable into a terminal InvalidArgument or the reverse. Pinning
// each arm here means that reword fails a test instead of changing production
// retry behaviour.
//
// The TRACEPARENT FALLBACK is what keeps an embedding's ledger row correlated
// with the caller's trace when the caller propagates context in gRPC metadata
// rather than on the proto field, which is what the consumption and creation
// embedders actually do.

// fakeEmbedder drives the adapter directly. The real domain.Service needs a
// wired EmbeddingClient and would only ever exercise its fail-loud arm here;
// the adapter's own mapping is what is under test.
type fakeEmbedder struct {
	resp    domain.EmbedFlowResponse
	err     error
	calls   int
	lastReq domain.EmbedFlowRequest
}

func (f *fakeEmbedder) Embed(ctx context.Context, req domain.EmbedFlowRequest) (domain.EmbedFlowResponse, error) {
	f.calls++
	f.lastReq = req
	if f.err != nil {
		return domain.EmbedFlowResponse{}, f.err
	}
	return f.resp, nil
}

// panicInvoker proves the Embed path never touches the Execute or GroundedSearch
// ports. NewServer requires all three, and a test that quietly shared one fake
// could not tell "Embed is wired" from "Embed fell through to something else".
type panicInvoker struct{}

func (panicInvoker) Execute(ctx context.Context, req domain.InvokeRequest) (domain.InvokeResponse, error) {
	panic("Embed must not reach the Execute port")
}

func (panicInvoker) GroundedSearch(ctx context.Context, req domain.GroundedSearchRequest) (domain.GroundedSearchResult, error) {
	panic("Embed must not reach the GroundedSearch port")
}

func newEmbedServer(t *testing.T, emb *fakeEmbedder) (mgv1.ModelGatewayServiceClient, func()) {
	t.Helper()
	srvAdapter, err := modelgatewaygrpc.NewServer(panicInvoker{}, panicInvoker{}, emb)
	require.NoError(t, err)

	lis := bufconn.Listen(1024 * 1024)
	gs := grpc.NewServer()
	mgv1.RegisterModelGatewayServiceServer(gs, srvAdapter)
	go func() { _ = gs.Serve(lis) }()

	conn, err := grpc.NewClient(
		"passthrough://bufnet",
		grpc.WithContextDialer(func(ctx context.Context, _ string) (net.Conn, error) { return lis.DialContext(ctx) }),
		grpc.WithTransportCredentials(insecure.NewCredentials()),
	)
	require.NoError(t, err)
	return mgv1.NewModelGatewayServiceClient(conn), func() { _ = conn.Close(); gs.Stop() }
}

func validEmbedRequest() *mgv1.EmbedRequest {
	return &mgv1.EmbedRequest{
		InvocationId:     "0198f000-0000-7000-8000-000000000001",
		TenantId:         "tenant-1",
		Gcid:             "gcid-1",
		AgentId:          "consumption_memory_embedder",
		CrewKind:         "familiar",
		LogicalModelId:   "text-embedding-004",
		Text:             "the learner practised bar charts",
		TaskType:         "RETRIEVAL_DOCUMENT",
		OutputDimensions: 768,
	}
}

func TestEmbed_MapsEveryFieldInBothDirections(t *testing.T) {
	completed := time.Date(2026, 8, 14, 10, 0, 0, 0, time.UTC)
	emb := &fakeEmbedder{resp: domain.EmbedFlowResponse{
		InvocationID:   "0198f000-0000-7000-8000-000000000001",
		Values:         []float32{0.5, -0.25, 0.125},
		Vendor:         "vertex_gemini",
		ModelVersion:   "text-embedding-004",
		Usage:          domain.TokenUsage{InputTokens: 7, CostMicros: 0},
		CompletedAt:    completed,
		GatewayVersion: "chora-model-gateway:test",
	}}
	client, cleanup := newEmbedServer(t, emb)
	defer cleanup()

	resp, err := client.Embed(context.Background(), validEmbedRequest())
	require.NoError(t, err)

	// Inbound: proto to domain.
	require.Equal(t, 1, emb.calls)
	got := emb.lastReq
	assert.Equal(t, "tenant-1", got.TenantID)
	assert.Equal(t, "gcid-1", got.GCID)
	assert.Equal(t, "consumption_memory_embedder", got.AgentID)
	assert.Equal(t, "familiar", got.CrewKind)
	assert.Equal(t, domain.LogicalModelID("text-embedding-004"), got.LogicalModelID)
	assert.Equal(t, "the learner practised bar charts", got.Text)
	assert.Equal(t, "RETRIEVAL_DOCUMENT", got.TaskType)
	assert.Equal(t, int32(768), got.OutputDimensions)

	// Outbound: domain to proto.
	assert.Equal(t, []float32{0.5, -0.25, 0.125}, resp.Values)
	assert.Equal(t, "vertex_gemini", resp.Vendor)
	assert.Equal(t, "text-embedding-004", resp.ModelVersion)
	require.NotNil(t, resp.Usage)
	assert.Equal(t, int64(7), resp.Usage.InputTokens)
	assert.Equal(t, int64(0), resp.Usage.CostMicros,
		"embeddings are unpriced by owner ruling; a non-zero cost here would be a billing defect")
	assert.Equal(t, completed, resp.CompletedAt.AsTime())
	assert.Equal(t, "chora-model-gateway:test", resp.GatewayVersion)
}

// Every required field is refused BEFORE the port is called, so a malformed
// caller never reaches the vendor and never bills.
func TestEmbed_RefusesMissingRequiredFieldsWithoutCallingThePort(t *testing.T) {
	cases := map[string]func(*mgv1.EmbedRequest){
		"tenant_id":       func(r *mgv1.EmbedRequest) { r.TenantId = "" },
		"gcid":            func(r *mgv1.EmbedRequest) { r.Gcid = "" },
		"agent_id":        func(r *mgv1.EmbedRequest) { r.AgentId = "" },
		"text":            func(r *mgv1.EmbedRequest) { r.Text = "" },
		"whitespace text": func(r *mgv1.EmbedRequest) { r.Text = "   \t\n  " },
	}
	for name, mutate := range cases {
		t.Run(name, func(t *testing.T) {
			emb := &fakeEmbedder{}
			client, cleanup := newEmbedServer(t, emb)
			defer cleanup()

			req := validEmbedRequest()
			mutate(req)
			_, err := client.Embed(context.Background(), req)

			require.Error(t, err)
			assert.Equal(t, codes.InvalidArgument, status.Code(err))
			assert.Zero(t, emb.calls, "a rejected request must never reach the embedding port")
		})
	}
}

// A nil message cannot arrive over a real connection (the codec always
// constructs one), so this arm is driven against the adapter directly. It exists
// because an in-process caller can pass nil, and the alternative is a panic on
// the chokepoint.
func TestEmbed_NilRequestIsInvalidArgumentNotAPanic(t *testing.T) {
	srv, err := modelgatewaygrpc.NewServer(panicInvoker{}, panicInvoker{}, &fakeEmbedder{})
	require.NoError(t, err)

	_, err = srv.Embed(context.Background(), nil)
	require.Error(t, err)
	assert.Equal(t, codes.InvalidArgument, status.Code(err))
}

// The two embedder callers propagate trace context in gRPC metadata. Without
// this fallback the ledger row is minted with a fresh root and the embedding
// stops correlating with the turn that caused it.
func TestEmbed_FallsBackToMetadataTraceContext(t *testing.T) {
	emb := &fakeEmbedder{}
	client, cleanup := newEmbedServer(t, emb)
	defer cleanup()

	const tp = "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01"
	ctx := metadata.NewOutgoingContext(context.Background(), metadata.Pairs(
		"traceparent", tp,
		"tracestate", "chora=1",
	))
	req := validEmbedRequest()
	req.Traceparent = "" // absent on the proto, present in metadata
	req.Tracestate = ""

	_, err := client.Embed(ctx, req)
	require.NoError(t, err)
	require.Equal(t, 1, emb.calls)
	assert.Equal(t, tp, emb.lastReq.Traceparent,
		"the metadata traceparent must reach the domain, or the ledger row loses its trace")
	assert.Equal(t, "chora=1", emb.lastReq.Tracestate)
}

// An explicit proto traceparent wins over metadata: the caller said which trace
// this embedding belongs to.
func TestEmbed_ProtoTraceparentWinsOverMetadata(t *testing.T) {
	emb := &fakeEmbedder{}
	client, cleanup := newEmbedServer(t, emb)
	defer cleanup()

	const onProto = "00-11111111111111111111111111111111-2222222222222222-01"
	ctx := metadata.NewOutgoingContext(context.Background(), metadata.Pairs(
		"traceparent", "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01",
	))
	req := validEmbedRequest()
	req.Traceparent = onProto

	_, err := client.Embed(ctx, req)
	require.NoError(t, err)
	assert.Equal(t, onProto, emb.lastReq.Traceparent)
}

// The error mapping decides whether a caller retries. Each arm is pinned so a
// reworded domain error fails a test rather than silently changing production
// retry behaviour.
func TestEmbed_MapsDomainErrorsOntoTheContractedCodes(t *testing.T) {
	cases := []struct {
		name string
		err  error
		want codes.Code
	}{
		{"a validation refusal is terminal", errors.New("model gateway: tenant_id required"), codes.InvalidArgument},
		{"a non-embedding model is terminal", errors.New("model gateway: gemini-2.5-pro is not an embedding model"), codes.InvalidArgument},
		{"a vendor dispatch failure is retryable", errors.New("model gateway: vendor dispatch failed: 503"), codes.Unavailable},
		{"anything else is internal", errors.New("model gateway: ledger write failed"), codes.Internal},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			client, cleanup := newEmbedServer(t, &fakeEmbedder{err: tc.err})
			defer cleanup()

			_, err := client.Embed(context.Background(), validEmbedRequest())
			require.Error(t, err)
			assert.Equal(t, tc.want, status.Code(err))
			assert.Contains(t, status.Convert(err).Message(), tc.err.Error(),
				"the caller needs the reason, not just the code")
		})
	}
}

// TestEmbed_UpstreamStatusRelay verifies the embed path relays a vendor
// failure carrying the provider's own HTTP status, mirroring the invoke path.
func TestEmbed_UpstreamStatusRelay(t *testing.T) {
	cases := []struct {
		name   string
		status int
		want   codes.Code
	}{
		{"401 → Unauthenticated", 401, codes.Unauthenticated},
		{"404 → NotFound", 404, codes.NotFound},
		{"429 → ResourceExhausted", 429, codes.ResourceExhausted},
		{"500 → Unavailable", 500, codes.Unavailable},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			client, cleanup := newEmbedServer(t, &fakeEmbedder{err: &upstreamError{status: tc.status, msg: "embed upstream blew up"}})
			defer cleanup()

			_, err := client.Embed(context.Background(), validEmbedRequest())
			require.Error(t, err)
			assert.Equal(t, tc.want, status.Code(err))
		})
	}
}

// TestEmbed_LedgerFailureIsInternal verifies the embed ledger failure is
// explicitly mapped: an un-ledgered embed must not succeed, and a store fault
// is an internal error the caller cannot fix (not a retryable vendor refusal,
// not a caller bug).
func TestEmbed_LedgerFailureIsInternal(t *testing.T) {
	client, cleanup := newEmbedServer(t, &fakeEmbedder{err: errors.New("embed: ledger enqueue failed (an un-ledgered embed must not succeed): connection refused")})
	defer cleanup()

	_, err := client.Embed(context.Background(), validEmbedRequest())
	require.Error(t, err)
	assert.Equal(t, codes.Internal, status.Code(err))
	// The message is sanitized: the raw upstream error body is NOT included.
	// The gateway-controlled message (ledger enqueue failed) IS included.
	assert.Contains(t, status.Convert(err).Message(), "ledger enqueue failed")
}

// NewServer must refuse a nil Embedder rather than accept it and fail per call:
// a gateway that boots without its embedding port is a silent stub.
func TestNewServer_RefusesANilEmbedder(t *testing.T) {
	_, err := modelgatewaygrpc.NewServer(panicInvoker{}, panicInvoker{}, nil)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "Embedder required")
}
