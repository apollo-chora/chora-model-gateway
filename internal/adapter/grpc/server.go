// Package modelgatewaygrpc is the gRPC server-side adapter for the
// ModelGatewayService contract (chora-contracts/proto/services/
// model_gateway_service.proto).
//
// The adapter does ONLY proto↔-domain translation and gRPC status mapping; all
// business logic lives in domain.Service.
//
// Two of the proto's RPCs are NOT implemented: GroundedSearch, which was a
// Google-Search-grounding surface on a now-removed provider, and nothing
// else. Because the struct embeds mgv1.UnimplementedModelGatewayServiceServer,
// GroundedSearch returns the standard gRPC Unimplemented code — a caller gets
// an honest "this gateway does not do that" rather than a panic or a hang.
package modelgatewaygrpc

import (
	"context"
	"errors"
	"fmt"

	"go.opentelemetry.io/otel/trace"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/metadata"
	"google.golang.org/grpc/status"
	"google.golang.org/protobuf/types/known/timestamppb"

	mgv1 "github.com/locoroco-git/Chora-LMS/chora-contracts/gen/go/chora/services/model_gateway/v1"

	"github.com/apollo-chora/chora-model-gateway/internal/domain"
)

// Invoker is the subset of domain.Service the adapter calls for Invoke.
type Invoker interface {
	Invoke(ctx context.Context, req domain.InvokeRequest) (domain.InvokeResponse, error)
}

// Embedder is the subset of domain.Service the adapter calls for Embed.
type Embedder interface {
	Embed(ctx context.Context, req domain.EmbedRequest) (domain.EmbedResponse, error)
}

// Server adapts mgv1.ModelGatewayServiceServer to an Invoker + Embedder.
type Server struct {
	mgv1.UnimplementedModelGatewayServiceServer

	svc      Invoker
	embedder Embedder
}

// NewServer constructs the gRPC adapter. Both collaborators MUST be non-nil;
// an adapter that silently served empty responses would look like a working
// gateway that had forgotten how to route.
func NewServer(svc Invoker, embedder Embedder) (*Server, error) {
	if svc == nil {
		return nil, errors.New("modelgatewaygrpc: Invoker required")
	}
	if embedder == nil {
		return nil, errors.New("modelgatewaygrpc: Embedder required")
	}
	return &Server{svc: svc, embedder: embedder}, nil
}

// Embed implements mgv1.ModelGatewayServiceServer.Embed.
func (s *Server) Embed(ctx context.Context, in *mgv1.EmbedRequest) (*mgv1.EmbedResponse, error) {
	if in == nil {
		return nil, status.Error(codes.InvalidArgument, "embed: request required")
	}
	req := domain.EmbedRequest{
		InvocationID:     in.GetInvocationId(),
		TenantID:         in.GetTenantId(),
		GCID:             in.GetGcid(),
		AgentID:          in.GetAgentId(),
		CrewKind:         in.GetCrewKind(),
		LogicalModelID:   domain.LogicalModelID(in.GetLogicalModelId()),
		Text:             in.GetText(),
		TaskType:         in.GetTaskType(),
		OutputDimensions: in.GetOutputDimensions(),
		Traceparent:      resolveTraceparent(ctx, in.GetTraceparent()),
		Tracestate:       in.GetTracestate(),
	}

	resp, err := s.embedder.Embed(ctx, req)
	if err != nil {
		return nil, mapError(err)
	}

	out := &mgv1.EmbedResponse{
		InvocationId:   resp.InvocationID,
		Vendor:         resp.Vendor,
		ModelVersion:   resp.ModelVersion,
		CompletedAt:    timestamppb.New(resp.CompletedAt),
		GatewayVersion: resp.GatewayVersion,
		Usage:          usageToProto(resp.Usage),
	}
	// The proto declares `repeated float values`; a 768-dimension embedding is
	// 768 elements, which is well inside the message limit.
	out.Values = make([]float32, 0, len(resp.Values))
	for _, v := range resp.Values {
		out.Values = append(out.Values, float32(v))
	}
	return out, nil
}

// Invoke implements mgv1.ModelGatewayServiceServer.Invoke.
func (s *Server) Invoke(ctx context.Context, in *mgv1.InvokeRequest) (*mgv1.InvokeResponse, error) {
	if in == nil {
		return nil, status.Error(codes.InvalidArgument, "invoke: request required")
	}

	req := domain.InvokeRequest{
		InvocationID:           in.GetInvocationId(),
		TenantID:               in.GetTenantId(),
		GCID:                   in.GetGcid(),
		AgentID:                in.GetAgentId(),
		CrewKind:               in.GetCrewKind(),
		LogicalModelID:         domain.LogicalModelID(in.GetLogicalModelId()),
		Prompt:                 in.GetPrompt(),
		SystemPrompt:           in.GetSystemPrompt(),
		ResponseModality:       in.GetResponseModality(),
		ActionCode:             in.GetActionCode(),
		ContentsJSON:           in.GetContentsJson(),
		ToolsJSON:              in.GetToolsJson(),
		Surface:                in.GetSurface(),
		DispatchIdempotencyKey: in.GetDispatchIdempotencyKey(),
		Traceparent:            resolveTraceparent(ctx, in.GetTraceparent()),
		Tracestate:             in.GetTracestate(),
	}
	for _, fb := range in.GetFallbackLogicalModelIds() {
		req.FallbackModelIDs = append(req.FallbackModelIDs, domain.LogicalModelID(fb))
	}

	// The proto's generation_config is a Struct; the domain carries a plain
	// map. A malformed Struct is a caller bug and is reported rather than
	// silently dropped — a dropped temperature changes the answer.
	if gc := in.GetGenerationConfig(); gc != nil {
		req.GenerationConfig = gc.AsMap()
	}

	resp, err := s.svc.Invoke(ctx, req)
	if err != nil {
		return nil, mapError(err)
	}
	return toProto(resp), nil
}

// ---------------------------------------------------------------------------
// Status mapping
// ---------------------------------------------------------------------------

// mapError translates a domain error to a gRPC status. A vendor failure that
// already carries an upstream HTTP status is relayed rather than flattened,
// so a caller can tell "your credential is wrong" from "the provider is down".
func mapError(err error) error {
	if err == nil {
		return nil
	}
	if _, ok := status.FromError(err); ok && status.Code(err) != codes.Unknown {
		return err
	}

	var upstream interface{ UpstreamStatusCode() int }
	if errors.As(err, &upstream) {
		switch upstream.UpstreamStatusCode() {
		case 401, 403:
			return status.Errorf(codes.Unauthenticated, "upstream credential rejected: %v", err)
		case 404:
			return status.Errorf(codes.NotFound, "upstream model or endpoint not found: %v", err)
		case 429:
			return status.Errorf(codes.ResourceExhausted, "upstream rate limited: %v", err)
		}
		return status.Errorf(codes.Unavailable, "upstream provider error: %v", err)
	}

	// A gateway-side misconfiguration is FailedPrecondition, not
	// Unavailable: it will not fix itself on a retry, and Unavailable would
	// invite the caller into a retry loop against a working provider.
	var configErr *domain.ConfigError
	if errors.As(err, &configErr) {
		return status.Error(codes.FailedPrecondition, configErr.Detail)
	}

	var invokeErr *domain.InvokeError
	if errors.As(err, &invokeErr) {
		detail := invokeErr.Detail
		if invokeErr.Inner != nil {
			detail = invokeErr.Inner.Error()
		}
		switch invokeErr.Reason {
		case domain.FinishReasonBudgetBlock:
			return status.Error(codes.ResourceExhausted, invokeErr.Detail)
		case domain.FinishReasonVendorError:
			return status.Error(codes.Unavailable, detail)
		default:
			return status.Error(codes.Internal, detail)
		}
	}

	// Bare validation errors from the domain (Embed's own guards).
	return status.Error(codes.InvalidArgument, err.Error())
}

// ---------------------------------------------------------------------------
// Proto conversion
// ---------------------------------------------------------------------------

func toProto(r domain.InvokeResponse) *mgv1.InvokeResponse {
	out := &mgv1.InvokeResponse{
		InvocationId:   r.InvocationID,
		Completion:     r.Completion,
		Usage:          usageToProto(r.Usage),
		Vendor:         r.Vendor,
		ModelVersion:   r.ModelVersion,
		FallbackChain:  r.FallbackChain,
		LatencyMs:      r.LatencyMs,
		FinishReason:   finishReasonToProto(r.FinishReason),
		FinishDetail:   r.FinishDetail,
		CompletedAt:    timestamppb.New(r.CompletedAt),
		GatewayVersion: r.GatewayVersion,
		ToolCallsJson:  r.ToolCallsJSON,
	}
	if len(r.ImageBytes) > 0 {
		out.ImageBytes = r.ImageBytes
		out.ImageMimeType = r.ImageMIMEType
	}
	return out
}

func usageToProto(u domain.TokenUsage) *mgv1.TokenUsage {
	return &mgv1.TokenUsage{
		InputTokens:  u.InputTokens,
		OutputTokens: u.OutputTokens,
		CachedTokens: u.CachedTokens,
		CostMicros:   u.CostMicros,
	}
}

func finishReasonToProto(r domain.FinishReason) mgv1.FinishReason {
	// The domain constants are pinned to the shared proto enum's numeric
	// values, so this is a checked conversion rather than a mapping.
	if int(r) < int(mgv1.FinishReason_FINISH_REASON_UNSPECIFIED) || int(r) > int(mgv1.FinishReason_FINISH_REASON_MANA_BLOCK) {
		return mgv1.FinishReason_FINISH_REASON_UNSPECIFIED
	}
	return mgv1.FinishReason(r)
}

// ---------------------------------------------------------------------------
// Trace propagation
// ---------------------------------------------------------------------------

// resolveTraceparent prefers the request's explicit traceparent field and
// falls back to the inbound gRPC metadata, so a caller that propagates W3C
// context over the wire gets a ledger row correlated with its own trace.
func resolveTraceparent(ctx context.Context, inbound string) string {
	if inbound != "" {
		return inbound
	}
	return PullTraceparent(ctx)
}

// PullTraceparent extracts the traceparent from inbound gRPC metadata,
// falling back to the active span on the context. Exported because the HTTP
// facade's equivalents live in a different package.
func PullTraceparent(ctx context.Context) string {
	if md, ok := metadata.FromIncomingContext(ctx); ok {
		if vals := md.Get("traceparent"); len(vals) > 0 && vals[0] != "" {
			return vals[0]
		}
	}
	spanCtx := trace.SpanContextFromContext(ctx)
	if !spanCtx.IsValid() {
		return ""
	}
	return fmt.Sprintf("00-%s-%s-%s", spanCtx.TraceID(), spanCtx.SpanID(), spanCtx.TraceFlags())
}

var _ mgv1.ModelGatewayServiceServer = (*Server)(nil)
