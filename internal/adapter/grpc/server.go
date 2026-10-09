// Package modelgatewaygrpc is the gRPC server-side adapter for the
// ModelGatewayService contract (chora-contracts/proto/services/
// model_gateway_service.proto).
//
// The adapter does ONLY proto-↔-ExecuteRequest/ExecuteResponse translation +
// gRPC status code mapping; all governance logic lives in the Executor
// (internal/executor), which composes the governance pipeline (suspension →
// mana → domain service). Both gRPC and HTTP adapters call the same
// Executor.Execute() path.
package modelgatewaygrpc

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"strings"

	"go.opentelemetry.io/otel/trace"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/metadata"
	"google.golang.org/grpc/status"
	"google.golang.org/protobuf/types/known/timestamppb"

	"github.com/apollo-chora/chora-common/tracing"
	mgv1 "github.com/apollo-chora/chora-contracts/gen/go/chora/services/model_gateway/v1"

	"github.com/apollo-chora/chora-model-gateway/internal/domain"
	"github.com/apollo-chora/chora-model-gateway/internal/executor"
)

// Invoker is the subset of the Executor the adapter calls. The Executor
// composes the governance pipeline (suspension → mana → domain service);
// *executor.Executor satisfies it.
type Invoker interface {
	Execute(ctx context.Context, req executor.ExecuteRequest) (executor.ExecuteResponse, error)
}

// GroundedSearcher is the subset of domain.Service the adapter calls for the
// ADR-231 grounded-search RPC. *domain.Service satisfies it DIRECTLY (NOT via
// the ManaMetering decorator — GroundedSearch meters its own high-price
// external_egress action in the domain chain, so it must not be double-metered).
type GroundedSearcher interface {
	GroundedSearch(ctx context.Context, req domain.GroundedSearchRequest) (domain.GroundedSearchResult, error)
}

// Embedder is the subset of domain.Service the adapter calls for the G1'-1
// embeddings RPC. *domain.Service satisfies it directly (no mana decorator:
// embeddings carry no price by owner ruling and self-ledger in the domain
// flow).
type Embedder interface {
	Embed(ctx context.Context, req domain.EmbedFlowRequest) (domain.EmbedFlowResponse, error)
}

// Server adapts mgv1.ModelGatewayServiceServer to an Invoker (the Executor)
// + GroundedSearcher + Embedder.
type Server struct {
	mgv1.UnimplementedModelGatewayServiceServer

	svc      Invoker
	grounded GroundedSearcher
	embedder Embedder
}

// NewServer constructs the gRPC adapter. All three MUST NOT be nil,
// fail-loud per feedback_no_stubs_real_wiring (no silent-stub fallback). In
// production wiring svc is the Executor (the governance composition root:
// CompanionSuspension → ManaMetering → domain.Service); grounded and
// embedder are the raw domain.Service (GroundedSearch self-meters; Embed
// self-ledgers, unpriced).
func NewServer(svc Invoker, grounded GroundedSearcher, embedder Embedder) (*Server, error) {
	if svc == nil {
		return nil, errors.New("modelgatewaygrpc: Invoker required")
	}
	if grounded == nil {
		return nil, errors.New("modelgatewaygrpc: GroundedSearcher required")
	}
	if embedder == nil {
		return nil, errors.New("modelgatewaygrpc: Embedder required")
	}
	return &Server{svc: svc, grounded: grounded, embedder: embedder}, nil
}

// Embed implements mgv1.ModelGatewayServiceServer.Embed (G1'-1).
func (s *Server) Embed(ctx context.Context, in *mgv1.EmbedRequest) (*mgv1.EmbedResponse, error) {
	if in == nil {
		return nil, status.Error(codes.InvalidArgument, "EmbedRequest required")
	}
	if in.TenantId == "" {
		return nil, status.Error(codes.InvalidArgument, "tenant_id required")
	}
	if in.Gcid == "" {
		return nil, status.Error(codes.InvalidArgument, "gcid required")
	}
	if in.AgentId == "" {
		return nil, status.Error(codes.InvalidArgument, "agent_id required")
	}
	if strings.TrimSpace(in.Text) == "" {
		return nil, status.Error(codes.InvalidArgument, "text required")
	}

	traceparent := in.Traceparent
	tracestate := in.Tracestate
	if traceparent == "" {
		if md, ok := metadata.FromIncomingContext(ctx); ok {
			if v := md.Get("traceparent"); len(v) > 0 {
				traceparent = v[0]
			}
			if v := md.Get("tracestate"); len(v) > 0 {
				tracestate = v[0]
			}
		}
	}
	traceparent = resolveTraceparent(ctx, traceparent)

	resp, err := s.embedder.Embed(ctx, domain.EmbedFlowRequest{
		InvocationID:     in.InvocationId,
		TenantID:         in.TenantId,
		GCID:             in.Gcid,
		AgentID:          in.AgentId,
		CrewKind:         in.CrewKind,
		LogicalModelID:   domain.LogicalModelID(in.LogicalModelId),
		Text:             in.Text,
		TaskType:         in.TaskType,
		OutputDimensions: in.OutputDimensions,
		Traceparent:      traceparent,
		Tracestate:       tracestate,
	})
	if err != nil {
		return nil, mapEmbedError(err)
	}

	return &mgv1.EmbedResponse{
		InvocationId: resp.InvocationID,
		Values:       resp.Values,
		Vendor:       resp.Vendor,
		ModelVersion: resp.ModelVersion,
		Usage: &mgv1.TokenUsage{
			InputTokens: resp.Usage.InputTokens,
			CostMicros:  resp.Usage.CostMicros,
		},
		CompletedAt:    timestamppb.New(resp.CompletedAt),
		GatewayVersion: resp.GatewayVersion,
	}, nil
}

// mapEmbedError translates a domain embed failure into the gRPC status code
// the proto contract documents. The mapping mirrors the Python gateway's
// embed arm: a vendor failure carrying the provider's own HTTP status is
// relayed; a validation refusal is terminal (InvalidArgument); a ledger
// failure is an internal store fault the caller cannot fix.
//
// ERROR SANITIZATION: the client-facing message NEVER includes the raw
// upstream error body. The upstream body may contain internal hostnames,
// stack traces, or other sensitive information. Gateway-controlled error
// messages (validation refusals, ledger failures) are safe to relay.
func mapEmbedError(err error) error {
	// Per-tenant concurrency ceiling: nothing was dispatched and nothing was
	// billed, so the refusal is RESOURCE_EXHAUSTED (retryable).
	var overloaded *domain.OverloadedError
	if errors.As(err, &overloaded) {
		return status.Errorf(codes.ResourceExhausted, "%s", overloaded.Error())
	}
	// Upstream HTTP status relay (mirrors the invoke path): a non-2xx
	// provider response is relayed rather than flattened. The message is
	// sanitized: the upstream error body is NOT included.
	if us := domain.UpstreamStatusOf(err); us != nil {
		return mapUpstreamStatus(*us, "")
	}
	// Registry / policy refusals discovered BEFORE any provider call: the
	// logical model id is unknown to the registry, the resolved entry does not
	// advertise the embeddings capability, its credential reference is empty,
	// or no embedding adapter is wired for the resolved provider. These are
	// configuration faults (the Invoke path maps the same set to
	// FailedPrecondition), not vendor failures — and nothing was billed.
	//
	// A dimension refusal belongs here too: the request asks for a vector
	// length the resolved model is known not to produce, or the route returned
	// a length that contradicts its own configuration. Either way the caller's
	// request cannot be served as configured, and the message says which.
	var cfgErr *domain.ConfigError
	var capErr *domain.CapabilityError
	var credErr *domain.CredentialError
	var noProvider *domain.NoProviderError
	var dimErr *domain.DimensionMismatchError
	if errors.As(err, &cfgErr) || errors.As(err, &capErr) ||
		errors.As(err, &credErr) || errors.As(err, &noProvider) ||
		errors.As(err, &dimErr) {
		return status.Error(codes.FailedPrecondition, err.Error())
	}
	// For non-upstream errors, the gateway-controlled message is safe to
	// relay. The full error (with the upstream body) is available in the
	// logs for internal debugging.
	msg := err.Error()
	switch {
	case strings.Contains(msg, "required") || strings.Contains(msg, "not an embedding model"):
		return status.Error(codes.InvalidArgument, msg)
	case strings.Contains(msg, "vendor dispatch"):
		return status.Error(codes.Unavailable, msg)
	case strings.Contains(msg, "ledger enqueue failed"):
		// An un-ledgered embed must not succeed; the ledger row is the
		// requirement this RPC exists to satisfy. A store fault is not a
		// caller bug, so it is Internal (not InvalidArgument) — the caller
		// cannot fix it, but it is not a retryable vendor refusal either.
		return status.Error(codes.Internal, msg)
	default:
		return status.Error(codes.Internal, msg)
	}
}

// Invoke — implements mgv1.ModelGatewayServiceServer.Invoke.
//
// Adapter responsibilities:
//  1. Validate required proto fields → INVALID_ARGUMENT.
//  2. Extract W3C trace context from gRPC metadata if absent on the proto.
//  3. Build executor.ExecuteRequest + call s.svc.Execute.
//  4. Map domain.InvokeError → gRPC status code per the Python gateway's
//     _map_error: relay a carried upstream HTTP status, else map by Kind
//     (config → FAILED_PRECONDITION, bare → INVALID_ARGUMENT, invoke →
//     UNAVAILABLE).
//  5. Map in-band executor.ExecuteResponse → proto response (BudgetBlock /
//     ArmorBlock paths flow as success-with-finish-reason).
func (s *Server) Invoke(ctx context.Context, in *mgv1.InvokeRequest) (*mgv1.InvokeResponse, error) {
	if in == nil {
		return nil, status.Error(codes.InvalidArgument, "InvokeRequest required")
	}
	if in.TenantId == "" {
		return nil, status.Error(codes.InvalidArgument, "tenant_id required")
	}
	if in.Gcid == "" {
		return nil, status.Error(codes.InvalidArgument, "gcid required")
	}
	if in.AgentId == "" {
		return nil, status.Error(codes.InvalidArgument, "agent_id required")
	}
	if in.LogicalModelId == "" {
		return nil, status.Error(codes.InvalidArgument, "logical_model_id required")
	}
	if in.Prompt == "" {
		return nil, status.Error(codes.InvalidArgument, "prompt required")
	}

	// Pull traceparent from proto first; fall back to gRPC metadata if
	// empty (some callers don't populate the proto field, only the
	// header).
	traceparent := in.Traceparent
	tracestate := in.Tracestate
	if traceparent == "" {
		if md, ok := metadata.FromIncomingContext(ctx); ok {
			if v := md.Get("traceparent"); len(v) > 0 {
				traceparent = v[0]
			}
			if v := md.Get("tracestate"); len(v) > 0 {
				tracestate = v[0]
			}
		}
	}
	// Guarantee a valid, caller-correlated W3C traceparent on the emitted
	// TokenUsageEvent. The shared outbox publisher REJECTS an empty traceparent
	// (mandatory envelope field per CLAUDE.md §6), which previously poisoned
	// every gateway-emitted token_usage row. resolveTraceparent prefers the
	// active OTel span (created by the otelgrpc server handler — a child of the
	// caller's trace when propagated) and falls back to minting/preserving the
	// inbound string. See HANDOFF_OBSERVABILITY_OUTBOX_JAM_2026-05-29 Fix 2.
	traceparent = resolveTraceparent(ctx, traceparent)

	// Convert GenerationConfig (proto Struct) to a generic map for the
	// vendor adapters' inspection.
	var genConfig map[string]any
	if in.GenerationConfig != nil {
		genConfig = in.GenerationConfig.AsMap()
	}

	// Agent-declared fallback chain (CR qgen 2026-06-01) — decode the
	// repeated string field into typed LogicalModelIDs for the policy loader.
	var fallbackModels []domain.LogicalModelID
	if len(in.FallbackLogicalModelIds) > 0 {
		fallbackModels = make([]domain.LogicalModelID, 0, len(in.FallbackLogicalModelIds))
		for _, m := range in.FallbackLogicalModelIds {
			fallbackModels = append(fallbackModels, domain.LogicalModelID(m))
		}
	}

	// Modality default: an absent response_modality means TEXT (mirrors the
	// Python adapter's `request.response_modality or "TEXT"`).
	modality := in.ResponseModality
	if modality == "" {
		modality = "TEXT"
	}

	// System-prompt prepending. The agent's instruction arrives in
	// `system_prompt`; when a structured conversation (contents_json) is
	// present, carry the instruction as the leading system message so the
	// model sees it. ONLY when there is a conversation to prepend to: with
	// contents_json empty the user turn is the flat prompt, and prepending a
	// system turn to an empty list would silently drop the user turn (the
	// vendor builders prefer a non-empty contents over the flat prompt).
	//
	// When the system message is prepended it rides in contents, so the
	// vendor-native system instruction is cleared to avoid stating the
	// instruction twice (the gemini vendor sets SystemInstruction from
	// SystemPrompt independently of contents).
	systemPrompt := in.SystemPrompt
	contentsJSON := in.ContentsJson
	if contentsJSON != "" {
		var contents []json.RawMessage
		if err := json.Unmarshal([]byte(contentsJSON), &contents); err != nil {
			// Graceful degradation: malformed contents_json decodes to empty,
			// so no system turn is prepended. The raw string is still passed
			// to the vendor, which re-parses it and falls back to the flat
			// prompt on the same malformed input.
			contents = nil
		}
		if systemPrompt != "" && len(contents) > 0 {
			systemMsg, err := json.Marshal(struct {
				Role    string `json:"role"`
				Content string `json:"content"`
			}{Role: "system", Content: systemPrompt})
			if err != nil {
				return nil, status.Errorf(codes.Internal, "marshal system message: %v", err)
			}
			prepended := make([]json.RawMessage, 0, len(contents)+1)
			prepended = append(prepended, systemMsg)
			prepended = append(prepended, contents...)
			encoded, err := json.Marshal(prepended)
			if err != nil {
				return nil, status.Errorf(codes.Internal, "marshal contents: %v", err)
			}
			contentsJSON = string(encoded)
			systemPrompt = ""
		}
	}

	// Tool declarations: validate tools_json, degrading to empty on malformed
	// input so a bad payload never reaches the vendor (mirrors the Python
	// adapter's _decode_json returning None on malformed input).
	toolsJSON := in.ToolsJson
	if toolsJSON != "" {
		var tools any
		if err := json.Unmarshal([]byte(toolsJSON), &tools); err != nil {
			toolsJSON = ""
		}
	}

	req := domain.InvokeRequest{
		InvocationID:     in.InvocationId,
		TenantID:         in.TenantId,
		GCID:             in.Gcid,
		AgentID:          in.AgentId,
		CrewKind:         in.CrewKind,
		LogicalModelID:   domain.LogicalModelID(in.LogicalModelId),
		FallbackModelIDs: fallbackModels,
		Prompt:           in.Prompt,
		ResponseModality: modality,
		SystemPrompt:     systemPrompt,
		ActionCode:       in.ActionCode,
		ContentsJSON:     contentsJSON,
		ToolsJSON:        toolsJSON,
		GenerationConfig: genConfig,
		Traceparent:      traceparent,
		Tracestate:       tracestate,
		// ADR-254 D7: the caller-stamped surface (ADR-252 Q1) and the
		// agent-forwarded dispatch key (R22) ride verbatim into the decorator
		// chain (CompanionSuspension reads Surface; ManaMetering claims on the
		// key). Validation of an ABSENT surface is the decorator's job, so the
		// refusal is one typed FAILED_PRECONDITION rather than an adapter-level
		// INVALID_ARGUMENT that would hide which gate refused.
		Surface:                in.Surface,
		DispatchIdempotencyKey: in.DispatchIdempotencyKey,
	}

	resp, err := s.svc.Execute(ctx, req)
	if err != nil {
		return nil, mapInvokeError(err)
	}
	return toProto(resp), nil
}

// mapInvokeError translates a domain.InvokeError into the gRPC status code
// the proto contract documents.
//
// The mapping mirrors the Python gateway's _map_error: a vendor failure
// carrying the provider's own HTTP status is RELAYED rather than flattened;
// otherwise the error's Kind decides — config (FailedPrecondition), bare
// (InvalidArgument), invoke (Unavailable).
//
// Budget block + Armor blocks flow in-band (the proto contract reserves
// PERMISSION_DENIED + RESOURCE_EXHAUSTED + FAILED_PRECONDITION for those,
// but the current service layer chose in-band to give callers the latency +
// ledger correlation). The adapter respects that choice.
func mapInvokeError(err error) error {
	// Per-tenant concurrency ceiling (CHORA_LLM_MAX_CONCURRENT_PER_TENANT): the
	// request never reached a provider and nothing was billed, so it is
	// RESOURCE_EXHAUSTED (retryable), not a vendor failure.
	var overloaded *domain.OverloadedError
	if errors.As(err, &overloaded) {
		return status.Errorf(codes.ResourceExhausted, "%s", overloaded.Error())
	}
	// ADR-254 D7 / ADR-252 D5: a typed precondition refusal. The message LEADS
	// with the machine token (PreconditionError.Reason), so callers prefix-
	// match companion_suspended / surface_unstamped off the status.
	var perr *domain.PreconditionError
	if errors.As(err, &perr) {
		switch perr.Reason {
		case domain.DenySuspensionUnreadable:
			// A store fault, not a verdict: retryable.
			return status.Errorf(codes.Unavailable, "%s: %s", perr.Reason, perr.Detail)
		default:
			return status.Errorf(codes.FailedPrecondition, "%s: %s", perr.Reason, perr.Detail)
		}
	}
	var ierr *domain.InvokeError
	if !errors.As(err, &ierr) {
		return status.Error(codes.Internal, "internal error")
	}
	// Upstream HTTP status relay: a non-2xx provider response is relayed
	// rather than flattened into one code, so a caller can distinguish an
	// auth rejection (retry won't help) from a rate limit (back off) from a
	// provider outage (retry later). The message is sanitized: the upstream
	// error body is NOT included.
	if ierr.UpstreamStatus != nil {
		return mapUpstreamStatus(*ierr.UpstreamStatus, domain.SanitizeUpstreamError(*ierr.UpstreamStatus, ""))
	}
	// Error taxonomy: config → FailedPrecondition, bare → InvalidArgument,
	// invoke (the zero value) → Unavailable. The Detail field is the
	// gateway-controlled message; the full error (with the upstream body)
	// is available in the logs for internal debugging.
	switch ierr.Kind {
	case domain.ErrorKindConfig:
		return status.Errorf(codes.FailedPrecondition, "%s", ierr.Detail)
	case domain.ErrorKindBare:
		return status.Errorf(codes.InvalidArgument, "%s", ierr.Detail)
	default:
		return status.Errorf(codes.Unavailable, "%s", ierr.Detail)
	}
}

// mapUpstreamStatus relays a provider's HTTP status to the closest gRPC
// status, mirroring the Python gateway's _map_error upstream branch.
// The message is sanitized: the upstream error body is NOT included.
func mapUpstreamStatus(code int, msg string) error {
	switch {
	case code == 401 || code == 403:
		return status.Errorf(codes.Unauthenticated, "upstream credential rejected")
	case code == 404:
		return status.Errorf(codes.NotFound, "upstream model or endpoint not found")
	case code == 429:
		return status.Errorf(codes.ResourceExhausted, "upstream rate limited")
	default:
		return status.Errorf(codes.Unavailable, "upstream provider error")
	}
}

// toProto maps a domain.InvokeResponse onto its proto counterpart.
func toProto(r domain.InvokeResponse) *mgv1.InvokeResponse {
	return &mgv1.InvokeResponse{
		InvocationId:          r.InvocationID,
		Completion:            r.Completion,
		ImageBytes:            r.ImageBytes,
		ImageMimeType:         r.ImageMIMEType,
		ToolCallsJson:         r.ToolCallsJSON,
		Usage:                 usageToProto(r.Usage),
		Vendor:                r.Vendor,
		ModelVersion:          r.ModelVersion,
		ModelArmorVerdictPre:  armorVerdictToProto(r.ArmorPre),
		ModelArmorVerdictPost: armorVerdictToProto(r.ArmorPost),
		FallbackChain:         r.FallbackChain,
		LatencyMs:             r.LatencyMs,
		FinishReason:          finishReasonToProto(r.FinishReason),
		FinishDetail:          r.FinishDetail,
		CompletedAt:           timestamppb.New(r.CompletedAt),
		GatewayVersion:        r.GatewayVersion,
	}
}

func usageToProto(u domain.TokenUsage) *mgv1.TokenUsage {
	return &mgv1.TokenUsage{
		InputTokens:  u.InputTokens,
		OutputTokens: u.OutputTokens,
		CachedTokens: u.CachedTokens,
		CostMicros:   u.CostMicros,
	}
}

func armorVerdictToProto(v domain.ArmorVerdict) mgv1.ModelArmorVerdict {
	// Domain values are numerically aligned with the proto enum, so the
	// cast is safe; gate via switch so an unrecognised future value
	// degrades to UNSPECIFIED rather than smuggling a bogus int32 onto
	// the wire.
	switch v {
	case domain.ArmorVerdictAllow:
		return mgv1.ModelArmorVerdict_MODEL_ARMOR_VERDICT_ALLOW
	case domain.ArmorVerdictBlock:
		return mgv1.ModelArmorVerdict_MODEL_ARMOR_VERDICT_BLOCK
	case domain.ArmorVerdictSanitise:
		return mgv1.ModelArmorVerdict_MODEL_ARMOR_VERDICT_SANITISE
	case domain.ArmorVerdictError:
		return mgv1.ModelArmorVerdict_MODEL_ARMOR_VERDICT_ERROR
	case domain.ArmorVerdictBypassed:
		return mgv1.ModelArmorVerdict_MODEL_ARMOR_VERDICT_BYPASSED
	default:
		return mgv1.ModelArmorVerdict_MODEL_ARMOR_VERDICT_UNSPECIFIED
	}
}

func finishReasonToProto(r domain.FinishReason) mgv1.FinishReason {
	switch r {
	case domain.FinishReasonComplete:
		return mgv1.FinishReason_FINISH_REASON_COMPLETE
	case domain.FinishReasonMaxTokens:
		return mgv1.FinishReason_FINISH_REASON_MAX_TOKENS
	case domain.FinishReasonModelArmorBlock:
		return mgv1.FinishReason_FINISH_REASON_MODEL_ARMOR_BLOCK
	case domain.FinishReasonBudgetBlock:
		return mgv1.FinishReason_FINISH_REASON_BUDGET_BLOCK
	case domain.FinishReasonVendorError:
		return mgv1.FinishReason_FINISH_REASON_VENDOR_ERROR
	case domain.FinishReasonManaBlock:
		return mgv1.FinishReason_FINISH_REASON_MANA_BLOCK
	default:
		return mgv1.FinishReason_FINISH_REASON_UNSPECIFIED
	}
}

// resolveTraceparent returns a guaranteed-valid W3C traceparent for the
// emitted TokenUsageEvent. Precedence:
//
//  1. The active OTel span context (set by the otelgrpc server handler in
//     cmd/server/main.go). When the caller propagated trace context this span
//     is a child of the caller's trace, so the cost event correlates
//     end-to-end AND points at the gateway's own Invoke span as parent.
//  2. tracing.EnsureTraceparent(inbound) — preserves a valid inbound string,
//     or mints a fresh valid root when there is neither a span nor a usable
//     inbound value.
//
// Either way the result is a structurally-valid traceparent, so the shared
// outbox publisher never rejects the row on the mandatory-traceparent rule.
func resolveTraceparent(ctx context.Context, inbound string) string {
	if sc := trace.SpanContextFromContext(ctx); sc.IsValid() {
		return fmt.Sprintf("00-%s-%s-%02x", sc.TraceID(), sc.SpanID(), byte(sc.TraceFlags()))
	}
	return tracing.EnsureTraceparent(inbound)
}

// PullTraceparent inspects a gRPC IncomingContext for a W3C traceparent
// header. Exported so the OTLP exporter wiring in cmd/server/main.go can
// reuse the same metadata-key convention.
func PullTraceparent(ctx context.Context) string {
	md, ok := metadata.FromIncomingContext(ctx)
	if !ok {
		return ""
	}
	if v := md.Get("traceparent"); len(v) > 0 {
		return strings.TrimSpace(v[0])
	}
	return ""
}

// GroundedSearch — implements mgv1.ModelGatewayServiceServer.GroundedSearch
// (ADR-231, the controlled web-egress surface).
//
// Adapter responsibilities mirror Invoke: validate required fields, resolve
// W3C trace context, build the domain request, call the (undecorated) grounded
// service, and map domain.GroundedSearchError → the gRPC status the proto
// contract documents. Mana exhaustion + zero citations are in-band (a success
// response with finish_reason / empty citations), NOT errors.
func (s *Server) GroundedSearch(ctx context.Context, in *mgv1.GroundedSearchRequest) (*mgv1.GroundedSearchResponse, error) {
	if in == nil {
		return nil, status.Error(codes.InvalidArgument, "GroundedSearchRequest required")
	}
	if in.TenantId == "" {
		return nil, status.Error(codes.InvalidArgument, "tenant_id required")
	}
	if in.Gcid == "" {
		return nil, status.Error(codes.InvalidArgument, "gcid required")
	}
	if in.AgentId == "" {
		return nil, status.Error(codes.InvalidArgument, "agent_id required")
	}
	if in.Directive == "" {
		return nil, status.Error(codes.InvalidArgument, "directive required")
	}
	if in.ActionCode == "" {
		return nil, status.Error(codes.InvalidArgument, "action_code required (un-priced egress not permitted)")
	}

	traceparent := in.Traceparent
	tracestate := in.Tracestate
	if traceparent == "" {
		if md, ok := metadata.FromIncomingContext(ctx); ok {
			if v := md.Get("traceparent"); len(v) > 0 {
				traceparent = v[0]
			}
			if v := md.Get("tracestate"); len(v) > 0 {
				tracestate = v[0]
			}
		}
	}
	traceparent = resolveTraceparent(ctx, traceparent)

	res, err := s.grounded.GroundedSearch(ctx, domain.GroundedSearchRequest{
		InvocationID:   in.InvocationId,
		TenantID:       in.TenantId,
		GCID:           in.Gcid,
		AgentID:        in.AgentId,
		CrewKind:       in.CrewKind,
		Directive:      in.Directive,
		MaxResults:     in.MaxResults,
		LogicalModelID: domain.LogicalModelID(in.LogicalModelId),
		ActionCode:     in.ActionCode,
		Traceparent:    traceparent,
		Tracestate:     tracestate,
	})
	if err != nil {
		return nil, mapGroundedError(err)
	}
	return groundedToProto(res), nil
}

// mapGroundedError translates a domain.GroundedSearchError into the gRPC status
// code the proto contract documents (ADR-231 / CHO-2113 acceptance criteria).
//
// ERROR SANITIZATION: the client-facing message NEVER includes the raw
// upstream error body. The gateway-controlled Detail field is used instead,
// led by the reason token so callers can prefix-match the denial reason.
func mapGroundedError(err error) error {
	// Per-tenant concurrency ceiling: no egress happened, so it is
	// RESOURCE_EXHAUSTED (retryable), not a governance denial.
	var overloaded *domain.OverloadedError
	if errors.As(err, &overloaded) {
		return status.Errorf(codes.ResourceExhausted, "%s", overloaded.Error())
	}
	var gerr *domain.GroundedSearchError
	if !errors.As(err, &gerr) {
		return status.Error(codes.Internal, "grounded search failed")
	}
	switch gerr.Reason {
	case "invalid_argument":
		return status.Errorf(codes.InvalidArgument, "%s: %s", gerr.Reason, gerr.Detail)
	case domain.EgressDenyDisabled, domain.EgressDenyKillSwitch, domain.EgressDenyBudget:
		// External egress not permitted / tenant budget exhausted — a
		// precondition the caller must resolve (enable entitlement / top up).
		return status.Errorf(codes.FailedPrecondition, "%s: %s", gerr.Reason, gerr.Detail)
	case domain.EgressDenyDailyCeil, domain.EgressDenyArmorPost:
		// Daily grounded-call ceiling hit, or Armor POST blocked the web answer.
		return status.Errorf(codes.ResourceExhausted, "%s: %s", gerr.Reason, gerr.Detail)
	case domain.EgressDenyArmorPre:
		return status.Errorf(codes.PermissionDenied, "%s: %s", gerr.Reason, gerr.Detail)
	case domain.EgressDenyVendorError:
		return status.Errorf(codes.Unavailable, "%s: %s", gerr.Reason, gerr.Detail)
	default:
		// "not_wired" + any unrecognised reason: a gateway misconfiguration.
		return status.Errorf(codes.Internal, "%s: %s", gerr.Reason, gerr.Detail)
	}
}

// groundedToProto maps a domain.GroundedSearchResult onto its proto counterpart.
func groundedToProto(r domain.GroundedSearchResult) *mgv1.GroundedSearchResponse {
	citations := make([]*mgv1.GroundedCitation, 0, len(r.Citations))
	for _, c := range r.Citations {
		citations = append(citations, &mgv1.GroundedCitation{
			Uri:        c.URI,
			Title:      c.Title,
			Domain:     c.Domain,
			Snippet:    c.Snippet,
			Confidence: c.Confidence,
		})
	}
	return &mgv1.GroundedSearchResponse{
		InvocationId:          r.InvocationID,
		Citations:             citations,
		GroundedAnswer:        r.GroundedAnswer,
		SearchEntryPointHtml:  r.SearchEntryPointHTML,
		WebSearchQueries:      r.WebSearchQueries,
		Usage:                 usageToProto(r.Usage),
		Vendor:                r.Vendor,
		ModelVersion:          r.ModelVersion,
		ModelArmorVerdictPre:  armorVerdictToProto(r.ArmorPre),
		ModelArmorVerdictPost: armorVerdictToProto(r.ArmorPost),
		FinishReason:          finishReasonToProto(r.FinishReason),
		FinishDetail:          r.FinishDetail,
		CompletedAt:           timestamppb.New(r.CompletedAt),
		GatewayVersion:        r.GatewayVersion,
	}
}
