package domain

import (
	"errors"
	"fmt"
	"strings"

	"context"
)

// Embed executes the embedding flow. Order of operations:
//
//  1. Validate required identity + text; refuse loud on any gap.
//  2. Resolve invocation_id (caller-supplied or generated).
//  3. Resolve the model through the registry; the target must advertise the
//     "embeddings" capability.
//  4. Vendor dispatch through the EmbeddingClient port.
//  5. Ledger: enqueue the token-usage event with the vendor-reported input
//     tokens and cost_micros 0. An un-ledgered embed FAILS — the ledger row
//     is the reason this RPC exists.
func (s *Service) Embed(ctx context.Context, req EmbedRequest) (EmbedResponse, error) {
	start := s.now()

	if s.embedder == nil {
		return EmbedResponse{}, errors.New("model gateway: Embed called but no EmbeddingClient is wired")
	}
	if strings.TrimSpace(req.TenantID) == "" {
		return EmbedResponse{}, errors.New("embed: tenant_id required")
	}
	if strings.TrimSpace(req.GCID) == "" {
		return EmbedResponse{}, errors.New("embed: gcid required")
	}
	if strings.TrimSpace(req.AgentID) == "" {
		return EmbedResponse{}, errors.New("embed: agent_id required")
	}
	if strings.TrimSpace(req.Text) == "" {
		return EmbedResponse{}, errors.New("embed: text required")
	}

	invocationID := req.InvocationID
	if invocationID == "" {
		invocationID = s.newID()
	}

	model := req.LogicalModelID
	if model == "" {
		model = DefaultEmbeddingModelID
	}

	policy, err := s.policies.ResolveAgentPolicy(ctx, req.AgentID, req.CrewKind, model, nil)
	if err != nil {
		return EmbedResponse{}, fmt.Errorf("embed: resolve model %q: %w", model, err)
	}
	target := policy.Target
	if !target.Supports(CapabilityEmbeddings) {
		return EmbedResponse{}, &ConfigError{
			Detail: fmt.Sprintf("embed: model %q does not advertise the %q capability", model, CapabilityEmbeddings),
		}
	}

	dims := req.OutputDimensions
	if dims <= 0 {
		dims = defaultEmbeddingDimensions
	}

	credential := ""
	if target.APIKeyRef != "" {
		credential, err = s.secrets.ResolveCredential(ctx, target.APIKeyRef)
		if err != nil {
			return EmbedResponse{}, fmt.Errorf("embed: resolve credential %q: %w", target.APIKeyRef, err)
		}
		if credential == "" {
			return EmbedResponse{}, &ConfigError{
				Detail: fmt.Sprintf("embed: credential reference %q is set but resolves to an empty value", target.APIKeyRef),
			}
		}
	}

	vendorResp, err := s.embedder.EmbedText(ctx, EmbedVendorRequest{
		Target:           target,
		Credential:       credential,
		LogicalModelID:   model,
		Text:             req.Text,
		TaskType:         req.TaskType,
		OutputDimensions: dims,
		TenantID:         req.TenantID,
		Traceparent:      req.Traceparent,
	})
	if err != nil {
		return EmbedResponse{}, fmt.Errorf("embed: vendor dispatch: %w", err)
	}

	usage := TokenUsage{
		InputTokens: vendorResp.InputTokens,
		CostMicros:  0, // no price attached: a priced embedding is user-visible
	}
	evt := TokenUsageEvent{
		UsageID:        invocationID,
		TenantID:       req.TenantID,
		GCID:           req.GCID,
		ModelID:        vendorResp.ModelVersion,
		InputTokens:    usage.InputTokens,
		OutputTokens:   0,
		CachedTokens:   0,
		CostMicros:     0,
		InvocationID:   invocationID,
		AgentRole:      req.AgentID,
		RecordedAt:     s.now(),
		Vendor:         string(target.Vendor),
		FallbackChain:  []string{fmt.Sprintf("%s:%s", target.Vendor, target.UpstreamModel)},
		GatewayVersion: s.gatewayVersion,
		Traceparent:    req.Traceparent,
		Tracestate:     req.Tracestate,
	}
	if err := s.outbox.EnqueueTokenUsageRecorded(ctx, evt); err != nil {
		return EmbedResponse{}, fmt.Errorf("embed: ledger enqueue failed (an un-ledgered embed must not succeed): %w", err)
	}

	return EmbedResponse{
		InvocationID:   invocationID,
		Values:         vendorResp.Values,
		Vendor:         string(target.Vendor),
		ModelVersion:   vendorResp.ModelVersion,
		Usage:          usage,
		LatencyMs:      elapsedMs(s.now().Sub(start)),
		CompletedAt:    s.now(),
		GatewayVersion: s.gatewayVersion,
	}, nil
}
