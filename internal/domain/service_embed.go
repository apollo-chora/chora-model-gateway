package domain

import (
	"errors"
	"fmt"
	"strings"

	"context"
)

// Embed executes the chokepoint embedding flow (G1'-1, owner-ruled
// 2026-08-07). Order of operations:
//
//  1. Validate required identity + text; refuse loud on any gap.
//  2. Resolve invocation_id (caller-supplied or generated).
//  3. Resolve model (default text-embedding-004) against the embedding
//     allowlist; a generation model here is refused, never re-routed.
//  4. Budget gate (fail-closed "budget required" mode only): with
//     CHORA_LLM_BUDGET_REQUIRED, a missing active budget window refuses the
//     embed before the vendor call. Default off — no budget read, historical
//     allow preserved.
//  5. Vendor dispatch through the EmbeddingClient port.
//  6. Ledger: enqueue the canonical token-usage event with the
//     vendor-reported input tokens, cost_micros 0 (no price attached by
//     ruling) and Bypassed Armor markers (permissive entry; no Armor leg
//     runs on a text-to-vector call). An un-ledgered embed FAILS: the
//     ledger row is the requirement this RPC exists to satisfy.
func (s *Service) Embed(ctx context.Context, req EmbedFlowRequest) (EmbedFlowResponse, error) {
	start := s.now()

	if s.embedder == nil {
		return EmbedFlowResponse{}, errors.New("model gateway: Embed called but no EmbeddingClient is wired (fail loud; the direct-Vertex fallback is closed)")
	}
	if strings.TrimSpace(req.TenantID) == "" {
		return EmbedFlowResponse{}, errors.New("embed: tenant_id required")
	}
	if strings.TrimSpace(req.GCID) == "" {
		return EmbedFlowResponse{}, errors.New("embed: gcid required")
	}
	if strings.TrimSpace(req.AgentID) == "" {
		return EmbedFlowResponse{}, errors.New("embed: agent_id required")
	}
	if strings.TrimSpace(req.Text) == "" {
		return EmbedFlowResponse{}, errors.New("embed: text required")
	}

	// Fail-closed "budget required" mode (CHORA_LLM_BUDGET_REQUIRED): with
	// the flag on, a tenant with NO active budget window is refused BEFORE
	// the vendor call — the absence of a policy must not restore unlimited
	// provider spending. Default (flag off) preserves the historical allow
	// and skips the budget read entirely.
	if s.budgetRequired {
		budget, err := s.budget.GetTenantBudget(ctx, req.TenantID)
		if err != nil {
			return EmbedFlowResponse{}, fmt.Errorf("embed: budget repo unavailable: %w", err)
		}
		if budget == nil {
			return EmbedFlowResponse{}, errors.New("embed: no active LLM budget window for tenant; budget required (CHORA_LLM_BUDGET_REQUIRED)")
		}
	}

	invocationID := req.InvocationID
	if invocationID == "" {
		invocationID = s.newID()
	}

	model := req.LogicalModelID
	if model == "" {
		model = DefaultEmbeddingModelID
	}
	if !embeddingModelAllowlist[model] {
		return EmbedFlowResponse{}, fmt.Errorf("embed: model %q is not an embedding model; refusing (never silently re-routed)", model)
	}

	dims := req.OutputDimensions
	if dims <= 0 {
		dims = defaultEmbeddingDimensions
	}

	vendorResp, err := s.embedder.EmbedText(ctx, EmbedVendorRequest{
		LogicalModelID:   model,
		Text:             req.Text,
		TaskType:         req.TaskType,
		OutputDimensions: dims,
		TenantID:         req.TenantID,
		Traceparent:      req.Traceparent,
	})
	if err != nil {
		return EmbedFlowResponse{}, fmt.Errorf("embed: vendor dispatch: %w", err)
	}

	usage := TokenUsage{
		InputTokens: vendorResp.InputTokens,
		CostMicros:  0, // no price attached (owner ruling; user-visible otherwise)
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
		AgentID:        req.AgentID,
		ActionCode:     req.AgentID,
		RecordedAt:     s.now(),
		Vendor:         string(s.embedder.Family()),
		FallbackChain:  []string{fmt.Sprintf("%s:%s", s.embedder.Family(), model)},
		ArmorPre:       ArmorVerdictBypassed,
		ArmorPost:      ArmorVerdictBypassed,
		GatewayVersion: s.gatewayVersion,
		Traceparent:    req.Traceparent,
		Tracestate:     req.Tracestate,
	}
	if err := s.outbox.EnqueueTokenUsageRecorded(ctx, evt); err != nil {
		return EmbedFlowResponse{}, fmt.Errorf("embed: ledger enqueue failed (an un-ledgered embed must not succeed): %w", err)
	}

	return EmbedFlowResponse{
		InvocationID:   invocationID,
		Values:         vendorResp.Values,
		Vendor:         string(s.embedder.Family()),
		ModelVersion:   vendorResp.ModelVersion,
		Usage:          usage,
		LatencyMs:      int32(s.now().Sub(start).Milliseconds()), // #nosec G115 -- elapsed ms, bounded far below int32
		CompletedAt:    s.now(),
		GatewayVersion: s.gatewayVersion,
	}, nil
}
