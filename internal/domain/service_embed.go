package domain

import (
	"errors"
	"fmt"
	"strings"

	"context"
)

// Embed executes the chokepoint embedding flow (G1'-1, owner-ruled
// 2026-08-07). The per-tenant concurrency ceiling
// (CHORA_LLM_MAX_CONCURRENT_PER_TENANT) is checked before the vendor call and
// is never billed. Order of operations:
//
//  1. Validate required identity + text; refuse loud on any gap.
//  2. Resolve invocation_id (caller-supplied or generated).
//  3. Resolve model (default text-embedding-004) against the embedding
//     allowlist; a generation model here is refused, never re-routed.
//  4. Budget gate (fail-closed "budget required" mode only): with
//     CHORA_LLM_BUDGET_REQUIRED, a missing active budget window refuses the
//     embed before the vendor call. Default off — no budget read, historical
//     allow preserved.
//  5. Route resolution: the logical model id is resolved through the model
//     registry — the same authoritative mechanism the Invoke flow uses —
//     and the embed is dispatched through the embedding adapter registered
//     for the RESOLVED provider family, with the resolved upstream model and
//     base URL. A logical id is an alias, not a route: nothing about the
//     id's name shape says which provider serves it.
//  6. Ledger: enqueue the canonical token-usage event with the
//     vendor-reported input tokens, cost_micros 0 (no price attached by
//     ruling) and Bypassed Armor markers (permissive entry; no Armor leg
//     runs on a text-to-vector call). An un-ledgered embed FAILS: the
//     ledger row is the requirement this RPC exists to satisfy.
func (s *Service) Embed(ctx context.Context, req EmbedFlowRequest) (EmbedFlowResponse, error) {
	start := s.now()

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

	// Per-tenant concurrency ceiling (CHORA_LLM_MAX_CONCURRENT_PER_TENANT),
	// taken before the vendor dispatch so a refused embed is never billed.
	// nil limiter = unlimited/off.
	release, ok := s.admission.Acquire(req.TenantID)
	if !ok {
		return EmbedFlowResponse{}, &OverloadedError{TenantID: req.TenantID, Limit: s.admission.Limit()}
	}
	defer release()

	// Fail-closed "budget required" mode: a tenant with NO active budget
	// window is refused BEFORE the vendor call — the absence of a policy must
	// not restore unlimited provider spending. The mode is global
	// (CHORA_LLM_BUDGET_REQUIRED) or per-tenant
	// (CHORA_LLM_BUDGET_REQUIRED_TENANTS); the decision is made per request
	// from req.TenantID. Default (both off) preserves the historical allow
	// and skips the budget read entirely.
	if budgetRequiredFor(s.budgetRequired, s.budgetRequiredTenants, req.TenantID) {
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
	// Demo-deployment route pin (CHORA_EMBEDDING_PIN_*): when wired, the ONLY
	// accepted model is the pinned logical id. A runtime override that would
	// redirect the demo's embedding route (including to another allowlisted
	// embedding model) is refused BEFORE dispatch, never silently re-routed.
	if s.embeddingPin != nil && model != s.embeddingPin.LogicalID {
		return EmbedFlowResponse{}, fmt.Errorf("embed: model %q is refused; the embedding route is pinned to %q (CHORA_EMBEDDING_PIN_LOGICAL_ID)", model, s.embeddingPin.LogicalID)
	}
	if !embeddingModelAllowlist[model] {
		return EmbedFlowResponse{}, fmt.Errorf("embed: model %q is not an embedding model; refusing (never silently re-routed)", model)
	}

	dims := req.OutputDimensions
	if dims <= 0 {
		dims = defaultEmbeddingDimensions
	}

	vendorResp, family, err := s.embedThroughRegistry(ctx, model, req, dims)
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
		Vendor:         string(family),
		FallbackChain:  []string{fmt.Sprintf("%s:%s", family, model)},
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
		Vendor:         string(family),
		ModelVersion:   vendorResp.ModelVersion,
		Usage:          usage,
		LatencyMs:      int32(s.now().Sub(start).Milliseconds()), // #nosec G115 -- elapsed ms, bounded far below int32
		CompletedAt:    s.now(),
		GatewayVersion: s.gatewayVersion,
	}, nil
}

// ----------------------------------------------------------------------------
// Registry-directed dispatch (Release B)
// ----------------------------------------------------------------------------

// embedThroughRegistry resolves the logical model through the model registry
// — the same authoritative mechanism the Invoke flow uses — and dispatches
// through the embedding adapter registered for the RESOLVED provider family.
//
// A logical id is an alias, not a route. Nothing about the id's name shape
// says which provider serves it: `text-embedding-004` is the deployment's
// embedding ROUTE, and the registry entry behind it names an OpenRouter
// upstream. The resolved configuration therefore decides provider, upstream
// model, base URL and credential — and an entry whose provider has no wired
// embedding adapter is refused rather than re-routed to whatever adapter
// happens to be constructed.
//
// Returns the vendor response plus the family of the adapter that actually
// served it, so the ledger attributes the route that ran.
func (s *Service) embedThroughRegistry(ctx context.Context, model LogicalModelID, req EmbedFlowRequest, dims int32) (EmbedVendorResponse, VendorFamily, error) {
	if s.models == nil {
		return EmbedVendorResponse{}, "", errors.New("embed: no model registry resolver is wired; refusing to dispatch through an unverified adapter")
	}
	primary, err := s.models.Resolve(ctx, model)
	if err != nil {
		return EmbedVendorResponse{}, "", &ConfigError{
			Detail: fmt.Sprintf("embed: model %q is not in the registry", model),
			Inner:  err,
		}
	}

	// The primary first, then the registry-declared fallbacks. Every target
	// goes through the SAME resolution + policy checks, so a fallback cannot
	// land on a provider or model the primary was refused for.
	ids := make([]LogicalModelID, 0, 1+len(primary.FallbackIDs))
	ids = append(ids, model)
	ids = append(ids, primary.FallbackIDs...)

	var lastErr error
	for i, id := range ids {
		info := primary
		if i > 0 {
			resolved, err := s.models.Resolve(ctx, id)
			if err != nil {
				lastErr = &ConfigError{
					Detail: fmt.Sprintf("embed: fallback %q is not in the registry", id),
					Inner:  err,
				}
				continue
			}
			info = resolved
		}
		if err := checkEmbedTarget(info, id); err != nil {
			lastErr = err
			continue
		}
		client, ok := s.embedders[info.Vendor]
		if !ok {
			lastErr = &NoProviderError{Family: info.Vendor}
			continue
		}
		resp, err := client.EmbedText(ctx, EmbedVendorRequest{
			LogicalModelID:   model,
			UpstreamModel:    info.UpstreamModel,
			BaseURL:          info.BaseURL,
			Text:             req.Text,
			TaskType:         req.TaskType,
			OutputDimensions: dims,
			TenantID:         req.TenantID,
			Traceparent:      req.Traceparent,
		})
		if err == nil {
			return resp, client.Family(), nil
		}
		lastErr = err
	}
	if lastErr == nil {
		lastErr = errors.New("embed: no embedding target could be resolved")
	}
	return EmbedVendorResponse{}, "", lastErr
}

// checkEmbedTarget applies the embed policy checks to one resolved registry
// entry: it must advertise the embeddings capability, carry a usable
// credential reference, and name a dispatchable endpoint. The base-URL check
// is the one that closes the "silent paid route" hole: an entry with no
// resolvable endpoint is refused rather than dispatched to whatever the
// adapter happens to be constructed with.
func checkEmbedTarget(info ModelInfo, model LogicalModelID) error {
	if !info.Supports("embeddings") {
		return &CapabilityError{Model: model, Capability: "embeddings"}
	}
	if info.APIKeyEnv != "" && info.APIKey == "" {
		return &CredentialError{Model: model, APIKeyEnv: info.APIKeyEnv}
	}
	if info.BaseURL == "" {
		return fmt.Errorf("embed: registry entry %q declares no base_url and no role base URL is configured; refusing rather than dispatching to an unverified endpoint", model)
	}
	return nil
}
