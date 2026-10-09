package domain

import (
	"context"
	"errors"
	"fmt"
	"net/url"
	"strings"
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
//     for the RESOLVED provider family, with the resolved upstream model,
//     base URL and credential. A logical id is an alias, not a route: nothing
//     about the id's name shape says which provider serves it.
//  6. Dimension policy (model-aware, ChatGPT round 12): the resolved model's
//     declared (or pinned) vector length decides what the response must
//     contain, and a `dimensions` parameter goes out only where the entry
//     declares the upstream accepts one. A wrong-length vector is refused,
//     never truncated or padded.
//  7. Ledger: enqueue the canonical token-usage event with the
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

	// The caller's requested dimension is passed through as-is: 0 means
	// "omitted". An omitted dimension must NOT be replaced by a gateway-side
	// default — a hard-coded 768 asked EVERY model for 768 dimensions, and the
	// pinned production route (a 1024-dimension model that rejects the
	// parameter outright) answered with a 400. What the response must contain
	// is the resolved model's own declared/pinned length; whether a parameter
	// goes out at all is decided per target in embedThroughRegistry.
	vendorResp, family, err := s.embedThroughRegistry(ctx, model, req, req.OutputDimensions)
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
func (s *Service) embedThroughRegistry(ctx context.Context, model LogicalModelID, req EmbedFlowRequest, requestedDims int32) (EmbedVendorResponse, VendorFamily, error) {
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
	//
	// The embedding-route pin DISABLES the fallback walk outright (ChatGPT
	// round 12, release blocker). A pinned request is a billing assertion —
	// "the embedding spend goes to the one approved free upstream" — and a
	// fallback is a second route the pin never approved. Re-applying the pin to
	// each fallback would still leave the walk to a target the operator never
	// named: the boot check refuses a pinned entry that DECLARES fallbacks, but
	// a registry swap after boot is not covered by a per-target assertion whose
	// allow-list is the primary's own fallback list. No fallback is walked
	// while the pin is wired.
	ids := make([]LogicalModelID, 0, 1+len(primary.FallbackIDs))
	ids = append(ids, model)
	if s.embeddingPin == nil {
		ids = append(ids, primary.FallbackIDs...)
	}

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
		if err := checkEmbedTarget(info, id, s.embeddingPin != nil); err != nil {
			lastErr = err
			continue
		}
		// Model-aware dimension policy: what the response MUST contain, and
		// whether a `dimensions` parameter goes out at all.
		expectedDims, paramDims, err := resolveEmbeddingDimensions(id, info, s.embeddingPin, requestedDims)
		if err != nil {
			lastErr = err
			continue
		}
		client, ok := s.embedders[info.Vendor]
		if !ok {
			lastErr = &NoProviderError{Family: info.Vendor}
			continue
		}
		resp, err := client.EmbedText(ctx, EmbedVendorRequest{
			LogicalModelID: model,
			UpstreamModel:  info.UpstreamModel,
			BaseURL:        info.BaseURL,
			// The credential belongs to the RESOLVED entry. Handing the
			// adapter the entry's own credential is what keeps a request
			// routed to a different host from carrying another provider's
			// key.
			APIKey:           info.APIKey,
			APIKeyEnv:        info.APIKeyEnv,
			Text:             req.Text,
			TaskType:         req.TaskType,
			OutputDimensions: paramDims,
			TenantID:         req.TenantID,
			Traceparent:      req.Traceparent,
		})
		if err != nil {
			lastErr = err
			continue
		}
		if err := checkEmbedResponseLength(resp.Values, id, expectedDims); err != nil {
			lastErr = err
			continue
		}
		return resp, client.Family(), nil
	}
	if lastErr == nil {
		lastErr = errors.New("embed: no embedding target could be resolved")
	}
	return EmbedVendorResponse{}, "", lastErr
}

// resolveEmbeddingDimensions applies the model-aware dimension policy to one
// resolved target and returns (expected, param):
//
//   - expected is the vector length the response MUST carry. 0 = unconstrained.
//   - param is the `dimensions` value to SEND to the provider. 0 = send none.
//
// The two are deliberately distinct. A model that always produces 1024 values
// needs no parameter at all, and forwarding one to a model that rejects the
// parameter turns a working route into a 400 — which is exactly how a
// hard-coded 768 default broke the pinned production route.
//
// Policy:
//   - the caller omitted dimensions: NO parameter is sent. The expected length
//     is the model's declared dimension (or the pin's).
//   - the caller requested dimensions and the entry declares the upstream
//     accepts an override: the request is honoured, and the response must carry
//     the requested length.
//   - the caller requested dimensions the model is known not to produce (it
//     declares a different length and takes no override): refused outright,
//     never dispatched.
//   - the caller requested dimensions and the entry declares no dimension
//     policy at all: no parameter is sent — the gateway must not forward a
//     parameter the entry does not declare support for — and the RESPONSE is
//     held to the requested length.
func resolveEmbeddingDimensions(id LogicalModelID, info ModelInfo, pin *EmbeddingRoutePin, requested int32) (int32, int32, error) {
	expected := int32(info.EmbeddingDimensions)
	paramOK := info.EmbeddingDimensionsParam
	if pin != nil {
		// The pin is authoritative for the pinned route: it names the approved
		// upstream, the vector width the deployment's columns hold, and the
		// fact that the approved upstream takes no `dimensions` parameter.
		if expected > 0 && pin.ExpectedDimensions > 0 && expected != pin.ExpectedDimensions {
			return 0, 0, &DimensionMismatchError{
				Model:    id,
				Expected: pin.ExpectedDimensions,
				Detail: fmt.Sprintf("embed: registry entry %q declares %d dimensions but the pinned route requires %d; refusing to dispatch a route whose vector width contradicts the pin",
					id, expected, pin.ExpectedDimensions),
			}
		}
		if expected == 0 {
			expected = pin.ExpectedDimensions
		}
		paramOK = false
	}
	switch {
	case requested <= 0:
		return expected, 0, nil
	case paramOK:
		return requested, requested, nil
	case expected > 0 && requested != expected:
		return 0, 0, &DimensionMismatchError{
			Model:    id,
			Expected: expected,
			Detail: fmt.Sprintf("embed: model %q produces %d-dimensional vectors and accepts no dimensions override; refusing the requested %d rather than returning a vector the caller cannot store",
				id, expected, requested),
		}
	default:
		return requested, 0, nil
	}
}

// checkEmbedResponseLength holds the returned vector to the expected length. A
// wrong-length vector is refused, never truncated or padded: the caller writes
// it into a fixed-width pgvector column, so a silent mismatch either corrupts
// the embedding or fails there with a message that no longer names the model.
func checkEmbedResponseLength(values []float32, id LogicalModelID, expected int32) error {
	if len(values) == 0 {
		return &DimensionMismatchError{
			Model:    id,
			Expected: expected,
			Detail:   fmt.Sprintf("embed: model %q returned no vector; refusing to ledger an empty embedding", id),
		}
	}
	if expected > 0 && int32(len(values)) != expected {
		return &DimensionMismatchError{
			Model:    id,
			Expected: expected,
			Actual:   int32(len(values)), // #nosec G115 -- a vector length, bounded far below int32
			Detail: fmt.Sprintf("embed: model %q returned a %d-dimensional vector, expected %d; refusing (never truncated or padded)",
				id, len(values), expected),
		}
	}
	return nil
}

// embedCredentialHosts lists the provider hosts that authenticate every
// request. The credential rule is per HOST, not per provider token: the token
// only says which wire shape is spoken, and the same `openai` shape is served
// by self-hosted servers that legitimately need no credential at all. The
// hosted endpoints this deployment can reach are listed, so an entry re-pointed
// at one of them without a credential is refused instead of dispatched
// anonymously.
var embedCredentialHosts = map[string]bool{
	"api.openai.com":    true,
	"api.anthropic.com": true,
	"openrouter.ai":     true,
	"api.longcat.ai":    true,
	"api.meta.ai":       true,
}

// hostRequiresCredential reports whether the entry's endpoint is a hosted
// provider that authenticates every request. A URL that does not parse has no
// host and is left to the base-URL/dispatch checks.
func hostRequiresCredential(baseURL string) bool {
	return embedCredentialHosts[hostOfBaseURL(baseURL)]
}

// hostOfBaseURL extracts the host of an absolute http(s) URL, lowercased.
// Returns "" for anything that does not parse.
func hostOfBaseURL(raw string) string {
	u, err := url.Parse(strings.TrimSpace(raw))
	if err != nil || u.Host == "" {
		return ""
	}
	return strings.ToLower(u.Hostname())
}

// checkEmbedTarget applies the embed policy checks to one resolved registry
// entry: it must advertise the embeddings capability, carry a usable
// credential, and name a dispatchable endpoint. The base-URL check is the one
// that closes the "silent paid route" hole: an entry with no resolvable
// endpoint is refused rather than dispatched to whatever the adapter happens
// to be constructed with.
//
// The credential rule is PROVIDER-SPECIFIC (ChatGPT round 12). The previous
// check refused only "a reference that resolves to empty", which let an entry
// with NO reference at all be dispatched anonymously to a provider that
// authenticates every request. A public or self-hosted OpenAI-compatible
// server legitimately needs no credential, so the requirement keys off the
// endpoint host; the pinned route additionally requires a configured reference,
// because the approved OpenRouter endpoint is never reached anonymously.
func checkEmbedTarget(info ModelInfo, model LogicalModelID, pinned bool) error {
	if !info.Supports("embeddings") {
		return &CapabilityError{Model: model, Capability: "embeddings"}
	}
	if info.BaseURL == "" {
		return fmt.Errorf("embed: registry entry %q declares no base_url and no role base URL is configured; refusing rather than dispatching to an unverified endpoint", model)
	}
	if info.APIKeyEnv != "" && info.APIKey == "" {
		return &CredentialError{Model: model, APIKeyEnv: info.APIKeyEnv}
	}
	if info.APIKeyEnv == "" && (pinned || hostRequiresCredential(info.BaseURL)) {
		return &CredentialError{
			Model: model,
			Detail: fmt.Sprintf("embed: registry entry %q declares no credential reference but dispatches to %s, which authenticates every request; refusing to call it anonymously (set the entry's api_key_env)",
				model, hostOfBaseURL(info.BaseURL)),
		}
	}
	return nil
}
