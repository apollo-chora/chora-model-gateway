package domain

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"strings"
	"time"
)

// Service is the core domain orchestrator for ModelGatewayService.Invoke.
// Composes the per-Invoke flow: budget gate → policy resolve → vendor
// dispatch (with fallback) → budget debit + ledger write → return.
//
// Service holds NO state across requests — all per-request state lives
// on the stack via the InvokeRequest/InvokeResponse pair. Thread-safe
// to call from multiple goroutines.
type Service struct {
	vendors        map[VendorFamily]VendorClient // dispatch registry
	budget         BudgetRepo
	outbox         OutboxWriter
	secrets        SecretClient
	policies       PolicyLoader
	gatewayVersion string // build identifier, e.g. "chora-model-gateway:local"
	now            func() time.Time
	newID          func() string // UUIDv7 generator for invocation_id default

	// claims is the keyed-idempotency seam. OPTIONAL: when nil the
	// DispatchIdempotencyKey field is ignored (a plain retried call debits
	// again, which is the honest behaviour for a key that was never claimed).
	claims ClaimRecorder

	// embedder is OPTIONAL at construction so an Invoke-only wiring and the
	// unit suite need not provide it; Embed fails loud at call-time when nil.
	embedder EmbeddingClient
}

// ServiceConfig groups the ports + cross-cutting deps the gateway needs.
type ServiceConfig struct {
	Vendors  []VendorClient // one per VendorFamily
	Budget   BudgetRepo
	Outbox   OutboxWriter
	Secrets  SecretClient
	Policies PolicyLoader

	GatewayVersion string

	// Claims is the keyed-idempotency recorder. OPTIONAL.
	Claims ClaimRecorder

	// Embedder is OPTIONAL; nil means Service.Embed fails loud.
	Embedder EmbeddingClient

	// Optional overrides for deterministic tests. Production wiring leaves
	// these nil and the constructor substitutes time.Now + UUIDv7.
	Now   func() time.Time
	NewID func() string
}

// NewService constructs a Service. Returns an error if any required port
// is nil — fail-loud, no silent fallback to a stub adapter.
func NewService(cfg ServiceConfig) (*Service, error) {
	if cfg.Budget == nil {
		return nil, errors.New("model gateway: BudgetRepo required")
	}
	if cfg.Outbox == nil {
		return nil, errors.New("model gateway: OutboxWriter required")
	}
	if cfg.Secrets == nil {
		return nil, errors.New("model gateway: SecretClient required")
	}
	if cfg.Policies == nil {
		return nil, errors.New("model gateway: PolicyLoader required")
	}
	if len(cfg.Vendors) == 0 {
		return nil, errors.New("model gateway: at least one VendorClient required")
	}
	if cfg.GatewayVersion == "" {
		return nil, errors.New("model gateway: GatewayVersion required")
	}

	vendors := make(map[VendorFamily]VendorClient, len(cfg.Vendors))
	for _, v := range cfg.Vendors {
		if v == nil {
			return nil, errors.New("model gateway: nil VendorClient in Vendors slice")
		}
		fam := v.Family()
		if fam == "" {
			return nil, errors.New("model gateway: VendorClient.Family() returned empty")
		}
		if _, dup := vendors[fam]; dup {
			return nil, fmt.Errorf("model gateway: duplicate VendorClient for family %q", fam)
		}
		vendors[fam] = v
	}

	now := cfg.Now
	if now == nil {
		now = time.Now
	}
	newID := cfg.NewID
	if newID == nil {
		newID = func() string {
			// Production wiring substitutes a real UUIDv7 generator in
			// cmd/server/main.go. The default here intentionally panics so a
			// no-op-default-substitution mistake fails loud at test time.
			panic("model gateway: NewID generator not wired — set ServiceConfig.NewID")
		}
	}

	return &Service{
		vendors:        vendors,
		budget:         cfg.Budget,
		outbox:         cfg.Outbox,
		secrets:        cfg.Secrets,
		policies:       cfg.Policies,
		gatewayVersion: cfg.GatewayVersion,
		now:            now,
		newID:          newID,
		claims:         cfg.Claims,
		embedder:       cfg.Embedder,
	}, nil
}

// InvokeError categorises terminal Invoke failures the adapters map back to
// a transport status code.
type InvokeError struct {
	Reason FinishReason
	Detail string
	Inner  error
	// FallbackLog names every target the gateway tried before giving up, so
	// a caller-facing 502 can say which models were unreachable.
	FallbackLog []string
}

func (e *InvokeError) Error() string {
	if e.Inner != nil {
		return fmt.Sprintf("invoke %s: %s: %v", e.Reason, e.Detail, e.Inner)
	}
	return fmt.Sprintf("invoke %s: %s", e.Reason, e.Detail)
}

func (e *InvokeError) Unwrap() error { return e.Inner }

// ConfigError marks a gateway-side misconfiguration — a registry entry that
// names a credential which is unset, an unknown capability, a downgrade
// target that is not in the catalogue.
//
// It is a distinct type from InvokeError because it is NOT an upstream
// failure: retrying will not help, and reporting it as 502 "bad gateway"
// sends the caller looking at a provider that is working perfectly. The HTTP
// facade maps it to 500, which says "the thing you called is misconfigured".
type ConfigError struct {
	Detail string
	Inner  error
}

func (e *ConfigError) Error() string {
	if e.Inner != nil {
		return fmt.Sprintf("config %s: %v", e.Detail, e.Inner)
	}
	return fmt.Sprintf("config %s", e.Detail)
}

func (e *ConfigError) Unwrap() error { return e.Inner }

func (fr FinishReason) String() string {
	switch fr {
	case FinishReasonComplete:
		return "complete"
	case FinishReasonMaxTokens:
		return "max_tokens"
	case FinishReasonBudgetBlock:
		return "budget_block"
	case FinishReasonVendorError:
		return "vendor_error"
	case FinishReasonContentBlock:
		return "content_block_retired"
	case FinishReasonMeteringBlock:
		return "metering_block_retired"
	default:
		return "unspecified"
	}
}

// Invoke executes the canonical per-request flow.
//
// Order of operations (each step is one port call — easy to trace):
//
//  1. Validate the request envelope; refuse loud on any gap.
//  2. Resolve invocation_id (caller-supplied or service-generated).
//  3. PolicyLoader.ResolveAgentPolicy — registry lookup: provider + upstream
//     model + endpoint + credential reference.
//  4. Take a keyed idempotency claim when the caller supplied a
//     dispatch key; a duplicate is short-circuited.
//  5. BudgetRepo.GetTenantBudget — load active budget window.
//  6. Decide budget action (BudgetAllow / BudgetDowngrade / BudgetAlert /
//     BudgetBlock). On Block: short-circuit with FinishReasonBudgetBlock.
//     On Downgrade: swap policy.ResolvedLogicalModelID before dispatch.
//  7. Resolve the credential for the target.
//  8. VendorClient.Generate — primary target. On error, walk
//     policy.FallbackChain until success or exhaustion.
//  9. BudgetRepo.DebitSpent + OutboxWriter.EnqueueTokenUsageRecorded in the
//     same transaction (the data-consistency outbox pattern).
//
// 10. Return InvokeResponse.
func (s *Service) Invoke(ctx context.Context, req InvokeRequest) (InvokeResponse, error) {
	start := s.now()

	// Step 1 — validate the envelope. A missing tenant or model is a caller
	// bug, and discovering it at the vendor would mean a billable request
	// with no attribution.
	if err := req.Validate(); err != nil {
		return InvokeResponse{}, &InvokeError{
			Reason: FinishReasonVendorError,
			Detail: "request validation failed",
			Inner:  err,
		}
	}

	// Step 2 — resolve invocation_id.
	invocationID := req.InvocationID
	if invocationID == "" {
		invocationID = s.newID()
	}

	// Step 3 — resolve the registry entry.
	policy, err := s.policies.ResolveAgentPolicy(ctx, req.AgentID, req.CrewKind, req.LogicalModelID, req.FallbackModelIDs)
	if err != nil {
		return InvokeResponse{}, &InvokeError{
			Reason: FinishReasonVendorError,
			Detail: "policy resolve failed",
			Inner:  err,
		}
	}

	// Step 4 — keyed idempotency. A redelivered dispatch with the same key
	// must bill once, so the claim is taken BEFORE any spend.
	deduped := false
	if req.DispatchIdempotencyKey != "" && s.claims != nil {
		claimed, cErr := s.claims.ClaimDebit(ctx, req.GCID, req.DispatchIdempotencyKey, req.ActionCode)
		if cErr != nil {
			return InvokeResponse{}, &InvokeError{
				Reason: FinishReasonVendorError,
				Detail: "idempotency claim failed",
				Inner:  cErr,
			}
		}
		deduped = !claimed
		if deduped {
			slog.InfoContext(ctx, "model gateway: duplicate dispatch key, debit skipped",
				"agent_id", req.AgentID, "tenant_id", req.TenantID,
				"invocation_id", invocationID, "dispatch_key", req.DispatchIdempotencyKey)
		}
	}

	// Step 5 — load tenant budget.
	budget, err := s.budget.GetTenantBudget(ctx, req.TenantID)
	if err != nil {
		return InvokeResponse{}, &InvokeError{
			Reason: FinishReasonVendorError,
			Detail: "budget repo unavailable",
			Inner:  err,
		}
	}

	// Step 6 — decide the budget action.
	decision := BudgetAllow
	if budget != nil {
		decision = budget.Decide()
		if decision == BudgetDowngrade && budget.DowngradeToModel != "" {
			policy.ResolvedLogicalModelID = budget.DowngradeToModel
			// The downgrade names a different registry entry, so the target
			// must be re-resolved. A downgrade to an unknown model is a
			// configuration fault: fail loud rather than dispatching the
			// expensive model the budget was meant to avoid.
			downgraded, dErr := s.policies.ResolveAgentPolicy(ctx, req.AgentID, req.CrewKind, budget.DowngradeToModel, nil)
			if dErr != nil {
				return InvokeResponse{}, &ConfigError{
					Detail: "budget downgrade target is not in the model registry: " + string(budget.DowngradeToModel),
					Inner:  dErr,
				}
			}
			downgraded.ResolvedLogicalModelID = budget.DowngradeToModel
			policy = downgraded
		}
	}
	if decision == BudgetBlock {
		return InvokeResponse{
			InvocationID:   invocationID,
			FallbackChain:  []string{},
			LatencyMs:      elapsedMs(s.now().Sub(start)),
			FinishReason:   FinishReasonBudgetBlock,
			FinishDetail:   "tenant LLM budget exhausted; policy=block",
			CompletedAt:    s.now(),
			GatewayVersion: s.gatewayVersion,
		}, nil
	}

	// Step 7 + 8 — resolve the credential and dispatch, walking the fallback
	// chain on vendor failure.
	targets := make([]TargetModel, 0, 1+len(policy.FallbackChain))
	targets = append(targets, policy.Target)
	for _, fb := range policy.FallbackChain {
		if fb.Target.LogicalModelID == "" {
			continue
		}
		targets = append(targets, fb.Target)
	}

	var (
		vendorResp    VendorResponse
		usedTarget    TargetModel
		fallbackChain []string
		lastErr       error
		dispatched    bool
	)
	for _, target := range targets {
		vendorClient, ok := s.vendors[target.Vendor]
		if !ok {
			lastErr = fmt.Errorf("no vendor adapter registered for family %q", target.Vendor)
			slog.WarnContext(ctx, "model gateway: no adapter for family",
				"family", target.Vendor, "model", target.LogicalModelID)
			continue
		}

		credential := ""
		if target.APIKeyRef != "" {
			credential, err = s.secrets.ResolveCredential(ctx, target.APIKeyRef)
			if err != nil {
				lastErr = fmt.Errorf("resolve credential %q: %w", target.APIKeyRef, err)
				slog.ErrorContext(ctx, "model gateway: credential resolution failed",
					"ref", target.APIKeyRef, "model", target.LogicalModelID, "error", err)
				continue
			}
			if credential == "" {
				// A configured credential reference that resolves empty is a
				// misconfiguration, not a free request: refuse rather than
				// sending an unauthenticated call that a public endpoint would
				// happily serve.
				slog.ErrorContext(ctx, "model gateway: configured credential is empty",
					"ref", target.APIKeyRef, "model", target.LogicalModelID)
				return InvokeResponse{}, &ConfigError{
					Detail: fmt.Sprintf("credential reference %q is set but resolves to an empty value", target.APIKeyRef),
				}
			}
		}

		// Modality gate. Refused against an entry that does not advertise the
		// capability rather than dispatched-and-ignored: a grounded call
		// silently downgraded to ungrounded is worse than a refusal, because
		// the caller gets an answer that LOOKS researched.
		if needed := RequiredCapability(req.ResponseModality); !target.Supports(needed) {
			return InvokeResponse{}, &ConfigError{
				Detail: fmt.Sprintf("model %q does not advertise the %q capability (it has: %s)",
					target.LogicalModelID, needed, strings.Join(target.Capabilities, ", ")),
			}
		}
		// Grounding needs a configured endpoint on top of the capability. A
		// provider can support search while the registry entry never said
		// where its tool lives, and there is nothing to dispatch to.
		if req.ResponseModality == ModalityGrounded && target.Grounding == nil {
			return InvokeResponse{}, &ConfigError{
				Detail: fmt.Sprintf(
					"model %q advertises %q but its registry entry configures no grounding endpoint; add a `grounding:` block",
					target.LogicalModelID, CapabilityWebSearch),
			}
		}

		fallbackChain = append(fallbackChain, fmt.Sprintf("%s:%s", target.Vendor, target.UpstreamModel))

		vendorResp, err = vendorClient.Generate(ctx, VendorRequest{
			Target:           target,
			Prompt:           req.Prompt,
			ResponseModality: req.ResponseModality,
			SystemPrompt:     req.SystemPrompt,
			GenerationConfig: applyOutputCeiling(req.GenerationConfig, target.MaxOutputTokens),
			ExtraHeaders:     mergeHeaders(target, req.ExtraHeaders),
			TenantID:         req.TenantID,
			Traceparent:      req.Traceparent,
			Tracestate:       req.Tracestate,
			ContentsJSON:     req.ContentsJSON,
			ToolsJSON:        req.ToolsJSON,
			// The credential travels on the request, not on the adapter, so
			// two registry entries can hold different keys.
			Credential: credential,
		})
		if err == nil {
			usedTarget = target
			dispatched = true
			break
		}
		lastErr = err
		slog.WarnContext(ctx, "model gateway: dispatch failed, walking fallback chain",
			"model", target.LogicalModelID, "vendor", target.Vendor, "error", err)
	}

	if !dispatched {
		if lastErr == nil {
			lastErr = errors.New("no dispatchable target resolved")
		}
		return InvokeResponse{}, &InvokeError{
			Reason:      FinishReasonVendorError,
			Detail:      "all targets exhausted",
			Inner:       lastErr,
			FallbackLog: fallbackChain,
		}
	}

	// Step 9 — settle the budget + the ledger atomically.
	if err := s.settle(ctx, req, invocationID, usedTarget, vendorResp, fallbackChain, deduped); err != nil {
		return InvokeResponse{}, &InvokeError{
			Reason:      FinishReasonVendorError,
			Detail:      "settle failed",
			Inner:       err,
			FallbackLog: fallbackChain,
		}
	}

	completion := vendorResp.Completion
	modelVersion := vendorResp.ModelVersion
	if modelVersion == "" {
		modelVersion = usedTarget.UpstreamModel
	}

	return InvokeResponse{
		InvocationID:       invocationID,
		Completion:         completion,
		ImageBytes:         vendorResp.ImageBytes,
		ImageMIMEType:      vendorResp.ImageMIMEType,
		ImageRevisedPrompt: vendorResp.ImageRevisedPrompt,
		Usage:              vendorResp.Usage,
		Vendor:             string(usedTarget.Vendor),
		ModelVersion:       modelVersion,
		FallbackChain:      fallbackChain,
		LatencyMs:          elapsedMs(s.now().Sub(start)),
		FinishReason:       vendorResp.FinishReason,
		FinishDetail:       vendorResp.FinishDetail,
		CompletedAt:        s.now(),
		GatewayVersion:     s.gatewayVersion,
		ToolCallsJSON:      vendorResp.ToolCallsJSON,
		Citations:          vendorResp.Citations,
		Messages:           vendorResp.Messages,
		SearchQueries:      vendorResp.SearchQueries,
		GroundingSurface:   groundingSurfaceOf(usedTarget, req.ResponseModality),
	}, nil
}

// settle writes the budget debit and the ledger row.
//
// When the repo is atomic (it implements Settler) both land in one
// transaction. Otherwise they are two writes, and the failure mode is
// asymmetric: a debit that lands without a ledger row is invisible spend,
// while a ledger row without a debit is merely an under-count. The ordering
// below therefore debits first — an operator reconciling a total sees spend
// they cannot explain, which is recoverable, rather than usage they never
// paid for.
func (s *Service) settle(
	ctx context.Context,
	req InvokeRequest,
	invocationID string,
	target TargetModel,
	vendorResp VendorResponse,
	fallbackChain []string,
	deduped bool,
) error {
	evt := TokenUsageEvent{
		UsageID:       invocationID,
		TenantID:      req.TenantID,
		GCID:          req.GCID,
		ModelID:       vendorResp.ModelVersion,
		InputTokens:   vendorResp.Usage.InputTokens,
		OutputTokens:  vendorResp.Usage.OutputTokens,
		CachedTokens:  vendorResp.Usage.CachedTokens,
		CostMicros:    vendorResp.Usage.CostMicros,
		InvocationID:  invocationID,
		AgentRole:     req.ActionCode,
		Surface:       req.Surface,
		Modality:      req.ResponseModality,
		RecordedAt:    s.now(),
		Vendor:        string(target.Vendor),
		FallbackChain: fallbackChain,
		DebitDeduped:  deduped,

		GatewayVersion: s.gatewayVersion,
		Traceparent:    req.Traceparent,
		Tracestate:     req.Tracestate,
	}

	// A deduped dispatch passes a zero delta. The provider already ran and was
	// already paid, so the usage is real and MUST be ledgered — but moving the
	// budget again would bill the same work twice, which is the entire thing
	// the claim exists to prevent.
	debit := vendorResp.Usage.CostMicros
	if deduped {
		debit = 0
	}

	if settler, ok := s.budget.(Settler); ok {
		if err := settler.Settle(ctx, evt, debit); err != nil {
			return fmt.Errorf("settle: %w", err)
		}
		return nil
	}

	if debit > 0 {
		if err := s.budget.DebitSpent(ctx, req.TenantID, debit); err != nil {
			return fmt.Errorf("debit budget: %w", err)
		}
	}
	if err := s.outbox.EnqueueTokenUsageRecorded(ctx, evt); err != nil {
		return fmt.Errorf("ledger enqueue: %w", err)
	}
	return nil
}

// applyOutputCeiling clamps a caller-supplied max_tokens to the registry
// entry's ceiling. A caller asking for MORE than the model allows is a
// configuration mistake worth correcting silently; a caller asking for less
// is honoured.
func applyOutputCeiling(cfg map[string]any, ceiling int) map[string]any {
	if ceiling <= 0 || cfg == nil {
		return cfg
	}
	requested, ok := numberFrom(cfg["max_tokens"])
	if !ok || requested <= float64(ceiling) {
		return cfg
	}
	out := make(map[string]any, len(cfg)+1)
	for k, v := range cfg {
		out[k] = v
	}
	out["max_tokens"] = ceiling
	return out
}

// mergeHeaders combines the registry entry's static headers with any
// per-request override. Registry values are the floor; the request may add
// but never silently drop them.
func mergeHeaders(target TargetModel, requestHeaders map[string]string) map[string]string {
	if len(target.ExtraHeaders) == 0 && len(requestHeaders) == 0 {
		return nil
	}
	out := make(map[string]string, len(target.ExtraHeaders)+len(requestHeaders))
	for k, v := range target.ExtraHeaders {
		out[k] = v
	}
	for k, v := range requestHeaders {
		out[k] = v
	}
	return out
}

// groundingSurfaceOf reports which search surface served the call, and "" for a
// call that did not use one.
//
// The modality is checked first on purpose: a model CAN be grounded without
// every call being grounded, so reporting a surface for a plain text call would
// tell an operator the search ran when it did not.
func groundingSurfaceOf(t TargetModel, modality string) string {
	if modality != ModalityGrounded || t.Grounding == nil {
		return ""
	}
	return string(t.Grounding.EffectiveSurface(t.Vendor))
}

func elapsedMs(d time.Duration) int32 {
	// #nosec G115 -- elapsed milliseconds since request start; only overflows
	// past ~24.8 days of single-request wall time.
	return int32(d.Milliseconds())
}
