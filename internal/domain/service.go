package domain

import (
	"context"
	"errors"
	"fmt"
	"time"
)

// Service is the core domain orchestrator for the model gateway.
//
// The Invoke flow is delegated to the Executor — the single governed
// execution path that wraps the governance pipeline (suspension, budget,
// Armor PRE/POST, provider dispatch with fallback, usage accounting, mana
// debit, outbox emit). The Executor is the security boundary: handlers
// (gRPC, HTTP) translate to ExecuteRequest and call Execute; they never
// hold a Provider client.
//
// Service holds NO state across requests — all per-request state lives
// on the stack via the InvokeRequest/InvokeResponse pair. Thread-safe
// to call from multiple goroutines.
type Service struct {
	// executor is the governed Invoke path. Constructed in NewService from
	// the ports below; Invoke delegates to executor.Execute. The Executor is
	// the security boundary — handlers never hold a Provider client.
	executor *Executor

	// Shared ports. The Invoke flow's ports live on the Executor (which
	// holds references to these same instances); the GroundedSearch / Embed
	// flows read them directly off the Service.
	vendors        map[VendorFamily]VendorClient // dispatch registry
	armor          ArmorClient
	budget         BudgetRepo
	outbox         OutboxWriter
	policies       PolicyLoader
	gatewayVersion string // build identifier, e.g. "chora-model-gateway:58b28eb"
	now            func() time.Time
	newID          func() string // UUIDv7 generator for invocation_id default
	models         ModelResolver  // model-registry resolver (the Invoke flow's authoritative route source)

	// GroundedSearch ports (ADR-231). OPTIONAL at construction so Invoke-only
	// wiring + tests need not provide them; GroundedSearch fails loud at
	// call-time when any is nil. Production (cmd/server/main.go) wires all four.
	egressGate  ExternalEgressGate
	grounded    GroundedVendorClient
	mana        ManaMeter
	egressAudit EgressAuditWriter

	// Embed ports (G1'-1). OPTIONAL at construction, mirroring the grounded
	// ports; Embed fails loud at call-time when nil (never a silent
	// fallback to direct Vertex).
	//
	// The registry-directed Embed flow resolves the logical model through
	// s.models and dispatches through the client registered for the RESOLVED
	// provider family, so the registry — not the logical id's name shape —
	// decides the embedding route. Optional: with no Embedders wired the
	// flow fails loud rather than dispatching through an unverified
	// adapter.
	embedders map[VendorFamily]EmbeddingClient

	// budgetRequired enables the fail-closed "budget required" mode for the
	// GroundedSearch + Embed flows (see ServiceConfig.BudgetRequired).
	budgetRequired bool

	// budgetRequiredTenants is the tenant-scoped override for the same mode
	// (CHORA_LLM_BUDGET_REQUIRED_TENANTS): listed tenants fail closed even
	// when the global flag is off. See BudgetRequiredTenants.
	budgetRequiredTenants BudgetRequiredTenants

	// admission caps the number of in-flight billable invocations per tenant
	// (CHORA_LLM_MAX_CONCURRENT_PER_TENANT). nil = unlimited/off. See
	// TenantConcurrencyLimiter for the single-instance assumption.
	admission *TenantConcurrencyLimiter

	// embeddingPin, when non-nil, refuses an Embed request whose model is not
	// the pinned logical id (CHORA_EMBEDDING_PIN_*). nil = unpinned.
	embeddingPin *EmbeddingRoutePin
}

// ServiceConfig groups the ports + cross-cutting deps the gateway needs.
type ServiceConfig struct {
	Vendors        []VendorClient // one per VendorFamily
	Armor          ArmorClient
	Budget         BudgetRepo
	Outbox         OutboxWriter
	Policies       PolicyLoader
	GatewayVersion string

	// GroundedSearch ports (ADR-231). OPTIONAL — Invoke-only callers may leave
	// them nil; Service.GroundedSearch fails loud if invoked without them.
	EgressGate  ExternalEgressGate
	Grounded    GroundedVendorClient
	Mana        ManaMeter
	EgressAudit EgressAuditWriter

	// Violations is the ADR-152 governance-violation producer. OPTIONAL, see
	// the field comment on Service.violations.
	Violations ViolationPublisher

	// Embed ports (G1'-1). OPTIONAL; nil means Service.Embed fails loud
	// rather than dispatching through an unverified adapter.
	Embedders []EmbeddingClient

	// EnforceContentsScreen promotes the G1'-3 contents leg from audit-only to
	// blocking. Default false, which is the owner's closing rule of 2026-08-07:
	// a newly screened path must not start refusing calls that pass today. When
	// false the verdict is still computed, reported and published as governance
	// evidence, so the promotion decision can be made on measured data rather
	// than on expectation. Sourced from CHORA_ARMOR_ENFORCE_CONTENTS_SCREEN.
	EnforceContentsScreen bool

	// EnforceToolCallScreen promotes the G1'-2 tool-call leg from audit-only to
	// blocking, on exactly the same terms as EnforceContentsScreen above:
	// tool-calling turns succeed today, so by default the verdict is computed,
	// reported and published as evidence while the turn proceeds. Sourced from
	// CHORA_ARMOR_ENFORCE_TOOL_CALL_SCREEN.
	EnforceToolCallScreen bool

	// EnforceToolsScreen promotes the CHO-2391 tool-DECLARATION leg from
	// audit-only to blocking, on exactly the same terms as the two above.
	// Default false is load-bearing here: tool declarations are
	// platform-authored today, so every tool-calling turn in production passes
	// this path, and a leg that starts refusing them would break working
	// callers to close a hole nothing currently reaches. Sourced from
	// CHORA_ARMOR_ENFORCE_TOOLS_SCREEN.
	EnforceToolsScreen bool

	// BudgetRequired enables the fail-closed "budget required" mode
	// (CHORA_LLM_BUDGET_REQUIRED) for the GroundedSearch + Embed flows: a
	// tenant with NO active budget window is refused instead of allowed, so
	// the absence of a policy cannot restore unlimited provider spending.
	// Default false preserves the historical fail-open behaviour.
	BudgetRequired bool

	// BudgetRequiredTenants is the tenant-scoped override
	// (CHORA_LLM_BUDGET_REQUIRED_TENANTS): a comma-separated list of tenant
	// UUIDs that fail closed even when BudgetRequired is false. Tenants not
	// in the list keep the historical fail-open behaviour until their budgets
	// have been provisioned.
	BudgetRequiredTenants BudgetRequiredTenants

	// MaxConcurrentPerTenant caps the number of in-flight billable invocations
	// per tenant across ALL THREE dispatch paths (Invoke, Embed,
	// GroundedSearch). 0 (the default) means unlimited — the historical
	// behaviour. Sourced from CHORA_LLM_MAX_CONCURRENT_PER_TENANT.
	//
	// SINGLE-INSTANCE ASSUMPTION: the cap is process-local. It is a true
	// ceiling only when every billable request for the capped tenant reaches
	// exactly one gateway process; see TenantConcurrencyLimiter.
	MaxConcurrentPerTenant int

	// EmbeddingPin, when non-nil, pins the Embed flow to one logical model id
	// (CHORA_EMBEDDING_PIN_*): a caller-supplied model that is not the pinned
	// id is refused rather than dispatched, so a runtime override cannot
	// redirect the demo's embedding route. nil = unpinned.
	EmbeddingPin *EmbeddingRoutePin

	// Optional overrides for deterministic tests. Production wiring leaves
	// these nil and the constructor substitutes time.Now + UUIDv7.
	Now   func() time.Time
	NewID func() string

	// Claims is the keyed debit-claim port (ADR-254 D7, R22). OPTIONAL; nil
	// means Invoke skips the dispatch dedup claim.
	Claims DebitClaimer

	// Models is the model-registry resolver. OPTIONAL; nil means the
	// capability / output-ceiling / credential checks are skipped.
	Models ModelResolver

	// FallbackRetries / FallbackRetryBackoff configure the FallbackEngine's
	// same-provider retry budget (see ExecutorConfig.FallbackRetries). OPTIONAL;
	// 0 retries means a retryable failure advances straight to the next target.
	FallbackRetries      int
	FallbackRetryBackoff time.Duration

	// RetryMatrix overrides the ratified retry table (DefaultRetryMatrix).
	// OPTIONAL; the zero value keeps the default.
	RetryMatrix RetryMatrix

	// AtomicOutbox is the transactional debit+outbox port. OPTIONAL; nil
	// means the service falls back to separate DebitSpent + outbox calls.
	AtomicOutbox AtomicBudgetOutbox

	// DefaultAgentID is the fallback agent id for IMAGE action_code
	// defaulting. Empty means no IMAGE defaulting.
	DefaultAgentID string
}

// NewService constructs a Service. Returns an error if any required port
// is nil — fail-loud per feedback_no_stubs_real_wiring (no silent fallback
// to a stub adapter).
func NewService(cfg ServiceConfig) (*Service, error) {
	if cfg.Armor == nil {
		return nil, errors.New("model gateway: ArmorClient required")
	}
	if cfg.Budget == nil {
		return nil, errors.New("model gateway: BudgetRepo required")
	}
	if cfg.Outbox == nil {
		return nil, errors.New("model gateway: OutboxWriter required")
	}
	if cfg.Policies == nil {
		return nil, errors.New("model gateway: PolicyLoader required")
	}
	if len(cfg.Vendors) == 0 {
		return nil, errors.New("model gateway: at least one VendorClient required")
	}
	if cfg.GatewayVersion == "" {
		return nil, errors.New("model gateway: GatewayVersion required (no inline-config-style empty defaults)")
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

	embedders := make(map[VendorFamily]EmbeddingClient, len(cfg.Embedders))
	for _, e := range cfg.Embedders {
		if e == nil {
			return nil, errors.New("model gateway: nil EmbeddingClient in Embedders slice")
		}
		fam := e.Family()
		if fam == "" {
			return nil, errors.New("model gateway: EmbeddingClient.Family() returned empty")
		}
		if _, dup := embedders[fam]; dup {
			return nil, fmt.Errorf("model gateway: duplicate EmbeddingClient for family %q", fam)
		}
		embedders[fam] = e
	}

	now := cfg.Now
	if now == nil {
		now = time.Now
	}
	newID := cfg.NewID
	if newID == nil {
		newID = func() string {
			// Production wiring substitutes a real UUIDv7 generator in
			// cmd/server/main.go. The default here intentionally panics so
			// a no-op-default-substitution mistake fails loud at test time.
			panic("model gateway: NewID generator not wired — set ServiceConfig.NewID")
		}
	}

	// Construct the Executor — the governed Invoke path — from the same
	// ports. The Executor holds its own references; the Service keeps the
	// ports the GroundedSearch / Embed flows read directly.
	providers := make([]Provider, 0, len(cfg.Vendors))
	for _, v := range cfg.Vendors {
		providers = append(providers, v) // VendorClient satisfies Provider
	}
	executor, err := NewExecutor(ExecutorConfig{
		Providers:      providers,
		Armor:          cfg.Armor,
		Budget:         cfg.Budget,
		Outbox:         cfg.Outbox,
		Policies:       cfg.Policies,
		Claims:         cfg.Claims,
		Models:         cfg.Models,
		AtomicOutbox:   cfg.AtomicOutbox,
		Violations:     cfg.Violations,
		GatewayVersion: cfg.GatewayVersion,
		Now:            now,
		NewID:          newID,
		DefaultAgentID: cfg.DefaultAgentID,

		FallbackRetries:      cfg.FallbackRetries,
		FallbackRetryBackoff: cfg.FallbackRetryBackoff,
		RetryMatrix:          cfg.RetryMatrix,

		EnforceContentsScreen: cfg.EnforceContentsScreen,
		EnforceToolCallScreen: cfg.EnforceToolCallScreen,
		EnforceToolsScreen:    cfg.EnforceToolsScreen,

		BudgetRequired: cfg.BudgetRequired,

		BudgetRequiredTenants: cfg.BudgetRequiredTenants,
	})
	if err != nil {
		return nil, err
	}

	return &Service{
		executor: executor,

		vendors:        vendors,
		armor:          cfg.Armor,
		budget:         cfg.Budget,
		outbox:         cfg.Outbox,
		policies:       cfg.Policies,
		gatewayVersion: cfg.GatewayVersion,
		now:            now,
		newID:          newID,
		egressGate:     cfg.EgressGate,
		grounded:       cfg.Grounded,
		mana:           cfg.Mana,
		egressAudit:    cfg.EgressAudit,
		embedders:      embedders,
		models:         cfg.Models,

		budgetRequired: cfg.BudgetRequired,

		budgetRequiredTenants: cfg.BudgetRequiredTenants,

		admission:    NewTenantConcurrencyLimiter(cfg.MaxConcurrentPerTenant),
		embeddingPin: cfg.EmbeddingPin,
	}, nil
}

// InvokeError categorises terminal Invoke failures the gRPC adapter maps
// back to the appropriate gRPC status code.
type InvokeError struct {
	Reason FinishReason
	Detail string
	Inner  error
	// UpstreamStatus carries the provider's own HTTP status when the failure
	// came from a non-2xx upstream response. The adapter relays it (401/403 →
	// Unauthenticated, 404 → NotFound, 429 → ResourceExhausted, else
	// Unavailable) instead of flattening every failure into one code. Nil when
	// the failure has no upstream status (transport error, config refusal).
	UpstreamStatus *int
	// Kind mirrors the Python gateway's error taxonomy for gRPC status
	// mapping: ErrorKindConfig (FailedPrecondition), ErrorKindInvoke
	// (Unavailable), ErrorKindBare (InvalidArgument). The zero value behaves
	// as ErrorKindInvoke so existing constructions keep their mapping.
	Kind ErrorKind
}

func (e *InvokeError) Error() string {
	if e.Inner != nil {
		return fmt.Sprintf("invoke %s: %s: %v", e.Reason, e.Detail, e.Inner)
	}
	return fmt.Sprintf("invoke %s: %s", e.Reason, e.Detail)
}

func (e *InvokeError) Unwrap() error { return e.Inner }

// ErrorKind is the invoke-error taxonomy the gRPC adapter maps to a gRPC
// status code. It mirrors the Python GatewayError.kind field.
type ErrorKind string

const (
	// ErrorKindInvoke — a vendor/dispatch failure. Maps to Unavailable.
	ErrorKindInvoke ErrorKind = "invoke"
	// ErrorKindConfig — a gateway misconfiguration (model not in registry,
	// capability missing, credential empty, vendor family not registered).
	// Maps to FailedPrecondition.
	ErrorKindConfig ErrorKind = "config"
	// ErrorKindBare — a bare request-validation refusal (the embed guards).
	// Maps to InvalidArgument.
	ErrorKindBare ErrorKind = "bare"
)

// ConfigError marks a gateway misconfiguration discovered during dispatch.
// The service tags the terminal InvokeError with ErrorKindConfig when the
// walk hit one of these; the adapter maps that to FailedPrecondition.
type ConfigError struct {
	Detail string
	Inner  error
}

func (e *ConfigError) Error() string {
	if e.Inner != nil {
		return e.Detail + ": " + e.Inner.Error()
	}
	return e.Detail
}

func (e *ConfigError) Unwrap() error { return e.Inner }

// UpstreamStatusProvider is implemented by vendor errors that carry the
// provider's own HTTP status code (openai/anthropic/gemini/vertexembed
// ProviderError). The service + adapter use it to relay the upstream status
// instead of flattening every failure into one gRPC code.
type UpstreamStatusProvider interface {
	UpstreamStatus() int
}

// UpstreamStatusOf extracts the provider's HTTP status from err when the
// error (or any error it wraps) carries one. Returns nil when the failure
// has no upstream status (e.g. a transport error or a config refusal).
func UpstreamStatusOf(err error) *int {
	var usp UpstreamStatusProvider
	if errors.As(err, &usp) {
		s := usp.UpstreamStatus()
		return &s
	}
	return nil
}

func (fr FinishReason) String() string {
	switch fr {
	case FinishReasonComplete:
		return "complete"
	case FinishReasonMaxTokens:
		return "max_tokens"
	case FinishReasonModelArmorBlock:
		return "model_armor_block"
	case FinishReasonBudgetBlock:
		return "budget_block"
	case FinishReasonVendorError:
		return "vendor_error"
	case FinishReasonManaBlock:
		return "mana_block"
	default:
		return "unspecified"
	}
}

// Invoke executes the canonical per-request flow per ADR-163 by delegating to
// the Executor — the single governed execution path. The Executor wraps the
// full governance pipeline (suspension, budget, Armor PRE/POST, provider
// dispatch with fallback, usage accounting, mana debit, outbox emit) and is
// the security boundary: handlers translate to ExecuteRequest and call
// Execute; they never hold a Provider client.
//
// Invoke is retained as the wire-level entry point the adapters' Invoker
// interface calls; it adds no logic of its own.
//
// The per-tenant concurrency ceiling is taken HERE, on the single path every
// Invoke rides (gRPC and HTTP both reach the domain service through the
// governance decorators), so the slot is held across the provider dispatch and
// released on every exit path. The check runs BEFORE any billable dispatch: a
// refused request never reaches a vendor and is never billed.
func (s *Service) Invoke(ctx context.Context, req InvokeRequest) (InvokeResponse, error) {
	release, ok := s.admission.Acquire(req.TenantID)
	if !ok {
		return InvokeResponse{}, &OverloadedError{TenantID: req.TenantID, Limit: s.admission.Limit()}
	}
	defer release()

	resp, err := s.executor.Execute(ctx, req)
	if err != nil {
		return InvokeResponse{}, err
	}
	return *resp, nil
}

// Execute is the transport-agnostic entry point to the domain service's
// Invoke flow. It delegates to Invoke so that *domain.Service satisfies the
// adapters' Invoker interface (which calls Execute). In production the
// adapters call Executor.Execute(), which wraps this method with the
// governance decorators (suspension → mana).
func (s *Service) Execute(ctx context.Context, req InvokeRequest) (InvokeResponse, error) {
	return s.Invoke(ctx, req)
}
