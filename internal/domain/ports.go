// Boundary interfaces — Go-hexagonal style: interfaces live in the
// CONSUMER's package (domain), implementations live in adapter packages
// that import domain. This avoids the import cycle that arises when a
// separate ports/ package tries to reference domain types while
// domain/service.go tries to import ports/.
//
// Hexagonal direction-of-dependency: adapter → domain. NEVER reverse.
//
// All methods take a context.Context first per Go convention. The domain
// service propagates the inbound gRPC context (carrying W3C trace headers,
// auth metadata, deadline) through every port call so downstream OTLP
// spans nest correctly + deadline propagation works.
package domain

import (
	"context"
)

// VendorClient dispatches an LLM call to a specific vendor. One
// implementation per VendorFamily lives under internal/adapter/vendor/.
//
// The service layer picks the implementation based on the resolved
// AgentPolicy.Vendor (or the fallback chain entries on retry).
type VendorClient interface {
	// Family returns the VendorFamily this client implements. The service
	// uses it to pick the right client from a registry at dispatch time.
	Family() VendorFamily

	// Generate executes the LLM call. Returns the completion + token usage
	// + finish reason. Errors are returned verbatim — the service layer
	// decides whether to walk the fallback chain.
	Generate(ctx context.Context, req VendorRequest) (VendorResponse, error)
}

// EmbeddingClient produces dense text embeddings via a vendor's embedding
// surface (G1'-1). Registered separately from VendorClient because
// embeddings carry no completion/finish-reason semantics. ONE
// implementation for v1 (internal/adapter/vendors/vertexembed).
type EmbeddingClient interface {
	// Family names the vendor family for ledger attribution.
	Family() VendorFamily

	// EmbedText produces one embedding vector. Errors return verbatim;
	// the service fails loud (no fallback chain on the embed path in v1).
	EmbedText(ctx context.Context, req EmbedVendorRequest) (EmbedVendorResponse, error)
}

// ArmorClient wraps Cloud Model Armor SDK calls. ONE implementation
// (internal/adapter/modelarmor/) — the gateway is the sole caller per
// ADR-152 + ADR-163.
type ArmorClient interface {
	// SanitizeUserPrompt is the PRE-LLM screening hop. Returns the verdict
	// + the (possibly sanitised) prompt text the service should forward
	// to the vendor.
	SanitizeUserPrompt(ctx context.Context, template, prompt string) (verdict ArmorVerdict, sanitisedPrompt string, err error)

	// SanitizeModelResponse is the POST-LLM screening hop. Returns the
	// verdict + the (possibly sanitised) response text the gateway
	// returns to the caller. Both verdicts get persisted on the
	// outbox + reported in InvokeResponse.
	SanitizeModelResponse(ctx context.Context, template, response string) (verdict ArmorVerdict, sanitisedResponse string, err error)
}

// BudgetRepo manages the per-tenant LLM budget row in
// chora_observability.per_tenant_llm_budget. Each Invoke wraps a
// transaction: GetTenantBudget → service decision → DebitSpent (on
// success) or no-op (on block).
//
// The implementation MUST SET LOCAL chora.tenant_id = $1 at the start of
// the transaction so the RLS policy from migration 0008 enforces
// per-tenant isolation.
type BudgetRepo interface {
	// GetTenantBudget loads the currently-active budget window for the
	// tenant. Returns nil + nil error when no budget is configured (gateway
	// treats no-budget as BudgetAllow — billing reconciles via the global
	// ledger).
	GetTenantBudget(ctx context.Context, tenantID string) (*BudgetState, error)

	// DebitSpent atomically adds usdMicrosDelta to spent_usd_micros for
	// the budget row identified by the prior GetTenantBudget snapshot.
	// Called after a successful vendor dispatch + Armor POST + before
	// the outbox emit; same transaction as the outbox row to preserve
	// transactional integrity per data-consistency skill.
	DebitSpent(ctx context.Context, tenantID string, usdMicrosDelta int64) error
}

// OutboxWriter enqueues the TokenUsageRecorded event onto the per-tenant
// outbox row inside the same transaction as the budget debit. The
// dedicated Pub/Sub publisher process drains the outbox → canonical
// topic chora.observability.token_usage.recorded.v1.
//
// Implementation lives in internal/adapter/events/.
type OutboxWriter interface {
	EnqueueTokenUsageRecorded(ctx context.Context, evt TokenUsageEvent) error
}

// PolicyLoader resolves the per-agent routing + guardrail policy at the
// start of every Invoke. Implementations may cache aggressively in
// memory (policies change rarely — at deploy time of the policy YAML).
type PolicyLoader interface {
	// ResolveAgentPolicy resolves the routing + guardrail policy for one
	// Invoke. requestedModel is the caller's primary logical model;
	// fallbackModels is the agent-declared ordered fallback chain (CR qgen
	// 2026-06-01 — AGENT-DRIVEN tiering) which the loader translates into
	// AgentPolicy.FallbackChain entries against the resolved vendor. Empty
	// fallbackModels = single-shot (no fallback).
	ResolveAgentPolicy(ctx context.Context, agentID, crewKind string, requestedModel LogicalModelID, fallbackModels []LogicalModelID) (AgentPolicy, error)
}

// SecretClient resolves BYOA tenant API keys + global vendor fallback
// keys via Secret Manager. ONLY the vendor adapters call this — the
// service layer doesn't touch secrets directly.
type SecretClient interface {
	// ResolveByoaKey returns the API key for the (tenant, vendor) pair if
	// the tenant has provisioned BYOA. Returns ("", nil) when no BYOA is
	// configured for that pair — vendor adapter falls back to global key.
	ResolveByoaKey(ctx context.Context, tenantID string, vendor VendorFamily) (string, error)

	// ResolveGlobalKey returns the platform-controlled global API key for
	// the vendor. Used when BYOA is absent.
	ResolveGlobalKey(ctx context.Context, vendor VendorFamily) (string, error)
}

// -----------------------------------------------------------------------------
// GroundedSearch ports (ADR-231) — the controlled web-egress surface.
// -----------------------------------------------------------------------------

// ExternalEgressGate is the fail-closed per-tenant + platform gate for grounded
// web egress (ADR-231 D6). It enforces, in one Authorize call:
//   - the platform-wide O+ kill-switch (all grounded egress off),
//   - the per-tenant external_egress entitlement (ADR-220 D4 default OFF for
//     franchise), and
//   - the per-tenant DAILY grounded-call ceiling (ADR-231 decision 6).
//
// Any error ⇒ DENY (the chokepoint does not trust an unreachable policy store).
// Sourced from chora_observability tenant policy, beside the budget.
type ExternalEgressGate interface {
	// Authorize returns whether a grounded egress is permitted for the tenant
	// RIGHT NOW. It does NOT increment the daily counter — call RecordEgress
	// after a successful grounded dispatch.
	Authorize(ctx context.Context, tenantID string) (EgressAuthorization, error)

	// RecordEgress increments the tenant's grounded-call counter for today so
	// the daily ceiling is enforced across calls. Called post-success only.
	RecordEgress(ctx context.Context, tenantID string) error
}

// GroundedVendorClient dispatches a grounded ("Grounding with Google Search")
// completion. ONLY the gemini adapter implements it — grounding rides the
// existing generateContent call with the google_search tool (ADR-231 D1).
type GroundedVendorClient interface {
	// Family returns the VendorFamily backing grounded search.
	Family() VendorFamily

	// GroundedGenerate runs the grounded web-search completion and parses the
	// structured citations + searchEntryPoint + queries out of
	// groundingMetadata. Errors are returned verbatim to the service.
	GroundedGenerate(ctx context.Context, req GroundedVendorRequest) (GroundedVendorResponse, error)
}

// ManaMeter is the domain-side port to the identity ManaService for the
// GroundedSearch chain (ADR-177 central metering). Grounded search is
// fail-closed: the service PRE-FLIGHTs Quote before the expensive egress and
// Debits post-success. Implemented by an adapter over clients.ManaClient.
type ManaMeter interface {
	// Quote runs a dry-run affordability check for action_code WITHOUT
	// debiting. UnknownAction ⇒ the code is unpriced.
	Quote(ctx context.Context, gcid, tenantID, actionCode string) (ManaQuote, error)

	// Debit charges the catalogue cost for action_code, idempotent on idemKey.
	// UnknownAction ⇒ unpriced (no charge).
	Debit(ctx context.Context, gcid, tenantID, actionCode, idemKey string) (ManaDebit, error)
}

// EgressAuditWriter enqueues the ExternalEgressAudited event onto the outbox
// (topic chora.governance.audit.external_egress.v1) — a first-class egress
// trail (ADR-231 D6). Implemented alongside OutboxWriter by the pg adapter.
type EgressAuditWriter interface {
	EnqueueExternalEgressAudited(ctx context.Context, evt ExternalEgressAuditEvent) error
}

// ViolationPublisher enqueues the PolicyViolationDetected event onto the outbox
// (topic chora.governance.policy.violation_detected.v1) whenever Cloud Model
// Armor BLOCKS a call: the ADR-152 producer, amended 2026-08-07 to sit at the
// gateway rather than behind a Cloud Logging sink. Implemented alongside
// OutboxWriter by the pg adapter.
type ViolationPublisher interface {
	EnqueuePolicyViolationDetected(ctx context.Context, evt PolicyViolationEvent) error
}

// DebitClaimer is the keyed debit-claim port (ADR-254 D7, R22): the domain
// service claims (gcid, dispatch_idempotency_key, action_code) BEFORE any
// spend so a redelivered dispatch bills once. claimed=true means this call
// won the key and must debit; claimed=false means the key was already billed
// by an earlier delivery (deduped) and the call proceeds unbilled.
//
// Implemented by adapter/pg.Repo.ClaimDebit (chora_observability migration
// 0019). The domain service takes the claim at the start of Invoke and
// communicates the result to the mana-metering decorator via
// InvokeResponse.Deduped, so ONE claim suppresses BOTH the tenant-budget
// debit and the mana debit.
type DebitClaimer interface {
	ClaimDebit(ctx context.Context, gcid, dispatchKey, actionCode, invocationID string) (bool, error)
}

// AtomicBudgetOutbox runs the budget debit + token-usage outbox insert in
// ONE database transaction (transactional outbox pattern). The pg adapter
// implements it; the service uses it on the happy path + Armor-POST-block
// path so the two writes either both commit or both abort.
type AtomicBudgetOutbox interface {
	DebitAndEnqueue(ctx context.Context, evt TokenUsageEvent, usdMicrosDelta int64) error
}
