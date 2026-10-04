// Boundary interfaces — Go-hexagonal style: interfaces live in the
// CONSUMER's package (domain), implementations live in adapter packages
// that import domain. This avoids the import cycle that arises when a
// separate ports/ package tries to reference domain types while
// domain/service.go tries to import ports/.
//
// Hexagonal direction-of-dependency: adapter → domain. NEVER reverse.
//
// All methods take a context.Context first per Go convention. The domain
// service propagates the inbound context (carrying W3C trace headers,
// deadline) through every port call so deadline propagation works.
package domain

import (
	"context"
)

// VendorClient dispatches an LLM call to a specific vendor. One
// implementation per VendorFamily lives under internal/adapter/vendors/.
//
// The service layer picks the implementation based on the resolved
// policy.Vendor, and the PER-REQUEST target (upstream model name, base URL,
// credential) arrives on VendorRequest.Target — so one adapter instance
// serves every registry entry that shares a provider.
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
// surface. Registered separately from VendorClient because embeddings carry
// no completion/finish-reason semantics.
type EmbeddingClient interface {
	// Family names the vendor family for ledger attribution.
	Family() VendorFamily

	// EmbedText produces one embedding vector. Errors return verbatim;
	// the service fails loud (no fallback chain on the embed path).
	EmbedText(ctx context.Context, req EmbedVendorRequest) (EmbedVendorResponse, error)
}

// BudgetRepo manages the per-tenant LLM budget row in the
// per_tenant_llm_budget table. Each Invoke wraps a transaction:
// GetTenantBudget → service decision → DebitSpent (on success).
//
// The implementation MUST SET LOCAL chora.tenant_id = $1 at the start of
// the transaction so the RLS policy enforces per-tenant isolation.
type BudgetRepo interface {
	// GetTenantBudget loads the currently-active budget window for the
	// tenant. Returns nil + nil error when no budget is configured (gateway
	// treats no-budget as BudgetAllow — billing reconciles via the ledger).
	GetTenantBudget(ctx context.Context, tenantID string) (*BudgetState, error)

	// DebitSpent atomically adds usdMicrosDelta to spent_usd_micros for
	// the budget row identified by the prior GetTenantBudget snapshot.
	// Called after a successful vendor dispatch and in the SAME transaction
	// as the outbox row.
	DebitSpent(ctx context.Context, tenantID string, usdMicrosDelta int64) error
}

// OutboxWriter enqueues the TokenUsageRecorded event onto the token-usage
// outbox row inside the same transaction as the budget debit. The budget
// debit and the ledger row MUST land atomically or not at all.
type OutboxWriter interface {
	EnqueueTokenUsageRecorded(ctx context.Context, evt TokenUsageEvent) error
}

// Settler is the atomic form of the budget debit + the ledger write.
//
// When the configured BudgetRepo also implements Settler, Service.settle
// uses it and both writes share one transaction. When it does not, the
// service falls back to two sequential writes, which is correct on a happy
// path but leaves a window in which a crash records spend with no ledger row
// (or the reverse). An implementation that CAN be atomic SHOULD implement
// this; the pg adapter does.
type Settler interface {
	Settle(ctx context.Context, evt TokenUsageEvent, usdMicrosDelta int64) error
}

// PolicyLoader resolves the per-agent routing + guardrail policy at the
// start of every Invoke. Implementations may cache aggressively in memory
// (policies change rarely — at deploy time of the config file).
type PolicyLoader interface {
	// ResolveAgentPolicy resolves the routing policy for one Invoke.
	// requestedModel is the caller's primary logical model; fallbackModels
	// is the agent-declared ordered fallback chain, which the loader
	// translates into AgentPolicy.FallbackChain entries. Each fallback id
	// is resolved through the registry exactly like the primary, so a chain
	// may legitimately cross providers. Empty fallbackModels = single-shot.
	ResolveAgentPolicy(ctx context.Context, agentID, crewKind string, requestedModel LogicalModelID, fallbackModels []LogicalModelID) (AgentPolicy, error)
}

// SecretClient resolves the credential for a resolved model target.
// Implementations MUST read from the environment (or an equivalent
// out-of-band source) — never from a value baked into source.
type SecretClient interface {
	// ResolveCredential returns the credential for the named secret/env
	// reference. Returns ("", nil) when the reference is unset, which the
	// service turns into a loud refusal (a configured-but-empty credential
	// is a misconfiguration, not a free request).
	ResolveCredential(ctx context.Context, ref string) (string, error)
}

// ClaimRecorder is the keyed-idempotency seam: a redelivered dispatch with
// the same (gcid, key, action) must bill once.
type ClaimRecorder interface {
	// ClaimDebit reports whether this caller now owns the debit. It returns
	// true on a fresh claim and false when the key was already claimed, in
	// which case the service skips the budget debit but still completes the
	// call and still writes a ledger row.
	ClaimDebit(ctx context.Context, gcid, idempotencyKey, actionCode string) (bool, error)
}
