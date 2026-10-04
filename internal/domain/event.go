package domain

import "time"

// TokenUsageEvent is the domain-shaped payload the gateway hands to the
// OutboxWriter port. The outbox row is the durable, per-call cost ledger:
// one row per completed dispatch, written in the same transaction as the
// budget debit.
type TokenUsageEvent struct {
	// UUIDv7 — the ledger entry identifier. Equal to InvocationID, which
	// makes it the natural idempotency key: a retry of the same logical call
	// collapses onto one row instead of double-billing.
	UsageID string

	TenantID string
	GCID     string

	// Upstream model actually dispatched (post-registry resolution).
	ModelID string

	// Token + cost accounting.
	InputTokens  int64
	OutputTokens int64
	CachedTokens int64
	CostMicros   int64

	// Correlation to the invocation chain.
	InvocationID string

	// Free-text role within the crew (e.g. "atom_validator", "router").
	AgentRole string

	// Surface the call was attributed to; empty records as "unspecified".
	Surface string

	// Requested output modality: "" / TEXT / IMAGE. Empty for embeddings.
	Modality string

	RecordedAt time.Time

	// Provider family that served the call ("openai", "anthropic").
	Vendor string

	// Every target the gateway tried, as "vendor:model". Length 1 on the
	// happy path; longer means a fallback fired.
	FallbackChain []string

	// DebitDeduped is true when the dispatch key had already been claimed,
	// so this row records usage without a second budget debit.
	DebitDeduped bool

	// GatewayVersion is the build that handled the call.
	GatewayVersion string

	// W3C trace context, so a cost row can be correlated with the gateway
	// span that produced it.
	Traceparent string
	Tracestate  string
}
