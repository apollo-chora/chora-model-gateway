package domain

import "time"

// TokenUsageEvent is the domain-shaped payload the gateway hands to the
// OutboxWriter port. The outbox publisher serialises this into the
// canonical chora.observability.token_usage.recorded.v1 Pub/Sub envelope
// (per chora-contracts/proto/events/observability/token_usage.proto +
// the ADR-163 additive v1 fields).
//
// Field names mirror the proto field names (snake_case → Go's
// CamelCase) so the events adapter is a near-1:1 mapping.
type TokenUsageEvent struct {
	// UUIDv7 — TokenUsageLedger entry identifier (same as InvocationID
	// post-cutover; pre-cutover the legacy ledger writer generated its
	// own value).
	UsageID string

	TenantID string
	GCID     string

	// Resolved model identifier post-optimizer / post-LoRA-resolution.
	ModelID string

	// Token + cost accounting.
	InputTokens  int64
	OutputTokens int64
	CachedTokens int64
	CostMicros   int64

	// Correlation to AI Kernel invocation chain.
	InvocationID string

	// Free-text role within the crew (e.g. "atom_validator", "router").
	AgentRole string

	// AgentID is the AI Kernel agent identity (req.AgentID) — stamped onto the
	// outbox row's agid column (migration 0006: "only AI Kernel agent emissions
	// carry it").
	AgentID string

	// ActionCode is the priced action label (chora.identity action_code) the
	// caller stamped on the turn. Carried onto the outbox event so the
	// observability projection can label usage by action. Empty when the
	// caller did not stamp one.
	ActionCode string

	// ManaUnits is the WS-1 umbrella-metered amount debited for the turn
	// (chora.identity mana). Zero when the turn was un-metered (no action_code
	// or an unpriced one) — the Python reference likewise defaults it to 0.
	ManaUnits int64

	RecordedAt time.Time

	// ADR-163 additive v1 fields.
	Vendor         string       // "vertex_ai_gemini", "openai_byoa", ...
	FallbackChain  []string     // ["vendor:model", ...] — length 1 on happy-path
	ArmorPre       ArmorVerdict // enum value, NOT a string — outbox serialises to proto enum
	ArmorPost      ArmorVerdict
	GatewayVersion string // e.g. "chora-model-gateway:58b28eb"

	// W3C trace context — propagated through Pub/Sub envelope so consumers
	// can correlate the cost-recorded event with the original gateway span.
	Traceparent string
	Tracestate  string
}
