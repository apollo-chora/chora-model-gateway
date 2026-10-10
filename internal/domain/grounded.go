package domain

import (
	"fmt"
	"time"
)

// =============================================================================
// GroundedSearch — the platform's single controlled web-egress call (ADR-231).
//
// The gateway grounds via Vertex AI "Grounding with Google Search" behind the
// SAME governed chain as Invoke, plus a fail-closed per-tenant external_egress
// gate. Types here are pure domain values (no proto/infra imports) — the grpc
// adapter maps proto ⇄ domain at the wire boundary.
// =============================================================================

// GroundedSearchRequest is the domain-shaped input to Service.GroundedSearch.
// Mirrors chora.services.model_gateway.v1.GroundedSearchRequest.
type GroundedSearchRequest struct {
	// Caller-supplied UUIDv7. If empty the service generates one + echoes it.
	// Idempotency key: a retried Seeker turn collapses to one ledger row + one
	// mana debit + one egress audit event.
	InvocationID string

	// REQUIRED — tenant scope. Drives the external_egress entitlement gate,
	// budget gate, RLS, metering, and Pub/Sub attribution. UUIDv7.
	TenantID string

	// REQUIRED — learner GCID. The per-GCID mana debit lands here. UUIDv7.
	GCID string

	// REQUIRED — calling skill identifier (e.g. "familiar_seeker"). Mesh authz
	// allow-lists which agent_id may reach GroundedSearch (Seeker-only egress).
	AgentID string

	// OPTIONAL — crew classification (e.g. "familiar").
	CrewKind string

	// REQUIRED — the screened, learner-pointed directive (a ConceptNode label
	// or ≤120-char free-text query). Armor PRE-INSPECTS it before egress.
	Directive string

	// OPTIONAL — citation cap. Default DefaultMaxResults; hard-capped at
	// MaxCitationsHardCap.
	MaxResults int32

	// OPTIONAL — grounding-capable logical model id (default resolved via the
	// agent policy / DefaultGroundedModel).
	LogicalModelID LogicalModelID

	// REQUIRED — the high-price external_egress mana action_code the gateway
	// debits (ADR-177 sole meter). An un-priced egress is not permitted.
	ActionCode string

	// REQUIRED for tracing — W3C traceparent.
	Traceparent string

	// OPTIONAL — W3C tracestate.
	Tracestate string
}

// GroundedSearchResult is the domain-shaped output of Service.GroundedSearch.
// Mirrors chora.services.model_gateway.v1.GroundedSearchResponse.
type GroundedSearchResult struct {
	InvocationID string

	// Structured citations (IMDA D2 mandate). EMPTY ⇒ the caller MUST hedge
	// (ADR-220 D3) — never presented as familiar knowledge.
	Citations []GroundedCitation

	// The model's grounded answer synthesised over the cited sources. Screened
	// by Armor POST. OPTIONAL for the caller (fact_check runs its own fenced
	// verify turn over Citations).
	GroundedAnswer string

	// Google-mandated Search-Suggestions chip HTML (searchEntryPoint
	// renderedContent). The Far Sight FE MUST render it when non-empty (D5).
	SearchEntryPointHTML string

	// The web search queries the model issued (O+ transparency).
	WebSearchQueries []string

	Usage        TokenUsage
	Vendor       string
	ModelVersion string
	ArmorPre     ArmorVerdict
	ArmorPost    ArmorVerdict
	LatencyMs    int32
	FinishReason FinishReason
	FinishDetail string
	CompletedAt  time.Time

	GatewayVersion string
}

// GroundedCitation is one renderable source (IMDA D2). URI+Title are the
// citation surface — a hit counts toward the mandate only when BOTH are set.
type GroundedCitation struct {
	// Vertex grounding-api-redirect URL — EPHEMERAL (~30-day expiry, ADR-231
	// D4). Persist Domain+Title durably; treat URI as an expiring convenience.
	URI string

	// Source page title.
	Title string

	// Publisher domain (e.g. "nature.com") — render THIS (durable, ADR-231 D4).
	Domain string

	// Grounded evidence segment (best-effort from groundingSupports; may be
	// empty). The mandate never depends on Snippet.
	Snippet string

	// groundingSupports confidence for this citation (0..1; 0 when absent).
	Confidence float32
}

// HasCitations reports whether the result carries at least one renderable
// citation (URI+Title both set). Mirrors the consumer-side mandate: EMPTY ⇒
// the caller must hedge (ADR-220 D3).
func (r GroundedSearchResult) HasCitations() bool {
	for _, c := range r.Citations {
		if c.URI != "" && c.Title != "" {
			return true
		}
	}
	return false
}

// Grounded-search config defaults (no inline magic numbers scattered in the
// chain). The hard cap bounds citation breadth regardless of caller input.
const (
	// DefaultMaxResults is the citation cap when the caller omits MaxResults.
	DefaultMaxResults int32 = 5
	// MaxCitationsHardCap is the ceiling the gateway enforces on MaxResults.
	MaxCitationsHardCap int32 = 10
	// DefaultGroundedModel is the default grounded-answer model (ADR-231 D6).
	// The grounded chain no longer rides a grounding-capable model's own web
	// tool: the exa vendor does Exa retrieval + this model synthesises the
	// grounded answer. Grounding is web-search capability, not a model
	// feature, so a budget downgrade is NOT applied here.
	DefaultGroundedModel LogicalModelID = "longcat-2.5-preview"
)

// GroundedVendorRequest is the contract between Service.GroundedSearch and the
// grounded vendor port (GroundedVendorClient.GroundedGenerate).
type GroundedVendorRequest struct {
	LogicalModelID LogicalModelID
	// Directive is the Armor-screened learner directive.
	Directive string
	// MaxResults is the (hard-capped) citation cap.
	MaxResults int32
	// SystemPrompt optionally carries a citation-mandate system instruction.
	SystemPrompt string
	Traceparent  string
	Tracestate   string
	TenantID     string
}

// GroundedVendorResponse is what the grounded vendor port returns.
type GroundedVendorResponse struct {
	// Answer is the grounded completion synthesised over the sources.
	Answer string
	// Citations are the structured sources parsed from groundingMetadata.
	Citations []GroundedCitation
	// SearchEntryPointHTML is the Google Search-Suggestions chip HTML.
	SearchEntryPointHTML string
	// WebSearchQueries are the queries the model issued.
	WebSearchQueries []string
	Usage            TokenUsage
	ModelVersion     string
	FinishReason     FinishReason
	FinishDetail     string
}

// EgressAuthorization is the ExternalEgressGate verdict.
type EgressAuthorization struct {
	// Allowed reports whether grounded web egress is permitted right now.
	Allowed bool
	// Reason is a machine token explaining a deny (empty when Allowed). One of
	// the EgressDenyReason* constants.
	Reason string
}

// EgressDenyReason machine tokens — surfaced on the audit event's denial_reason
// and mapped to gRPC status by the adapter.
const (
	EgressDenyKillSwitch  = "kill_switch_engaged"
	EgressDenyDisabled    = "external_egress_disabled"
	EgressDenyDailyCeil   = "daily_ceiling_exceeded"
	EgressDenyBudget      = "budget_exhausted"
	EgressDenyManaBlock   = "insufficient_mana"
	EgressDenyArmorPre    = "model_armor_pre_block"
	EgressDenyArmorPost   = "model_armor_post_block"
	EgressDenyVendorError = "vendor_unavailable"
	EgressDenyZeroCite    = "zero_citations"
	// EgressDenyOverloaded — the tenant is at its per-tenant concurrency
	// ceiling (CHORA_LLM_MAX_CONCURRENT_PER_TENANT). Nothing was dispatched and
	// nothing was billed; the caller retries later.
	EgressDenyOverloaded = "tenant_concurrency_limit"
)

// EgressAuditResult mirrors chora.governance.v1.AuditResult — the outcome of a
// grounded egress for the external_egress audit trail (ADR-231 D6). Kept as a
// domain value; the pg adapter maps it to the governance proto enum.
type EgressAuditResult int

const (
	EgressAuditUnspecified EgressAuditResult = 0
	EgressAuditAllowed     EgressAuditResult = 1
	EgressAuditDenied      EgressAuditResult = 2
	EgressAuditAnomaly     EgressAuditResult = 3
)

// ExternalEgressAuditEvent is the domain-shaped payload the gateway hands to
// the EgressAuditWriter port. The outbox publisher serialises it into the
// chora.governance.audit.external_egress.v1 Pub/Sub envelope.
//
// PRIVACY: the raw directive is NEVER persisted — only DirectiveHash rides the
// trail. WebSearchQueries (model-emitted) ARE recorded for O+ transparency.
type ExternalEgressAuditEvent struct {
	AuditID  string // UUIDv7
	TenantID string
	// ActorGCID — the learner GCID that triggered the egress.
	ActorGCID string
	AgentID   string
	// ActionCode — the metered high-price egress action.
	ActionCode string
	// DirectiveHash — SHA-256 hex of the screened directive (not the raw text).
	DirectiveHash string
	// WebSearchQueries — the queries the model issued (empty on pre-egress deny).
	WebSearchQueries []string
	// CitationCount — renderable citations returned (0 ⇒ hedge or denied).
	CitationCount int32
	Result        EgressAuditResult
	// DenialReason — machine token for a DENIED/ANOMALY result (empty when ALLOWED).
	DenialReason string
	// ArmorVerdictPre/Post — string tokens mirroring ArmorVerdict.String()
	// ("allow"|"block"|"sanitise"|"error"|"bypassed"); governance never imports
	// a service proto so these ride as strings.
	ArmorVerdictPre  string
	ArmorVerdictPost string
	Vendor           string
	ModelVersion     string
	OccurredAt       time.Time
	Traceparent      string
	Tracestate       string
}

// ManaQuote is the result of a dry-run mana affordability check (domain view).
type ManaQuote struct {
	Affordable     bool
	RequiredUnits  int64
	AvailableUnits int64
	// UnknownAction ⇒ the action_code is unpriced (treated as un-metered).
	UnknownAction bool
}

// ManaDebit is the result of an actual mana debit (domain view).
type ManaDebit struct {
	Success           bool
	RequiredUnits     int64
	BalanceAfterUnits int64
	UnknownAction     bool
}

// GroundedSearchError categorises terminal GroundedSearch failures the gRPC
// adapter maps to a status code. In-band outcomes (mana block, zero citations)
// are NOT errors — they return a GroundedSearchResult with a finish reason /
// empty citations.
type GroundedSearchError struct {
	// Reason is one of the EgressDenyReason* / EgressDeny* tokens above.
	Reason string
	Detail string
	Inner  error
}

func (e *GroundedSearchError) Error() string {
	if e.Inner != nil {
		return fmt.Sprintf("grounded_search %s: %s: %v", e.Reason, e.Detail, e.Inner)
	}
	return fmt.Sprintf("grounded_search %s: %s", e.Reason, e.Detail)
}

func (e *GroundedSearchError) Unwrap() error { return e.Inner }
